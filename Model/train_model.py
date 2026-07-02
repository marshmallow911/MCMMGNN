from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from sklearn.metrics import average_precision_score, roc_auc_score

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model import IrregularLatentStateHeteroGNN

EdgeType = Tuple[str, str, str]

FLOW_KEYWORDS = (
    "amount",
    "burn",
    "collect",
    "count",
    "delta",
    "fee0_delta",
    "fee1_delta",
    "feesusd",
    "flow",
    "growth",
    "mint",
    "net_flow",
    "route_edge_count",
    "route_edge_volume",
    "swap",
    "tx",
    "unique",
    "volume",
    "withdraw",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_payload(dataset_path: Path) -> Dict:
    return torch.load(dataset_path, map_location="cpu")


def _is_flow_col(name: str) -> bool:
    lower = name.lower()
    if lower == "feetier":
        return False
    return any(key in lower for key in FLOW_KEYWORDS)


def _edge_feature_key(edge_type: EdgeType) -> str:
    src, _, dst = edge_type
    pair = {src, dst}
    if pair == {"position", "pool"}:
        return "edge_position_pool"
    if pair == {"pool", "token"}:
        return "edge_pool_token"
    if src == "pool" and dst == "pool":
        return "edge_pool_pool_route"
    return ""


def build_flow_idx_dict(feature_cols: Mapping[str, List[str]], sample) -> Dict[Union[str, EdgeType], List[int]]:
    flow_idx: Dict[Union[str, EdgeType], List[int]] = {}
    for node_type in sample.node_types:
        cols = feature_cols.get(node_type, [])
        flow_idx[node_type] = [i for i, col in enumerate(cols) if _is_flow_col(col)]
    for edge_type in sample.edge_types:
        cols = feature_cols.get(_edge_feature_key(edge_type), [])
        flow_idx[edge_type] = [i for i, col in enumerate(cols) if _is_flow_col(col)]
    return flow_idx


def valid_indices(indices: Sequence[int], time_window: int) -> List[int]:
    return [idx for idx in indices if idx - time_window + 1 >= 0]


def make_window(snapshots: Sequence, idx: int, time_window: int) -> List:
    return list(snapshots[idx - time_window + 1 : idx + 1])


def make_window_batch(snapshots: Sequence, indices: Sequence[int], time_window: int) -> List:
    if len(indices) == 1:
        return make_window(snapshots, indices[0], time_window)
    # For each relative time offset, batch the same batch item order. PyG creates
    # node-store .batch vectors [0, ..., batch_size-1], and model.py offsets
    # global_id by this batch id before aligning recurrent caches/states.
    return [
        Batch.from_data_list([snapshots[idx - time_window + 1 + offset] for idx in indices])
        for offset in range(time_window)
    ]


def batched(iterable: Sequence[int], batch_size: int) -> List[List[int]]:
    return [list(iterable[i : i + batch_size]) for i in range(0, len(iterable), batch_size)]


def class_weights(snapshots: Sequence, indices: Sequence[int], node_type: str, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(2, dtype=torch.float32)
    for idx in indices:
        store = snapshots[idx][node_type]
        if not hasattr(store, "y") or not hasattr(store, "label_mask"):
            continue
        mask = store.label_mask.bool()
        if not mask.any():
            continue
        y = store.y[mask].long()
        counts += torch.bincount(y.clamp_min(0), minlength=2).float()[:2]
    weights = counts.sum().clamp_min(1.0) / (counts.clamp_min(1.0) * 2.0)
    return weights.to(device)


def class_logits_loss(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(logits, y.long().view(-1), weight=weight, reduction=reduction)


def multitask_loss(
    out: Dict[str, torch.Tensor],
    target,
    pool_weight: torch.Tensor,
    position_weight: torch.Tensor,
    position_loss_weight: float,
) -> torch.Tensor:
    losses = []
    if hasattr(target["pool"], "label_mask") and target["pool"].label_mask.any():
        mask = target["pool"].label_mask.to(out["pool_pred"].device).bool()
        y = target["pool"].y.to(out["pool_pred"].device).long()
        losses.append(class_logits_loss(out["pool_pred"][mask], y[mask], pool_weight))
    if position_loss_weight > 0 and hasattr(target["position"], "label_mask") and target["position"].label_mask.any():
        mask = target["position"].label_mask.to(out["position_pred"].device).bool()
        y = target["position"].y.to(out["position_pred"].device).long()
        losses.append(position_loss_weight * class_logits_loss(out["position_pred"][mask], y[mask], position_weight))
    if not losses:
        return out["pool_pred"].sum() * 0.0
    return sum(losses)


def batch_loss(
    model: nn.Module,
    snapshots: Sequence,
    idx_batch: Sequence[int],
    time_window: int,
    pool_weight: torch.Tensor,
    position_weight: torch.Tensor,
    position_loss_weight: float,
) -> torch.Tensor:
    window = make_window_batch(snapshots, idx_batch, time_window)
    out = model(window)
    return multitask_loss(out, window[-1], pool_weight, position_weight, position_loss_weight)


def binary_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if logits.dim() == 2 and logits.size(-1) == 2:
        prob = torch.softmax(logits, dim=-1)[:, 1]
        pred = torch.argmax(logits, dim=-1).long()
    else:
        logits = logits.view(-1)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()
    y = y.long().view(-1)
    correct = (pred == y).sum().item()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    y_np = y.cpu().numpy()
    prob_np = prob.detach().cpu().numpy()
    try:
        roc_auc = float(roc_auc_score(y_np, prob_np))
        pr_auc = float(average_precision_score(y_np, prob_np))
    except ValueError:
        roc_auc = float("nan")
        pr_auc = float("nan")
    return {
        "acc": correct / max(int(y.numel()), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n": int(y.numel()),
    }


def format_metric(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def finite_metric(value: float, default: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else default


def validation_selection_score(metrics: Mapping[str, float], position_loss_weight: float) -> float:
    return finite_metric(metrics.get("pool_pr_auc", float("nan"))) + position_loss_weight * finite_metric(
        metrics.get("position_pr_auc", float("nan"))
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, transition_lr_mult: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    transition_params = []
    transition_prefixes = (
        "latent_transition.C",
        "latent_transition.retention_logit.",
        "latent_transition.message_beta_logit.",
        "latent_transition.rollout_beta_logit",
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(transition_prefixes):
            transition_params.append(param)
        elif param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    if transition_params:
        groups.append({"params": transition_params, "lr": lr * transition_lr_mult, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def nonfinite_grad_report(model: nn.Module, limit: int = 12) -> str:
    bad = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad)
        if finite.all():
            continue
        grad_detached = grad.detach()
        bad.append(
            f"{name}: nan={int(torch.isnan(grad_detached).sum().item())} "
            f"inf={int(torch.isinf(grad_detached).sum().item())} "
            f"shape={tuple(grad_detached.shape)}"
        )
        if len(bad) >= limit:
            break
    return "; ".join(bad) if bad else "no non-finite parameter gradients found"


def print_run_config(args: argparse.Namespace, model: nn.Module, train_size: int, val_size: int, flow_idx_dict) -> None:
    param_count = count_parameters(model)
    config = {
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "time_window": args.time_window,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size or args.batch_size,
        "max_train_batches": args.max_train_batches,
        "detect_anomaly": args.detect_anomaly,
        "hidden_dim": args.hidden_dim,
        "state_steps": args.state_steps,
        "state_grad_steps": args.state_grad_steps,
        "snapshot_encoder": "light",
        "c_transition_mask": False,
        "state_predictor": False,
        "encoder_dropout": args.encoder_dropout,
        "transition_dropout": args.transition_dropout,
        "readout_dropout": args.readout_dropout,
        "head_dropout": args.head_dropout,
        "lr": args.lr,
        "transition_lr_mult": args.transition_lr_mult,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "position_loss_weight": args.position_loss_weight,
        "early_stop_patience": args.early_stop_patience,
        "flow_split": True,
        "device": args.device,
        "seed": args.seed,
        "train_windows": train_size,
        "val_windows": val_size,
        "test_after_train": True,
        "num_node_types": len(getattr(model, "node_types", [])),
        "num_edge_types": len(getattr(model, "edge_types", [])),
        "parameters_total": param_count["total"],
        "parameters_trainable": param_count["trainable"],
    }
    if flow_idx_dict is not None:
        config["flow_dims"] = {str(k): len(v) for k, v in flow_idx_dict.items()}
    print("run_config:")
    print(json.dumps(config, indent=2, sort_keys=True))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    snapshots: Sequence,
    indices: Sequence[int],
    time_window: int,
    device: torch.device,
    batch_size: int,
    position_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_by_type = {"pool": [], "position": []}
    y_by_type = {"pool": [], "position": []}
    valid = valid_indices(indices, time_window)
    for idx_batch in batched(valid, batch_size):
        window = make_window_batch(snapshots, idx_batch, time_window)
        target = window[-1]
        out = model(window)
        for node_type, key, weight in [
            ("pool", "pool_pred", 1.0),
            ("position", "position_pred", position_loss_weight),
        ]:
            if weight <= 0:
                continue
            store = target[node_type]
            if not hasattr(store, "label_mask") or not store.label_mask.any():
                continue
            mask = store.label_mask.to(device).bool()
            y = store.y.to(device).long()
            logits = out[key][mask]
            loss = class_logits_loss(logits, y[mask], reduction="sum") * weight
            total_loss += float(loss.item())
            total_examples += int(mask.sum().item())
            logits_by_type[node_type].append(logits.cpu())
            y_by_type[node_type].append(y[mask].cpu())
    metrics = {"loss": total_loss / max(total_examples, 1)}
    for node_type in ["pool", "position"]:
        if y_by_type[node_type]:
            node_metrics = binary_metrics(torch.cat(logits_by_type[node_type], dim=0), torch.cat(y_by_type[node_type], dim=0))
            metrics.update({f"{node_type}_{name}": value for name, value in node_metrics.items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Irregular Latent-State HeteroGNN on Uniswap HeteroData snapshots.")
    parser.add_argument("--dataset-path", default="pull_data/uniswap_pyg_dataset_precise_masks/uniswap_hetero_snapshots_h6.pt") # pull_data\uniswap_pyg_dataset_precise_masks\uniswap_hetero_snapshots_h6.pt
    parser.add_argument("--output-dir", default="MCMMGNN/checkpoints/TRSM")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--time-window", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--state-steps", type=int, default=1)
    parser.add_argument("--state-grad-steps", type=int, default=0, help="With detached recurrent caches, keep gradients through this many recent transitions inside each window.")
    parser.add_argument("--encoder-dropout", type=float, default=0.1, help="Dropout inside raw node/edge feature encoders.")
    parser.add_argument("--transition-dropout", type=float, default=0.2, help="Dropout inside TRSM relation-message MLPs and route rollout.")
    parser.add_argument("--readout-dropout", type=float, default=0.3, help="Dropout inside slot-attention scoring MLPs.")
    parser.add_argument("--head-dropout", type=float, default=0.5, help="Dropout inside final pool/position prediction heads.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transition-lr-mult", type=float, default=4.0, help="LR multiplier for C/rho/beta transition-control parameters; these parameters use weight_decay=0.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Clip global gradient norm. Use 0 to disable.")
    parser.add_argument("--position-loss-weight", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=15, help="Stop after this many validation checks without val loss improvement. Use 0 to disable.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of target-hour windows per optimizer step. Windows are PyG-batched by relative time offset.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug option: limit train batches per epoch.")
    parser.add_argument("--detect-anomaly", action="store_true", help="Enable torch autograd anomaly detection for debugging NaN gradients.")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    payload = load_payload(Path(args.dataset_path))
    snapshots = payload["snapshots"]
    splits = payload["splits"]
    flow_idx_dict = build_flow_idx_dict(payload.get("feature_cols", {}), snapshots[0])
    train_indices = valid_indices(splits["train"], args.time_window)
    val_indices = valid_indices(splits["val"], args.time_window)
    test_indices = valid_indices(splits["test"], args.time_window)
    eval_batch_size = args.eval_batch_size or args.batch_size

    device = torch.device(args.device)
    model = IrregularLatentStateHeteroGNN.from_sample_data(
        snapshots[0],
        hidden_dim=args.hidden_dim,
        out_dim_pool=2,
        out_dim_position=2,
        flow_idx_dict=flow_idx_dict,
        encoder_dropout=args.encoder_dropout,
        transition_dropout=args.transition_dropout,
        readout_dropout=args.readout_dropout,
        head_dropout=args.head_dropout,
        state_steps=args.state_steps,
        state_grad_steps=args.state_grad_steps,
    ).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay, args.transition_lr_mult)
    pool_weight = class_weights(snapshots, train_indices, "pool", device)
    # pool_weight = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    position_weight = class_weights(snapshots, train_indices, "position", device)
    print_run_config(args, model, len(train_indices), len(val_indices), flow_idx_dict)
    print(f"pool_class_weight={pool_weight.detach().cpu().tolist()} position_class_weight={position_weight.detach().cpu().tolist()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_score = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    last_epoch = 0
    history = []
    epoch_train_times = []
    val_inference_times = []
    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        synchronize_if_cuda(device)
        train_start_time = time.perf_counter()
        model.train()
        random.shuffle(train_indices)
        total_loss = 0.0
        train_batches = batched(train_indices, args.batch_size)
        if args.max_train_batches is not None:
            train_batches = train_batches[: max(int(args.max_train_batches), 0)]
        for step, idx_batch in enumerate(train_batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            if args.detect_anomaly:
                with torch.autograd.detect_anomaly():
                    loss = batch_loss(
                        model,
                        snapshots,
                        idx_batch,
                        args.time_window,
                        pool_weight,
                        position_weight,
                        args.position_loss_weight,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss at epoch={epoch} step={step} idx_batch={list(idx_batch)} "
                            f"loss={float(loss.detach().cpu())}"
                        )
                    loss.backward()
            else:
                loss = batch_loss(
                    model,
                    snapshots,
                    idx_batch,
                    args.time_window,
                    pool_weight,
                    position_weight,
                    args.position_loss_weight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch={epoch} step={step} idx_batch={list(idx_batch)} "
                        f"loss={float(loss.detach().cpu())}"
                    )
                loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"non-finite grad norm at epoch={epoch} step={step} "
                        f"idx_batch={list(idx_batch)} grad_norm={float(grad_norm.detach().cpu())}; "
                        f"{nonfinite_grad_report(model)}"
                    )

            optimizer.step()
            total_loss += float(loss.item())
            if step % 100 == 0:
                seen = min(step * args.batch_size, len(train_indices))
                print(
                    f"epoch={epoch} step={step}/{len(train_batches)} "
                    f"windows={seen}/{len(train_indices)} train_loss={total_loss / step:.6f}"
                )

        synchronize_if_cuda(device)
        train_time_sec = time.perf_counter() - train_start_time
        epoch_train_times.append(train_time_sec)
        row = {"epoch": epoch, "train_loss": total_loss / max(len(train_batches), 1)}
        row["train_time_sec"] = train_time_sec
        if epoch % args.eval_every == 0:
            synchronize_if_cuda(device)
            val_inference_start_time = time.perf_counter()
            val_metrics = evaluate(model, snapshots, val_indices, args.time_window, device, eval_batch_size, args.position_loss_weight)
            synchronize_if_cuda(device)
            val_inference_time_sec = time.perf_counter() - val_inference_start_time
            val_inference_times.append(val_inference_time_sec)
            val_score = validation_selection_score(val_metrics, args.position_loss_weight)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["val_inference_time_sec"] = val_inference_time_sec
            row["val_selection_score"] = val_score
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_score={val_score:.6f} "
                f"val_pool_acc={val_metrics.get('pool_acc', 0.0):.4f} "
                f"val_pool_f1={format_metric(val_metrics.get('pool_f1', float('nan')))} "
                f"val_pool_prauc={format_metric(val_metrics.get('pool_pr_auc', float('nan')))} "
                f"val_pool_rocauc={format_metric(val_metrics.get('pool_roc_auc', float('nan')))} "
                f"val_position_acc={val_metrics.get('position_acc', 0.0):.4f} "
                f"val_position_f1={format_metric(val_metrics.get('position_f1', float('nan')))} "
                f"val_position_prauc={format_metric(val_metrics.get('position_pr_auc', float('nan')))} "
                f"val_position_rocauc={format_metric(val_metrics.get('position_roc_auc', float('nan')))}"
            )
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "best_val_score": best_val_score,
                        "best_val_score_metric": "val_pool_pr_auc + position_loss_weight * val_position_pr_auc",
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                    },
                    output_dir / "best.pt",
                )
                print(f"saved new best checkpoint: epoch={best_epoch} val_score={best_val_score:.6f} val_loss={best_val_loss:.6f}")
            else:
                epochs_without_improvement += 1
                if args.early_stop_patience > 0:
                    print(
                        f"early_stop patience={epochs_without_improvement}/{args.early_stop_patience} "
                        f"best_epoch={best_epoch} best_val_score={best_val_score:.6f} best_val_loss={best_val_loss:.6f}"
                    )
        history.append(row)
        with (output_dir / "history.json").open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"early stopping at epoch={epoch}: no val score improvement for "
                f"{epochs_without_improvement} validation checks"
            )
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "best_val_score": best_val_score,
            "best_val_score_metric": "val_pool_pr_auc + position_loss_weight * val_position_pr_auc",
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "last_epoch": last_epoch,
        },
        output_dir / "last.pt",
    )
    print(f"saved checkpoints to {output_dir}")
    completed_epochs = len(epoch_train_times)
    avg_epoch_train_time_sec = sum(epoch_train_times) / max(completed_epochs, 1)
    avg_val_inference_time_per_eval_sec = sum(val_inference_times) / max(len(val_inference_times), 1)
    avg_val_inference_time_per_epoch_sec = sum(val_inference_times) / max(completed_epochs, 1)
    print(
        f"efficiency_summary completed_epochs={completed_epochs} "
        f"avg_epoch_train_time_sec={avg_epoch_train_time_sec:.6f} "
        f"val_inference_calls={len(val_inference_times)} "
        f"avg_val_inference_time_per_eval_sec={avg_val_inference_time_per_eval_sec:.6f} "
        f"avg_val_inference_time_per_epoch_sec={avg_val_inference_time_per_epoch_sec:.6f}"
    )

    ckpt_path = output_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = output_dir / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, snapshots, test_indices, args.time_window, device, eval_batch_size, args.position_loss_weight)
    print(
        f"test_checkpoint={ckpt_path} "
        f"test_loss={test_metrics['loss']:.6f} "
        f"test_pool_acc={test_metrics.get('pool_acc', 0.0):.4f} "
        f"test_pool_f1={format_metric(test_metrics.get('pool_f1', float('nan')))} "
        f"test_pool_prauc={format_metric(test_metrics.get('pool_pr_auc', float('nan')))} "
        f"test_pool_rocauc={format_metric(test_metrics.get('pool_roc_auc', float('nan')))} "
        f"test_position_acc={test_metrics.get('position_acc', 0.0):.4f} "
        f"test_position_f1={format_metric(test_metrics.get('position_f1', float('nan')))} "
        f"test_position_prauc={format_metric(test_metrics.get('position_pr_auc', float('nan')))} "
        f"test_position_rocauc={format_metric(test_metrics.get('position_roc_auc', float('nan')))}"
    )
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, indent=2)


if __name__ == "__main__":
    main()

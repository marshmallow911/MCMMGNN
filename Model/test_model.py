from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import torch
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
    return [
        Batch.from_data_list([snapshots[idx - time_window + 1 + offset] for idx in indices])
        for offset in range(time_window)
    ]


def batched(iterable: Sequence[int], batch_size: int) -> List[List[int]]:
    return [list(iterable[i : i + batch_size]) for i in range(0, len(iterable), batch_size)]


def class_logits_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y.long().view(-1), reduction="sum")


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


@torch.no_grad()
def evaluate(
    model,
    snapshots: Sequence,
    indices: Sequence[int],
    time_window: int,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    logits_by_type = {"pool": [], "position": []}
    y_by_type = {"pool": [], "position": []}
    loss_sum = {"pool": 0.0, "position": 0.0}
    n_sum = {"pool": 0, "position": 0}
    valid = valid_indices(indices, time_window)
    for idx_batch in batched(valid, batch_size):
        window = make_window_batch(snapshots, idx_batch, time_window)
        target = window[-1]
        out = model(window)
        for node_type, key in [("pool", "pool_pred"), ("position", "position_pred")]:
            store = target[node_type]
            if not hasattr(store, "label_mask") or not store.label_mask.any():
                continue
            mask = store.label_mask.to(device).bool()
            y = store.y.to(device).long()[mask]
            logits = out[key][mask]
            loss_sum[node_type] += float(class_logits_loss(logits, y).item())
            n_sum[node_type] += int(y.numel())
            logits_by_type[node_type].append(logits.cpu())
            y_by_type[node_type].append(y.cpu())

    result: Dict[str, Dict[str, float]] = {}
    for node_type in ["pool", "position"]:
        if y_by_type[node_type]:
            logits = torch.cat(logits_by_type[node_type], dim=0)
            y = torch.cat(y_by_type[node_type], dim=0)
            result[node_type] = binary_metrics(logits, y)
            result[node_type]["loss"] = loss_sum[node_type] / max(n_sum[node_type], 1)
        else:
            result[node_type] = {
                "acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "loss": 0.0,
                "n": 0,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained Irregular Latent-State HeteroGNN on Uniswap snapshots.")
    parser.add_argument("--dataset-path", default="pull_data/uniswap_pyg_dataset_precise_masks/uniswap_hetero_snapshots_h6.pt")
    parser.add_argument("--checkpoint", default="MCMMGNN/checkpoints/TRSM/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of target-hour windows per PyG-batched evaluation step.")
    parser.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-flow-split", action="store_true", help="Disable automatic flow_idx_dict construction; all features are decayed.")
    args = parser.parse_args()

    payload = torch.load(args.dataset_path, map_location="cpu")
    snapshots = payload["snapshots"]
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print(
        "checkpoint "
        f"path={args.checkpoint} "
        f"best_epoch={ckpt.get('best_epoch')} "
        f"best_val_loss={ckpt.get('best_val_loss')} "
        f"last_epoch={ckpt.get('last_epoch')}"
    )
    train_args = ckpt.get("args", {})
    time_window = int(train_args.get("time_window", 6))
    indices = valid_indices(payload["splits"][args.split], time_window)
    if args.max_snapshots is not None:
        indices = indices[: args.max_snapshots]
    no_flow_split = bool(train_args.get("no_flow_split", args.no_flow_split))
    flow_idx_dict = None if no_flow_split else build_flow_idx_dict(payload.get("feature_cols", {}), snapshots[0])

    device = torch.device(args.device)
    fallback_dropout = float(train_args.get("dropout", 0.0))
    model = IrregularLatentStateHeteroGNN.from_sample_data(
        snapshots[0],
        hidden_dim=int(train_args.get("hidden_dim", 64)),
        out_dim_pool=int(train_args.get("out_dim_pool", 2)),
        out_dim_position=int(train_args.get("out_dim_position", 2)),
        flow_idx_dict=flow_idx_dict,
        encoder_dropout=float(train_args.get("encoder_dropout", fallback_dropout)),
        transition_dropout=float(train_args.get("transition_dropout", fallback_dropout)),
        readout_dropout=float(train_args.get("readout_dropout", fallback_dropout)),
        head_dropout=float(train_args.get("head_dropout", fallback_dropout)),
        state_steps=int(train_args.get("state_steps", 1)),
        state_grad_steps=int(train_args.get("state_grad_steps", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    result = evaluate(model, snapshots, indices, time_window, device, args.batch_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

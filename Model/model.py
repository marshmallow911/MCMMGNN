from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


EdgeType = Tuple[str, str, str]
ALIGNMENT_STRIDE = 1_000_000


def _edge_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def _clean(x: Tensor) -> Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sign(x) * torch.log1p(torch.abs(x).clamp_max(1.0e12))


def _idx_tensor(idx: Optional[Sequence[int]], dim: int, device: torch.device) -> Tensor:
    if idx is None:
        return torch.arange(dim, device=device)
    return torch.as_tensor(idx, dtype=torch.long, device=device)


def _get_optional_column(store, name: str, n: int, device: torch.device, default: float) -> Tensor:
    value = getattr(store, name, None)
    if value is None:
        return torch.full((n, 1), default, device=device)
    value = torch.nan_to_num(value.float(), nan=default, posinf=default, neginf=default).to(device)
    if value.dim() == 1:
        value = value.unsqueeze(-1)
    if value.size(1) != 1:
        value = value[:, :1]
    return value


def _get_observed_mask(store, fallback: Tensor, n: int, device: torch.device) -> Tensor:
    value = getattr(store, "state_observed_mask", None)
    if value is None:
        return fallback
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=1.0, neginf=0.0).to(device)
    if value.dim() == 1:
        value = value.unsqueeze(-1)
    if value.size(1) != 1:
        value = value[:, :1]
    if value.size(0) != n:
        return fallback
    return value


def _alignment_id(store, device: torch.device) -> Optional[Tensor]:
    """Return ids that are unique inside a PyG Batch and stable across hours."""
    if not hasattr(store, "global_id"):
        return None
    gid = store.global_id.to(device).long()
    batch = getattr(store, "batch", None)
    if batch is None:
        return gid
    # PyG Batch concatenates graphs and keeps local global_id values unchanged.
    # Offset by graph id so equal global_id values from different batch items do
    # not collide when caches/states are aligned across a window.
    return batch.to(device).long() * ALIGNMENT_STRIDE + gid


def _edge_alignment_id(data: HeteroData, edge_type: EdgeType, device: torch.device) -> Optional[Tensor]:
    """Stable edge ids from source/destination node global ids.

    Edge caches must not be aligned by row number because edge order can change
    across hours and across PyG batches. If node global ids are unavailable,
    the caller should skip edge-cache reuse.
    """
    src_type, _, dst_type = edge_type
    src_node_id = _alignment_id(data[src_type], device)
    dst_node_id = _alignment_id(data[dst_type], device)
    if src_node_id is None or dst_node_id is None:
        return None
    edge_index = data[edge_type].edge_index.to(device)
    return src_node_id[edge_index[0]] * ALIGNMENT_STRIDE + dst_node_id[edge_index[1]]


def _align_by_global_id(
    cache_entry: Optional[Mapping[str, Tensor]],
    store,
    n: int,
    fallback: Tensor,
) -> Tensor:
    """Return previous values aligned to current local rows when global_id exists."""
    if cache_entry is None:
        return fallback.expand(n, -1)
    prev_value = cache_entry["value"].to(fallback.device)
    prev_gid = cache_entry.get("align_id")
    if prev_gid is None:
        prev_gid = cache_entry.get("global_id")
    cur_gid = _alignment_id(store, fallback.device)
    if prev_gid is None or cur_gid is None:
        if prev_value.size(0) == n:
            return prev_value
        return fallback.expand(n, -1)

    prev_gid = prev_gid.to(fallback.device).long()
    out = fallback.expand(n, -1).clone()
    if prev_gid.numel() == 0 or cur_gid.numel() == 0:
        return out
    order = torch.argsort(prev_gid)
    sorted_gid = prev_gid[order]
    pos = torch.searchsorted(sorted_gid, cur_gid)
    valid = (pos < sorted_gid.numel()) & (sorted_gid[pos.clamp_max(sorted_gid.numel() - 1)] == cur_gid)
    if valid.any():
        out[valid] = prev_value[order[pos[valid]]]
    return out


def _cache_value(value: Tensor, detach: bool) -> Tensor:
    return value.detach() if detach else value


def _state_cache(
    state_dict: Mapping[str, Mapping[str, Tensor]],
    detach: bool,
    previous_state: Optional[Mapping[str, Mapping[str, Tensor]]] = None,
) -> Dict[str, Dict[str, Tensor]]:
    cache: Dict[str, Dict[str, Tensor]] = {}
    for nt, entry in state_dict.items():
        value = _cache_value(entry["value"], detach)
        align_id = entry.get("align_id")
        prev_entry = None if previous_state is None else previous_state.get(nt)
        prev_align_id = None
        if prev_entry is not None:
            prev_align_id = prev_entry.get("align_id")
            if prev_align_id is None:
                prev_align_id = prev_entry.get("global_id")

        if align_id is None or prev_entry is None or prev_align_id is None:
            cached_entry = {"value": value}
            if align_id is not None:
                cached_entry["align_id"] = align_id.detach()
            cache[nt] = cached_entry
            continue

        prev_value = _cache_value(prev_entry["value"].to(value.device), detach)
        prev_align_id = prev_align_id.to(value.device).long()
        cur_align_id = align_id.to(value.device).long()
        combined_value = torch.cat([prev_value, value], dim=0)
        combined_align = torch.cat([prev_align_id, cur_align_id], dim=0)
        if combined_align.numel() == 0:
            cache[nt] = {"value": combined_value, "align_id": combined_align.detach()}
            continue

        priority = torch.cat(
            [
                torch.zeros(prev_align_id.numel(), dtype=torch.long, device=value.device),
                torch.ones(cur_align_id.numel(), dtype=torch.long, device=value.device),
            ],
            dim=0,
        )
        order = torch.argsort(combined_align * 2 + priority)
        sorted_align = combined_align[order]
        sorted_value = combined_value[order]
        keep = torch.ones(sorted_align.numel(), dtype=torch.bool, device=value.device)
        if sorted_align.numel() > 1:
            keep[:-1] = sorted_align[:-1] != sorted_align[1:]
        cache[nt] = {"value": sorted_value[keep], "align_id": sorted_align[keep].detach()}
    return cache


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        activate_last: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 0) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if activate_last or not is_last:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecayAwareFeatureEncoder(nn.Module):
    """Per-type irregular observation encoder for node and edge features.

    State features use exponential decay toward a learnable type mean:
        gamma = exp(-softplus(w_type) * delta_t)
        x_tilde = mask * x + (1 - mask) * (gamma * x_prev + (1 - gamma) * x_bar)

    Flow features are not decayed. The final encoded input is:
        MLP([x_tilde_state, x_flow, mask, delta_t, log1p(count)])
    """

    def __init__(
        self,
        node_input_dims: Mapping[str, int],
        edge_input_dims: Mapping[EdgeType, int],
        hidden_dim: int,
        state_idx_dict: Optional[Mapping[Union[str, EdgeType], Sequence[int]]] = None,
        flow_idx_dict: Optional[Mapping[Union[str, EdgeType], Sequence[int]]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_types = list(node_input_dims.keys())
        self.edge_types = list(edge_input_dims.keys())
        self.node_input_dims = dict(node_input_dims)
        self.edge_input_dims = dict(edge_input_dims)
        self.state_idx_dict = dict(state_idx_dict or {})
        self.flow_idx_dict = dict(flow_idx_dict or {})

        self.node_decay_w = nn.ParameterDict()
        self.node_bar = nn.ParameterDict()
        self.node_mlps = nn.ModuleDict()
        for nt, dim in self.node_input_dims.items():
            state_idx, flow_idx = self._resolve_indices(nt, dim, torch.device("cpu"))
            state_dim = int(state_idx.numel())
            flow_dim = int(flow_idx.numel())
            self.node_decay_w[nt] = nn.Parameter(torch.zeros(state_dim))
            self.node_bar[nt] = nn.Parameter(torch.zeros(state_dim))
            # self.node_mlps[nt] = MLP(state_dim + flow_dim + 3, hidden_dim, hidden_dim, dropout=dropout)
            self.node_mlps[nt] = MLP(state_dim + flow_dim, hidden_dim, hidden_dim, dropout=dropout)

        self.edge_decay_w = nn.ParameterDict()
        self.edge_bar = nn.ParameterDict()
        self.edge_mlps = nn.ModuleDict()
        for et, dim in self.edge_input_dims.items():
            key = _edge_key(et)
            state_idx, flow_idx = self._resolve_indices(et, dim, torch.device("cpu"))
            state_dim = int(state_idx.numel())
            flow_dim = int(flow_idx.numel())
            self.edge_decay_w[key] = nn.Parameter(torch.zeros(state_dim))
            self.edge_bar[key] = nn.Parameter(torch.zeros(state_dim))
            # self.edge_mlps[key] = MLP(state_dim + flow_dim + 3, hidden_dim, hidden_dim, dropout=dropout)
            self.edge_mlps[key] = MLP(state_dim + flow_dim, hidden_dim, hidden_dim, dropout=dropout)

    def _raw_indices(self, mapping: Mapping[Union[str, EdgeType], Sequence[int]], key: Union[str, EdgeType]) -> Optional[Sequence[int]]:
        value = mapping.get(key)
        if value is None and isinstance(key, tuple):
            value = mapping.get(_edge_key(key))
        return value

    def _resolve_indices(self, key: Union[str, EdgeType], dim: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        state_raw = self._raw_indices(self.state_idx_dict, key)
        flow_raw = self._raw_indices(self.flow_idx_dict, key)
        flow_idx = _idx_tensor(flow_raw, 0, device)
        # If only flow indices are provided, state indices are the complement.
        # This keeps flow columns out of the decayed state channel.
        if state_raw is None and flow_raw is not None:
            is_flow = torch.zeros(dim, dtype=torch.bool, device=device)
            is_flow[flow_idx] = True
            state_idx = torch.arange(dim, device=device)[~is_flow]
        else:
            state_idx = _idx_tensor(state_raw, dim, device)
        return state_idx, flow_idx

    def _split_indices(self, key: Union[str, EdgeType], dim: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        return self._resolve_indices(key, dim, device)

    def forward(
        self,
        data: HeteroData,
        decay_cache: Optional[Mapping[str, Dict[Union[str, EdgeType], Mapping[str, Tensor]]]] = None,
        detach_cache: bool = True,
    ) -> Tuple[Dict[str, Tensor], Dict[EdgeType, Tensor], Dict[str, Dict[Union[str, EdgeType], Dict[str, Tensor]]]]:
        node_cache = dict((decay_cache or {}).get("node", {}))
        edge_cache = dict((decay_cache or {}).get("edge", {}))
        new_node_cache: Dict[str, Dict[str, Tensor]] = {}
        new_edge_cache: Dict[EdgeType, Dict[str, Tensor]] = {}
        x_encoded: Dict[str, Tensor] = {}
        edge_encoded: Dict[EdgeType, Tensor] = {}

        for nt in self.node_types:
            x = _clean(data[nt].x)
            device = x.device
            n, dim = x.shape
            state_idx, flow_idx = self._split_indices(nt, dim, device)
            x_state = x[:, state_idx] if state_idx.numel() else x.new_zeros(n, 0)
            x_flow = x[:, flow_idx] if flow_idx.numel() else x.new_zeros(n, 0)
            mask = _get_optional_column(data[nt], "mask", n, device, 1.0)
            observed_mask = _get_observed_mask(data[nt], mask, n, device)
            delta_t = _get_optional_column(data[nt], "delta_t", n, device, 1.0)
            delta_t = torch.log1p(delta_t)
            # count = _get_optional_column(data[nt], "count", n, device, 0.0)

            x_bar = self.node_bar[nt].to(device).unsqueeze(0)
            # Node caches are aligned by global_id when available because the
            # set/order of position nodes may change across hourly snapshots.
            prev = _align_by_global_id(node_cache.get(nt), data[nt], n, x_bar)
            gamma = torch.exp(-F.softplus(self.node_decay_w[nt]).to(device).unsqueeze(0) * delta_t)
            # Use current state when the row is observed; event/update mask is
            # still passed to the MLP as an irregular-activity signal.
            decayed_state = gamma * prev + (1.0 - gamma) * x_bar
            x_tilde = observed_mask * x_state + (1.0 - observed_mask) * decayed_state
            # enc_in = torch.cat([x_tilde, x_flow, mask, delta_t, torch.log1p(count.clamp_min(0.0))], dim=-1)
            enc_in = torch.cat([x_tilde, x_flow], dim=-1)
            x_encoded[nt] = self.node_mlps[nt](enc_in)
            entry = {"value": _cache_value(x_tilde, detach_cache)}
            align_id = _alignment_id(data[nt], device)
            if align_id is not None:
                entry["align_id"] = align_id.detach()
            new_node_cache[nt] = entry

        for et in self.edge_types:
            key = _edge_key(et)
            store = data[et]
            edge_index = store.edge_index
            e = getattr(store, "edge_attr", None)
            if e is None:
                e = torch.zeros(edge_index.size(1), self.edge_input_dims[et], device=edge_index.device)
            e = _clean(e).to(edge_index.device)
            if e.dim() == 1:
                e = e.unsqueeze(-1)
            if e.size(1) == 0:
                e = e.new_zeros(e.size(0), self.edge_input_dims[et])
            n, dim = e.shape
            state_idx, flow_idx = self._split_indices(et, dim, e.device)
            e_state = e[:, state_idx] if state_idx.numel() else e.new_zeros(n, 0)
            e_flow = e[:, flow_idx] if flow_idx.numel() else e.new_zeros(n, 0)
            mask = _get_optional_column(store, "mask", n, e.device, 1.0)
            observed_mask = _get_observed_mask(store, mask, n, e.device)
            delta_t = _get_optional_column(store, "delta_t", n, e.device, 1.0)
            delta_t = torch.log1p(delta_t)
            # count = _get_optional_column(store, "count", n, e.device, 0.0)

            e_bar = self.edge_bar[key].to(e.device).unsqueeze(0)
            prev_entry = edge_cache.get(et) or edge_cache.get(key)
            edge_align_id = _edge_alignment_id(data, et, e.device)
            if prev_entry is not None and edge_align_id is not None and "align_id" in prev_entry:
                prev_value = prev_entry["value"].to(e.device)
                prev_gid = prev_entry["align_id"].to(e.device).long()
                cur_gid = edge_align_id.long()
                prev = e_bar.expand(n, -1).clone()
                if prev_gid.numel() and cur_gid.numel():
                    order = torch.argsort(prev_gid)
                    sorted_gid = prev_gid[order]
                    pos = torch.searchsorted(sorted_gid, cur_gid)
                    pos_safe = pos.clamp_max(sorted_gid.numel() - 1)
                    valid = (pos < sorted_gid.numel()) & (sorted_gid[pos_safe] == cur_gid)
                    if valid.any():
                        prev[valid] = prev_value[order[pos[valid]]]
            else:
                # Without stable edge ids, do not reuse row-wise edge cache.
                prev = e_bar.expand(n, -1)
            gamma = torch.exp(-F.softplus(self.edge_decay_w[key]).to(e.device).unsqueeze(0) * delta_t)
            decayed_state = gamma * prev + (1.0 - gamma) * e_bar
            e_tilde = observed_mask * e_state + (1.0 - observed_mask) * decayed_state
            # enc_in = torch.cat([e_tilde, e_flow, mask, delta_t, torch.log1p(count.clamp_min(0.0))], dim=-1)
            enc_in = torch.cat([e_tilde, e_flow], dim=-1)
            edge_encoded[et] = self.edge_mlps[key](enc_in)
            edge_entry = {"value": _cache_value(e_tilde, detach_cache)}
            if edge_align_id is not None:
                edge_entry["align_id"] = edge_align_id.detach()
            new_edge_cache[et] = edge_entry

        return x_encoded, edge_encoded, {"node": new_node_cache, "edge": new_edge_cache}


class LightHeteroSnapshotEncoder(nn.Module):
    """Light relation-gated heterogeneous message passing for one snapshot.

    For each directed edge type r = (src_type, rel, dst_type):
        gate_r = sigmoid(MLP_r([x_src, x_dst, e_r]))  # [E, 1]
        msg_r = gate_r * W_r x_src
        h_dst = LayerNorm(W_self x_dst + sum_r scatter_add(msg_r))

    Relation names are only used to build per-edge-type modules; no fixed
    relation string is assumed.
    """

    def __init__(self, node_types: Sequence[str], edge_types: Sequence[EdgeType], hidden_dim: int) -> None:
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.self_proj = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types})
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types})
        self.rel_proj = nn.ModuleDict({_edge_key(et): nn.Linear(hidden_dim, hidden_dim) for et in self.edge_types})
        self.gate = nn.ModuleDict({_edge_key(et): nn.Linear(3 * hidden_dim, 1) for et in self.edge_types})

    def edge_gate(self, edge_type: EdgeType, h_src: Tensor, h_dst: Tensor, e: Tensor) -> Tensor:
        return torch.sigmoid(self.gate[_edge_key(edge_type)](torch.cat([h_src, h_dst, e], dim=-1)))

    def forward(self, data: HeteroData, x_encoded: Mapping[str, Tensor], edge_encoded: Mapping[EdgeType, Tensor]) -> Dict[str, Tensor]:
        agg = {
            nt: torch.zeros(x_encoded[nt].size(0), self.hidden_dim, device=x_encoded[nt].device)
            for nt in self.node_types
        }
        out = {nt: self.self_proj[nt](x_encoded[nt]) for nt in self.node_types}
        for et in self.edge_types:
            src_type, _, dst_type = et
            edge_index = data[et].edge_index
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            x_src = x_encoded[src_type][src]  # [E, H]
            x_dst = x_encoded[dst_type][dst]  # [E, H]
            e = edge_encoded[et]              # [E, H]
            # Directed relation-specific scalar gate: [E, 1] broadcasts over H.
            msg = self.edge_gate(et, x_src, x_dst, e) * self.rel_proj[_edge_key(et)](x_src)
            agg[dst_type].index_add_(0, dst, msg)
        for nt in self.node_types:
            out[nt] = self.norm[nt](out[nt] + agg[nt])
        return out


class TemporalRelationSequenceMemory(nn.Module):
    """Relation-indexed temporal memory over directed heterogeneous edges.

    Memory slots are no longer abstract latent channels.  Each node stores:
        S_v,t in R^{(1 + |R_e|) x H}

    Slot 0 is START, which injects the current encoded node feature z_v,t.
    Slot 1 + b stores historical information that most recently arrived through
    edge relation b.  A learned relation-pair matrix C[a, b] decides which
    previous-relation slots should be read when traversing current relation b.
    """

    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[EdgeType],
        hidden_dim: int,
        transition_dropout: float = 0.0,
        state_steps: int = 2,
    ) -> None:
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.num_edge_relations = len(self.edge_types)
        self.num_slots = self.num_edge_relations + 1
        self.start_slot = 0
        self.edge_slot = {_edge_key(et): i + 1 for i, et in enumerate(self.edge_types)}
        self.edge_rel_index = {_edge_key(et): i for i, et in enumerate(self.edge_types)}
        self.state_steps = max(int(state_steps), 1)
        # c_init = torch.full((self.num_slots, self.num_edge_relations), 0.0)
        # if self.num_edge_relations > 0:
        #     c_init[self.start_slot, :] = 0.1
        #     for et in self.edge_types:
        #         key = _edge_key(et)
        #         slot = self.edge_slot[key]
        #         rel_idx = self.edge_rel_index[key]
        #         c_init[slot, rel_idx] = 0.4
        # self.C = nn.Parameter(c_init)
        self.C = nn.Parameter(torch.zeros(self.num_slots, self.num_edge_relations))
        # self.state_gate = nn.ModuleDict()
        self.msg_mlp = nn.ModuleDict()
        for et in self.edge_types:
            key = _edge_key(et)
            # self.state_gate[key] = nn.Linear(3 * hidden_dim, 1)
            self.msg_mlp[key] = MLP(4 * hidden_dim, hidden_dim, hidden_dim, dropout=transition_dropout)
        beta_init = math.log(0.1 / 0.9)
        self.retention_logit = nn.ParameterDict({nt: nn.Parameter(torch.zeros(self.num_slots)) for nt in self.node_types})
        self.message_beta_logit = nn.ParameterDict({
            nt: nn.Parameter(torch.full((self.num_slots,), float(beta_init)))
            for nt in self.node_types
        })
        self.rollout_beta_logit = nn.Parameter(torch.tensor(float(beta_init)))
        self.start_inject = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types})
        self.update_norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types})
        self.rollout_norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types})
        self.rollout_dropout = nn.Dropout(transition_dropout)

    def _c_alpha(self, rel_idx: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        logits = self.C[:, rel_idx].to(device=device, dtype=dtype)
        return torch.softmax(logits, dim=0)

    @staticmethod
    def _safe_slot_norm(norm: nn.LayerNorm, x: Tensor) -> Tensor:
        active = x.detach().abs().sum(dim=-1) > 1.0e-12
        if not active.any():
            return x
        out = x.clone()
        out[active] = norm(x[active])
        return out

    def _initial_or_aligned_state(
        self,
        prev_state_entry: Optional[Mapping[str, Tensor]],
        store,
        n: int,
        device: torch.device,
    ) -> Tensor:
        fallback = torch.zeros(1, self.num_slots * self.hidden_dim, device=device)
        flat = _align_by_global_id(prev_state_entry, store, n, fallback)
        return flat.view(n, self.num_slots, self.hidden_dim)

    def _inject(self, z_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        injected: Dict[str, Tensor] = {}
        for nt in self.node_types:
            z = z_dict[nt]
            value = z.new_zeros(z.size(0), self.num_slots, self.hidden_dim)
            value[:, self.start_slot, :] = self.start_inject[nt](z)
            injected[nt] = value
        return injected

    def _transition(
        self,
        data: HeteroData,
        z_dict: Mapping[str, Tensor],
        edge_encoded: Mapping[EdgeType, Tensor],
        source_state: Mapping[str, Tensor],
        active_edge_types: Sequence[EdgeType],
    ) -> Dict[str, Tensor]:
        s_hat = {
            nt: torch.zeros(z_dict[nt].size(0), self.num_slots, self.hidden_dim, device=z_dict[nt].device)
            for nt in self.node_types
        }
        deg = {
            nt: torch.zeros(z_dict[nt].size(0), self.num_slots, 1, device=z_dict[nt].device)
            for nt in self.node_types
        }
        for et in active_edge_types:
            src_type, _, dst_type = et
            key = _edge_key(et)
            rel_idx = self.edge_rel_index[key]
            dst_slot = self.edge_slot[key]
            edge_index = data[et].edge_index
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            src_state = source_state[src_type][src]   # [E, S, H]
            z_src = z_dict[src_type][src]             # [E, H]
            z_dst = z_dict[dst_type][dst]             # [E, H]
            e = edge_encoded[et]                      # [E, H]
            # gate = torch.sigmoid(self.state_gate[key](torch.cat([z_src, z_dst, e], dim=-1)))  # [E, 1]
            gate = z_src.new_ones(z_src.size(0), 1)
            z_src_k = z_src.unsqueeze(1).expand(-1, self.num_slots, -1)
            z_dst_k = z_dst.unsqueeze(1).expand(-1, self.num_slots, -1)
            e_k = e.unsqueeze(1).expand(-1, self.num_slots, -1)
            # phi_i = MLP([s_src_i, z_src, z_dst, edge_embedding]).
            phi = self.msg_mlp[key](torch.cat([src_state, z_src_k, z_dst_k, e_k], dim=-1))  # [E, S, H]
            alpha = self._c_alpha(rel_idx, e.device, e.dtype).view(1, self.num_slots, 1)
            msg = gate * (alpha * phi).sum(dim=1)  # [E, H]
            s_hat[dst_type][:, dst_slot, :].index_add_(0, dst, msg)
            deg[dst_type][:, dst_slot, :].index_add_(0, dst, gate)

        return {nt: s_hat[nt] / deg[nt].clamp_min(1.0e-6) for nt in self.node_types}

    def _update(self, prev_state: Mapping[str, Tensor], injected: Mapping[str, Tensor], message: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        next_state: Dict[str, Tensor] = {}
        for nt in self.node_types:
            rho = torch.sigmoid(self.retention_logit[nt]).to(prev_state[nt].device).view(1, self.num_slots, 1)
            beta = torch.sigmoid(self.message_beta_logit[nt]).to(prev_state[nt].device).view(1, self.num_slots, 1)
            beta = beta.clone()
            beta[:, self.start_slot, :] = 0.0
            mixed = rho * prev_state[nt] + injected[nt] + beta * message[nt]
            next_state[nt] = self._safe_slot_norm(self.update_norm[nt], mixed)
        return next_state

    def _route_rollout(
        self,
        data: HeteroData,
        z_dict: Mapping[str, Tensor],
        edge_encoded: Mapping[EdgeType, Tensor],
        current_state: Mapping[str, Tensor],
    ) -> Dict[str, Tensor]:
        route_edge_types = [et for et in self.edge_types if et[0] == "pool" and et[2] == "pool"]
        if not route_edge_types or self.state_steps <= 1:
            return dict(current_state)

        rolled = dict(current_state)
        for _ in range(self.state_steps - 1):
            route_msg = self._transition(data, z_dict, edge_encoded, rolled, route_edge_types)
            beta_roll = torch.sigmoid(self.rollout_beta_logit).to(rolled["pool"].device)
            next_pool = rolled["pool"].clone()
            for et in route_edge_types:
                slot = self.edge_slot[_edge_key(et)]
                value = rolled["pool"][:, slot, :] + beta_roll * self.rollout_dropout(route_msg["pool"][:, slot, :])
                next_pool[:, slot, :] = self._safe_slot_norm(self.rollout_norm["pool"], value)
            rolled["pool"] = next_pool
        return rolled

    def forward(
        self,
        data: HeteroData,
        z_dict: Mapping[str, Tensor],
        edge_encoded: Mapping[EdgeType, Tensor],
        state_dict: Optional[Mapping[str, Mapping[str, Tensor]]] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        prev_state = dict(state_dict or {})
        s_prev: Dict[str, Tensor] = {}
        for nt in self.node_types:
            s_prev[nt] = self._initial_or_aligned_state(prev_state.get(nt), data[nt], z_dict[nt].size(0), z_dict[nt].device)

        injected = self._inject(z_dict)
        message = self._transition(data, z_dict, edge_encoded, s_prev, self.edge_types)
        current_state = self._update(s_prev, injected, message)
        current_state = self._route_rollout(data, z_dict, edge_encoded, current_state)

        new_state: Dict[str, Dict[str, Tensor]] = {}
        for nt in self.node_types:
            n = z_dict[nt].size(0)
            value = current_state[nt]
            entry = {"value": value.view(n, self.num_slots * self.hidden_dim)}
            align_id = _alignment_id(data[nt], z_dict[nt].device)
            if align_id is not None:
                entry["align_id"] = align_id.detach()
            new_state[nt] = entry
        return new_state



class MultiTaskReadout(nn.Module):
    """Pool and position readouts from temporal memory only.

    TRSM stores graph-encoder embeddings, so the final prediction intentionally
    uses only a state summary r. Slot aggregation is content-based and does not
    query S with the current snapshot embedding h.
    """

    def __init__(
        self,
        node_types: Sequence[str],
        hidden_dim: int,
        num_slots: int,
        out_dim_pool: int = 2,
        out_dim_position: int = 2,
        readout_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_types = list(node_types)
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.state_proj = nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types})
        self.slot_score = nn.ModuleDict({nt: MLP(hidden_dim, 1, hidden_dim, dropout=readout_dropout) for nt in self.node_types})
        self.slot_prior = nn.ParameterDict({nt: nn.Parameter(torch.zeros(num_slots)) for nt in self.node_types})
        self.summary_norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types})
        self.pool_head = MLP(hidden_dim, out_dim_pool, hidden_dim, dropout=head_dropout)
        self.position_head = MLP(hidden_dim, out_dim_position, hidden_dim, dropout=head_dropout)

    def _state_tensor(self, state_entry: Mapping[str, Tensor]) -> Tensor:
        value = state_entry["value"]
        return value.reshape(value.size(0), self.num_slots, self.hidden_dim)

    def _summarize_slots(self, node_type: str, s: Tensor, prior: Tensor) -> Tensor:
        score = self.slot_score[node_type](torch.tanh(self.state_proj[node_type](s))).squeeze(-1)
        score = score + prior.view(1, s.size(1))
        alpha = torch.softmax(score, dim=-1)
        r = (alpha.unsqueeze(-1) * s).sum(dim=1)
        return self.summary_norm[node_type](r)

    def pool_summary(self, state_entry: Mapping[str, Tensor]) -> Tensor:
        s = self._state_tensor(state_entry)
        prior = self.slot_prior["pool"].to(s.device)
        if s.size(1) > 1:
            s = s[:, 1:, :]
            prior = prior[1:]
        return self._summarize_slots("pool", s, prior)

    def state_summary(self, node_type: str, state_entry: Mapping[str, Tensor]) -> Tensor:
        s = self._state_tensor(state_entry)
        prior = self.slot_prior[node_type].to(s.device)
        return self._summarize_slots(node_type, s, prior)

    def forward(
        self,
        data: HeteroData,
        _x_encoded: Mapping[str, Tensor],
        h_dict: Mapping[str, Tensor],
        state_dict: Mapping[str, Mapping[str, Tensor]],
        temporal_dict: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        r_pool = self.pool_summary(state_dict["pool"])
        pool_pred = self.pool_head(r_pool)

        r_pos = self.state_summary("position", state_dict["position"])
        position_pred = self.position_head(r_pos)
        return {"pool_pred": pool_pred, "position_pred": position_pred}


class IrregularLatentStateHeteroGNN(nn.Module):
    """End-to-end wrapper for single snapshots or ordered snapshot sequences.

    Data flow for each hourly snapshot:
        raw HeteroData
        -> DecayAwareFeatureEncoder
        -> LightHeteroSnapshotEncoder
        -> TemporalRelationSequenceMemory
        -> MultiTaskReadout

    For a sequence, decay caches and latent states are recurrently passed from
    one snapshot to the next. Loss computation is intentionally left outside.
    """

    def __init__(
        self,
        node_input_dims: Mapping[str, int],
        edge_input_dims: Mapping[EdgeType, int],
        hidden_dim: int = 64,
        out_dim_pool: int = 2,
        out_dim_position: int = 2,
        state_idx_dict: Optional[Mapping[Union[str, EdgeType], Sequence[int]]] = None,
        flow_idx_dict: Optional[Mapping[Union[str, EdgeType], Sequence[int]]] = None,
        encoder_dropout: float = 0.0,
        transition_dropout: float = 0.0,
        readout_dropout: float = 0.0,
        head_dropout: float = 0.0,
        state_steps: int = 1,
        state_grad_steps: int = 0,
    ) -> None:
        super().__init__()
        self.node_types = list(node_input_dims.keys())
        self.edge_types = list(edge_input_dims.keys())
        self.hidden_dim = hidden_dim
        self.state_grad_steps = max(int(state_grad_steps), 0)
        self.decay_encoder = DecayAwareFeatureEncoder(
            node_input_dims,
            edge_input_dims,
            hidden_dim,
            state_idx_dict=state_idx_dict,
            flow_idx_dict=flow_idx_dict,
            dropout=encoder_dropout,
        )
        self.snapshot_encoder = LightHeteroSnapshotEncoder(self.node_types, self.edge_types, hidden_dim)
        self.latent_transition = TemporalRelationSequenceMemory(
            self.node_types,
            self.edge_types,
            hidden_dim,
            transition_dropout=transition_dropout,
            state_steps=state_steps,
        )
        self.readout = MultiTaskReadout(
            self.node_types,
            hidden_dim,
            num_slots=self.latent_transition.num_slots,
            out_dim_pool=out_dim_pool,
            out_dim_position=out_dim_position,
            readout_dropout=readout_dropout,
            head_dropout=head_dropout,
        )

    @classmethod
    def from_sample_data(
        cls,
        sample_data: HeteroData,
        hidden_dim: int = 64,
        out_dim_pool: int = 2,
        out_dim_position: int = 2,
        state_idx_dict: Optional[Mapping[Union[str, EdgeType], Sequence[int]]] = None,
        flow_idx_dict: Optional[Mapping[Union[str, EdgeType], Sequence[int]]] = None,
        encoder_dropout: float = 0.0,
        transition_dropout: float = 0.0,
        readout_dropout: float = 0.0,
        head_dropout: float = 0.0,
        state_steps: int = 1,
        state_grad_steps: int = 0,
    ) -> "IrregularLatentStateHeteroGNN":
        node_input_dims = {nt: int(sample_data[nt].x.size(-1)) for nt in sample_data.node_types}
        edge_input_dims: Dict[EdgeType, int] = {}
        for et in sample_data.edge_types:
            edge_attr = getattr(sample_data[et], "edge_attr", None)
            edge_input_dims[et] = 1 if edge_attr is None or edge_attr.dim() == 1 else max(int(edge_attr.size(-1)), 1)
        return cls(
            node_input_dims,
            edge_input_dims,
            hidden_dim=hidden_dim,
            out_dim_pool=out_dim_pool,
            out_dim_position=out_dim_position,
            state_idx_dict=state_idx_dict,
            flow_idx_dict=flow_idx_dict,
            encoder_dropout=encoder_dropout,
            transition_dropout=transition_dropout,
            readout_dropout=readout_dropout,
            head_dropout=head_dropout,
            state_steps=state_steps,
            state_grad_steps=state_grad_steps,
        )

    def _detach_cache_after_sequence_step(self, step_idx: int, total_steps: int) -> bool:
        remaining = total_steps - 1 - step_idx
        return not (0 < remaining <= self.state_grad_steps)

    def _encode_one(
        self,
        data: HeteroData,
        decay_cache: Optional[Mapping[str, Dict[Union[str, EdgeType], Mapping[str, Tensor]]]] = None,
        state_dict: Optional[Mapping[str, Mapping[str, Tensor]]] = None,
        detach_cache: Optional[bool] = None,
    ) -> Dict[str, Union[HeteroData, Dict]]:
        device = next(self.parameters()).device
        data = data.clone().to(device)
        should_detach_cache = True if detach_cache is None else detach_cache
        x_encoded, edge_encoded, new_decay_cache = self.decay_encoder(data, decay_cache, detach_cache=should_detach_cache)
        h_dict = self.snapshot_encoder(data, x_encoded, edge_encoded)
        new_state = self.latent_transition(data, h_dict, edge_encoded, state_dict)
        state_cache = _state_cache(new_state, should_detach_cache, previous_state=state_dict)
        return {
            "data": data,
            "x_encoded": x_encoded,
            "edge_encoded": edge_encoded,
            "h_dict": h_dict,
            "state_dict": new_state,
            "state_cache": state_cache,
            "decay_cache": new_decay_cache,
        }

    def _forward_one(
        self,
        data: HeteroData,
        decay_cache: Optional[Mapping[str, Dict[Union[str, EdgeType], Mapping[str, Tensor]]]] = None,
        state_dict: Optional[Mapping[str, Mapping[str, Tensor]]] = None,
        temporal_dict: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Union[Tensor, Dict]]:
        encoded = self._encode_one(data, decay_cache=decay_cache, state_dict=state_dict)
        data = encoded["data"]  # type: ignore[assignment]
        x_encoded = encoded["x_encoded"]  # type: ignore[assignment]
        h_dict = encoded["h_dict"]  # type: ignore[assignment]
        new_state = encoded["state_dict"]  # type: ignore[assignment]
        state_cache = encoded["state_cache"]  # type: ignore[assignment]
        new_decay_cache = encoded["decay_cache"]  # type: ignore[assignment]
        # 4) Pool main prediction and position auxiliary prediction.
        pred = self.readout(data, x_encoded, h_dict, new_state, temporal_dict=temporal_dict)
        pred.update({"state_dict": state_cache, "decay_cache": new_decay_cache})
        return pred

    def forward(
        self,
        data_or_sequence: Union[HeteroData, Sequence[HeteroData]],
        decay_cache: Optional[Mapping[str, Dict[Union[str, EdgeType], Mapping[str, Tensor]]]] = None,
        state_dict: Optional[Mapping[str, Mapping[str, Tensor]]] = None,
        return_all: bool = False,
    ) -> Dict[str, Union[Tensor, Dict, List[Dict[str, Union[Tensor, Dict]]]]]:
        if isinstance(data_or_sequence, HeteroData):
            return self._forward_one(data_or_sequence, decay_cache=decay_cache, state_dict=state_dict)

        outputs: List[Dict[str, Union[Tensor, Dict]]] = []
        last: Optional[Dict[str, Union[Tensor, Dict]]] = None
        cur_cache = decay_cache
        cur_state = state_dict
        total_steps = len(data_or_sequence)
        for step_idx, snapshot in enumerate(data_or_sequence):
            detach_cache = self._detach_cache_after_sequence_step(step_idx, total_steps)
            encoded = self._encode_one(snapshot, decay_cache=cur_cache, state_dict=cur_state, detach_cache=detach_cache)
            cur_cache = encoded["decay_cache"]  # type: ignore[assignment]
            cur_state = encoded["state_cache"]  # type: ignore[assignment]
            last = self.readout(
                encoded["data"],  # type: ignore[arg-type]
                encoded["x_encoded"],  # type: ignore[arg-type]
                encoded["h_dict"],  # type: ignore[arg-type]
                encoded["state_dict"],  # type: ignore[arg-type]
            )
            last.update({"state_dict": cur_state, "decay_cache": cur_cache})
            if return_all:
                outputs.append(last)
        if last is None:
            raise ValueError("data_or_sequence is empty.")
        if return_all:
            last = dict(last)
            last["all_outputs"] = outputs
        return last


def minimal_dummy_test() -> None:
    data = HeteroData()
    data["pool"].x = torch.randn(3, 8)
    data["token"].x = torch.randn(2, 5)
    data["position"].x = torch.randn(4, 6)
    data["pool"].global_id = torch.arange(3)
    data["token"].global_id = torch.arange(2)
    data["position"].global_id = torch.arange(4)

    data["position", "belongs_to", "pool"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2]])
    data["position", "belongs_to", "pool"].edge_attr = torch.randn(4, 10)
    data["pool", "rev_belongs_to", "position"].edge_index = torch.tensor([[0, 1, 1, 2], [0, 1, 2, 3]])
    data["pool", "rev_belongs_to", "position"].edge_attr = torch.randn(4, 10)
    data["pool", "has_token", "token"].edge_index = torch.tensor([[0, 1, 2, 2], [0, 0, 1, 0]])
    data["pool", "has_token", "token"].edge_attr = torch.randn(4, 6)
    data["token", "rev_has_token", "pool"].edge_index = torch.tensor([[0, 0, 1, 0], [0, 1, 2, 2]])
    data["token", "rev_has_token", "pool"].edge_attr = torch.randn(4, 6)
    data["pool", "routes_to", "pool"].edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    data["pool", "routes_to", "pool"].edge_attr = torch.randn(3, 10)

    model = IrregularLatentStateHeteroGNN.from_sample_data(data, hidden_dim=32)
    out_single = model(data)
    print("single pool_pred:", out_single["pool_pred"].shape)
    print("single position_pred:", out_single["position_pred"].shape)

    out_seq = model([data, data])
    print("sequence pool_pred:", out_seq["pool_pred"].shape)
    print("sequence position_pred:", out_seq["position_pred"].shape)


if __name__ == "__main__":
    minimal_dummy_test()

"""Short-sequence reference trainer with a shared sparse delta at every occurrence.

No prefix state is reused. Router weights are frozen; selection is recomputed.
The dense-attention replica is permitted only when all causal keys fit inside
the actual sparse-attention top-k. Long-sequence training is rejected.
"""
from .environment import worker_imports
worker_imports()
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from engraft.replica.model import Replica, LayerState
from engraft.replica.layers import hc_mix

class SharedRows(torch.nn.Module):
    def __init__(self, table, rows):
        super().__init__()
        rows = np.asarray(rows, dtype=np.int64)
        if not len(rows) or np.any(np.diff(rows) <= 0):
            raise ValueError("Select sorted, unique existing rows")
        self.register_buffer("rows", torch.from_numpy(rows.copy()))
        self.register_buffer("anchors", torch.from_numpy(table.read_global(rows)))
        self.register_buffer("scale", self.anchors.square().mean(-1).sqrt().clamp_min(1e-6))
        self.normalized_delta = torch.nn.Parameter(torch.zeros_like(self.anchors, dtype=torch.float32))
        self.table = table

    @property
    def delta(self):
        return self.normalized_delta * self.scale[:, None]

    def gather(self, tokens):
        addresses = self.table.global_addresses(tokens)
        flat = addresses.reshape(-1)
        base = torch.from_numpy(self.table.read_global(flat))
        ids = torch.from_numpy(flat.copy())
        indices = torch.searchsorted(self.rows, ids)
        safe = indices.clamp_max(len(self.rows) - 1)
        matched = (indices < len(self.rows)) & (self.rows[safe] == ids)
        # Every occurrence points at the same parameter row, including prompt positions.
        changed = base + self.delta[safe] * matched[:, None]
        return changed.reshape(len(tokens), -1)

    @torch.no_grad()
    def constrain(self, max_normalized_rms=0.1):
        norm = self.normalized_delta.square().mean(-1).sqrt().clamp_min(1e-12)
        self.normalized_delta.mul_((max_normalized_rms / norm).clamp_max(1)[:, None])

class SequenceReplica(Replica):
    def __init__(self, hp, weights, table, max_tokens=128, checkpoint_layers=True):
        super().__init__(hp, weights, table)
        keys = [k for k in table.metadata if "indexer" in k and ("top_k" in k or "topk" in k)]
        if len(keys) != 1:
            raise ValueError(f"Cannot establish exact sparse-attention range: {keys}")
        self.max_tokens = min(max_tokens, int(table.metadata[keys[0]]))
        self.checkpoint_layers = checkpoint_layers

    def embed(self, tokens):
        return torch.from_numpy(self.w.embedding_rows(tokens))

    def full(self, tokens, rows=None, capture=None, routing_source=None):
        if not 0 < len(tokens) <= self.max_tokens:
            raise ValueError(f"Sequence exceeds verified short-attention range (limit {self.max_tokens})")
        hp = self.hp
        x = self.embed(tokens).unsqueeze(1).repeat(1, hp.hc_mult, 1)
        pos = torch.arange(len(tokens), dtype=torch.float64)
        ple = rows.gather(tokens) if rows is not None else torch.from_numpy(self.table.read_global(self.table.global_addresses(tokens).reshape(-1)).reshape(len(tokens), -1))
        if capture is not None: capture["ple_embd"] = ple.detach().clone()
        for il in range(hp.n_layer):
            from .resources import check_budget
            check_budget()
            def layer(hidden, memory, il=il):
                result, _, routing = self.run_layer(il, hidden, pos, LayerState(), routing_source,
                    memory if hp.is_ple(il) else None, need_ffn_output=True, persist_experts=True)
                if capture is not None and routing is not None: capture[f'routing_{il}']=routing.detach().clone()
                return result
            if self.checkpoint_layers and torch.is_grad_enabled() and (x.requires_grad or (hp.is_ple(il) and ple.requires_grad)):
                x = checkpoint(layer, x, ple if hp.is_ple(il) else ple.detach(), use_reentrant=False)
            else:
                x = layer(x, ple)
            if capture is not None:
                capture[f"layer_{il}"] = x.detach().clone()
                if hasattr(capture, 'on_layer'): capture.on_layer(il, x)
        norm = torch.from_numpy(self.w.tensor("output_hc_norm.weight"))
        down = torch.from_numpy(self.w.tensor("output_hc_down.weight"))
        up = torch.from_numpy(self.w.tensor("output_hc_up.weight"))
        mixed, _ = hc_mix(x, norm, down, up, None, hp.f_norm_rms_eps, hp.hc_mult)
        output = torch.from_numpy(self.w.tensor("output.weight"))
        return mixed @ output.T

def objective(logits, tokens, assistant_mask, base_logits, rows, retain_weight=0.3, anchor_weight=0.01):
    """Mask labels by who GENERATED the target token, not by the predictor role."""
    target = torch.as_tensor(tokens[1:], dtype=torch.int64)
    mask = torch.as_tensor(assistant_mask[1:], dtype=torch.bool)
    if len(assistant_mask) != len(tokens) or not bool(mask.any()):
        raise ValueError("No assistant targets, or invalid role mask")
    answer = F.cross_entropy(logits[:-1][mask], target[mask])
    # base_logits must come from a fresh full forward with ORIGINAL rows.
    retain = F.kl_div(F.log_softmax(logits, -1), F.softmax(base_logits.detach(), -1), reduction="batchmean")
    anchor = rows.normalized_delta.square().mean()
    return answer + retain_weight * retain + anchor_weight * anchor, {"answer": float(answer.detach()), "retain": float(retain.detach()), "anchor": float(anchor.detach())}

def require_training_evidence(evidence, identity_sha256):
    required = ("empty_overlay", "original_row_overlay", "addresses", "intermediate_parity", "logit_parity", "directional_gradient", "live_routing")
    if evidence.get("identity_sha256") != identity_sha256 or any(evidence.get(k) is not True for k in required):
        raise RuntimeError("Training blocked: actual-model compatibility and gradient evidence is incomplete")

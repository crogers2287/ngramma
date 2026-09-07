# Adapted from flash_memory; see repository NOTICE for ENGRAFT attribution.
"""Short-sequence forward diagnostic adapted from the historical replica.

Not qualified for training; all full() calls require torch.no_grad().
"""
from .environment import worker_imports
worker_imports()
import numpy as np
import torch
from engraft.replica.model import Replica, LayerState
from engraft.replica.layers import hc_mix

class SequenceReplica(Replica):
    def __init__(self, hp, weights, table, max_tokens=128, checkpoint_layers=False):
        if checkpoint_layers:
            raise ValueError("This diagnostic harness is forward-only; activation checkpointing is unavailable")
        super().__init__(hp, weights, table)
        keys = [k for k in table.metadata if "indexer" in k and ("top_k" in k or "topk" in k)]
        if len(keys) != 1:
            raise ValueError(f"Cannot establish exact sparse-attention range: {keys}")
        self.max_tokens = min(max_tokens, int(table.metadata[keys[0]]))
        self.checkpoint_layers = checkpoint_layers

    def embed(self, tokens):
        return torch.from_numpy(self.w.embedding_rows(tokens))

    def full(self, tokens, rows=None, capture=None, routing_source=None):
        if torch.is_grad_enabled():
            raise RuntimeError("SequenceReplica is forward-only; enter torch.no_grad() first. Numerical qualification remains failed.")
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

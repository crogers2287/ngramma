"""Engine-compatible PLE scalar for forward diagnosis; no backward admission.

PLE transcription derives from ENGRAFT, Copyright 2026 fulvian, Apache-2.0;
see NOTICE and licenses/ENGRAFT-*.txt. Only the summed-dot scaling expression
differs from that implementation. Use inside NativeForward for other primitives.
"""
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch


class NativePleScale:
    def __init__(self):
        self.calls = 0
        self.coefficient = None
        self.stack = None

    def __enter__(self):
        if torch.is_grad_enabled():
            raise RuntimeError('PLE scalar reference is forward-only; enter torch.no_grad() first')
        if self.stack is not None:
            raise RuntimeError('Use a new PLE reference for each forward')
        import engraft.replica.layers as layers
        import engraft.replica.model as model
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(layers, 'ple_forward', self.forward))
        self.stack.enter_context(patch.object(model, 'ple_forward', self.forward))
        return self

    def __exit__(self, *exc):
        self.stack.close()
        if exc[0] is None and self.calls != 1:
            raise RuntimeError('Expected exactly one PLE invocation in the audited model')

    def forward(self, emb, hidden, w, hist, hp):
        if torch.is_grad_enabled():
            raise RuntimeError('PLE scalar reference is forward-only')
        if hist.numel() or self.calls:
            raise ValueError('Only one complete fresh sequence is supported')
        import engraft.replica.layers as L
        hc, n_embd, eps = hp.hc_mult, hp.n_embd, hp.f_norm_rms_eps
        n = emb.shape[0]
        key = (emb @ w.w_key.T).reshape(n,hc,n_embd)
        key = L.rmsnorm_grouped(key,w.norm_key,eps,hc)
        query = L.rmsnorm_grouped(hidden,w.norm_query,eps,hc)
        # The engine multiplies by 1.0f/sqrtf(n_embd). Dividing by the
        # double-derived square root is a different FP32 operation.
        self.coefficient = float(np.float32(1)/np.sqrt(np.float32(n_embd)))
        s = (key*query).sum(dim=-1) * self.coefficient
        mag = torch.sqrt(torch.clamp(s.abs(),min=1e-6))
        gate = torch.sigmoid(torch.sign(s)*mag)
        value = emb @ w.w_value.T
        gated = value.unsqueeze(1)*gate.unsqueeze(-1)
        normed = L.rmsnorm_grouped(gated,w.norm_conv,eps,hc)
        normed_flat = normed.reshape(n,hc*n_embd)
        hist_flat = hist.reshape(0,hc*n_embd)
        conv = L.causal_depthwise_conv(normed_flat,hist_flat,w.conv1d,dilation=hp.ple_ngram_size)
        conv = torch.nn.functional.silu(conv).reshape(n,hc,n_embd)
        output = hidden+(gated+conv)
        hist_len = (w.conv1d.shape[1]-1)*hp.ple_ngram_size
        hist_full = torch.cat([hist,normed],dim=0)
        self.calls += 1
        return output, hist_full[-hist_len:] if hist_len > 0 else hist_full[:0]

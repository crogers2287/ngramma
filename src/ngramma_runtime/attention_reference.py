"""Native full-attention path for separately verified short CPU fixtures only."""
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .artifacts import file_hash


def verified_layouts(reference, hp, tokens):
    """Require observed dense-causal support at every full-attention layer.

    This is an empirical check on this exact sequence and engine capture, not a
    proof that sparse selection can be skipped for arbitrary short sequences.
    Raw probabilities are used only to verify support and cache width; numerical
    attention values are recomputed by the replica during the forward.
    """
    reference = Path(reference)
    metadata = json.loads((reference/'tensors.json').read_text())
    if json.loads((reference/'tokens.json').read_text()) != tokens or not 0 < len(tokens) <= 128:
        raise ValueError('Short fixture tokens must match the capture exactly')
    result = []
    for layer in range(hp.n_layer):
        if hp.is_recr(layer):
            continue
        items = [item for item in metadata if item['name'] == f'kq_soft_max-{layer}']
        if len(items) != 1 or items[0]['type'] != 0:
            raise ValueError(f'Missing unique float32 attention capture at layer {layer}')
        item = items[0]
        shape = tuple(reversed(item['shape']))
        if len(shape) != 4 or shape[:3] != (1, hp.n_head, len(tokens)) or not len(tokens) <= shape[3] <= 4096:
            raise ValueError('Attention capture head/token/cache dimensions do not match')
        probabilities = np.ndarray(shape, dtype='<f4', buffer=(reference/item['file']).read_bytes(),
                                   strides=tuple(reversed(item['strides'])))[0]
        visible = np.arange(shape[3])[None, :] <= np.arange(len(tokens))[:, None]
        expected = np.broadcast_to(visible, probabilities.shape)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.array_equal(probabilities > 0, expected):
            raise ValueError(f'Dense causal support is not established at layer {layer}')
        result.append({'layer': layer, 'cache_slots': shape[3],
                       'probability_sha256': file_hash(reference/item['file'])})
    if not result:
        raise ValueError('No full-attention layers were verified')
    return result


class NativeAttentionReference:
    """Replace composite attention inside one isolated, fresh-sequence forward.

    Enter inside NativeForward so weighted projections and normalizations retain
    the same native implementation. No cached prefixes, multimodal positions,
    edited rows, or general sparse selector are qualified by this fixture path.
    """
    def __init__(self, ops, layouts, tokens):
        self.ops = ops
        self.layouts = list(layouts)
        self.tokens = list(tokens)
        self.used = 0
        self.stack = None

    def __enter__(self):
        if torch.is_grad_enabled():
            raise RuntimeError('Native attention reference has no backward; enter torch.no_grad() first')
        if self.stack is not None or self.used:
            raise RuntimeError('Use a new attention reference for each forward')
        import engraft.replica.layers as layers
        import engraft.replica.model as model
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(layers, 'attention_full', self.attention))
        self.stack.enter_context(patch.object(model, 'attention_full', self.attention))
        return self

    def __exit__(self, *exc):
        self.stack.close()
        if exc[0] is None and self.used != len(self.layouts):
            raise RuntimeError('Full-attention call count does not match the verified fixture')

    def attention(self, x, w, positions, hp, k_cache, v_cache, cache_positions):
        import engraft.replica.layers as layers
        if any(value is not None for value in (k_cache, v_cache, cache_positions)):
            raise ValueError('Only complete fresh sequences are supported')
        if self.used >= len(self.layouts) or len(x) != len(self.tokens):
            raise ValueError('Attention call does not match the verified sequence')
        if not torch.equal(positions.cpu(), torch.arange(len(x), dtype=positions.dtype)):
            raise ValueError('Only sequential plain-text positions starting at zero are supported')
        slots = self.layouts[self.used]['cache_slots']
        n, hq, hk, d = len(x), hp.n_head, hp.n_head_kv, hp.n_embd_head
        q, gate = layers.split_q_gate(x @ w.wq.T, hq, d)
        q = layers.rmsnorm(q, w.q_norm, hp.f_norm_rms_eps)
        k = layers.rmsnorm((x @ w.wk.T).reshape(n,hk,d), w.k_norm, hp.f_norm_rms_eps)
        v = (x @ w.wv.T).reshape(n,hk,d)
        q = self.ops.rope(q, positions, hp)
        k = self.ops.rope(k, positions, hp)
        kp = torch.zeros((slots,hk,d), dtype=torch.float32)
        vp = torch.zeros_like(kp)
        kp[:n], vp[:n] = k, v
        scores = self.ops.scores(kp.permute(1,0,2), q)
        allowed = torch.arange(slots)[None,:] <= torch.arange(n)[:,None]
        mask = torch.where(allowed, 0., float('-inf')).float()
        probabilities = self.ops.softmax(scores, mask, 1/(d**.5))
        value = self.ops.matmul(vp.permute(1,2,0), probabilities).permute(1,0,2).reshape(n,-1)
        output = (value * torch.sigmoid(gate)) @ w.wo.T
        self.used += 1
        return output, k, v

#!/usr/bin/env python3
"""Trace layer-3 attention from saved exact inputs; never train or edit weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--native-library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-rope', action='store_true')
    p.add_argument('--native-attention', action='store_true')
    args = p.parse_args()
    import numpy as np
    import torch
    from ngramma_runtime.environment import worker_imports
    from ngramma_runtime.artifacts import atomic_json, file_hash
    worker_imports()
    from ngramma_runtime.native_forward import NativeEngineWeights, NativeForward
    from engraft.replica.hparams import Hparams
    from engraft.replica.model import load_attn_weights, load_hc, Replica, LayerState
    import engraft.replica.layers as L

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    metadata = json.loads((args.reference/'tensors.json').read_text())
    tokens = json.loads((args.reference/'tokens.json').read_text())
    identity = json.loads(args.manifest.read_text())
    paths = [x['path'] for x in identity['shards']]
    source_hash = file_hash(Path(__file__))
    start = time.monotonic()
    weights = NativeEngineWeights(paths, ram_cache_bytes=2 << 30)
    hp = Hparams.from_gguf_paths(paths[0], paths[1])
    mode = NativeForward(weights, args.native_library, recurrent=True, repack=True, reductions=True)
    if args.native_rope or args.native_attention:
        from ngramma_runtime.native_attention import AttentionOps, RopeConfig
        from ngramma_runtime._table import field_value
        model_metadata = {key: field_value(field) for key, field in weights.readers[0].fields.items()
                          if key.startswith('qwen4exp.') or key == 'general.architecture'}
        ops = AttentionOps(args.native_library, rope_config=RopeConfig.from_metadata(model_metadata), threads=4)
    else:
        ops = None

    def ref(name, occurrence=0):
        item = [x for x in metadata if x['name'] == name][occurrence]
        if item['type'] != 0:
            raise ValueError('This probe requires float32 references')
        array = np.ndarray(tuple(reversed(item['shape'])), dtype='<f4',
            buffer=(args.reference/item['file']).read_bytes(), strides=tuple(reversed(item['strides'])))
        return torch.from_numpy(array.copy().squeeze())

    metrics = {}
    def compare(name, actual, expected):
        a = actual.detach().numpy().astype(np.float64)
        b = expected.detach().numpy().reshape(a.shape).astype(np.float64)
        diff = np.abs(a-b)
        value = {'max_abs': float(diff.max()),
                 'relative_rms': float(np.sqrt(np.mean(diff**2))/max(np.sqrt(np.mean(b**2)), 1e-12)),
                 'different_values': int(np.count_nonzero(diff)), 'values': diff.size}
        metrics[name] = value
        print(name, json.dumps(value), flush=True)

    n, hq, hk, d = len(tokens), hp.n_head, hp.n_head_kv, hp.n_embd_head
    position = torch.arange(n, dtype=torch.float64)
    slots = ref('kq-3').shape[-1]
    allowed = torch.arange(slots)[None,:] <= torch.arange(n)[:,None]
    mask = torch.where(allowed, 0., float('-inf')).float()
    with torch.no_grad(), mode:
        residual = ref('l_last-2').reshape(n, hp.hc_mult, hp.n_embd)
        mixed, injection = L.hc_mix(residual, *load_hc(weights, 3, 'attn'), hp.f_norm_rms_eps, hp.hc_mult)
        compare('hc_mixed', mixed, ref('hc_mixed-3'))
        compare('hc_inject', injection, ref('hc_inject-3'))
        w = load_attn_weights(weights, 3)
        qfull = mixed@w.wq.T
        compare('query_projection', qfull, ref('Qcur_full-3'))
        q, gate = L.split_q_gate(qfull, hq, d)
        compare('query_split', q, ref('Qcur_reshaped-3'))
        q = L.rmsnorm(q, w.q_norm, hp.f_norm_rms_eps)
        compare('query_norm', q, ref('Qcur_normed-3'))
        kraw = mixed@w.wk.T
        compare('key_projection', kraw, ref('Kcur-3'))
        k = L.rmsnorm(kraw.reshape(n,hk,d), w.k_norm, hp.f_norm_rms_eps)
        compare('key_norm', k, ref('Kcur_normed-3'))
        v = (mixed@w.wv.T).reshape(n,hk,d)
        compare('value_projection', v, ref('Vcur-3', 1))
        if args.native_rope:
            q = ops.rope(q, position, hp)
            k = ops.rope(k, position, hp)
        else:
            cos, sin = L.rope_cos_sin(position, hp.rope_dim, hp.rope_freq_base)
            q, k = L.apply_rope(q,cos,sin), L.apply_rope(k,cos,sin)
        compare('query_rope', q, ref('Qcur-3'))
        compare('key_rope', k, ref('Kcur-3', 1))

        def contractions(query, key, value, prefix):
            if args.native_attention:
                kp = torch.zeros((slots,hk,d)); vp = torch.zeros_like(kp)
                kp[:n], vp[:n] = key, value
                scores = ops.matmul(kp.permute(1,0,2), query.permute(1,0,2))
                probs = ops.softmax(scores, mask, 1/(d**.5))
                out = ops.matmul(vp.permute(1,2,0), probs).permute(1,0,2)
            else:
                kr = key.repeat_interleave(hq//hk, 1)
                vr = value.repeat_interleave(hq//hk, 1)
                scores = torch.einsum('thd,shd->hts', query, kr)
                probs = torch.softmax(scores*(1/(d**.5))+mask[:,:n], -1)
                out = torch.einsum('hts,shd->thd', probs, vr)
            compare(prefix+'/scores', scores[:,:,:n], ref('kq-3')[:,:,:n])
            compare(prefix+'/probabilities', probs[:,:,:n], ref('kq_soft_max-3')[:,:,:n])
            compare(prefix+'/values', out, ref('attn_pregate-3'))
            return out

        out = contractions(q,k,v,'chained')
        contractions(ref('Qcur-3'),ref('Kcur-3',1),ref('Vcur-3',1),'exact_rotary_input')
        compare('gate_input', gate, ref('gate_reshaped-3'))
        sigmoid = torch.sigmoid(gate)
        compare('gate_sigmoid', sigmoid, ref('gate_sigmoid-3'))
        gated = out.reshape(n,-1)*sigmoid
        compare('gated_output', gated, ref('attn_gated-3'))
        projected = gated@w.wo.T
        compare('projected_output', projected, ref('attn_output-3'))
        # The unmodified full layer is a separate baseline, even in primitive
        # ablations above. It closes the chain only for the baseline condition.
        if not args.native_rope and not args.native_attention:
            replica = Replica(hp, weights, None)
            layer, _, _ = replica.run_layer(3, residual, position, LayerState(), None, None,
                                            need_ffn_output=True, persist_experts=True)
            compare('complete_layer3', layer, ref('l_last-3'))

    engine_probs = ref('kq_soft_max-3').numpy()
    expected_visibility = allowed.numpy()[None].repeat(hq, 0)
    result = {'schema':'ngramma-attention-stages/v1', 'identity_sha256':identity['identity_sha256'],
              'tokens':tokens, 'layer':3, 'cache_slots':slots, 'query_heads':hq, 'kv_heads':hk,
              'head_dim':d, 'rope_dim':hp.rope_dim, 'rope_sections':list(hp.rope_sections),
              'rope_freq_base':hp.rope_freq_base, 'native_rope':args.native_rope,
              'native_attention':args.native_attention, 'metrics':metrics,
              'engine_positive_probability_mask_matches_dense_causal':bool(np.array_equal(engine_probs>0, expected_visibility)),
              'source_sha256':source_hash, 'native_library_sha256':mode.library_sha256,
              'native_build_record':mode.build_record, 'native_calls':dict(mode.calls),
              'attention_calls':dict(ops.calls) if ops else {},
              'rope_config':ops.settings() if ops else None,
              'reference_metadata_sha256':file_hash(args.reference/'tensors.json'),
              'seconds_including_initialization':time.monotonic()-start,
              'diagnostic_only':True, 'admitted_for_gradients':False}
    atomic_json(args.output, result)


if __name__ == '__main__': main()

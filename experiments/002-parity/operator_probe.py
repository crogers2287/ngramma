"""Compare the complete first-layer chain with captured engine operations."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from ngramma_runtime.environment import worker_imports
    worker_imports()
    from ngramma_runtime.weights import EngineWeights
    from ngramma_runtime.activation_reference import ActivationReference
    from engraft.replica.hparams import Hparams
    from engraft.replica.model import load_hc, load_delta_net_weights, load_moe_nonexpert
    from engraft.replica.layers import hc_mix, hc_combine, rmsnorm_grouped, linear_attn_layer, delta_net_init_state, moe_ffn

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    identity = json.loads(args.manifest.read_text())
    paths = [x['path'] for x in identity['shards']]
    weights = EngineWeights(paths, ram_cache_bytes=2 << 30)
    hp = Hparams.from_gguf_paths(paths[0], paths[1])
    meta = json.loads((args.reference/'tensors.json').read_text())
    tokens = json.loads((args.reference/'tokens.json').read_text())

    def ref(name, occurrence=0):
        item = [x for x in meta if x['name'] == name][occurrence]
        raw = (args.reference/item['file']).read_bytes()
        dtype = '<i4' if item['type'] == 26 else '<f4'
        view = np.ndarray(tuple(reversed(item['shape'])), dtype=dtype, buffer=raw,
                          strides=tuple(reversed(item['strides'])))
        return torch.from_numpy(view.copy().squeeze())

    results = {}

    def compare(name, actual, expected):
        expected = expected.reshape(actual.shape)
        diff = (actual - expected).abs()
        metric = {'max_abs': float(diff.max()),
                  'relative_rms': float(diff.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)),
                  'different_values': int(torch.count_nonzero(diff)), 'values': diff.numel()}
        results[name] = metric
        print(name, json.dumps(metric), flush=True)
        return metric

    def mix_stages(x, kind, occurrence, prefix):
        norm, down, up, injection = load_hc(weights, 0, kind)
        xn = rmsnorm_grouped(x, norm, hp.f_norm_rms_eps, hp.hc_mult)
        compare(prefix+'/norm', xn, ref('hc_norm-0', occurrence))
        flat = xn.reshape(len(tokens), -1)
        low = torch.nn.functional.silu((flat @ down.T) / hp.hc_mult)
        gate = torch.sigmoid(low @ up.T).reshape(xn.shape)
        compare(prefix+'/gate', gate, ref('hc_gate-0', occurrence))
        mixed = (xn * gate).mean(dim=1)
        inject = flat @ injection.T
        compare(prefix+'/mixed', mixed, ref('hc_mixed-0', occurrence))
        compare(prefix+'/inject', inject, ref('hc_inject-0', occurrence))
        return mixed, inject

    start = time.monotonic()
    with torch.no_grad(), ActivationReference(weights):
        x = torch.from_numpy(weights.embedding_rows(tokens)).unsqueeze(1).repeat(1, hp.hc_mult, 1)
        compare('embedding', x, ref('hc_init'))
        mixed, inject = mix_stages(x, 'attn', 0, 'chained/attention_mix')
        dn = load_delta_net_weights(weights, 0)
        attention, _ = linear_attn_layer(mixed, dn, delta_net_init_state(hp), hp)
        compare('chained/attention_output', attention, ref('linear_attn_out-0'))
        exact_attention, _ = linear_attn_layer(ref('hc_mixed-0'), dn, delta_net_init_state(hp), hp)
        compare('exact_input/attention_output', exact_attention, ref('linear_attn_out-0'))
        combined = hc_combine(x, attention, inject, hp.hc_mult)
        compare('chained/attention_combine', combined, ref('hc_combine-0'))
        exact_combine = hc_combine(ref('hc_init'), ref('linear_attn_out-0'), ref('hc_inject-0'), hp.hc_mult)
        compare('exact_input/attention_combine', exact_combine, ref('hc_combine-0'))
        mixed2, inject2 = mix_stages(combined, 'ffn', 1, 'chained/ffn_mix')
        mix_stages(ref('hc_combine-0'), 'ffn', 1, 'exact_input/ffn_mix')
        moe = load_moe_nonexpert(weights, 0)
        routing = torch.topk(torch.softmax(mixed2 @ moe.gate_inp.T, -1), hp.n_expert_used, -1).indices
        expected_routing = ref('ffn_moe_topk-0').long()
        def expert(name):
            return lambda e: torch.from_numpy(weights.expert('blk.0.'+name+'.weight', e, persist=True))
        output = moe_ffn(mixed2, moe.gate_inp, expert('ffn_gate_exps'), expert('ffn_up_exps'),
                         expert('ffn_down_exps'), routing, moe.up_shexp, moe.gate_shexp,
                         moe.down_shexp, moe.gate_inp_shexp, hp.n_expert_used)
        compare('chained/ffn_output', output, ref('ffn_out-0'))
        final = hc_combine(combined, output, inject2, hp.hc_mult)
        compare('chained/layer0_output', final, ref('l_last-0'))
        exact_final = hc_combine(ref('hc_combine-0'), ref('ffn_out-0'), ref('hc_inject-0', 1), hp.hc_mult)
        compare('exact_input/final_combine', exact_final, ref('l_last-0'))

    result = {'schema':'ngramma-operator-probe/v1', 'tokens': tokens,
              'identity_sha256': identity['identity_sha256'], 'seconds': time.monotonic()-start,
              'routing_equal': bool(torch.equal(routing, expected_routing)),
              'metrics': results, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference_metadata_sha256': hashlib.sha256((args.reference/'tensors.json').read_bytes()).hexdigest(),
              'diagnostic_only': True, 'training_qualified': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compare a short full-model CPU forward with saved engine outputs; no training."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import resource
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-library', type=Path)
    p.add_argument('--native-recurrent', action='store_true')
    p.add_argument('--native-repack', action='store_true')
    p.add_argument('--native-reductions', action='store_true')
    p.add_argument('--native-attention', action='store_true')
    p.add_argument('--native-ple-scale', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--cache-gib', type=float, default=4)
    args = p.parse_args()
    if (args.native_recurrent or args.native_repack or args.native_reductions) and not args.native_library:
        p.error('Native recurrent/repack flags require --native-library')
    if args.threads < 1 or not 0 <= args.cache_gib <= 16:
        p.error('Require positive threads and cache-gib between 0 and 16')

    if args.native_attention and not (args.native_library and args.native_recurrent and args.native_repack and args.native_reductions):
        p.error('Native attention requires the complete previous native condition')
    if args.native_ple_scale and not args.native_attention:
        p.error('PLE scalar correction requires the complete native attention condition')

    import numpy as np
    import torch
    from ngramma_runtime.environment import worker_imports, config
    from ngramma_runtime.artifacts import atomic_json, file_hash
    from ngramma_runtime.resources import check_budget
    worker_imports()
    from engraft.replica.hparams import Hparams
    from ngramma_runtime.table import ModelTable
    from ngramma_runtime.sequence import SequenceReplica
    from ngramma_runtime.weights import EngineWeights
    from ngramma_runtime.activation_reference import ActivationReference
    from ngramma_runtime.capture_identity import validate_capture

    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), *sorted((root/'src/ngramma_runtime').rglob('*.py')),
               root/'src/ngramma_runtime/native/ggml_forward.cpp']
    source_hashes = {str(path.relative_to(root)): file_hash(path) for path in sources if path.is_file()}
    import engraft
    import gguf
    external_sources = {}
    for name, package in (('engraft', engraft), ('gguf', gguf)):
        package_root = Path(package.__file__).resolve().parent
        for path in sorted(package_root.rglob('*.py')):
            external_sources[name+'/'+str(path.relative_to(package_root))] = path
    external_hashes = {label: file_hash(path) for label, path in external_sources.items()}
    runtime = config().runtime
    dependency_hashes = {path.name: file_hash(path) for path in sorted(runtime.glob('*.so'))}

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1234)
    start = time.monotonic()
    inventory = check_budget()
    identity = json.loads(args.manifest.read_text())
    capture_identity = validate_capture(args.reference, identity['identity_sha256'])
    tokens = json.loads((args.reference/'tokens.json').read_text())
    metadata = json.loads((args.reference/'tensors.json').read_text())
    paths = [s['path'] for s in identity['shards']]
    table = ModelTable(paths)
    if args.native_library:
        from ngramma_runtime.native_forward import NativeEngineWeights, NativeForward
        weights = NativeEngineWeights(paths, ram_cache_bytes=int(args.cache_gib*(1 << 30)))
        mode = NativeForward(weights, args.native_library, threads=args.threads,
                             recurrent=args.native_recurrent, repack=args.native_repack, reductions=args.native_reductions)
    else:
        weights = EngineWeights(paths, ram_cache_bytes=int(args.cache_gib*(1 << 30)))
        mode = ActivationReference(weights)
    hp = Hparams.from_gguf_paths(paths[0], paths[1])
    replica = SequenceReplica(hp, weights, table, max_tokens=128)
    metrics = []
    compared_tensor_hashes = {}
    attention_mode = nullcontext()
    attention_ops = None
    layouts = None
    if args.native_attention:
        from ngramma_runtime.native_attention import AttentionOps, RopeConfig
        from ngramma_runtime.attention_reference import NativeAttentionReference, verified_layouts
        layouts = verified_layouts(args.reference, hp, tokens)
        attention_ops = AttentionOps(args.native_library, rope_config=RopeConfig.from_metadata(table.metadata), threads=args.threads)
        attention_mode = NativeAttentionReference(attention_ops, layouts, tokens)
    ple_mode = nullcontext()
    if args.native_ple_scale:
        from ngramma_runtime.ple_reference import NativePleScale
        ple_mode = NativePleScale()

    def compare(name, actual):
        item = next(x for x in metadata if x['name'] == name)
        if item['type'] != 0:
            raise ValueError('Reference must contain float32 diagnostic tensors')
        expected = np.ndarray(tuple(reversed(item['shape'])), dtype='<f4',
            buffer=(args.reference/item['file']).read_bytes(), strides=tuple(reversed(item['strides'])))
        actual = actual.detach().numpy()
        expected = expected.reshape(actual.shape)
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('Nonfinite reference or actual intermediate values')
        compared_tensor_hashes[name] = file_hash(args.reference/item['file'])
        difference = actual.astype(np.float64)-expected.astype(np.float64)
        result = {'name': name, 'max_abs': float(np.abs(difference).max()),
                  'relative_rms': float(np.sqrt(np.mean(difference**2))/max(np.sqrt(np.mean(expected.astype(np.float64)**2)), 1e-12)),
                  'different_values': int(np.count_nonzero(difference)), 'values': actual.size}
        result['actual_logical_sha256'] = hashlib.sha256(actual.astype('<f4',copy=False).tobytes()).hexdigest()
        result['reference_logical_sha256'] = hashlib.sha256(expected.astype('<f4',copy=False).tobytes()).hexdigest()
        result['bitwise_equal'] = result['actual_logical_sha256'] == result['reference_logical_sha256']
        metrics.append(result)
        print(json.dumps({**result, 'elapsed_seconds': time.monotonic()-start}), flush=True)

    class Captures(dict):
        # Compare immediately instead of retaining all layers/expert routes.
        def __setitem__(self, key, value):
            if key == 'ple_embd':
                compare('ple_embd', value)

        def on_layer(self, layer, value):
            compare(f'l_last-{layer}', value)

    initialized = time.monotonic()
    with torch.no_grad(), mode, attention_mode, ple_mode:
        logits = replica.full(tokens, capture=Captures())
    finished = time.monotonic()
    expected = np.fromfile(args.reference/'logits.f32', dtype='<f4').reshape(len(tokens), -1)
    if tuple(logits.shape) != expected.shape:
        raise ValueError('Logit shapes do not match')
    if not np.isfinite(logits.numpy()).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite reference or actual logits')
    selected = expected.argmax(axis=-1)
    with torch.no_grad():
        lp = torch.log_softmax(logits, -1).numpy()
        ref_lp = torch.log_softmax(torch.from_numpy(expected), -1).numpy()
    selected_difference = np.abs(lp[np.arange(len(tokens)), selected]-ref_lp[np.arange(len(tokens)), selected])
    logit_difference = np.abs(logits.numpy()-expected)
    logit_pass = bool(selected_difference.max() < 0.02)
    intermediate_pass = bool(len(metrics) == hp.n_layer+1 and max(x['relative_rms'] for x in metrics) < 0.01)
    changed_sources = [str(path.relative_to(root)) for path in sources if not path.is_file()
                       or file_hash(path) != source_hashes[str(path.relative_to(root))]]
    changed_sources += [label for label, path in external_sources.items() if not path.is_file() or file_hash(path) != external_hashes[label]]
    result = {'schema': 'ngramma-full-parity/v1', 'identity_sha256': identity['identity_sha256'],
              'tokens': tokens, 'threads': args.threads, 'cache_gib': args.cache_gib,
              'initialization_seconds': initialized-start, 'forward_seconds': finished-initialized,
              'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1 << 20),
              'initial_available_gib': inventory['available_bytes']/(1 << 30),
              'max_logit_error': float(logit_difference.max()), 'mean_logit_error': float(logit_difference.mean()),
              'actual_logits_sha256': hashlib.sha256(logits.numpy().astype('<f4',copy=False).tobytes()).hexdigest(),
              'logits_bitwise_equal': logits.numpy().astype('<f4',copy=False).tobytes() == expected.astype('<f4',copy=False).tobytes(),
              'selected_logprob_error_max': float(selected_difference.max()),
              'selected_logprob_error_by_position': selected_difference.tolist(),
              'top1_agreement': float(np.mean(logits.argmax(-1).numpy() == selected)),
              'intermediates': metrics, 'weight_io': weights.stats,
              'logit_parity': logit_pass, 'intermediate_parity': intermediate_pass,
              'thresholds': {'selected_logprob_nats_strictly_below': 0.02, 'relative_rms_strictly_below': 0.01},
              'native_library_sha256': file_hash(args.native_library) if args.native_library else None,
              'native_attention': args.native_attention,
              'native_ple_scale': args.native_ple_scale,
              'ple_scale_calls': ple_mode.calls if args.native_ple_scale else 0,
              'ple_scale_coefficient': ple_mode.coefficient if args.native_ple_scale else None,
              'attention_calls': dict(attention_ops.calls) if attention_ops else {},
              'rope_config': attention_ops.settings() if attention_ops else None,
              'verified_attention_layouts': layouts,
              'native_recurrent': args.native_recurrent,
              'native_repack': args.native_repack,
              'native_reductions': args.native_reductions,
              'native_build_record': mode.build_record if args.native_library else None,
              'native_calls': dict(mode.calls) if args.native_library else {},
              'native_misses': mode.misses if args.native_library else [],
              'source_sha256': source_hashes, 'dependency_sha256': dependency_hashes,
              'external_source_sha256': external_hashes,
              'compared_tensor_sha256': compared_tensor_hashes,
              'capture_identity': capture_identity,
              'source_hash_scope': 'Before model initialization; native build identity is recorded separately.',
              'sources_changed_during_run': changed_sources,
              'reference_sha256': {name: file_hash(args.reference/name) for name in ('tokens.json','tensors.json','logits.f32')},
              'diagnostic_only': True, 'admitted_for_gradients': False,
              'note': 'One short CPU sequence. Native operators have no backward implementation. This is not broad forward or training qualification.'}
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ('top1_agreement', 'selected_logprob_error_max', 'logit_parity', 'intermediate_parity', 'forward_seconds')}), flush=True)


if __name__ == '__main__':
    main()

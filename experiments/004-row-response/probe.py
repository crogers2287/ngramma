#!/usr/bin/env python3
"""Single existing-row finite response; model inputs and binaries stay local."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'reference', 'native-library', 'activation-library', 'local', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    from ngramma_runtime.environment import worker_imports
    worker_imports()
    from ngramma_runtime.artifacts import atomic_json, file_hash
    from ngramma_runtime.capture_identity import validate_capture
    from ngramma_runtime.native_forward import NativeEngineWeights, NativeForward
    from ngramma_runtime.activation_encoding import ActivationEncoding
    from ngramma_runtime.row_patch import RowPatch
    from ngramma_runtime.table import ModelTable
    from ngramma_runtime.resources import check_budget
    from engraft.replica.hparams import Hparams
    import engraft.replica.layers as L
    import engraft.replica.model as M

    if args.output.exists():
        raise ValueError('Use a new result file; retain earlier experiments')
    args.local.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    budget = check_budget()
    identity = json.loads(args.manifest.read_text())
    capture = validate_capture(args.reference, identity['identity_sha256'])
    tokens = capture['tokens']
    if tokens != [18374,50203,198,220,1330,1838,26453,220,248045,846,198,5834,248046,198,248045,74455,198]:
        raise ValueError('This preregistered experiment requires the Unicode/chat fixture')
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(__file__).with_name('PLAN.md'),
               *[root/'src/ngramma_runtime'/name for name in
                 ('row_patch.py','native_forward.py','activation_encoding.py','_weights.py','_table.py','environment.py','resources.py')],
               Path(L.__file__), Path(M.__file__)]
    hashes = {str(path): file_hash(path) for path in sources}
    metadata = json.loads((args.reference/'tensors.json').read_text())
    ref_hashes = {}

    def ref(name, shape):
        items = [x for x in metadata if x['name'] == name]
        if len(items) != 1 or items[0]['type'] != 0:
            raise ValueError('Require one F32 tensor '+name)
        item = items[0]
        path = (args.reference/item['file']).resolve()
        if not path.is_relative_to(args.reference.resolve()):
            raise ValueError('Unsafe capture path')
        raw = path.read_bytes()
        a = np.ndarray(tuple(reversed(item['shape'])), dtype='<f4', buffer=raw,
                       strides=tuple(reversed(item['strides'])))
        ref_hashes[name] = file_hash(path)
        if a.size != int(np.prod(shape)) or not np.isfinite(a).all():
            raise ValueError('Invalid reference tensor')
        return torch.from_numpy(a.copy().reshape(shape))

    paths = [s['path'] for s in identity['shards']]
    table = ModelTable(paths)
    weights = NativeEngineWeights(paths, ram_cache_bytes=2<<30)
    hp = Hparams.from_gguf(weights.readers[0])
    n, hc, d = len(tokens), hp.hc_mult, hp.n_embd
    if (hp.ple_layer, hc, d, table.dim, table.n_heads) != (1,4,2560,160,16):
        raise ValueError('Unqualified geometry')
    memory = ref('ple_embd', (n,d))
    hidden = ref('l_last-0', (n,hc,d))
    expected = hidden + (ref('ple_gated_value-1',(n,hc,d)) + ref('ple_conv_out-1',(n,hc,d)))
    addresses = table.global_addresses(tokens)
    gathered = table.read_global(addresses.reshape(-1)).reshape(n,16,160)
    if not np.array_equal(gathered.reshape(n,d), memory.numpy()):
        raise ValueError('Offline gathered rows differ from engine')
    row = int(addresses[6,8])
    occurrences = np.argwhere(addresses == row).tolist()
    anchor = table.read_global([row])[0]
    scale = float(np.sqrt(np.mean(anchor.astype(np.float64)**2)))
    direction = (np.where(np.arange(160)%2 == 0, 1., -1.)*scale).astype(np.float32)
    magnitudes = [2.**e for e in range(-20,-3,2)]
    epsilons = [0., *[s*e for e in magnitudes for s in (-1,1)]]
    logits = np.fromfile(args.reference/'logits.f32', dtype='<f4').reshape(n,-1)
    top = np.argsort(-logits[-1].astype(np.float64), kind='stable')[:2].tolist()
    margin = float(logits[-1,top[0]]) - float(logits[-1,top[1]])
    spec = {'row_id': row, 'head': 8, 'token_index': 6, 'occurrences': occurrences,
            'observed_token_triple': tokens[4:7], 'row_rms': scale,
            'anchor_sha256': hashlib.sha256(anchor.tobytes()).hexdigest(),
            'direction_sha256': hashlib.sha256(direction.tobytes()).hexdigest(),
            'direction': 'alternating +1/-1 multiplied by original row RMS, FP32',
            'epsilons': epsilons, 'local_scalar_index': [6,0,0],
            'engine_margin': {'position': n-1, 'token_ids': top, 'baseline': margin},
            'identity_sha256': identity['identity_sha256'], 'tokens': tokens}
    atomic_json(args.local/'spec.json', spec)  # Fixed before any edited forward.
    w = M.load_ple_weights(weights,1)
    weight_types = {name:int(weights.tensor_type('blk.1.ple_'+name+'.weight')) for name in ('key','value')}
    if set(weight_types.values()) != {8}:
        raise ValueError('Activation replay qualified only for actual Q8_0 projections')
    for name in ('key','value'):
        sources_found = [s for item in metadata for s in item.get('weight_sources',[])
                         if s['name'] == 'blk.1.ple_'+name+'.weight']
        if not sources_found or any(s['type'] != 8 or s['buffer'] != 'CPU_Mapped' for s in sources_found):
            raise ValueError('Require captured CPU_Mapped Q8_0 PLE projection')
    encoder = ActivationEncoding(args.activation_library)
    coefficient = float(np.float32(1)/np.sqrt(np.float32(d)))

    # ENGRAFT-derived PLE transcription; see NOTICE. The coefficient matches
    # experiment003. The same formula serves the explicit smooth FP64 control
    # and (under NativeForward) the quantized CPU forward diagnostic.
    def ple(emb, residual, pw):
        key = emb @ pw.w_key.T
        value = emb @ pw.w_value.T
        nk = L.rmsnorm_grouped(key.reshape(n,hc,d), pw.norm_key, hp.f_norm_rms_eps,hc)
        nq = L.rmsnorm_grouped(residual, pw.norm_query, hp.f_norm_rms_eps,hc)
        s = (nk*nq).sum(dim=-1)*coefficient
        gate = torch.sigmoid(torch.sign(s)*torch.sqrt(s.abs().clamp_min(1e-6)))
        gated = value.unsqueeze(1)*gate.unsqueeze(-1)
        normed = L.rmsnorm_grouped(gated,pw.norm_conv,hp.f_norm_rms_eps,hc)
        history = residual.new_zeros((0,hc*d))
        conv = L.causal_depthwise_conv(normed.reshape(n,hc*d),history,pw.conv1d,dilation=hp.ple_ngram_size)
        output = residual + (gated + torch.nn.functional.silu(conv).reshape(n,hc,d))
        return {'key':key, 'value':value, 'output':output, 'gate_sum':s}

    # Isolated smooth double-precision derivative: no activation quantization.
    wd = dataclasses.replace(w, **{f.name:getattr(w,f.name).double() for f in dataclasses.fields(w)})
    md, hd = memory.double(), hidden.double()
    tangent = torch.zeros_like(md).reshape(n,16,160)
    for ti, hi in occurrences:
        tangent[ti,hi] = torch.from_numpy(direction.astype(np.float64))
    tangent = tangent.reshape(n,d)
    alpha = torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    smooth_base = ple(md+alpha*tangent,hd,wd)['output'][6,0,0]
    derivative = float(torch.autograd.grad(smooth_base,alpha)[0])
    smooth = []
    with torch.no_grad():
        for e in magnitudes:
            plus = float(ple(md+e*tangent,hd,wd)['output'][6,0,0])
            minus = float(ple(md-e*tangent,hd,wd)['output'][6,0,0])
            secant = (plus-minus)/(2*e)
            smooth.append({'epsilon':e, 'central_difference':secant,
                           'relative_error':abs(secant-derivative)/max(abs(derivative),1e-12)})
    del wd, md, hd, tangent, smooth_base
    points, overlays = [], []
    native_hashes = {}
    for e in epsilons:
        check_budget()
        patch = RowPatch.from_direction(table,row,direction,e)
        edited = patch.apply(addresses,gathered).reshape(n,d)
        encoded = encoder.encode(8,edited)
        mode = NativeForward(weights,args.native_library,recurrent=True,repack=True,reductions=True,threads=4)
        with torch.no_grad(),mode:
            stages = {k:v.numpy().copy() for k,v in ple(torch.from_numpy(edited),hidden,w).items()}
        if mode.misses:
            raise ValueError('Unregistered native matrix multiply')
        if e == 0:
            if not np.array_equal(stages['output'],expected.numpy()):
                raise ValueError('Native baseline PLE output does not match engine')
            baseline, baseline_encoded = stages, encoded
        delta_out = stages['output'].astype(np.float64)-baseline['output']
        changed_bytes = encoded != baseline_encoded
        blocks = encoded.reshape(n,-1,34)
        base_blocks = baseline_encoded.reshape(n,-1,34)
        point = {'epsilon':e, 'ple_output_rms':float(np.sqrt(np.mean(delta_out**2))),
                 'key_changed_count':int(np.count_nonzero(stages['key'] != baseline['key'])),
                 'value_changed_count':int(np.count_nonzero(stages['value'] != baseline['value'])),
                 'local_scalar_delta':float(delta_out[6,0,0]),
                 'ple_changed_counts_by_token':np.count_nonzero(delta_out.reshape(n,-1),axis=1).tolist(),
                 'activation_changed_bytes':int(changed_bytes.sum()),
                 'activation_scale_changed_bytes':int(np.count_nonzero(blocks[:,:,:2] != base_blocks[:,:,:2])),
                 'activation_code_changed_bytes':int(np.count_nonzero(blocks[:,:,2:] != base_blocks[:,:,2:])),
                 'activation_sha256':hashlib.sha256(encoded.tobytes()).hexdigest(),
                 'replacement_changed_values':int(np.count_nonzero(patch.replacement != anchor)),
                 'realized_normalized_rms':float(np.sqrt(np.mean((patch.replacement.astype(np.float64)-anchor)**2))/scale),
                 'native_calls':dict(mode.calls),
                 'gate_sign_changed_count':int(np.count_nonzero(np.sign(stages['gate_sum']) != np.sign(baseline['gate_sum']))),
                 'stage_sha256':{k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in stages.items()}}
        label = 'zero' if e == 0 else ('plus-' if e>0 else 'minus-')+str(-int(round(np.log2(abs(e)))))
        overlays.append(patch.export(args.local/label,identity,table,{'experiment':'004-row-response','epsilon':e,'spec_sha256':file_hash(args.local/'spec.json')}))
        point['overlay_label'] = label
        point['overlay_sha256'] = overlays[-1]['overlay_sha256']
        points.append(point)
        native_hashes[label] = point['stage_sha256']['output']
        print(json.dumps({k:point[k] for k in ('epsilon','activation_changed_bytes','key_changed_count','ple_output_rms')}),flush=True)

    by_e = {point['epsilon']:point for point in points}
    def changes(e):
        return any(by_e[s*e]['key_changed_count'] or by_e[s*e]['value_changed_count'] for s in (-1,1))
    plateaus = [e for e in magnitudes if not changes(e)]
    changed = [e for e in magnitudes if changes(e)]
    selected = [max(plateaus)] if plateaus else []
    if changed:
        first = changed[0]
        selected.append(first)
        larger = [e for e in magnitudes if e>first]
        if larger: selected.append(larger[0])
        if not plateaus: selected.append(magnitudes[-1])
    selected = sorted(set(selected))
    selected_labels = ['zero',*[by_e[s*e]['overlay_label'] for e in selected for s in (-1,1)]]
    changed_sources = [str(path) for path in sources if hashes[str(path)] != file_hash(path)]
    if changed_sources:
        raise ValueError('Source changed during execution: '+str(changed_sources))
    result = {'schema':'ngramma.row-response/v1', 'evidence_kind':'measured',
              'title':'One existing trigram row through Q8_0 memory projections',
              'experiment':{**spec, 'row_id':str(row)}, 'points':points,
              'interpretation':{'response_kind':'finite_step','notes':'One row and fixed direction. No training, task improvement, or serving gradient established.'},
              'smooth_control':{'dtype':'float64','activation_quantization':False,'scalar_index':[6,0,0],
                                'autograd_directional_derivative':derivative,'finite_differences':smooth},
              'native_baseline_matches_engine':True, 'selected_engine_labels':selected_labels,
              'weight_types':weight_types,'activation_encoding':encoder.metadata,
              'activation_evidence_kind':'replay of actual dispatched encoder, not captured graph workspace',
              'capture_identity':capture,'reference_tensor_sha256':ref_hashes,
              'source_sha256':{str(path.relative_to(root)) if path.is_relative_to(root) else 'engraft/'+path.name:hashes[str(path)] for path in sources},
              'native_library_sha256':mode.library_sha256,'native_build_record':mode.build_record,
              'activation_library_sha256':file_hash(args.activation_library),
              'seconds':time.monotonic()-start,'initial_available_bytes':budget['available_bytes'],
              'sources_changed_during_run':changed_sources,'training_admitted':False}
    atomic_json(args.output,result)
    table.close()
    print('Full engine labels:',selected_labels,flush=True)


if __name__ == '__main__': main()

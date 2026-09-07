#!/usr/bin/env python3
"""Compare two PLE gate scalars from exact saved inputs; forward-only CPU probe.

The local PLE transcription derives from ENGRAFT (Copyright 2026 fulvian,
Apache-2.0); see NOTICE and licenses/ENGRAFT-*.txt. All other operations and
addition ordering are held fixed. No rows or checkpoint files are edited.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import struct
import time
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','reference','native-library','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    # Keep --help safe: numerical/local engine dependencies load only on execution.
    import numpy as np
    import torch
    from ngramma_runtime.environment import worker_imports
    from ngramma_runtime.artifacts import atomic_json,file_hash
    from ngramma_runtime.capture_identity import validate_capture
    from ngramma_runtime.resources import check_budget
    worker_imports()
    from ngramma_runtime.native_forward import NativeEngineWeights,NativeForward
    from engraft.replica.hparams import Hparams
    import engraft.replica.layers as L
    import engraft.replica.model as M
    import engraft
    import gguf

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1234)
    start=time.monotonic()
    budget=check_budget()
    identity=json.loads(args.manifest.read_text())
    capture_identity=validate_capture(args.reference,identity['identity_sha256'])
    tokens=json.loads((args.reference/'tokens.json').read_text())
    metadata=json.loads((args.reference/'tensors.json').read_text())
    root=Path(__file__).resolve().parents[2]
    sources=[Path(__file__),*sorted((root/'src/ngramma_runtime').rglob('*.py')),
             root/'src/ngramma_runtime/native/ggml_forward.cpp']
    source_hashes={str(path.relative_to(root)):file_hash(path) for path in sources}
    external_sources={}
    for name,package in (('engraft',engraft),('gguf',gguf)):
        package_root=Path(package.__file__).resolve().parent
        external_sources.update({name+'/'+str(path.relative_to(package_root)):path
                                 for path in sorted(package_root.rglob('*.py'))})
    external_hashes={name:file_hash(path) for name,path in external_sources.items()}
    reference_hashes={name:file_hash(args.reference/name)
                      for name in ('tokens.json','tensors.json','logits.f32','capture-summary.json')}
    tensor_hashes={}

    def ref(name,shape):
        items=[item for item in metadata if item['name']==name]
        if len(items)!=1 or items[0]['type']!=0:
            raise ValueError(f'Require one float32 captured tensor: {name}')
        item=items[0]
        path=(args.reference/item['file']).resolve()
        if not path.is_relative_to(args.reference.resolve()):
            raise ValueError('Reference tensor path escapes capture directory')
        raw=path.read_bytes()
        array=np.ndarray(tuple(reversed(item['shape'])),dtype='<f4',buffer=raw,
                         strides=tuple(reversed(item['strides'])))
        if array.size!=int(np.prod(shape)) or not np.isfinite(array).all():
            raise ValueError(f'Invalid/nonfinite reference shape for {name}')
        tensor_hashes[name]=file_hash(path)
        return torch.from_numpy(array.copy().reshape(shape))

    def compare(actual,expected):
        if tuple(actual.shape)!=tuple(expected.shape):raise ValueError('Compared shapes differ')
        a=actual.detach().numpy().astype(np.float64)
        b=expected.detach().numpy().astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite PLE comparison')
        difference=a-b
        changed=np.any(difference.reshape(len(tokens),-1)!=0,axis=1)
        indices=np.flatnonzero(changed)
        return {'max_abs':float(np.abs(difference).max()),
                'relative_rms':float(np.sqrt(np.mean(difference**2))/max(np.sqrt(np.mean(b**2)),1e-12)),
                'different_values':int(np.count_nonzero(difference)),'values':int(difference.size),
                'first_differing_token':int(indices[0]) if len(indices) else None,
                'different_values_by_token':[int(value) for value in np.count_nonzero(difference.reshape(len(tokens),-1),axis=1)]}

    paths=[entry['path'] for entry in identity['shards']]
    weights=NativeEngineWeights(paths,ram_cache_bytes=2<<30)
    hp=Hparams.from_gguf(weights.readers[0])
    hp.n_vocab=weights.shape('output.weight')[1]
    if hp.ple_layer!=1 or not hp.is_recr(1):
        raise ValueError('This probe is scoped to audited recurrent PLE layer 1')
    n,h,d=len(tokens),hp.hc_mult,hp.n_embd
    residual=ref('l_last-0',(n,h,d))
    memory=ref('ple_embd',(n,d))
    expected={'gate':ref('ple_gate-1',(n,h)),
              'gated_value':ref('ple_gated_value-1',(n,h,d)),
              'conv_out':ref('ple_conv_out-1',(n,h,d)),
              'full_layer1':ref('l_last-1',(n,h,d))}
    coefficient=np.float32(1)/np.sqrt(np.float32(d))
    original_ple=L.ple_forward
    original_model_ple=M.ple_forward
    conditions={}
    for corrected,label in ((False,'original_division'),(True,'float32_reciprocal_multiply')):
        check_budget()
        mode=NativeForward(weights,args.native_library,recurrent=True,repack=True,reductions=True,threads=4)
        stages={}

        # Transcribed from pinned engraft.replica.layers.ple_forward, with the
        # gate-scalar line as the sole ablation and snapshots for three stages.
        def ple(emb,hidden,w,hist,hparams):
            hc,n_embd,eps=hparams.hc_mult,hparams.n_embd,hparams.f_norm_rms_eps
            t_len=emb.shape[0]
            key=(emb@w.w_key.T).reshape(t_len,hc,n_embd)
            key=L.rmsnorm_grouped(key,w.norm_key,eps,hc)
            query=L.rmsnorm_grouped(hidden,w.norm_query,eps,hc)
            summed=(key*query).sum(dim=-1)
            s=summed*float(coefficient) if corrected else summed/(n_embd**0.5)
            mag=torch.sqrt(torch.clamp(s.abs(),min=1e-6))
            gate=torch.sigmoid(torch.sign(s)*mag)
            value=emb@w.w_value.T
            gated=value.unsqueeze(1)*gate.unsqueeze(-1)
            normed=L.rmsnorm_grouped(gated,w.norm_conv,eps,hc)
            normed_flat=normed.reshape(t_len,hc*n_embd)
            hist_flat=hist.reshape(hist.shape[0],hc*n_embd) if hist.numel() else hist.reshape(0,hc*n_embd)
            conv=L.causal_depthwise_conv(normed_flat,hist_flat,w.conv1d,dilation=hparams.ple_ngram_size)
            conv=torch.nn.functional.silu(conv).reshape(t_len,hc,n_embd)
            result=hidden+(gated+conv)
            hist_len=(w.conv1d.shape[1]-1)*hparams.ple_ngram_size
            hist_full=torch.cat([hist,normed],dim=0) if hist.numel() or hist.shape[0]==0 else normed
            stages.update(gate=gate.detach().clone(),gated_value=gated.detach().clone(),conv_out=conv.detach().clone())
            return result,hist_full[-hist_len:] if hist_len>0 else hist_full[:0]

        condition_start=time.monotonic()
        with torch.no_grad(),mode:
            ple_weights=M.load_ple_weights(weights,1)
            zero_history=torch.zeros(0,h,d)
            result,history=ple(memory,residual,ple_weights,zero_history,hp)
            if not corrected:
                upstream,upstream_history=original_ple(memory,residual,ple_weights,zero_history,hp)
                if not torch.equal(result,upstream) or not torch.equal(history,upstream_history):
                    raise ValueError('Local original PLE transcription does not match pinned upstream')
            replica=M.Replica(hp,weights,None)
            with ExitStack() as patches:
                patches.enter_context(patch.object(L,'ple_forward',ple))
                patches.enter_context(patch.object(M,'ple_forward',ple))
                layer,_,_=replica.run_layer(1,residual,torch.arange(n,dtype=torch.float64),
                    M.LayerState(),None,memory,need_ffn_output=True,persist_experts=True)
            stages['full_layer1']=layer
            metrics={name:compare(stages[name],value) for name,value in expected.items()}
        if L.ple_forward is not original_ple or M.ple_forward is not original_model_ple:
            raise RuntimeError('PLE interposition did not restore original functions')
        conditions[label]={'metrics':metrics,'seconds':time.monotonic()-condition_start,
                           'native_calls':dict(mode.calls),'native_misses':list(mode.misses),
                           'fresh_layer_state':True}
        print(label,json.dumps(metrics),flush=True)

    changed=[name for name,digest in source_hashes.items() if file_hash(root/name)!=digest]
    changed_external=[name for name,digest in external_hashes.items() if file_hash(external_sources[name])!=digest]
    if changed or changed_external:raise ValueError('Source changed during PLE comparison')
    result={'schema':'ngramma-ple-scalar-probe/v1','identity_sha256':identity['identity_sha256'],
            'tokens':tokens,'layer':1,'threads':4,'ram_cache_bytes':2<<30,
            'n_embd':d,'conditions':conditions,'original_divisor':d**.5,
            'corrected_float32_coefficient':float(coefficient),
            'corrected_coefficient_ieee754_le_hex':struct.pack('<f',float(coefficient)).hex(),
            'original_transcription_matches_upstream':True,
            'capture_identity':capture_identity,'source_sha256':source_hashes,
            'external_source_sha256':external_hashes,'reference_sha256':reference_hashes,
            'compared_tensor_sha256':tensor_hashes,'sources_changed_during_run':changed,
            'external_sources_changed_during_run':changed_external,
            'native_library_sha256':mode.library_sha256,'native_build_record':mode.build_record,
            'seconds_including_initialization':time.monotonic()-start,
            'initial_available_bytes':budget['available_bytes'],
            'diagnostic_only':True,'admitted_for_gradients':False}
    atomic_json(args.output,result)


if __name__=='__main__':main()

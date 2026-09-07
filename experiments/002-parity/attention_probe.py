"""Compare recurrent-attention stages against saved CPU-engine tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--native-library',type=Path,required=True)
    p.add_argument('--native-recurrent',action='store_true')
    p.add_argument('--native-repack',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import torch
    from ngramma_runtime.environment import worker_imports
    worker_imports()
    from ngramma_runtime.native_forward import NativeEngineWeights,NativeForward
    from engraft.replica.hparams import Hparams
    from engraft.replica.model import load_delta_net_weights
    import engraft.replica.layers as L
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    identity=json.loads(args.manifest.read_text());paths=[s['path'] for s in identity['shards']]
    w=NativeEngineWeights(paths,ram_cache_bytes=1<<30)
    hp=Hparams.from_gguf_paths(paths[0],paths[1]);d=hp.ssm_d_state;hk=hp.ssm_n_group;hv=hp.ssm_dt_rank
    meta=json.loads((args.reference/'tensors.json').read_text())
    def ref(name):
        item=next(x for x in meta if x['name']==name)
        return torch.from_numpy(np.ndarray(tuple(reversed(item['shape'])),dtype='<f4',
            buffer=(args.reference/item['file']).read_bytes(),strides=tuple(reversed(item['strides']))).copy().squeeze())
    results={}
    def check(name,actual,expected):
        expected=expected.reshape(actual.shape);diff=(actual-expected).abs()
        results[name]={'max_abs':float(diff.max()),'relative_rms':float(diff.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-12)),
                       'different_values':int(torch.count_nonzero(diff)),'values':diff.numel()}
        print(name,json.dumps(results[name]),flush=True)
    mode=NativeForward(w,args.native_library,recurrent=args.native_recurrent,repack=args.native_repack)
    start=time.monotonic()
    with torch.no_grad(),mode:
        x=ref('hc_mixed-0');weight=load_delta_net_weights(w,0);state=L.delta_net_init_state(hp);n=len(x)
        qkv=x@weight.wqkv.T;check('qkv',qkv,ref('linear_attn_qkv_mixed-0'))
        z=x@weight.wqkv_gate.T;check('z',z,ref('z-0'))
        beta_raw=x@weight.ssm_beta.T;check('beta_raw',beta_raw,ref('beta-0'))
        beta=torch.sigmoid(beta_raw);check('beta',beta,ref('beta_sigmoid-0'))
        alpha=x@weight.ssm_alpha.T;check('alpha',alpha,ref('alpha-0'))
        soft=torch.nn.functional.softplus(alpha+weight.ssm_dt_bias);check('softplus',soft,ref('a_softplus-0'))
        gate=soft*weight.ssm_a;check('log_decay',gate,ref('gate-0'))
        conv=L.causal_depthwise_conv(qkv,state.conv_hist,weight.ssm_conv1d);check('conv_raw',conv,ref('conv_output_raw-0'))
        conv=torch.nn.functional.silu(conv);check('conv_silu',conv,ref('conv_output_silu-0'))
        q=L.l2norm(conv[:,:d*hk].reshape(n,hk,d),hp.f_norm_rms_eps)
        k=L.l2norm(conv[:,d*hk:2*d*hk].reshape(n,hk,d),hp.f_norm_rms_eps)
        v=conv[:,2*d*hk:].reshape(n,hv,d)
        check('q_normalized',q,ref('q_conv_predelta-0'));check('k_normalized',k,ref('k_conv_predelta-0'));check('v',v,ref('v_conv_predelta-0'))
        output,snew=L.gated_delta_net_recurrence(q.tile((1,hv//hk,1)),k.tile((1,hv//hk,1)),v,gate,beta,state.s)
        check('recurrent_output',output,ref('attn_output-0'))
        check('recurrent_state',snew,ref('new_state-0').transpose(-1,-2).contiguous())
        final=(L.rmsnorm(output,weight.ssm_norm,hp.f_norm_rms_eps)*torch.sigmoid(z.reshape(n,hv,d))).reshape(n,-1)
        check('gated_normalized_output',final,ref('final_output-0'))
        result=final@weight.ssm_out.T;check('projected_output',result,ref('linear_attn_out-0'))
    evidence={'schema':'ngramma-attention-probe/v1','identity_sha256':identity['identity_sha256'],
              'native_recurrent':args.native_recurrent,'metrics':results,'seconds':time.monotonic()-start,
              'native_repack':args.native_repack,'native_build_record':mode.build_record,
              'native_calls':dict(mode.calls),'native_library_sha256':mode.library_sha256,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'diagnostic_only':True,'training_qualified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':main()

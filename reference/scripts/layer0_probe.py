from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
from flash_memory.environment import ROOT,worker_imports
worker_imports()
import numpy as np
import torch
from engraft.replica.hparams import Hparams
from engraft.replica.model import Replica,LayerState,load_hc,load_delta_net_weights,load_moe_nonexpert
from engraft.replica.layers import hc_mix,hc_combine,linear_attn_layer,delta_net_init_state,moe_ffn
from flash_memory.weights import EngineWeights
from flash_memory.activation_reference import ActivationReference
from flash_memory.artifacts import atomic_json
torch.set_num_threads(4)
identity=json.loads((ROOT/'artifacts/model-identity.json').read_text());paths=[s['path'] for s in identity['shards']]
w=EngineWeights(paths,ram_cache_bytes=4<<30);hp=Hparams.from_gguf_paths(paths[0],paths[1])
directory=ROOT/'artifacts/layer0-reference';meta=json.loads((directory/'tensors.json').read_text())
def tensor(name, occurrence=0):
    item=[m for m in meta if m['name']==name][occurrence]
    raw=(directory/item['file']).read_bytes();dtype='<i4' if item['type']==26 else '<f4'
    return torch.from_numpy(np.ndarray(tuple(reversed(item['shape'])),dtype=dtype,buffer=raw,strides=tuple(reversed(item['strides']))).copy().squeeze())
results={}
def compare(name,actual,expected):
    d=(actual-expected).abs();result={'max_abs':float(d.max()),'relative_rms':float(d.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-12))};results[name]=result;print(name,result,flush=True)
with torch.no_grad(),ActivationReference(w):
    x=tensor('hc_init');mixed,inject=hc_mix(x,*load_hc(w,0,'attn'),hp.f_norm_rms_eps,hp.hc_mult)
    tokens=json.loads((ROOT/'artifacts/reference-short/tokens.json').read_text())
    embeddings=torch.from_numpy(w.embedding_rows(tokens)).unsqueeze(1).repeat(1,hp.hc_mult,1)
    compare('embedding',embeddings,x)
    compare('hc_mix',mixed,tensor('hc_mixed-0'))
    output,_=linear_attn_layer(tensor('hc_mixed-0'),load_delta_net_weights(w,0),delta_net_init_state(hp),hp)
    compare('linear_attention_from_exact_input',output,tensor('linear_attn_out-0'))
    moe=load_moe_nonexpert(w,0);x=tensor('hc_mixed-0',1)
    routing=torch.topk(torch.softmax(x@moe.gate_inp.T,-1),hp.n_expert_used,-1).indices
    expected_routing=tensor('ffn_moe_topk-0').long()
    results['routing_equal']=bool(torch.equal(routing,expected_routing));print('routing_equal',results['routing_equal'],flush=True)
    def expert(name):return lambda e:torch.from_numpy(w.expert('blk.0.'+name+'.weight',e,persist=True))
    output=moe_ffn(x,moe.gate_inp,expert('ffn_gate_exps'),expert('ffn_up_exps'),expert('ffn_down_exps'),expected_routing,moe.up_shexp,moe.gate_shexp,moe.down_shexp,moe.gate_inp_shexp,hp.n_expert_used)
    compare('moe_from_exact_input_and_routing',output,tensor('ffn_out-0'))
    rep=Replica(hp,w,None)
    whole,_,selected=rep.run_layer(0,embeddings,torch.arange(len(tokens),dtype=torch.float64),LayerState(),None,None,persist_experts=True)
    compare('whole_layer0',whole,tensor('l_last-0'))
    results['whole_layer_routing_equal']=bool(torch.equal(selected,expected_routing))
    results['routing_differences']=[{'position':i,'actual':selected[i].tolist(),'expected':expected_routing[i].tolist()} for i in range(len(tokens)) if not torch.equal(selected[i],expected_routing[i])]
    print('whole_layer_routing',results['routing_differences'],flush=True)
atomic_json(ROOT/'artifacts/layer0-quantized-diagnosis.json',results)

#!/usr/bin/env python3
"""Actual-checkpoint forward parity; no training examples or teacher calls."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import resource
import re
import time
from contextlib import nullcontext
from flash_memory.environment import ROOT, worker_imports
worker_imports()
import numpy as np
import torch
from engraft.replica.hparams import Hparams
from flash_memory.table import ModelTable
from flash_memory.weights import EngineWeights
from flash_memory.sequence import SequenceReplica, SharedRows
from flash_memory.artifacts import atomic_json
from flash_memory.artifacts import file_hash

p=argparse.ArgumentParser();p.add_argument('--reference',default='artifacts/reference-short');p.add_argument('--gradient',action='store_true');p.add_argument('--activation-quantization',action='store_true');args=p.parse_args()
if args.activation_quantization and args.gradient: raise SystemExit('Quantized forward reference has no admitted gradient; cannot request backward')
reference=ROOT/args.reference
tokens=json.loads((reference/'tokens.json').read_text())
identity=json.loads((ROOT/'artifacts/model-identity.json').read_text())
paths=[s['path'] for s in identity['shards']]
source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'flash_memory/sequence.py',ROOT/'flash_memory/weights.py',ROOT/'flash_memory/activation_reference.py',ROOT/'vendor/engraft/replica/layers.py',ROOT/'runtime/libflash-memory-dequant.so']}
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.manual_seed(1234)
table=ModelTable(paths);weights=EngineWeights(paths,ram_cache_bytes=4<<30)
hp=Hparams.from_gguf_paths(paths[0],paths[1])
replica=SequenceReplica(hp,weights,table,max_tokens=128)
class Captures(dict):
    def on_layer(self,il,x):
        item=next(t for t in json.loads((reference/'tensors.json').read_text()) if t['name']==f'l_last-{il}')
        raw=(reference/item['file']).read_bytes()
        expected=np.ndarray(tuple(reversed(item['shape'])),dtype='<f4',buffer=raw,strides=tuple(reversed(item['strides']))).reshape(tuple(x.shape))
        actual=x.detach().numpy();difference=np.abs(actual-expected)
        print(json.dumps({'layer':il,'max_abs':float(difference.max()),'relative_rms':float(np.sqrt(np.mean(difference**2))/max(np.sqrt(np.mean(expected**2)),1e-9)),'seconds':time.monotonic()-start}),flush=True)
start=time.monotonic();capture=Captures()
from flash_memory.activation_reference import ActivationReference
mode=ActivationReference(weights) if args.activation_quantization else nullcontext()
with torch.no_grad(),mode: logits=replica.full(tokens,capture=capture)
elapsed=time.monotonic()-start
stem='replica-quantized' if args.activation_quantization else 'replica'
np.save(ROOT/f'artifacts/{stem}-logits.npy',logits.numpy())
np.savez(ROOT/f'artifacts/{stem}-intermediates.npz',**{k:v.numpy() for k,v in capture.items()})
expected=np.fromfile(reference/'logits.f32',dtype='<f4').reshape(len(tokens),-1)
diff=np.abs(logits.numpy()-expected)
selected=np.argmax(expected,axis=-1)
lp=torch.log_softmax(logits,-1).numpy();ref_lp=torch.log_softmax(torch.from_numpy(expected),-1).numpy()
selected_diff=np.abs(lp[np.arange(len(tokens)),selected]-ref_lp[np.arange(len(tokens)),selected])
intermediates=[]
for item in json.loads((reference/'tensors.json').read_text()):
    name=item['name']
    if name=='ple_embd': key='ple_embd'
    elif name.startswith('l_last-'): key='layer_'+re.match(r'l_last-(\d+)',name).group(1)
    else: continue
    if item['type']!=0: raise ValueError('Expected f32 diagnostic tensor')
    raw=(reference/item['file']).read_bytes()
    view=np.ndarray(tuple(reversed(item['shape'])),dtype='<f4',buffer=raw,strides=tuple(reversed(item['strides']))).squeeze(axis=0)
    actual=capture[key].numpy()
    if view.size != actual.size: continue  # cropped diagnostic alias, not a full-layer output
    view=view.reshape(actual.shape)
    error=np.abs(actual-view)
    intermediates.append({'name':name,'max_abs':float(error.max()),'relative_rms':float(np.sqrt(np.mean(error**2))/max(np.sqrt(np.mean(view**2)),1e-9))})
result={'identity_sha256':identity['identity_sha256'],'tokens':tokens,'seconds':elapsed,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1<<20),'max_logit_error':float(diff.max()),'mean_logit_error':float(diff.mean()),'selected_logprob_error_max':float(selected_diff.max()),'top1_agreement':float(np.mean(logits.argmax(-1).numpy()==selected)),'intermediates':intermediates,'weight_io':weights.stats,'logit_parity':bool(selected_diff.max()<0.02),'intermediate_parity':bool(intermediates and max(x['relative_rms'] for x in intermediates)<0.01),'note':'Single short CPU sequence. Must expand coverage before training admission.'}
result['activation_quantization']=args.activation_quantization
result['source_sha256']=source_hashes
result['admitted_for_gradients']=False
atomic_json(ROOT/f'artifacts/{stem}-parity.json',result);print(json.dumps(result),flush=True)
if args.gradient:
    if not result['logit_parity'] or not result['intermediate_parity']: raise SystemExit('Forward parity failed; gradient run refused')
    addresses=table.global_addresses(tokens)
    # Earlier prompt row, not just final-position rows: tests the full history path.
    selected_rows=np.unique(addresses[:max(1,len(tokens)//2),8:].reshape(-1))
    rows=SharedRows(table,selected_rows)
    target=int(selected[-1]);start=time.monotonic()
    output=replica.full(tokens,rows)
    loss=-torch.log_softmax(output[-1],-1)[target]
    loss.backward()
    grad=rows.normalized_delta.grad.detach().clone()
    direction=grad/grad.norm().clamp_min(1e-20)
    predicted=float((grad*direction).sum());measurements=[]
    for epsilon in (1e-3,3e-3,1e-2):
        vals=[]
        for sign in (-1,1):
            with torch.no_grad():
                rows.normalized_delta.copy_(sign*epsilon*direction)
                value=-torch.log_softmax(replica.full(tokens,rows)[-1],-1)[target]
                vals.append(float(value))
        measured=(vals[1]-vals[0])/(2*epsilon)
        measurements.append({'epsilon':epsilon,'predicted':predicted,'measured':measured,'relative_error':abs(measured-predicted)/max(abs(predicted),1e-9)})
    rows.normalized_delta.data.zero_()
    atomic_json(ROOT/'artifacts/gradient-probe.json',{'diagnostic_only':True,'target_source':'original student top-1, no teacher training data','selected_rows':selected_rows.tolist(),'seconds':time.monotonic()-start,'gradient_norm':float(grad.norm()),'finite_differences':measurements,'passed':bool(predicted>0 and any(m['relative_error']<0.1 for m in measurements)),'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1<<20)})

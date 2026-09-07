#!/usr/bin/env python3
"""Exercise actual model outputs and row addresses across inference paths."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import os
import subprocess
import time
import numpy as np
from flash_memory.environment import ROOT,config
from flash_memory.artifacts import atomic_json
from flash_memory.table import ModelTable

cfg=config();identity=json.loads((ROOT/'artifacts/model-identity.json').read_text())
table=ModelTable([s['path'] for s in identity['shards']])
reference=json.loads((ROOT/'artifacts/reference-short/tokens.json').read_text())
unicode_tokens=json.loads((ROOT/'artifacts/reference-lens.jsonl').read_text().splitlines()[-1])['tokens']
eos=table.eos_token_id
cases=[('short',reference,32),('unicode-chat',unicode_tokens,5),('eos-repeat',[reference[0],eos,*reference[:4],*reference[:4],eos,reference[0]],1)]
summary={'identity_sha256':identity['identity_sha256'],'cases':[(n,len(t),c) for n,t,c in cases],'conditions':{},'note':'Actual-model CPU f32 KV. This does not validate GPU training or production persistence.'}
root=ROOT/'artifacts/compatibility';root.mkdir(exist_ok=True)
for condition in ('original','disk','empty','original-rows'):
    condition_dir=root/condition;condition_dir.mkdir(exist_ok=True)
    jobs=[{'tokens':tokens,'chunk_size':chunk,'output_dir':str(condition_dir/name),'capture':['ple_embd','ple_gate','ple_conv_out']} for name,tokens,chunk in cases]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['LD_LIBRARY_PATH']=str(Path(cfg['reference_runtime']) if condition=='original' else ROOT/'runtime')
    env.pop('FLASH_MEMORY_OVERLAY',None)
    trace=condition_dir/'addresses.jsonl';trace.unlink(missing_ok=True)
    env['FLASH_MEMORY_TRACE']=str(trace)
    if condition in ('empty','original-rows'):env['FLASH_MEMORY_OVERLAY']=str(ROOT/'artifacts/overlays'/condition/'rows.fml')
    cmd=[str(ROOT/'runtime/flash-memory-lens'),'-m',identity['shards'][0]['path'],'--device','none','--fit','off','-ngl','0','-c','128','-b','32','-ub','32','-t','8','-tb','8','-fa','off','-ctk','f32','-ctv','f32']
    if condition!='original':cmd+=['--ngram-on-disk','--ngram-io-threads','4','--ngram-cache','64']
    start=time.monotonic();print('Starting '+condition,flush=True)
    with (condition_dir/'stderr.log').open('w') as err:
        result=subprocess.run(cmd,input=''.join(json.dumps(j)+'\n' for j in jobs),text=True,capture_output=False,stdout=subprocess.PIPE,stderr=err,env=env,timeout=1800)
    (condition_dir/'responses.jsonl').write_text(result.stdout)
    responses=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    if result.returncode or len(responses)!=len(jobs)+1 or any(x.get('ok') is not True for x in responses[1:]):raise RuntimeError(f'{condition} failed; inspect {condition_dir}')
    errors=[]
    for name,tokens,chunk in cases:
        actual=np.fromfile(condition_dir/name/'logits.f32',dtype='<f4')
        if condition!='original':
            expected=np.fromfile(root/'original'/name/'logits.f32',dtype='<f4')
            errors.append({'case':name,'max_abs':float(np.max(np.abs(actual-expected))),'exact':bool(np.array_equal(actual,expected))})
    address_count=0
    if condition!='original':
        records=[json.loads(line) for line in trace.read_text().splitlines()]
        cursor=0
        for name,tokens,chunk in cases:
            got=records[cursor:cursor+len(tokens)];cursor+=len(tokens)
            if [r['position'] for r in got]!=list(range(len(tokens))) or not np.array_equal([r['rows'] for r in got],table.global_addresses(tokens)):
                raise RuntimeError('Address mismatch: '+name)
            address_count+=len(tokens)*table.n_heads
        if cursor!=len(records):raise RuntimeError('Unexpected traced positions')
    summary['conditions'][condition]={'seconds':time.monotonic()-start,'logit_comparisons':errors,'address_count':address_count,'passed':all(e['exact'] for e in errors)}
    atomic_json(ROOT/'artifacts/compatibility-progress.json',summary)
    print(condition+': '+json.dumps(summary['conditions'][condition]),flush=True)
summary['empty_overlay']=summary['conditions']['empty']['passed']
summary['original_row_overlay']=summary['conditions']['original-rows']['passed']
summary['addresses']=True
summary['passed']=all(c['passed'] for c in summary['conditions'].values())
atomic_json(ROOT/'artifacts/engine-compatibility.json',summary)

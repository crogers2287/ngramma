"""Export original-row controls and one clearly marked non-training perturbation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from flash_memory.environment import ROOT
from flash_memory.table import ModelTable
from flash_memory.overlay import export_overlay,load_overlay
m=json.loads((ROOT/'artifacts/model-identity.json').read_text());table=ModelTable([s['path'] for s in m['shards']])
tokens=json.loads((ROOT/'artifacts/reference-short/tokens.json').read_text());addresses=table.global_addresses(tokens)
cases=[('empty',np.array([],np.int64)),('original-rows',np.unique(addresses.reshape(-1))),('diagnostic-perturbation',np.array([addresses[2,8]],np.int64))]
for name,rows in cases:
    directory=ROOT/'artifacts/overlays'/name
    if directory.exists():load_overlay(directory,m,table);continue
    delta=np.zeros((len(rows),160),np.float32)
    if name=='diagnostic-perturbation':delta[0,0]=float(np.sqrt(np.mean(table.read_global(rows)**2)))*0.1
    export_overlay(directory,m,table,rows,delta,{'kind':'diagnostic_only','teacher':None,'purpose':'Original-row controls and an inference-path perturbation; no improvement claim'})
    print(name,len(rows),flush=True)

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import os
import struct
import subprocess
import numpy as np
from flash_memory.environment import ROOT
from flash_memory.overlay import MAGIC
from flash_memory.artifacts import canonical,atomic_json

base=ROOT/'artifacts/overlays/original-rows/rows.fml';raw=base.read_bytes();size=struct.unpack('<I',raw[8:12])[0]
header=json.loads(raw[12:12+size]);payload=raw[12+size:]
out=ROOT/'artifacts/negative-overlays';out.mkdir(exist_ok=True)
cases=[]
def write(name,h,p,corrupt=False):
    h=json.loads(json.dumps(h));h['payload_sha256']=hashlib.sha256(p).hexdigest()
    j=canonical(h);data=MAGIC+struct.pack('<I',len(j))+j+p
    if corrupt:data=data[:-1]+bytes([data[-1]^1])
    path=out/(name+'.fml');path.write_bytes(data);cases.append((name,path))
write('payload-corruption',header,payload,True)
write('truncated',header,payload[:-3])
h=json.loads(json.dumps(header));h['row_dim']=159;write('wrong-dimension',h,payload)
p=bytearray(payload);p[4:8]=p[:4];write('duplicate-row',header,bytes(p))
p=bytearray(payload);offset=header['row_count']*4;p[offset:offset+4]=struct.pack('<f',float('nan'));write('nonfinite-row',header,bytes(p))
h=json.loads(json.dumps(header));h['model_identity']['shards'][0]['sha256']='0'*64;write('wrong-checkpoint',h,payload)
results=[]
for name,path in cases:
    env=os.environ.copy();env['FLASH_MEMORY_OVERLAY']=str(path)
    cmd=[str(ROOT/'runtime/overlay-inspect')]
    if name=='wrong-checkpoint':cmd += [s['path'] for s in header['model_identity']['shards']]
    r=subprocess.run(cmd,text=True,capture_output=True,env=env,timeout=60)
    result={'case':name,'rejected':r.returncode!=0,'message':r.stderr.strip()};results.append(result);print(result,flush=True)
atomic_json(ROOT/'artifacts/overlay-negative-checks.json',{'passed':all(r['rejected'] for r in results),'cases':results})
if not all(r['rejected'] for r in results):raise SystemExit('Invalid overlay accepted')

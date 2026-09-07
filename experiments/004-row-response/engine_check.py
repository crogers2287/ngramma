#!/usr/bin/env python3
"""Fresh full-engine overlays with live routing; append measured response evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('response','manifest','reference','runtime','lens','local','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    import numpy as np
    from ngramma_runtime.artifacts import atomic_json, file_hash
    from ngramma_runtime.resources import check_budget
    if args.output.exists():
        raise ValueError('Use a new result path')
    result = json.loads(args.response.read_text())
    root = Path(__file__).resolve().parents[2]
    capture_script = root/'scripts/capture_reference.py'
    source_hashes = {str(path.relative_to(root)):file_hash(path) for path in (Path(__file__),capture_script)}
    runtime_hashes = {path.name:file_hash(path) for path in sorted(args.runtime.glob('*.so'))}
    identity = json.loads(args.manifest.read_text())
    if result['experiment']['identity_sha256'] != identity['identity_sha256']:
        raise ValueError('Response and model identities differ')
    spec = result['experiment']
    tokens = spec['tokens']
    token_file = args.local/'tokens.json'
    atomic_json(token_file,tokens)
    n = len(tokens)
    original = np.fromfile(args.reference/'logits.f32',dtype='<f4').reshape(n,-1)
    if file_hash(args.reference/'logits.f32') != result['capture_identity']['logits_sha256']:
        raise ValueError('Baseline logits changed')
    selected = result['selected_engine_labels']
    by_label = {point['overlay_label']:point for point in result['points']}
    baseline_memory = None
    records = []
    capture_prefixes = ['ple_embd','ple_gated_value','ple_conv_out','l_last-0']

    def logical(directory, name, shape):
        meta = json.loads((directory/'tensors.json').read_text())
        entries = [item for item in meta if item['name'] == name]
        if len(entries) != 1 or entries[0]['type'] != 0:
            raise ValueError('Expected one F32 tensor '+name)
        item = entries[0]
        path = (directory/item['file']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Unsafe tensor path')
        a = np.ndarray(tuple(reversed(item['shape'])),dtype='<f4',buffer=path.read_bytes(),
                       strides=tuple(reversed(item['strides'])))
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite engine tensor')
        return a.copy().reshape(shape)

    for label in ['unmodified',*selected]:
        check_budget()
        directory = args.local/('engine-'+label)
        command = [sys.executable,str(capture_script),'--manifest',str(args.manifest),
                   '--runtime',str(args.runtime),'--lens',str(args.lens),'--tokens',str(token_file),
                   '--output',str(directory),'--threads','8','--timeout','900']
        for prefix in capture_prefixes:
            command.extend(['--capture',prefix])
        if label != 'unmodified':
            overlay = args.local/label/'rows.fml'
            if file_hash(overlay) != by_label[label]['overlay_sha256']:
                raise ValueError('Overlay changed since local response probe')
            command.extend(['--overlay',str(overlay)])
        start = time.monotonic()
        print('Running fresh full engine:',label,flush=True)
        completed = subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=930)
        (args.local/(label+'-capture-driver.log')).write_text(completed.stdout+completed.stderr)
        if completed.returncode:
            raise RuntimeError('Engine capture failed for '+label+'; inspect local logs')
        summary = json.loads((directory/'capture-summary.json').read_text())
        values = np.fromfile(directory/'logits.f32',dtype='<f4').reshape(original.shape)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite engine logits')
        mem = logical(directory,'ple_embd',(n,16,160))
        hidden = logical(directory,'l_last-0',(n,4,2560))
        ple = hidden + (logical(directory,'ple_gated_value-1',(n,4,2560)) + logical(directory,'ple_conv_out-1',(n,4,2560)))
        ple_hash = hashlib.sha256(ple.tobytes()).hexdigest()
        if label == 'unmodified':
            baseline_memory = mem
            baseline_hidden = hidden
            if not np.array_equal(values,original):
                raise ValueError('Fresh unmodified engine differs from pinned baseline')
            target_hash = by_label['zero']['stage_sha256']['output']
        else:
            point = by_label[label]
            target_hash = point['stage_sha256']['output']
            expected_memory = baseline_memory.copy()
            raw = (args.local/label/'rows.fml').read_bytes()
            header_size = int.from_bytes(raw[8:12],'little')
            payload = raw[12+header_size:]
            if int.from_bytes(payload[:4],'little',signed=True) != int(spec['row_id']):
                raise ValueError('Unexpected overlay row')
            replacement = np.frombuffer(payload[4:],dtype='<f4')
            for ti,hi in spec['occurrences']:
                expected_memory[ti,hi] = replacement
            if not np.array_equal(mem,expected_memory):
                raise ValueError('Engine gathered values do not match the intended row replacement')
            if not np.array_equal(hidden,baseline_hidden):
                raise ValueError('Changing memory altered the earlier layer0 residual')
        if ple_hash != target_hash:
            raise ValueError('Local native PLE and full engine PLE differ for '+label)
        ids = spec['engine_margin']['token_ids']
        margin = float(values[-1,ids[0]]) - float(values[-1,ids[1]])
        unchanged = np.array_equal(values,original)
        if label == 'zero' and not unchanged:
            raise ValueError('Zero overlay changed engine outputs')
        if label != 'unmodified' and not (by_label[label]['key_changed_count'] or by_label[label]['value_changed_count']) and not unchanged:
            raise ValueError('Unchanged projections produced changed full logits: isolation failure')
        record = {'label':label,'capture':summary,'ple_matches_local_native':True,
                  'ple_output_sha256':ple_hash,'logits_bitwise_equal_to_original':unchanged,
                  'changed_logit_values':int(np.count_nonzero(values != original)),
                  'max_abs_logit_change':float(np.abs(values.astype(np.float64)-original).max()),
                  'top1_agree_positions':int(np.count_nonzero(values.argmax(-1) == original.argmax(-1))),
                  'final_top1_token':int(values[-1].argmax()),
                  'logit_margin':margin,'logit_margin_delta':margin-spec['engine_margin']['baseline'],
                  'fresh_process':True,'live_routing_and_attention':True,
                  'gathered_replacement_verified':True,'seconds':time.monotonic()-start}
        if label != 'unmodified':
            by_label[label]['engine_logit_margin_delta'] = record['logit_margin_delta']
        records.append(record)
        atomic_json(args.local/'engine-progress.json',records)
        print(json.dumps({k:record[k] for k in ('label','changed_logit_values','logit_margin_delta','top1_agree_positions')}),flush=True)
    if any(file_hash(root/name) != value for name,value in source_hashes.items()):
        raise ValueError('Engine driver source changed during execution')
    if any(file_hash(args.runtime/name) != value for name,value in runtime_hashes.items()):
        raise ValueError('Engine runtime changed during execution')
    result['engine_checks'] = records
    result['engine_source_sha256'] = source_hashes
    result['engine_runtime_sha256'] = runtime_hashes
    result['local_response_sha256'] = file_hash(args.response)
    result['engine_shard_authentication'] = 'Every overlay load verifies complete shard sizes, paths, and SHA256 using the archived engine overlay hook.'
    atomic_json(args.output,result)


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Replay saved PLE activation encodings; no model weights or forwards."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ngramma_runtime.activation_encoding import ActivationEncoding


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference','overlay-root','response','activation-library','cpu-library','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new output; preserve prior controls')
    spec_path=args.overlay_root/'spec.json'
    spec=json.loads(spec_path.read_text())
    if spec['occurrences'] != [[6,8]]:raise ValueError('Expected the single preregistered occurrence')
    meta_path=args.reference/'tensors.json'
    meta=json.loads(meta_path.read_text())
    entry,=[item for item in meta if item['name']=='ple_embd']
    if entry['type']!=0 or entry['shape']!=[2560,17,1,1]:raise ValueError('Unexpected capture geometry/type')
    tensor_path=(args.reference/entry['file']).resolve()
    if not tensor_path.is_relative_to(args.reference.resolve()):raise ValueError('Unsafe capture path')
    memory=np.ndarray(tuple(reversed(entry['shape'])),dtype='<f4',buffer=tensor_path.read_bytes(),
                      strides=tuple(reversed(entry['strides']))).copy().reshape(17,16,160)
    if not np.isfinite(memory).all():raise ValueError('Nonfinite saved activation')
    build_path=Path(str(args.activation_library)+'.build.json')
    build=json.loads(build_path.read_text())
    if build['library_sha256']!=sha(args.activation_library):raise ValueError('Activation library/build mismatch')
    if build['linked_libraries']['libggml-cpu.so']!=sha(args.cpu_library):raise ValueError('CPU library/build mismatch')
    response=json.loads(args.response.read_text())
    if response['activation_library_sha256']!=build['library_sha256']:raise ValueError('Response/library mismatch')
    if response['native_build_record']['linked_libraries']!=build['linked_libraries']:raise ValueError('Native projection/encoding runtime mismatch')
    encoder=ActivationEncoding(args.activation_library)
    points={x['overlay_label']:x for x in response['points']}
    encodings={};inputs={};overlay_hashes={}
    for label in ('zero','minus-10','plus-10'):
        path=args.overlay_root/label/'rows.fml';raw=path.read_bytes()
        if raw[:8]!=b'FMLROW1\0':raise ValueError('Unexpected overlay magic')
        length=int.from_bytes(raw[8:12],'little');header=json.loads(raw[12:12+length]);payload=raw[12+length:]
        if len(payload)!=644 or header['row_count']!=1 or header['row_dim']!=160:raise ValueError('Unexpected overlay payload')
        if hashlib.sha256(payload).hexdigest()!=header['payload_sha256']:raise ValueError('Overlay payload hash mismatch')
        if int.from_bytes(payload[:4],'little',signed=True)!=spec['row_id']:raise ValueError('Unexpected row')
        if header['model_identity']['identity_sha256']!=spec['identity_sha256']:raise ValueError('Overlay model identity mismatch')
        overlay_hashes[label]=sha(path)
        if overlay_hashes[label]!=points[label]['overlay_sha256']:raise ValueError('Overlay response hash mismatch')
        replacement=np.frombuffer(payload[4:],dtype='<f4')
        value=memory.copy();value[6,8]=replacement
        if label=='zero' and value.tobytes()!=memory.tobytes():raise ValueError('Zero overlay differs from captured input')
        inputs[label]=value.reshape(17,2560)
        encodings[label]=encoder.encode(8,inputs[label]).reshape(17,80,34)
        if hashlib.sha256(encodings[label].tobytes()).hexdigest()!=points[label]['activation_sha256']:raise ValueError('Replay differs from response encoding')
    base=encodings['zero'];transitions={}
    for label in ('minus-10','plus-10'):
        encoded=encodings[label];changes=encoded!=base;records=[]
        for ti,block in np.argwhere(changes.any(-1)):
            before=base[ti,block];after=encoded[ti,block]
            before_input=inputs['zero'][ti,block*32:(block+1)*32]
            after_input=inputs[label][ti,block*32:(block+1)*32]
            before_maxima=np.flatnonzero(np.abs(before_input)==np.abs(before_input).max())
            after_maxima=np.flatnonzero(np.abs(after_input)==np.abs(after_input).max())
            scale_before=float(np.frombuffer(before[:2].tobytes(),dtype='<f2')[0])
            scale_after=float(np.frombuffer(after[:2].tobytes(),dtype='<f2')[0])
            records.append({'token_index':int(ti),'global_block_index':int(block),'head':int(block//5),
                'head_block_index':int(block%5),
                'absolute_maximum_indices_before':before_maxima.tolist(),
                'absolute_maximum_indices_after':after_maxima.tolist(),
                'maximum_signs_before':[int(np.sign(before_input[i])) for i in before_maxima],
                'maximum_signs_after':[int(np.sign(after_input[i])) for i in after_maxima],
                'scale_before':scale_before,'scale_after':scale_after,
                'scale_delta':scale_after-scale_before,'scale_before_bits_hex':f'{int.from_bytes(before[:2].tobytes(),"little"):04x}',
                'scale_after_bits_hex':f'{int.from_bytes(after[:2].tobytes(),"little"):04x}',
                'scale_changed_bytes':int(np.count_nonzero(changes[ti,block,:2])),
                'code_changed_bytes':int(np.count_nonzero(changes[ti,block,2:])),
                'code_unchanged_count':int(np.count_nonzero(~changes[ti,block,2:]))})
        transitions[label]={'changed_scale_fields':int(changes[:,:,:2].any(-1).sum()),
            'changed_scale_bytes':int(changes[:,:,:2].sum()),'changed_code_bytes':int(changes[:,:,2:].sum()),
            'changed_blocks':records,'input_sha256':hashlib.sha256(inputs[label].tobytes()).hexdigest(),
            'encoded_sha256':hashlib.sha256(encoded.tobytes()).hexdigest(),
            'local_scalar_delta_from_response':points[label]['local_scalar_delta']}
    changed_sets={label:{(x['token_index'],x['global_block_index']) for x in record['changed_blocks']} for label,record in transitions.items()}
    result={'schema':'ngramma.scale-transition/v1','evidence_kind':'saved-activation encoder replay; no model forward',
        'scope':{'occurrences':spec['occurrences'],'epsilon_magnitude':2**-10,'token_count':17,'block_elements':32,'block_bytes':34},
        'transitions':transitions,'both_signs_change_same_block_set':changed_sets['minus-10']==changed_sets['plus-10'],
        'all_replays_match_saved_response_hashes':True,
        'baseline_encoded_sha256':hashlib.sha256(base.tobytes()).hexdigest(),
        'baseline_input_sha256':hashlib.sha256(inputs['zero'].tobytes()).hexdigest(),
        'provenance':{'script_sha256':sha(__file__),'wrapper_sha256':sha(Path(__import__('ngramma_runtime.activation_encoding',fromlist=['x']).__file__)),
            'reference_metadata_sha256':sha(meta_path),'reference_ple_embd_sha256':sha(tensor_path),
            'reference_capture_summary_sha256':sha(args.reference/'capture-summary.json'),
            'spec_sha256':sha(spec_path),'response_sha256':sha(args.response),'overlay_sha256':overlay_hashes,
            'activation_library_sha256':sha(args.activation_library),'cpu_library_sha256':sha(args.cpu_library),
            'activation_build_record_sha256':sha(build_path),'activation_build_record':build},
        'interpretation':'Block41 has tied positive absolute maxima at local indices0 and31; opposite alternating edits raise opposite members, increasing its FP16scale by one ULP for both signs. Block40 moves in opposite directions. The encoded perturbations therefore are not opposite vectors. A same-sign scalar response to opposite row edits is compatible with such asymmetric finite steps; the encoder alone does not attribute the scalar response to individual projection weights or nonlinear operations.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'transitions':transitions,'both_signs_change_same_block_set':result['both_signs_change_same_block_set']},indent=2))


if __name__=='__main__':main()

"""Model-free FML wire-format and manifest tests; no NumPy or real shards."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest
from ngramma_runtime.artifacts import digest
from ngramma_runtime.overlay_inspect import inspect_overlay,MAX_HEADER_BYTES,MAX_ROWS


def sign(identity):
    identity['identity_sha256']=digest({key:value for key,value in identity.items() if key!='identity_sha256'})
    return identity


@pytest.fixture
def identity():
    return sign({'schema':'flash-memory-model/v1','shards':[{'path':'/unmounted/private/model.gguf',
        'bytes':10000,'sha256':'1'*64}], 'table':[{'path':'/unmounted/private/model.gguf',
        'offset':100,'bytes':900,'shape':[160,10],'type':20,'sha256':'2'*64}],
        'tokenizer_sha256':'3'*64,'chat_template_sha256':'4'*64,
        'engine':{'commit':'5'*40,'patch_sha256':'6'*64,'runtime':{'libggml-cpu.so':'7'*64}},
        'architecture_metadata':{'qwen4exp.ple.head_offsets':[0,6],'qwen4exp.ple.head_vocab_sizes':[3,4]}})


def wire(identity,rows=(1,6),values=None,**header_changes):
    if values is None:values=[.25]*(len(rows)*160)
    payload=struct.pack('<'+'i'*len(rows),*rows)+struct.pack('<'+'f'*len(values),*values)
    header={'schema':'flash-memory-overlay/v1','row_count':len(rows),'row_dim':160,
            'model_identity':identity,'payload_sha256':hashlib.sha256(payload).hexdigest(),
            'status':'experiment'}
    header.update(header_changes)
    return header,payload


def write(path,header,payload):
    encoded=json.dumps(header,separators=(',',':')).encode()
    path.write_bytes(b'FMLROW1\0'+struct.pack('<I',len(encoded))+encoded+payload)
    return path


def test_multiline_overlay_geometry_hashes_and_no_vector_or_path_disclosure(identity,tmp_path):
    path=write(tmp_path/'rows.fml',*wire(identity))
    before=path.read_bytes();result=inspect_overlay(path)
    assert result['rows']==[1,6] and result['row_count']==2 and result['row_dim']==160
    assert result['overlay_sha256']==hashlib.sha256(before).hexdigest()
    assert result['identity_sha256']==identity['identity_sha256']
    assert result['model_files_read'] is False and result['engine_model_authentication_required'] is True
    assert '/unmounted/' not in json.dumps(result) and 'values' not in result
    assert path.read_bytes()==before


def test_optional_same_identity_matches(identity,tmp_path):
    path=write(tmp_path/'rows.fml',*wire(identity))
    supplied=tmp_path/'identity.json';supplied.write_text(json.dumps(identity))
    assert inspect_overlay(path,supplied)['supplied_identity_matched'] is True


def test_optional_valid_but_different_identity_rejected(identity,tmp_path):
    path=write(tmp_path/'rows.fml',*wire(identity))
    different=copy.deepcopy(identity);different['tokenizer_sha256']='a'*64;sign(different)
    supplied=tmp_path/'identity.json';supplied.write_text(json.dumps(different))
    with pytest.raises(ValueError,match='checksums differ'):inspect_overlay(path,supplied)


def test_optional_identity_bad_self_digest_rejected(identity,tmp_path):
    path=write(tmp_path/'rows.fml',*wire(identity))
    identity['identity_sha256']='a'*64
    supplied=tmp_path/'identity.json';supplied.write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='self-digest'):inspect_overlay(path,supplied)


@pytest.mark.parametrize('data',[b'',b'FMLROW1\0',b'WRONG!!!'+struct.pack('<I',1)+b'{}',
    b'FMLROW1\0'+struct.pack('<I',0),b'FMLROW1\0'+struct.pack('<I',MAX_HEADER_BYTES+1),
    b'FMLROW1\0'+struct.pack('<I',100)+b'{}'])
def test_bad_magic_header_bounds_and_truncated_header(tmp_path,data):
    path=tmp_path/'bad.fml';path.write_bytes(data)
    with pytest.raises(ValueError):inspect_overlay(path)


@pytest.mark.parametrize('field,value',[('schema','wrong'),('row_dim',159),('row_dim',True),
    ('row_count',-1),('row_count',MAX_ROWS+1),('row_count',True)])
def test_bad_overlay_geometry(identity,tmp_path,field,value):
    path=write(tmp_path/'bad.fml',*wire(identity,**{field:value}))
    with pytest.raises(ValueError):inspect_overlay(path)


@pytest.mark.parametrize('change',['truncate','trailing','wrong_size','payload_hash'])
def test_payload_size_and_checksum(identity,tmp_path,change):
    header,payload=wire(identity)
    if change=='truncate':payload=payload[:-1]
    elif change=='trailing':payload+=b'x'
    elif change=='wrong_size':header['row_count']=3
    else:header['payload_sha256']='f'*64
    path=write(tmp_path/'bad.fml',header,payload)
    with pytest.raises(ValueError,match='payload|SHA-256'):inspect_overlay(path)


@pytest.mark.parametrize('rows',[(-1,),(1,1),(6,1),(3,),(5,),(10,)])
def test_negative_duplicate_unsorted_and_out_of_head_rows(identity,tmp_path,rows):
    path=write(tmp_path/'bad.fml',*wire(identity,rows=rows))
    with pytest.raises(ValueError,match='rows|range|padding'):inspect_overlay(path)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_vectors_rejected_even_with_matching_payload_hash(identity,tmp_path,value):
    path=write(tmp_path/'bad.fml',*wire(identity,rows=(1,),values=[value]+[0.]*159))
    with pytest.raises(ValueError,match='Nonfinite FML'):inspect_overlay(path)


@pytest.mark.parametrize('field',['shards','table','architecture_metadata','tokenizer_sha256','engine'])
def test_incomplete_self_signed_manifest_rejected(identity,tmp_path,field):
    identity.pop(field);sign(identity)
    path=write(tmp_path/'bad.fml',*wire(identity))
    with pytest.raises(ValueError):inspect_overlay(path)


@pytest.mark.parametrize('kind',['bad_digest','table_range','table_path','head_overlap','head_overrun','missing_head_sizes'])
def test_inconsistent_identity_geometry(identity,tmp_path,kind):
    if kind=='bad_digest':identity['identity_sha256']='f'*64
    else:
        if kind=='table_range':identity['table'][0]['offset']=9999
        elif kind=='table_path':identity['table'][0]['path']='/another/model.gguf'
        elif kind=='head_overlap':identity['architecture_metadata']['qwen4exp.ple.head_offsets']=[0,2]
        elif kind=='head_overrun':identity['architecture_metadata']['qwen4exp.ple.head_vocab_sizes']=[3,5]
        else:identity['architecture_metadata'].pop('qwen4exp.ple.head_vocab_sizes')
        sign(identity)
    path=write(tmp_path/'bad.fml',*wire(identity))
    with pytest.raises(ValueError):inspect_overlay(path)


@pytest.mark.parametrize('extra',['"row_dim":160','"provenance":{"x":1,"x":2}',
                                  '"note":NaN','"note":1e999'])
def test_duplicate_json_keys_and_nonfinite_json(identity,tmp_path,extra):
    header,payload=wire(identity)
    encoded=(json.dumps(header,separators=(',',':'))[:-1]+','+extra+'}').encode()
    path=tmp_path/'bad.fml';path.write_bytes(b'FMLROW1\0'+struct.pack('<I',len(encoded))+encoded+payload)
    with pytest.raises(ValueError,match='Duplicate|Nonfinite'):inspect_overlay(path)


def test_empty_overlay_is_valid_but_still_requires_model_authentication(identity,tmp_path):
    path=write(tmp_path/'empty.fml',*wire(identity,rows=()))
    result=inspect_overlay(path)
    assert result['rows']==[] and result['engine_model_authentication_required'] is True


def test_model_paths_never_opened(identity,tmp_path,monkeypatch):
    path=write(tmp_path/'rows.fml',*wire(identity))
    original=Path.open
    def checked_open(self,*args,**kwargs):
        assert self==path,'Unexpected path read, including a declared model shard'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',checked_open)
    assert inspect_overlay(path)['model_files_read'] is False


def test_cli_runs_without_site_packages(identity,tmp_path):
    path=write(tmp_path/'rows.fml',*wire(identity))
    root=Path(__file__).resolve().parents[1]
    code="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('ngramma_runtime.overlay_inspect',run_name='__main__')"
    process=subprocess.run([sys.executable,'-S','-c',code,str(root/'src'),str(path)],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    assert json.loads(process.stdout)['rows']==[1,6]

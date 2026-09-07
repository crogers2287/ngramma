"""Small-file capture authentication checks using only Python's standard library."""
import hashlib
import json

import pytest
from ngramma_runtime.capture_identity import validate_capture

IDENTITY='a'*64


@pytest.fixture
def capture(tmp_path):
    (tmp_path/'tensors.json').write_text('[]\n')
    (tmp_path/'logits.f32').write_bytes(b'\x00\x00\x80\x3f')
    tokens=[12,23,34]
    (tmp_path/'tokens.json').write_text(json.dumps(tokens))
    record={'identity_sha256':IDENTITY,'device':'CPU','cache_type':'f32','repack':True,
            'flash_attention':False,'rope_overrides':False,'context':128,'batch':32,
            'microbatch':32,'tokens':tokens,'chunk_size':32,'lens_sha256':'b'*64,
            'capture_script_sha256':'c'*64,
            'metadata_sha256':hashlib.sha256((tmp_path/'tensors.json').read_bytes()).hexdigest(),
            'logits_sha256':hashlib.sha256((tmp_path/'logits.f32').read_bytes()).hexdigest()}
    def write(value=record):
        (tmp_path/'capture-summary.json').write_text(json.dumps(value))
    write()
    return tmp_path,record,write


def test_valid_capture_binds_profile_tokens_and_small_file_hashes(capture):
    path,record,_=capture
    assert validate_capture(path,IDENTITY)==record


def test_wrong_requested_model_identity_rejected(capture):
    path,_,_=capture
    with pytest.raises(ValueError,match='identity_sha256'):
        validate_capture(path,'d'*64)


def test_edited_capture_cannot_be_used_as_unmodified_reference(capture):
    path,record,write=capture
    record['overlay_sha256']='e'*64
    write()
    with pytest.raises(ValueError,match='must not contain a memory overlay'):
        validate_capture(path,IDENTITY)


def test_explicit_absence_of_overlay_is_a_valid_baseline(capture):
    path,record,write=capture
    record['overlay_sha256']=None
    write()
    assert validate_capture(path,IDENTITY)==record


@pytest.mark.parametrize('field,value',[('identity_sha256','d'*64),('device','CUDA'),
    ('cache_type','f16'),('repack',False),('repack',1),('flash_attention',True),
    ('rope_overrides',True),('context',256),('context',128.),('batch',64),('microbatch',16)])
def test_wrong_capture_profile_rejected(capture,field,value):
    path,record,write=capture
    record[field]=value;write()
    with pytest.raises(ValueError,match=field):validate_capture(path,IDENTITY)


@pytest.mark.parametrize('filename',['tensors.json','logits.f32'])
def test_tampered_metadata_or_logits_rejected(capture,filename):
    path,_,_=capture
    with (path/filename).open('ab') as output:output.write(b'changed')
    with pytest.raises(ValueError,match='content hash mismatch'):
        validate_capture(path,IDENTITY)


def test_tampered_token_file_rejected(capture):
    path,_,_=capture
    (path/'tokens.json').write_text('[12,23,35]')
    with pytest.raises(ValueError,match='matching tokens'):
        validate_capture(path,IDENTITY)


@pytest.mark.parametrize('chunk',[0,2,33])
def test_chunk_size_requires_one_complete_bounded_prefill(capture,chunk):
    path,record,write=capture
    record['chunk_size']=chunk;write()
    with pytest.raises(ValueError,match='complete prefill'):
        validate_capture(path,IDENTITY)


@pytest.mark.parametrize('tokens',[[],list(range(33))])
def test_empty_or_long_token_sequences_rejected_even_if_both_copies_match(capture,tokens):
    path,record,write=capture
    record['tokens']=tokens;write()
    (path/'tokens.json').write_text(json.dumps(tokens))
    with pytest.raises(ValueError,match='complete prefill'):
        validate_capture(path,IDENTITY)


@pytest.mark.parametrize('field',['lens_sha256','capture_script_sha256'])
@pytest.mark.parametrize('digest',[None,'short','g'*64,123])
def test_missing_or_malformed_capture_source_identity(capture,field,digest):
    path,record,write=capture
    if digest is None:record.pop(field)
    else:record[field]=digest
    write()
    with pytest.raises(ValueError,match='Missing capture source identity'):
        validate_capture(path,IDENTITY)


@pytest.mark.parametrize('tokens',[[True],[False],[1.0],[-1],[2**31],'123',{'0':12}])
def test_token_list_rejects_bool_noninteger_negative_overflow_and_nonlist(capture,tokens):
    path,record,write=capture
    record['tokens']=tokens;write()
    (path/'tokens.json').write_text(json.dumps(tokens))
    with pytest.raises(ValueError,match='integer nonnegative int32'):
        validate_capture(path,IDENTITY)


@pytest.mark.parametrize('chunk',[True,False,3.5,32.,'32',None])
def test_chunk_size_must_be_an_integer_not_bool_or_coercible_value(capture,chunk):
    path,record,write=capture
    record['chunk_size']=chunk;write()
    with pytest.raises(ValueError,match='complete prefill'):
        validate_capture(path,IDENTITY)


def test_int32_edge_ids_and_exact_chunk_length_are_valid(capture):
    path,record,write=capture
    tokens=[0,2**31-1]
    record.update(tokens=tokens,chunk_size=len(tokens));write()
    (path/'tokens.json').write_text(json.dumps(tokens))
    assert validate_capture(path,IDENTITY)['tokens']==tokens

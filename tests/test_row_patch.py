"""Sparse replacement and archived FML wire-contract tests; synthetic rows only."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import struct

import pytest
np=pytest.importorskip('numpy')
from ngramma_runtime.artifacts import digest
from ngramma_runtime.row_patch import RowPatch


class Table:
    def __init__(self):self.rows=np.arange(6*160,dtype=np.float32).reshape(6,160)/1000
    def read_global(self,rows):
        if any(type(row) is not int or not 0<=row<len(self.rows) for row in rows):
            raise ValueError('Row outside original table')
        return self.rows[rows].copy()


@pytest.fixture
def table():return Table()


@pytest.fixture
def identity(tmp_path):
    shard=tmp_path/'synthetic-shard.gguf';shard.write_bytes(b'synthetic fixture, not GGUF')
    record={'schema':'flash-memory-model/v1','shards':[{'path':str(shard),'bytes':shard.stat().st_size,
        'sha256':hashlib.sha256(shard.read_bytes()).hexdigest()}],
        'table':[{'path':str(shard),'offset':0,'type':20,'shape':[160,6]}],
        'architecture_metadata':{'qwen4exp.ple.head_offsets':[0],'qwen4exp.ple.head_vocab_sizes':[6]}}
    record['identity_sha256']=digest(record)
    return record


def patch(table):return RowPatch.from_direction(table,2,np.ones(160,np.float32),.25)


def test_replaces_all_occurrences_and_preserves_other_rows_and_inputs(table):
    edit=patch(table)
    addresses=np.array([[2,1,2],[3,2,0]],np.int64)
    gathered=table.rows[addresses].copy();before=gathered.copy();original=table.rows.copy()
    result=edit.apply(addresses,gathered)
    np.testing.assert_array_equal(result[addresses==2],np.broadcast_to(edit.replacement,(3,160)))
    np.testing.assert_array_equal(result[addresses!=2],before[addresses!=2])
    np.testing.assert_array_equal(gathered,before);np.testing.assert_array_equal(table.rows,original)
    assert not np.shares_memory(result,gathered)


def test_no_occurrence_is_unchanged_independent_copy(table):
    addresses=np.array([[0,1]],np.int32);gathered=table.rows[addresses]
    result=patch(table).apply(addresses,gathered)
    np.testing.assert_array_equal(result,gathered)
    assert not np.shares_memory(result,gathered)


def test_patch_copies_and_irreversibly_freezes_row_arrays(table):
    anchor=table.rows[2].copy();replacement=anchor+1
    edit=RowPatch(2,anchor,replacement)
    anchor[:]=99;replacement[:]=99
    np.testing.assert_array_equal(edit.anchor,table.rows[2])
    for array in (edit.anchor,edit.replacement):
        with pytest.raises(ValueError):array[0]=0
        with pytest.raises(ValueError):array.flags.writeable=True
    with pytest.raises(FrozenInstanceError):edit.row_id=3


def test_wrong_anchor_in_any_occurrence_rejected(table):
    addresses=np.array([[2,2]],np.int64);gathered=table.rows[addresses].copy()
    gathered[0,1,7]+=1
    with pytest.raises(ValueError,match='anchor'):patch(table).apply(addresses,gathered)


@pytest.mark.parametrize('row',[-1,2**31,True,2.])
def test_row_identifier_bounds_and_type(table,row):
    with pytest.raises(ValueError,match='Row ID'):RowPatch(row,table.rows[0],table.rows[1])


@pytest.mark.parametrize('kind',['dtype','shape','nan','inf'])
@pytest.mark.parametrize('which',['anchor','replacement'])
def test_invalid_row_values_rejected(table,kind,which):
    value=table.rows[2].copy()
    if kind=='dtype':value=value.astype(np.float64)
    elif kind=='shape':value=value[:-1]
    elif kind=='nan':value[0]=np.nan
    else:value[0]=np.inf
    args={'row_id':2,'anchor':table.rows[2],'replacement':table.rows[2]}
    args[which]=value
    with pytest.raises(ValueError,match='160 finite FP32'):RowPatch(**args)


@pytest.mark.parametrize('epsilon',[float('nan'),float('inf'),True,1e300])
def test_nonfinite_or_overflowing_perturbations_rejected(table,epsilon):
    with pytest.raises(ValueError):RowPatch.from_direction(table,2,np.ones(160,np.float32),epsilon)


def test_table_bounds_checked_before_perturbation(table):
    with pytest.raises(ValueError,match='outside original table'):
        RowPatch.from_direction(table,6,np.ones(160,np.float32),.1)


@pytest.mark.parametrize('direction',[np.ones(159,np.float32),np.ones(160,np.float64),np.full(160,np.inf,np.float32)])
def test_direction_geometry_dtype_finiteness(table,direction):
    with pytest.raises(ValueError,match='Direction'):RowPatch.from_direction(table,2,direction,.1)


@pytest.mark.parametrize('kind',['float_addresses','wrong_rank','wrong_shape','wrong_dtype'])
def test_apply_rejects_incompatible_gathers(table,kind):
    addresses=np.array([[2,1]],np.int64);rows=table.rows[addresses]
    if kind=='float_addresses':addresses=addresses.astype(np.float32)
    elif kind=='wrong_rank':addresses=addresses[0]
    elif kind=='wrong_shape':rows=rows[...,:159]
    else:rows=rows.astype(np.float64)
    with pytest.raises(ValueError,match='addresses'):patch(table).apply(addresses,rows)


def test_zero_step_preserves_numeric_anchor(table):
    edit=RowPatch.from_direction(table,2,np.ones(160,np.float32),0.)
    np.testing.assert_array_equal(edit.anchor,edit.replacement)


def test_export_matches_engine_fml_layout_and_hashes(table,identity,tmp_path):
    edit=patch(table);destination=tmp_path/'overlay'
    summary=edit.export(destination,identity,table,{'kind':'synthetic-test'})
    raw=(destination/'rows.fml').read_bytes()
    assert raw[:8]==b'FMLROW1\0'
    header_length=struct.unpack_from('<I',raw,8)[0]
    assert 0<header_length<=4<<20
    header=json.loads(raw[12:12+header_length]);payload=raw[12+header_length:]
    assert header['schema']=='flash-memory-overlay/v1'
    assert header['row_count']==1 and header['row_dim']==160
    assert len(payload)==4+160*4
    assert struct.unpack_from('<i',payload)[0]==edit.row_id
    np.testing.assert_array_equal(np.frombuffer(payload,dtype='<f4',offset=4),edit.replacement)
    assert header['payload_sha256']==hashlib.sha256(payload).hexdigest()
    assert header['model_identity']==identity
    assert summary['overlay_sha256']==hashlib.sha256(raw).hexdigest()
    assert summary['anchor_sha256']==hashlib.sha256(edit.anchor.tobytes()).hexdigest()
    assert summary['replacement_sha256']==hashlib.sha256(payload[4:]).hexdigest()
    assert json.loads((destination/'manifest.json').read_text())==summary
    assert summary['identity_sha256']==digest({k:v for k,v in identity.items() if k!='identity_sha256'})
    # This checks the engine's published wire contract; it does not load a model
    # or claim native compatibility from the synthetic GGUF placeholder.


def test_export_refuses_modified_table_anchor(table,identity,tmp_path):
    edit=patch(table);table.rows[2,0]+=1
    with pytest.raises(ValueError,match='anchor'):edit.export(tmp_path/'bad',identity,table,{})
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('mutation',['digest','unsigned','schema','shards'])
def test_export_rejects_bad_identity_before_writes(table,identity,tmp_path,mutation):
    if mutation=='digest':identity['identity_sha256']='0'*64
    elif mutation=='unsigned':identity['shards'][0]['bytes']+=1
    elif mutation=='schema':identity['schema']='different'
    else:identity['shards']=[]
    with pytest.raises(ValueError):patch(table).export(tmp_path/'bad',identity,table,{})
    assert not (tmp_path/'bad').exists()


def test_duplicate_export_cannot_overwrite_prior_overlay(table,identity,tmp_path):
    destination=tmp_path/'overlay';patch(table).export(destination,identity,table,{})
    before={path.name:path.read_bytes() for path in destination.iterdir()}
    different=RowPatch.from_direction(table,2,np.ones(160,np.float32),.5)
    with pytest.raises(FileExistsError):different.export(destination,identity,table,{})
    assert before=={path.name:path.read_bytes() for path in destination.iterdir()}

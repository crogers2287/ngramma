"""Synthetic in-memory rows; declared model paths are never opened."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
np = pytest.importorskip('numpy')
from ngramma_runtime.artifacts import digest
from ngramma_runtime.multi_row_patch import MultiRowPatch
from ngramma_runtime.overlay_inspect import inspect_overlay
from ngramma_runtime.row_patch import RowPatch


def sign(identity):
    identity['identity_sha256'] = digest({k: v for k, v in identity.items() if k != 'identity_sha256'})
    return identity


@pytest.fixture
def fixture():
    identity = sign({'schema': 'flash-memory-model/v1',
        'shards': [{'path': '/never-opened/synthetic.gguf', 'bytes': 10000, 'sha256': '1'*64}],
        'table': [{'name': 'per_layer_token_embd.weight', 'path': '/never-opened/synthetic.gguf',
                   'offset': 100, 'bytes': 900, 'shape': [160, 10], 'type': 20, 'sha256': '2'*64}],
        'tokenizer_sha256': '3'*64, 'chat_template_sha256': '4'*64,
        'engine': {'commit': '5'*40, 'patch_sha256': '6'*64, 'runtime': {'libggml-cpu.so': '7'*64}},
        'architecture_metadata': {'qwen4exp.ple.head_offsets': [0, 6],
                                  'qwen4exp.ple.head_vocab_sizes': [4, 4]}})
    rows = np.arange(1600, dtype=np.float32).reshape(10, 160)/100
    table = SimpleNamespace(layout='joined', dim=160, n_rows=10, head_offsets=[0, 6],
        head_vocab_sizes=[4, 4], paths=[Path('/never-opened/synthetic.gguf')],
        metadata=copy.deepcopy(identity['architecture_metadata']),
        tensors={'per_layer_token_embd.weight': (Path('/never-opened/synthetic.gguf'),
            SimpleNamespace(data_offset=100, n_bytes=900, shape=[160, 10], tensor_type=20))},
        rows=rows, read_global=lambda ids: rows[ids].copy())
    patches = [RowPatch(7, rows[7], rows[7]+.25), RowPatch(1, rows[1], rows[1]-.5)]
    return identity, table, patches


def test_sorted_immutable_snapshot(fixture):
    _, table, patches = fixture
    edit = MultiRowPatch(patches)
    assert edit.row_ids == (1, 7)
    patches.clear()
    assert len(edit.patches) == 2
    with pytest.raises(FrozenInstanceError): edit.patches = ()
    for patch in edit.patches:
        for row in (patch.anchor, patch.replacement):
            with pytest.raises(ValueError): row.flags.writeable = True
    np.testing.assert_array_equal(edit.patches[0].anchor, table.rows[1])


def test_all_occurrences_original_input_and_untouched_rows(fixture):
    _, table, patches = fixture
    edit = MultiRowPatch(patches)
    ids = np.array([[7, 1, 2], [1, 6, 7]])
    gathered = table.rows[ids].copy()
    before = gathered.copy()
    result = edit.apply(ids, gathered)
    for patch in edit.patches:
        np.testing.assert_array_equal(result[ids == patch.row_id], np.tile(patch.replacement, (2, 1)))
    np.testing.assert_array_equal(result[(ids != 1) & (ids != 7)], before[(ids != 1) & (ids != 7)])
    np.testing.assert_array_equal(gathered, before)
    assert not np.shares_memory(result, gathered)


def test_no_match_still_copies(fixture):
    _, table, patches = fixture
    ids = np.array([[2, 3]])
    gathered = table.rows[ids]
    result = MultiRowPatch(patches).apply(ids, gathered)
    np.testing.assert_array_equal(result, gathered)
    assert not np.shares_memory(result, gathered)


def test_wrong_second_anchor_fails_without_partial_mutation(fixture):
    _, table, patches = fixture
    ids = np.array([[1, 7, 7]])
    gathered = table.rows[ids].copy()
    gathered[0, 2, 4] += 1
    before = gathered.copy()
    with pytest.raises(ValueError, match='anchor'): MultiRowPatch(patches).apply(ids, gathered)
    np.testing.assert_array_equal(before, gathered)


@pytest.mark.parametrize('kind', ['empty', 'too_many', 'duplicate', 'invalid'])
def test_invalid_patch_collections(fixture, kind):
    _, table, patches = fixture
    if kind == 'empty': patches = []
    elif kind == 'too_many': patches = [RowPatch(i, table.rows[i], table.rows[i]) for i in range(9)]
    elif kind == 'duplicate': patches *= 2
    else: patches = [object()]
    with pytest.raises(ValueError): MultiRowPatch(patches)


@pytest.mark.parametrize('which', ['anchor', 'replacement'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_rows_rejected(fixture, which, value):
    _, table, _ = fixture
    rows = {'anchor': table.rows[0].copy(), 'replacement': table.rows[0].copy()}
    rows[which][0] = value
    with pytest.raises(ValueError): MultiRowPatch([RowPatch(0, **rows)])


@pytest.mark.parametrize('kind', ['negative', 'overflow', 'float', 'bool', 'rank', 'shape', 'dtype', 'nan'])
def test_bad_gathers_rejected(fixture, kind):
    _, table, patches = fixture
    ids = np.array([[1, 7]])
    gathered = table.rows[ids].copy()
    if kind == 'negative': ids[0, 0] = -1
    elif kind == 'overflow': ids[0, 0] = 2**31
    elif kind == 'float': ids = ids.astype(float)
    elif kind == 'bool': ids = ids.astype(bool)
    elif kind == 'rank': ids = ids[0]
    elif kind == 'shape': gathered = gathered[..., :-1]
    elif kind == 'dtype': gathered = gathered.astype(np.float64)
    else: gathered[0, 0, 0] = np.nan
    with pytest.raises(ValueError): MultiRowPatch(patches).apply(ids, gathered)


def test_export_fml_id_block_then_rows_and_inspector(fixture, tmp_path):
    identity, table, patches = fixture
    edit = MultiRowPatch(patches)
    destination = tmp_path / 'overlay'
    summary = edit.export(destination, identity, table, {'kind': 'synthetic-test'})
    raw = (destination / 'rows.fml').read_bytes()
    length = struct.unpack_from('<I', raw, 8)[0]
    header = json.loads(raw[12:12+length])
    payload = raw[12+length:]
    assert struct.unpack_from('<ii', payload) == (1, 7)
    expected = np.stack([p.replacement for p in edit.patches])
    np.testing.assert_array_equal(np.frombuffer(payload, '<f4', offset=8).reshape(2,160), expected)
    assert header['payload_sha256'] == hashlib.sha256(payload).hexdigest()
    assert summary['overlay_sha256'] == hashlib.sha256(raw).hexdigest()
    assert summary == json.loads((destination / 'manifest.json').read_text())
    result = inspect_overlay(destination / 'rows.fml')
    assert result['rows'] == [1, 7] and result['model_files_read'] is False
    assert result['engine_model_authentication_required'] is True


def test_all_eight_valid_rows_export(fixture, tmp_path):
    identity, table, _ = fixture
    patches = [RowPatch(i, table.rows[i], table.rows[i]+1) for i in (9,8,7,6,3,2,1,0)]
    summary = MultiRowPatch(patches).export(tmp_path/'eight', identity, table, {})
    assert summary['row_ids'] == [0,1,2,3,6,7,8,9]


@pytest.mark.parametrize('row', [4, 5, 10, 2**31-1])
def test_head_padding_and_bounds_rejected_before_read(fixture, tmp_path, row):
    identity, table, _ = fixture
    table.read_global = lambda ids: pytest.fail('Must validate bounds before table read')
    edit = MultiRowPatch([RowPatch(row, np.zeros(160,np.float32), np.ones(160,np.float32))])
    with pytest.raises(ValueError, match='range'): edit.export(tmp_path/'bad', identity, table, {})
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('kind', ['digest', 'missing_engine', 'offset', 'path', 'shape', 'bytes', 'type', 'heads', 'metadata', 'shards'])
def test_table_identity_mismatch_before_writes(fixture, tmp_path, kind):
    identity, table, patches = fixture
    if kind == 'digest': identity['identity_sha256'] = 'a'*64
    elif kind == 'missing_engine': del identity['engine']; sign(identity)
    elif kind in ('offset', 'bytes', 'type'): identity['table'][0][kind] += 1; sign(identity)
    elif kind == 'path': table.tensors['per_layer_token_embd.weight'] = (Path('/wrong'), table.tensors['per_layer_token_embd.weight'][1])
    elif kind == 'shape': table.tensors['per_layer_token_embd.weight'][1].shape = [160, 9]
    elif kind == 'heads': table.head_vocab_sizes = [3, 4]
    elif kind == 'metadata': table.metadata['qwen4exp.ple.head_offsets'] = [0, 7]
    else: table.paths = [Path('/wrong')]
    with pytest.raises(ValueError): MultiRowPatch(patches).export(tmp_path/'bad', identity, table, {})
    assert not (tmp_path/'bad').exists()


def test_anchor_tampering_and_signed_zero_are_exact(fixture, tmp_path):
    identity, table, patches = fixture
    edit = MultiRowPatch(patches)
    table.rows[7, 0] += 1
    with pytest.raises(ValueError, match='anchor'): edit.export(tmp_path/'bad', identity, table, {})
    zeros = np.zeros(160, np.float32)
    signed = zeros.copy(); signed[0] = -0.0
    edit = MultiRowPatch([RowPatch(0, zeros, zeros)])
    with pytest.raises(ValueError, match='anchor'): edit.apply(np.array([[0]]), signed.reshape(1,1,160))


def test_duplicate_export_and_invalid_provenance(fixture, tmp_path):
    identity, table, patches = fixture
    edit = MultiRowPatch(patches)
    edit.export(tmp_path/'overlay', identity, table, {})
    before = (tmp_path/'overlay/rows.fml').read_bytes()
    with pytest.raises(FileExistsError): edit.export(tmp_path/'overlay', identity, table, {})
    assert before == (tmp_path/'overlay/rows.fml').read_bytes()
    with pytest.raises(ValueError): edit.export(tmp_path/'bad', identity, table, {'value': float('nan')})
    assert not (tmp_path/'bad').exists()

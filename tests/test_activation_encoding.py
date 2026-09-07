"""Synthetic opt-in tests: no checkpoint or model execution."""
import ctypes
import os
from pathlib import Path
import numpy as np
import pytest
from ngramma_runtime.activation_encoding import ActivationEncoding


@pytest.fixture
def encoder():
    library = os.environ.get('NGRAMMA_ACTIVATION_LIBRARY')
    if not library:
        pytest.skip('Set NGRAMMA_ACTIVATION_LIBRARY to the isolated activation probe')
    if not Path(library).is_file():
        pytest.fail('Configured activation probe does not exist')
    return ActivationEncoding(library)


def test_known_block_includes_scale_and_codes(encoder):
    # Exactly representable scale 1, and exact integral codes.
    x = np.arange(-16, 16, dtype=np.float32)
    x[0] = 127
    result = encoder.encode(8, x[None])
    expected = np.concatenate((np.array([1], np.float16).view(np.uint8), x.astype(np.int8).view(np.uint8)))
    np.testing.assert_array_equal(result[0], expected)
    assert result.shape == (1, 34)
    assert encoder.metadata['auxiliary_sums'] is None


def test_zero_block_all_bytes_initialized(encoder):
    np.testing.assert_array_equal(encoder.encode(8, np.zeros((3, 2560), np.float32)), np.zeros((3,2720), np.uint8))


def test_scale_only_change_visible(encoder):
    x = np.zeros((1,32),np.float32);x[0,0] = 127
    a,b = encoder.encode(8,x),encoder.encode(8,x*2)
    np.testing.assert_array_equal(a[:,2:],b[:,2:])
    assert not np.array_equal(a[:,:2],b[:,:2])


def test_block_and_row_partition_invariance(encoder):
    x = np.random.default_rng(7).normal(size=(3,2560)).astype(np.float32)
    whole = encoder.encode(8,x)
    blocks = encoder.encode(8,x.reshape(-1,32)).reshape(3,-1)
    rows = np.concatenate([encoder.encode(8,row[None]) for row in x])
    np.testing.assert_array_equal(whole,blocks)
    np.testing.assert_array_equal(whole,rows)


def test_noncontiguous_input(encoder):
    x = np.random.default_rng(2).normal(size=(64,3)).astype(np.float32).T
    assert not x.flags.c_contiguous
    np.testing.assert_array_equal(encoder.encode(8,x),encoder.encode(8,x.copy()))


@pytest.mark.parametrize('value,error',[
    (np.zeros((1,32),np.float16),TypeError),
    (np.zeros((1,32),np.float64),TypeError),
    (np.zeros((32,),np.float32),ValueError),
    (np.zeros((0,32),np.float32),ValueError),
    (np.zeros((1,31),np.float32),ValueError),
    (np.full((1,32),np.nan,np.float32),ValueError),
    (np.full((1,32),np.inf,np.float32),ValueError),
])
def test_reject_before_native(encoder,value,error):
    encoder._encode = lambda *args: pytest.fail('Native encoder must not be called')
    with pytest.raises(error):encoder.encode(8,value)


@pytest.mark.parametrize('weight_type',[0,2,15,True])
def test_reject_unsupported_weight(encoder,weight_type):
    with pytest.raises(ValueError):encoder.encode(weight_type,np.zeros((1,32),np.float32))


def test_c_boundary_guards(encoder):
    x=np.zeros((1,32),np.float32);y=np.zeros(34,np.uint8)
    call=encoder._encode
    assert call(8,None,y.ctypes.data,32,1,34)==-2
    assert call(8,x.ctypes.data,y.ctypes.data,32,1,33)==-3
    assert call(8,x.ctypes.data,y.ctypes.data,31,1,34)==-2
    assert call(0,x.ctypes.data,y.ctypes.data,32,1,34)==-1

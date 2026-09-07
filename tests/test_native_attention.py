"""Synthetic CPU attention primitives. No models; opt in with NGRAMMA_FORWARD_LIBRARY."""
import ctypes as ct
import os
from pathlib import Path
import pytest


@pytest.fixture(scope='module')
def bridge():
    path = os.environ.get('NGRAMMA_FORWARD_LIBRARY')
    if not path:
        pytest.skip('Set NGRAMMA_FORWARD_LIBRARY to an attention-enabled local bridge')
    if not Path(path).is_file():
        pytest.fail('Configured bridge does not exist')
    lib = ct.CDLL(path)
    if not hasattr(lib, 'ngramma_rope_multi'):
        pytest.skip('Configured bridge predates optional attention primitives')
    lib.ngramma_last_error.restype = ct.c_char_p
    lib.ngramma_rope_multi.argtypes = [ct.c_void_p]*4 + [ct.c_int64]*3 + [ct.c_int]*3 + [ct.c_float]*6 + [ct.c_int]
    lib.ngramma_rope_multi.restype = ct.c_int
    lib.ngramma_batched_matmul.argtypes = [ct.c_void_p]*3 + [ct.c_int64]*5 + [ct.c_int]
    lib.ngramma_batched_matmul.restype = ct.c_int
    lib.ngramma_softmax_ext.argtypes = [ct.c_void_p]*3 + [ct.c_int64]*3 + [ct.c_float, ct.c_int]
    lib.ngramma_softmax_ext.restype = ct.c_int
    return lib


@pytest.fixture(scope='module')
def np(bridge):
    return pytest.importorskip('numpy')


def ptrs(*arrays):
    return [a.ctypes.data for a in arrays]


def okay(bridge, result):
    assert result == 0, bridge.ngramma_last_error().decode()


def rope(bridge, np, values, positions, sections, n_rot, threads):
    output = np.empty_like(values)
    t, h, d = values.shape
    okay(bridge, bridge.ngramma_rope_multi(*ptrs(values, positions, sections, output),
        t, h, d, n_rot, 40, 4096, 10000., 1., 0., 1., 32., 1., threads))
    return output


@pytest.mark.parametrize('threads', [1, 2])
@pytest.mark.parametrize('dim,n_rot', [(8, 4), (16, 16), (256, 64)])
def test_rope_zero_position_identity(bridge, np, threads, dim, n_rot):
    values = np.random.default_rng(13).normal(size=(3,2,dim)).astype(np.float32)
    positions = np.zeros((4,3), np.int32)
    sections = np.array([n_rot//2,0,0,0], np.int32)
    output = rope(bridge, np, values, positions, sections, n_rot, threads)
    np.testing.assert_array_equal(output, values)


@pytest.mark.parametrize('threads', [1, 2])
def test_rope_partial_nonzero_rotation(bridge, np, threads):
    values = np.random.default_rng(15).normal(size=(3,2,16)).astype(np.float32)
    positions = np.tile(np.arange(3, dtype=np.int32), (4,1))
    sections = np.array([2,1,1,0], np.int32)
    n_rot = 8
    output = rope(bridge, np, values, positions, sections, n_rot, threads)
    frequencies = 10000.**(-2*np.arange(n_rot//2)/n_rot)
    theta = np.arange(3)[:,None,None]*frequencies[None,None,:]
    expected = values.copy()
    a, b = values[...,:n_rot//2], values[...,n_rot//2:n_rot]
    expected[...,:n_rot//2] = a*np.cos(theta)-b*np.sin(theta)
    expected[...,n_rot//2:n_rot] = a*np.sin(theta)+b*np.cos(theta)
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_array_equal(output[...,n_rot:], values[...,n_rot:])


@pytest.mark.parametrize('threads', [1, 2])
@pytest.mark.parametrize('hw,hx,n,m,k', [(1,2,3,4,7), (2,4,5,10,32), (2,24,256,10,256)])
def test_batched_matmul_contiguous_head_groups(bridge, np, threads, hw, hx, n, m, k):
    rng = np.random.default_rng(16)
    weights = rng.normal(size=(hw,n,k)).astype(np.float32)*.1
    inputs = rng.normal(size=(hx,m,k)).astype(np.float32)*.1
    output = np.empty((hx,m,n), np.float32)
    okay(bridge, bridge.ngramma_batched_matmul(*ptrs(weights, inputs, output), k,n,m,hw,hx,threads))
    expected = np.stack([inputs[h]@weights[h//(hx//hw)].T for h in range(hx)])
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize('threads', [1, 2])
@pytest.mark.parametrize('keys', [13,256])
def test_masked_softmax_causal_padding(bridge, np, threads, keys):
    heads,tokens = 2,10
    scores = np.random.default_rng(17).normal(size=(heads,tokens,keys)).astype(np.float32)
    mask = np.full((tokens,keys), -np.inf, np.float32)
    for t in range(tokens):
        mask[t,:t+1] = 0
    scale = np.float32(.125)
    output = np.empty_like(scores)
    okay(bridge, bridge.ngramma_softmax_ext(*ptrs(scores,mask,output), heads,tokens,keys,scale,threads))
    adjusted = scores*scale+mask
    exponent = np.exp(adjusted-adjusted.max(-1,keepdims=True))
    expected = exponent/exponent.sum(-1,keepdims=True)
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=1e-7)
    np.testing.assert_array_equal(output[...,tokens:], np.zeros_like(output[...,tokens:]))
    np.testing.assert_allclose(output.sum(-1), 1, atol=1e-6)


def test_invalid_shapes_and_rope_parameters(bridge,np):
    buf = np.ones((1024,), np.float32); out=np.empty_like(buf)
    pos = np.zeros((4,2),np.int32); sections=np.array([2,0,0,0],np.int32)
    for n_rot,mode in [(3,40),(10,40),(4,1)]:
        assert bridge.ngramma_rope_multi(*ptrs(buf,pos,sections,out),2,2,8,n_rot,mode,4096,10000.,1.,0.,1.,32.,1.,1)==-1
        assert bridge.ngramma_last_error()
    bad_sections=np.zeros((4,),np.int32)
    assert bridge.ngramma_rope_multi(*ptrs(buf,pos,bad_sections,out),2,2,8,4,40,4096,10000.,1.,0.,1.,32.,1.,1)==-1
    assert bridge.ngramma_batched_matmul(*ptrs(buf,buf,out),8,2,2,2,3,1)==-1
    assert bridge.ngramma_batched_matmul(*ptrs(buf,buf,out),0,2,2,1,2,1)==-1
    assert bridge.ngramma_softmax_ext(*ptrs(buf,buf,out),2,2,0,1.,1)==-1
    assert bridge.ngramma_softmax_ext(*ptrs(buf,buf,out),2,2,8,float('nan'),1)==-1

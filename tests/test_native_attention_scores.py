"""Optional stride-sensitive attention fixtures; no model or checkpoint reads."""
import ctypes as ct
import os
from pathlib import Path
import pytest


@pytest.fixture(scope='module')
def bridge():
    path = os.environ.get('NGRAMMA_FORWARD_LIBRARY')
    if not path:
        pytest.skip('Set NGRAMMA_FORWARD_LIBRARY to the stride-preserving bridge')
    if not Path(path).is_file():
        pytest.fail('Configured bridge does not exist')
    lib = ct.CDLL(path)
    if not hasattr(lib, 'ngramma_attention_scores'):
        pytest.skip('Configured bridge predates attention score primitive')
    lib.ngramma_attention_scores.argtypes = [ct.c_void_p]*3 + [ct.c_int64]*5 + [ct.c_int]
    lib.ngramma_attention_scores.restype = ct.c_int
    lib.ngramma_last_error.restype = ct.c_char_p
    # This symbol resolves through the same linked GGML CPU dependency.
    lib.ggml_vec_dot_f32.argtypes = [ct.c_int,ct.c_void_p,ct.c_size_t,
        ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t,ct.c_int]
    lib.ggml_vec_dot_f32.restype = None
    return lib


@pytest.fixture(scope='module')
def np(bridge):
    return pytest.importorskip('numpy')


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('tokens,heads,key_heads,keys,dim', [(3,4,2,7,32),(10,24,2,256,256)])
def test_query_layout_matches_native_row_dots(bridge,np,threads,tokens,heads,key_heads,keys,dim):
    rng = np.random.default_rng(317)
    k = rng.normal(size=(key_heads,keys,dim)).astype(np.float32)
    q = rng.normal(size=(tokens,heads,dim)).astype(np.float32)
    result = np.empty((heads,tokens,keys),np.float32)
    status = bridge.ngramma_attention_scores(k.ctypes.data,q.ctypes.data,result.ctypes.data,
        dim,keys,tokens,key_heads,heads,threads)
    assert status == 0, bridge.ngramma_last_error().decode()
    expected = np.empty_like(result)
    for h in range(heads):
        for t in range(tokens):
            for j in range(keys):
                offset = ((h*tokens+t)*keys+j)*4
                bridge.ggml_vec_dot_f32(dim,expected.ctypes.data+offset,0,
                    k[h//(heads//key_heads),j].ctypes.data,0,q[t,h].ctypes.data,0,1)
    np.testing.assert_array_equal(result,expected)
    # Independent semantics check catches a matching but wrongly indexed oracle.
    independent = np.einsum('thd,hnd->htn',q.astype(np.float64),
        np.repeat(k.astype(np.float64),heads//key_heads,axis=0))
    np.testing.assert_allclose(result,independent,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize('invalid', ['heads','zero_tokens','null'])
def test_invalid_score_contract_rejected(bridge,np,invalid):
    k = np.ones((2,3,8),np.float32)
    q = np.ones((2,4,8),np.float32)
    output = np.empty((4,2,3),np.float32)
    status = bridge.ngramma_attention_scores(
        None if invalid=='null' else k.ctypes.data,q.ctypes.data,output.ctypes.data,
        8,3,0 if invalid=='zero_tokens' else 2,2,3 if invalid=='heads' else 4,1)
    assert status != 0
    assert bridge.ngramma_last_error()

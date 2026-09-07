"""Optional synthetic CPU checks; no model weights or inference services.

Run with NGRAMMA_FORWARD_LIBRARY pointing to the locally built forward bridge.
The independent NumPy references check layout and operation semantics; tolerance
agreement does not qualify full-model numerical parity or any gradient path.
"""
import ctypes as ct
import os
from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def bridge():
    path = os.environ.get('NGRAMMA_FORWARD_LIBRARY')
    if not path:
        pytest.skip('Set NGRAMMA_FORWARD_LIBRARY to run local native CPU fixtures')
    if not Path(path).is_file():
        pytest.fail(f'Configured forward bridge does not exist: {path}')
    lib = ct.CDLL(path)
    lib.ngramma_last_error.restype = ct.c_char_p
    lib.ngramma_matmul.argtypes = [ct.c_int] + [ct.c_void_p]*3 + [ct.c_int64]*3 + [ct.c_int]
    lib.ngramma_matmul.restype = ct.c_int
    lib.ngramma_unary.argtypes = [ct.c_int] + [ct.c_void_p]*2 + [ct.c_int64]*2 + [ct.c_float, ct.c_int]
    lib.ngramma_unary.restype = ct.c_int
    lib.ngramma_ssm_conv.argtypes = [ct.c_void_p]*3 + [ct.c_int64]*3 + [ct.c_int]
    lib.ngramma_ssm_conv.restype = ct.c_int
    lib.ngramma_gdn.argtypes = [ct.c_void_p]*8 + [ct.c_int64]*4 + [ct.c_int]
    lib.ngramma_gdn.restype = ct.c_int
    return lib


@pytest.fixture(scope='module')
def np(bridge):
    # Resolve the bridge first so an unconfigured machine needs no NumPy.
    return pytest.importorskip('numpy')


def pointers(*arrays):
    return [value.ctypes.data for value in arrays]


def okay(bridge, code):
    assert code == 0, bridge.ngramma_last_error().decode()


@pytest.mark.parametrize('threads', [1, 2])
def test_f32_matmul(bridge, np, threads):
    weights = np.arange(64, dtype=np.float32).reshape(2, 32)/16
    values = np.linspace(-1, 1, 96, dtype=np.float32).reshape(3, 32)
    output = np.empty((3, 2), dtype=np.float32)
    okay(bridge, bridge.ngramma_matmul(0, *pointers(weights, values, output), 32, 2, 3, threads))
    np.testing.assert_allclose(output, values@weights.T, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('threads', [1, 2])
@pytest.mark.parametrize('width', [7, 32, 160])
@pytest.mark.parametrize('op', range(7))
def test_unary_row_semantics(bridge, np, threads, width, op):
    values = np.linspace(-2, 2, width*3, dtype=np.float32).reshape(3, width)
    output = np.empty_like(values)
    epsilon = np.float32(1e-6)
    okay(bridge, bridge.ngramma_unary(op, *pointers(values, output), width, 3, epsilon, threads))
    if op == 0:
        expected = values/np.sqrt(np.mean(values*values, axis=-1, keepdims=True)+epsilon)
    elif op == 1:
        expected = values/(1+np.exp(-values))
    elif op == 2:
        expected = 1/(1+np.exp(-values))
    elif op == 3:
        expected = values/np.maximum(np.sqrt(np.sum(values*values, axis=-1, keepdims=True)), epsilon)
    elif op == 4:
        exponential = np.exp(values-values.max(-1, keepdims=True))
        expected = exponential/exponential.sum(-1, keepdims=True)
    elif op == 5:
        expected = np.exp(values)
    else:
        expected = np.logaddexp(values, 0)
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('threads', [1, 2])
@pytest.mark.parametrize('tokens,channels,kernel', [(1, 3, 1), (3, 5, 4), (4, 7, 3)])
def test_ssm_windows(bridge, np, threads, tokens, channels, kernel):
    rng = np.random.default_rng(23)
    values = rng.normal(size=(channels, tokens+kernel-1)).astype(np.float32)
    weights = rng.normal(size=(channels, kernel)).astype(np.float32)
    before = values.copy()
    output = np.empty((tokens, channels), dtype=np.float32)
    okay(bridge, bridge.ngramma_ssm_conv(*pointers(values, weights, output), tokens, channels, kernel, threads))
    expected = np.stack([(values[:,t:t+kernel]*weights).sum(-1) for t in range(tokens)])
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(values, before)


def run_gdn(bridge, np, q, k, v, g, beta, state, threads):
    tokens, key_heads, dim = q.shape
    value_heads = v.shape[1]
    output = np.empty_like(v)
    new_state = np.empty_like(state)
    okay(bridge, bridge.ngramma_gdn(*pointers(q, k, v, g, beta, state, output, new_state),
                                  tokens, key_heads, value_heads, dim, threads))
    return output, new_state


@pytest.mark.parametrize('threads', [1, 2])
@pytest.mark.parametrize('tokens,key_heads,value_heads,dim', [(1, 1, 1, 4), (3, 2, 4, 7), (2, 2, 4, 32)])
def test_gdn_tiled_heads_nonzero_state_and_split_invariance(bridge, np, threads, tokens, key_heads, value_heads, dim):
    rng = np.random.default_rng(31)
    q = rng.normal(size=(tokens, key_heads, dim)).astype(np.float32)*.1
    k = rng.normal(size=q.shape).astype(np.float32)*.1
    v = rng.normal(size=(tokens, value_heads, dim)).astype(np.float32)
    g = -rng.random((tokens, value_heads), dtype=np.float32)
    beta = rng.random((tokens, value_heads), dtype=np.float32)
    state = rng.normal(size=(value_heads, dim, dim)).astype(np.float32)*.1
    initial = state.copy()
    output, new_state = run_gdn(bridge, np, q, k, v, g, beta, state, threads)
    expected = np.empty_like(v)
    reference_state = state.copy()
    for token in range(tokens):
        for head in range(value_heads):
            # Engine state[value,key] is the transpose of conceptual state[key,value].
            reference_state[head] *= np.exp(g[token,head])
            delta = (v[token,head]-reference_state[head]@k[token,head % key_heads])*beta[token,head]
            reference_state[head] += delta[:,None]*k[token,head % key_heads][None,:]
            expected[token,head] = (reference_state[head]@q[token,head % key_heads])/np.sqrt(np.float32(dim))
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(new_state, reference_state, rtol=2e-5, atol=1e-6)
    np.testing.assert_array_equal(state, initial)
    step_state = state.copy()
    steps = []
    for token in range(tokens):
        args = [array[token:token+1] for array in (q, k, v, g, beta)]
        step_output, step_state = run_gdn(bridge, np, *args, step_state, threads)
        steps.append(step_output)
    np.testing.assert_array_equal(np.concatenate(steps), output)
    np.testing.assert_array_equal(step_state, new_state)


def test_gdn_float32_engine_transpose_convention(bridge, np):
    # beta=0, g=0 leave a nonsymmetric state unchanged. A basis query selects
    # a KEY column of stored state[value,key], not one of its value rows.
    state = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    q = np.array([[[1, 0]]], dtype=np.float32)
    k = np.zeros_like(q)
    v = np.zeros_like(q)
    g = np.zeros((1, 1), dtype=np.float32)
    beta = np.zeros_like(g)
    output, new_state = run_gdn(bridge, np, q, k, v, g, beta, state, 1)
    scale = np.float32(1)/np.sqrt(np.float32(2))
    expected = np.array([[[1, 3]]], dtype=np.float32)*scale
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(new_state, state)
    assert output.dtype == np.float32 and new_state.dtype == np.float32


def test_guarded_invalid_ids_and_shapes(bridge, np):
    # All pointers reference real arrays; invalid parameters are rejected before
    # any kernel uses them. Never pass unsupported ABI addresses to probe errors.
    buf = np.ones((128,), dtype=np.float32)
    other = np.empty_like(buf)
    for args in [(-1, 32, 2, 1, 1), (999, 32, 2, 1, 1),
                 (0, 0, 2, 1, 1), (0, 32, 2, 1, 0), (2, 7, 2, 1, 1)]:
        kind, width, rows, tokens, threads = args
        assert bridge.ngramma_matmul(kind, *pointers(buf, buf, other), width, rows, tokens, threads) == -1
        assert bridge.ngramma_last_error()
    for op, width, epsilon in [(7, 4, 1e-6), (0, 0, 1e-6), (0, 4, -1)]:
        assert bridge.ngramma_unary(op, *pointers(buf, other), width, 1, epsilon, 1) == -1
        assert bridge.ngramma_last_error()
    assert bridge.ngramma_ssm_conv(*pointers(buf, buf, other), 1, 2, 0, 1) == -1
    # Valid storage, but incompatible head ratio (Hv=3 is not divisible by Hk=2).
    state_output = np.empty_like(buf)
    assert bridge.ngramma_gdn(*pointers(buf, buf, buf, buf, buf, buf, other, state_output), 1, 2, 3, 2, 1) == -1
    assert b'divisible' in bridge.ngramma_last_error()


@pytest.fixture(scope='module')
def repack_bridge(bridge):
    if not hasattr(bridge, 'ngramma_matmul_repack'):
        pytest.skip('Configured bridge predates the optional repack API')
    bridge.ngramma_matmul_repack.argtypes = bridge.ngramma_matmul.argtypes
    bridge.ngramma_matmul_repack.restype = ct.c_int
    bridge.ngramma_last_buffer_type.restype = ct.c_char_p
    return bridge


@pytest.mark.parametrize('tokens', [1, 4, 10, 12])
def test_repack_q4_random_preserves_token_rows(repack_bridge, np, tokens):
    bridge = repack_bridge
    bridge.ggml_row_size.argtypes = [ct.c_int, ct.c_int64]
    bridge.ggml_row_size.restype = ct.c_size_t
    bridge.ggml_quantize_chunk.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p,
                                          ct.c_int64, ct.c_int64, ct.c_int64, ct.c_void_p]
    bridge.ggml_quantize_chunk.restype = ct.c_size_t
    rng = np.random.default_rng(902+tokens)
    width, channels, qtype = 128, 32, 2  # Actual source Q4_0 bytes, not decoded F32.
    weights = rng.normal(size=(channels, width)).astype(np.float32)*.1
    raw = np.empty(bridge.ggml_row_size(qtype, width)*channels, dtype=np.uint8)
    assert bridge.ggml_quantize_chunk(qtype, weights.ctypes.data, raw.ctypes.data,
                                     0, channels, width, None) == raw.size
    original = raw.copy()
    values = rng.normal(size=(tokens, width)).astype(np.float32)
    raw_output = np.empty((tokens, channels), dtype=np.float32)
    # Sentinel rows check that non-tile token lengths do not overrun output.
    guarded_output = np.full((tokens+2, channels), 12345, dtype=np.float32)
    output = guarded_output[1:-1]
    okay(bridge, bridge.ngramma_matmul(qtype, *pointers(raw, values, raw_output), width, channels, tokens, 2))
    okay(bridge, bridge.ngramma_matmul_repack(qtype, *pointers(raw, values, output), width, channels, tokens, 2))
    name = bridge.ngramma_last_buffer_type().decode()
    assert name, 'Buffer selection must be explicit even if repacking is unavailable'
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(guarded_output[[0,-1]], np.full((2, channels), 12345, dtype=np.float32))
    # Extra-buffer kernels can round activation scales differently. This checks
    # numerical sanity, not bit equality or full-engine parity qualification.
    np.testing.assert_allclose(output, raw_output, rtol=2e-3, atol=2e-3)


def test_repack_f32_explicit_plain_cpu_fallback(repack_bridge, np):
    bridge = repack_bridge
    weights = np.arange(64, dtype=np.float32).reshape(2,32)/16
    values = np.ones((10,32), dtype=np.float32)
    output = np.empty((10,2), dtype=np.float32)
    okay(bridge, bridge.ngramma_matmul_repack(0, *pointers(weights, values, output), 32,2,10,1))
    assert bridge.ngramma_last_buffer_type().decode() == 'CPU'
    np.testing.assert_allclose(output, values@weights.T, rtol=1e-6, atol=1e-6)

"""Optional adapter-only CPU fixtures: no checkpoint, table, or model loads.

Configure NGRAMMA_FORWARD_LIBRARY (with sum_rows support), NGRAMMA_ENGINE_SOURCE,
and NGRAMMA_ENGRAFT_SOURCE to run. Unconfigured CI imports no research packages.
"""
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest


@pytest.fixture(scope='module')
def runtime():
    required = ('NGRAMMA_FORWARD_LIBRARY', 'NGRAMMA_ENGINE_SOURCE',
                'NGRAMMA_ENGRAFT_SOURCE')
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip('Configure optional native adapter fixtures: ' + ', '.join(missing))
    library = Path(os.environ['NGRAMMA_FORWARD_LIBRARY'])
    if not library.is_file():
        pytest.fail('Configured native forward library does not exist')
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    from ngramma_runtime.native_forward import NativeForward
    import engraft.replica.layers as layers
    return SimpleNamespace(np=np, torch=torch, adapter=NativeForward,
                           library=library, layers=layers)


def make_adapter(runtime, raw_weight=None, **options):
    registry = {}
    if raw_weight is not None:
        registry[raw_weight.ctypes.data] = (
            weakref.ref(raw_weight), raw_weight, 0, raw_weight.shape)
    return runtime.adapter(SimpleNamespace(native_weight_views=registry),
                           runtime.library, primitives=False, threads=1, **options)


def forbid_call(monkeypatch, adapter, name):
    calls = []
    def forbidden(*args):
        calls.append(args)
        raise AssertionError('Invalid input reached native code')
    monkeypatch.setattr(adapter.lib, name, forbidden)
    return calls


def test_registered_shared_gate_vector_uses_native_projection(runtime):
    np, torch = runtime.np, runtime.torch
    raw_weight = np.array([.5, -1, 2, -.25], dtype=np.float32)
    weight = torch.from_numpy(raw_weight)
    values = torch.tensor([[2., 3., 4., 8.], [-4., 2., 1., 4.], [0., 0., 0., 0.]])
    expected = np.array([4., -3., 0.], dtype=np.float32)
    adapter = make_adapter(runtime, raw_weight, reductions=True)
    with torch.no_grad(), adapter:
        actual = values @ weight
    assert actual.shape == (3,)
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert adapter.calls['registered_vector_weight'] == 1
    assert adapter.calls['matmul_type_0'] == 1
    assert adapter.calls['unregistered_mm_or_mv'] == 0


def test_dead_matrix_registration_does_not_hide_live_vector_weight(runtime):
    np, torch = runtime.np, runtime.torch
    raw_weight = np.array([.5, -1, 2, -.25], dtype=np.float32)
    values = torch.tensor([[2., 3., 4., 8.]])
    old_decoded = np.zeros((1, 4), dtype=np.float32)
    expired = weakref.ref(old_decoded)
    del old_decoded
    assert expired() is None
    adapter = make_adapter(runtime, raw_weight, reductions=True)
    adapter.weights.native_weight_views[values.data_ptr()] = (
        expired, np.zeros(1, dtype=np.uint8), 0, (1, 4))
    with torch.no_grad(), adapter:
        actual = values @ torch.from_numpy(raw_weight)
    np.testing.assert_array_equal(actual.numpy(), [4.])
    assert adapter.calls['registered_vector_weight'] == 1
    assert adapter.calls['unregistered_mm_or_mv'] == 0


@pytest.mark.parametrize('vector_weight', [False, True])
def test_float16_matrix_input_rejected_before_native(runtime, monkeypatch, vector_weight):
    np, torch = runtime.np, runtime.torch
    shape = (4,) if vector_weight else (2, 4)
    raw_weight = np.ones(shape, dtype=np.float32)
    weight = torch.from_numpy(raw_weight)
    values = torch.ones((3, 4), dtype=torch.float16)
    adapter = make_adapter(runtime, raw_weight, reductions=True)
    calls = forbid_call(monkeypatch, adapter, 'ngramma_matmul')
    with torch.no_grad(), adapter:
        with pytest.raises(ValueError, match='CPU float32'):
            values @ (weight if vector_weight else weight.T)
    assert not calls


def test_grad_enabled_entry_refused_without_patching(runtime):
    torch = runtime.torch
    original = runtime.layers.causal_depthwise_conv
    adapter = make_adapter(runtime, recurrent=True)
    with torch.enable_grad():
        with pytest.raises(RuntimeError, match='no backward'):
            with adapter:
                pytest.fail('Gradient-enabled context entered')
    assert runtime.layers.causal_depthwise_conv is original
    assert not adapter.calls


@pytest.mark.parametrize('problem', ['history', 'channels', 'rank', 'half_history'])
def test_bad_convolution_inputs_rejected_before_native(runtime, monkeypatch, problem):
    torch = runtime.torch
    value = torch.ones((2, 3))
    history = torch.zeros((2, 3))
    kernel = torch.ones((3, 3))
    if problem == 'history':
        history = torch.zeros((3, 3))
    elif problem == 'channels':
        kernel = torch.ones((4, 3))
    elif problem == 'rank':
        value = value.unsqueeze(0)
    else:
        history = history.half()
    original = runtime.layers.causal_depthwise_conv
    adapter = make_adapter(runtime, recurrent=True)
    calls = forbid_call(monkeypatch, adapter, 'ngramma_ssm_conv')
    with torch.no_grad(), adapter:
        with pytest.raises(ValueError, match='shapes|float32'):
            runtime.layers.causal_depthwise_conv(value, history, kernel)
    assert not calls
    assert runtime.layers.causal_depthwise_conv is original


@pytest.mark.parametrize('history_length,dilation', [(0, 1), (1, 1), (2, 1), (0, 3)])
def test_convolution_accepts_missing_history(runtime, history_length, dilation):
    np, torch = runtime.np, runtime.torch
    values = torch.arange(1, 16, dtype=torch.float32).reshape(5, 3)
    history = torch.arange(-3 * history_length, 0, dtype=torch.float32).reshape(history_length, 3)
    kernel = torch.tensor([[1., 2., 3.], [-1., 1., 2.], [2., -2., 1.]])
    original = runtime.layers.causal_depthwise_conv
    # Run the original function before entering the patching context.
    with torch.no_grad():
        expected_original = original(values, history, kernel, dilation)
    # Independent causal indexing also verifies absent history means zeros.
    expected = np.zeros((5, 3), dtype=np.float32)
    source = np.concatenate((history.numpy(), values.numpy()), axis=0)
    for t in range(5):
        for tap in range(3):
            position = history_length + t - (2 - tap) * dilation
            if position >= 0:
                expected[t] += source[position] * kernel.numpy()[:, tap]
    adapter = make_adapter(runtime, recurrent=True)
    with torch.no_grad(), adapter:
        actual = runtime.layers.causal_depthwise_conv(values, history, kernel, dilation)
    np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(actual.numpy(), expected_original.numpy())
    assert adapter.calls['ssm_conv'] == (1 if dilation == 1 else 0)
    assert adapter.calls['fallback_dilated_conv'] == (0 if dilation == 1 else 1)
    assert runtime.layers.causal_depthwise_conv is original


@pytest.mark.parametrize('problem', ['key', 'heads', 'gate', 'state', 'half'])
def test_bad_gdn_inputs_rejected_before_native(runtime, monkeypatch, problem):
    torch = runtime.torch
    q = torch.ones((2, 2, 4))
    k = q.clone()
    v = torch.ones((2, 4, 4))
    g = torch.zeros((2, 4))
    beta = torch.ones((2, 4))
    state = torch.zeros((4, 4, 4))
    if problem == 'key':
        k = torch.ones((1, 2, 4))
    elif problem == 'heads':
        v = torch.ones((2, 3, 4))
    elif problem == 'gate':
        g = torch.zeros((2, 1))
    elif problem == 'state':
        state = torch.zeros((4, 4, 3))
    else:
        q = q.half()
    original = runtime.layers.gated_delta_net_recurrence
    adapter = make_adapter(runtime, recurrent=True)
    calls = forbid_call(monkeypatch, adapter, 'ngramma_gdn')
    with torch.no_grad(), adapter:
        with pytest.raises(ValueError, match='shapes|float32'):
            runtime.layers.gated_delta_net_recurrence(q, k, v, g, beta, state)
    assert not calls
    assert runtime.layers.gated_delta_net_recurrence is original


@pytest.mark.parametrize('keepdim', [False, True])
@pytest.mark.parametrize('dimension', [-1, 2])
def test_row_sum_preserves_dimensions_and_double_accumulation(runtime, keepdim, dimension):
    np, torch = runtime.np, runtime.torch
    raw = np.array([[[1e8, 1, -1e8], [2, 3, 4]],
                    [[-1e8, -1, 1e8], [8, -2, 1]]], dtype=np.float32)
    expected = raw.sum(axis=-1, dtype=np.float64, keepdims=keepdim).astype(np.float32)
    adapter = make_adapter(runtime, reductions=True)
    with torch.no_grad(), adapter:
        actual = torch.from_numpy(raw).sum(dim=dimension, keepdim=keepdim)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert adapter.calls['sum_rows'] == 1

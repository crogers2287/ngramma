"""Forward-only PLE lifecycle checks using tiny synthetic data, no model/native library."""
import sys
from types import ModuleType,SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('numpy')
from ngramma_runtime.ple_reference import NativePleScale


@pytest.fixture
def fake_engraft(monkeypatch):
    engraft=ModuleType('engraft');engraft.__path__=[]
    replica=ModuleType('engraft.replica');replica.__path__=[]
    layers=ModuleType('engraft.replica.layers');model=ModuleType('engraft.replica.model')
    original=lambda *args:None
    layers.ple_forward=original;model.ple_forward=original
    layers.rmsnorm_grouped=lambda value,*_:value
    layers.causal_depthwise_conv=lambda value,*args,**kwargs:torch.zeros_like(value)
    engraft.replica=replica;replica.layers=layers;replica.model=model
    for module in (engraft,replica,layers,model):monkeypatch.setitem(sys.modules,module.__name__,module)
    return layers,model,original


def inputs():
    hp=SimpleNamespace(hc_mult=1,n_embd=4,f_norm_rms_eps=1e-6,ple_ngram_size=2)
    w=SimpleNamespace(w_key=torch.eye(4),w_value=torch.eye(4),norm_key=None,norm_query=None,
                      norm_conv=None,conv1d=torch.ones(4,2))
    return torch.ones(3,4),torch.ones(3,1,4),w,torch.empty(0,1,4),hp


def test_enter_and_direct_forward_require_no_grad():
    with torch.enable_grad():
        with pytest.raises(RuntimeError,match='forward-only'):NativePleScale().__enter__()
        with pytest.raises(RuntimeError,match='forward-only'):NativePleScale().forward(*inputs())


def test_successful_single_forward_records_scalar_and_restores(fake_engraft):
    layers,model,original=fake_engraft
    reference=NativePleScale()
    with torch.no_grad(),reference:
        assert layers.ple_forward==reference.forward and model.ple_forward==reference.forward
        output,history=layers.ple_forward(*inputs())
        assert output.shape==(3,1,4) and history.shape==(2,1,4)
        assert torch.isfinite(output).all() and not output.requires_grad
    assert reference.calls==1 and reference.coefficient==.5
    assert layers.ple_forward is original and model.ple_forward is original
    with torch.no_grad(),pytest.raises(RuntimeError,match='new PLE reference'):
        reference.__enter__()


def test_exception_restores_both_patch_targets(fake_engraft):
    layers,model,original=fake_engraft
    with torch.no_grad(),pytest.raises(ValueError,match='deliberate'):
        with NativePleScale():raise ValueError('deliberate')
    assert layers.ple_forward is original and model.ple_forward is original


def test_no_invocation_fails_and_restores(fake_engraft):
    layers,model,original=fake_engraft
    with torch.no_grad(),pytest.raises(RuntimeError,match='exactly one PLE'):
        with NativePleScale():pass
    assert layers.ple_forward is original and model.ple_forward is original


def test_cached_history_rejected(fake_engraft):
    args=list(inputs());args[3]=torch.ones(1,1,4)
    with torch.no_grad(),pytest.raises(ValueError,match='fresh sequence'):
        NativePleScale().forward(*args)


def test_second_forward_in_same_context_rejected(fake_engraft):
    with torch.no_grad(),NativePleScale() as reference:
        reference.forward(*inputs())
        with pytest.raises(ValueError,match='fresh sequence'):
            reference.forward(*inputs())

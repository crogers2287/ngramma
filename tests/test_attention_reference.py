"""Synthetic layout evidence and fresh-sequence gates, without ENGRAFT/models."""
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

np=pytest.importorskip('numpy')
torch=pytest.importorskip('torch')
from ngramma_runtime.attention_reference import NativeAttentionReference, verified_layouts


@pytest.fixture
def capture(tmp_path):
    tokens=[12,23,34]
    hp=SimpleNamespace(n_layer=2,n_head=2,is_recr=lambda layer: layer==0)
    array=np.zeros((1,2,3,8),dtype='<f4')
    for token in range(3):array[:,:,token,:token+1]=1/(token+1)
    item={'name':'kq_soft_max-1','type':0,'shape':list(reversed(array.shape)),
          'strides':list(reversed(array.strides)),'file':'probabilities.bin'}
    def write(values=array,items=None,record_tokens=tokens):
        (tmp_path/'probabilities.bin').write_bytes(values.tobytes())
        (tmp_path/'tensors.json').write_text(json.dumps([item] if items is None else items))
        (tmp_path/'tokens.json').write_text(json.dumps(record_tokens))
    write()
    return tmp_path,tokens,hp,array,item,write


def test_verified_layouts_dense_causal_fixture_and_hash(capture):
    path,tokens,hp,array,_,_=capture
    assert verified_layouts(path,hp,tokens)==[{'layer':1,'cache_slots':8,
        'probability_sha256':hashlib.sha256(array.tobytes()).hexdigest()}]


@pytest.mark.parametrize('failure',['missing','duplicate','wrong_type','wrong_heads','wrong_token_shape','short_cache'])
def test_layout_rejects_missing_or_wrong_tensor_shape(capture,failure):
    path,tokens,hp,_,item,write=capture
    item=dict(item)
    if failure=='missing':items=[]
    elif failure=='duplicate':items=[item,item]
    else:
        if failure=='wrong_type':item['type']=1
        elif failure=='wrong_heads':item['shape']=[8,3,4,1]
        elif failure=='wrong_token_shape':item['shape']=[8,2,2,1]
        elif failure=='short_cache':item['shape']=[2,3,2,1]
        items=[item]
    write(items=items)
    with pytest.raises(ValueError):verified_layouts(path,hp,tokens)


@pytest.mark.parametrize('failure',['negative','nan','infinity','missing_allowed_key','future_key'])
def test_layout_rejects_invalid_probabilities_or_support(capture,failure):
    path,tokens,hp,array,_,write=capture
    array=array.copy()
    if failure=='negative':array[0,0,0,0]=-1
    elif failure=='nan':array[0,0,0,0]=np.nan
    elif failure=='infinity':array[0,0,0,0]=np.inf
    elif failure=='missing_allowed_key':array[0,0,2,0]=0
    else:array[0,0,0,7]=.1
    write(values=array)
    with pytest.raises(ValueError,match='Dense causal support'):
        verified_layouts(path,hp,tokens)


def test_layout_requires_matching_nonempty_bounded_token_sequence(capture):
    path,tokens,hp,_,_,write=capture
    for bad in ([12,23,35],[],list(range(129))):
        write(record_tokens=bad)
        with pytest.raises(ValueError,match='tokens must match'):
            verified_layouts(path,hp,bad if len(bad)!=3 else tokens)


def test_no_full_attention_layers_is_not_success(capture):
    path,tokens,hp,_,_,_=capture
    hp.is_recr=lambda _:True
    with pytest.raises(ValueError,match='No full-attention layers'):
        verified_layouts(path,hp,tokens)


@pytest.fixture
def fake_engraft(monkeypatch):
    # Only the patch targets are faked. No computation is substituted for tests
    # claiming numerical correctness: these exercise lifecycle/early guards only.
    engraft=ModuleType('engraft');engraft.__path__=[]
    replica=ModuleType('engraft.replica');replica.__path__=[]
    layers=ModuleType('engraft.replica.layers');model=ModuleType('engraft.replica.model')
    original=lambda *args:None
    layers.attention_full=original;model.attention_full=original
    engraft.replica=replica;replica.layers=layers;replica.model=model
    for module in (engraft,replica,layers,model):monkeypatch.setitem(sys.modules,module.__name__,module)
    return layers,model,original


def reference():
    return NativeAttentionReference(None,[{'layer':1,'cache_slots':8}],[12,23,34])


def test_reference_requires_no_grad_context():
    with torch.enable_grad(),pytest.raises(RuntimeError,match='no backward'):
        reference().__enter__()


def test_reference_restores_functions_on_exception_and_cannot_reenter(fake_engraft):
    layers,model,original=fake_engraft
    ref=reference()
    with torch.no_grad():
        with pytest.raises(ValueError,match='deliberate'):
            with ref:
                assert layers.attention_full==ref.attention and model.attention_full==ref.attention
                raise ValueError('deliberate')
        assert layers.attention_full is original and model.attention_full is original
        with pytest.raises(RuntimeError,match='new attention reference'):
            ref.__enter__()


def test_reference_missing_calls_fails_and_restores(fake_engraft):
    layers,model,original=fake_engraft
    with torch.no_grad(),pytest.raises(RuntimeError,match='call count'):
        with reference():pass
    assert layers.attention_full is original and model.attention_full is original


@pytest.mark.parametrize('cache_index',[0,1,2])
def test_cached_prefixes_rejected(fake_engraft,cache_index):
    caches=[None,None,None];caches[cache_index]=torch.zeros(1)
    with torch.no_grad(),pytest.raises(ValueError,match='fresh sequences'):
        reference().attention(torch.zeros(3,4),None,torch.arange(3),None,*caches)


@pytest.mark.parametrize('failure',['wrong_length','wrong_positions','exhausted_layouts'])
def test_reference_sequence_boundaries(fake_engraft,failure):
    ref=reference();values=torch.zeros(3,4);positions=torch.arange(3)
    if failure=='wrong_length':values=values[:2]
    elif failure=='wrong_positions':positions+=1
    else:ref.used=1
    with torch.no_grad(),pytest.raises(ValueError):
        ref.attention(values,None,positions,None,None,None,None)

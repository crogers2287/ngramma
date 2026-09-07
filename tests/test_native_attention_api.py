"""Python attention ABI boundaries with synthetic tensors only; no model/GPU."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def api():
    path = os.environ.get('NGRAMMA_FORWARD_LIBRARY')
    if not path:
        pytest.skip('Set NGRAMMA_FORWARD_LIBRARY for native attention API tests')
    torch = pytest.importorskip('torch')
    from ngramma_runtime.native_attention import AttentionOps, RopeConfig
    # Absence of a configured file is an error, never a successful skip.
    metadata = {'general.architecture': 'qwen4exp', 'qwen4exp.context_length': 4096,
                'qwen4exp.rope.dimension_count': 4,
                'qwen4exp.rope.dimension_sections': [2,0,0,0],
                'qwen4exp.rope.freq_base': 10000.}
    ops = AttentionOps(path, rope_config=RopeConfig.from_metadata(metadata), threads=2)
    hp = SimpleNamespace(rope_sections=(2,0,0,0), rope_dim=4, rope_freq_base=10000.)
    return torch, ops, hp, metadata


def test_requires_no_grad_context(api):
    torch,ops,_,_=api
    with torch.enable_grad(), pytest.raises(RuntimeError,match='no backward'):
        ops.matmul(torch.ones(1,2,4),torch.ones(1,2,4))


@pytest.mark.parametrize('kind',['requires_grad','float64','integer','empty','meta'])
def test_input_tensor_guards(api,kind):
    torch,ops,_,_=api
    tensors = {
        'requires_grad': lambda: torch.ones(1,2,4,requires_grad=True),
        'float64': lambda: torch.ones(1,2,4,dtype=torch.float64),
        'integer': lambda: torch.ones(1,2,4,dtype=torch.int32),
        'empty': lambda: torch.empty(1,0,4),
        # Meta tensors test the CPU boundary without accessing a GPU/device.
        'meta': lambda: torch.empty(1,2,4,device='meta'),
    }
    with torch.no_grad(),pytest.raises(ValueError,match='CPU float32'):
        ops.matmul(tensors[kind](),torch.ones(1,2,4))


@pytest.mark.parametrize('wshape,xshape', [((2,4),(1,2,4)),((1,2,4),(2,4)),
                                           ((1,2,4),(1,2,5)),((2,2,4),(3,2,4))])
def test_matmul_shape_guards(api,wshape,xshape):
    torch,ops,_,_=api
    with torch.no_grad(),pytest.raises(ValueError,match='Expected weights'):
        ops.matmul(torch.ones(wshape),torch.ones(xshape))


def test_text_position_zero_identity_and_metadata_settings(api):
    torch,ops,hp,_=api
    values=torch.arange(48,dtype=torch.float32).reshape(3,2,8)
    with torch.no_grad():
        output=ops.rope(values,torch.zeros(3,dtype=torch.int64),hp)
    torch.testing.assert_close(output,values,rtol=0,atol=0)
    assert output.dtype==torch.float32 and not output.requires_grad
    assert ops.settings()=={'mode':40,'original_context':4096,'frequency_scale':1.,
                            'extension_factor':0.,'attention_factor':1.,'beta_fast':32.,'beta_slow':1.}


@pytest.mark.parametrize('bad_position',[.25,-1.,float('nan'),float('inf'),2147483648.])
def test_rejects_noninteger_negative_nonfinite_or_overflow_positions(api,bad_position):
    torch,ops,hp,_=api
    positions=torch.tensor([0.,bad_position],dtype=torch.float64)
    with torch.no_grad(),pytest.raises(ValueError,match='nonnegative int32'):
        ops.rope(torch.ones(2,2,8),positions,hp)


@pytest.mark.parametrize('kind',['value_rank','position_rank','position_length','position_grad','position_meta'])
def test_rope_tensor_shapes_and_position_device(api,kind):
    torch,ops,hp,_=api
    values=torch.ones(2,2,8)
    positions=torch.zeros(2)
    if kind=='value_rank': values=values[0]
    elif kind=='position_rank': positions=positions[:,None]
    elif kind=='position_length': positions=torch.zeros(3)
    elif kind=='position_grad': positions.requires_grad_(True)
    elif kind=='position_meta': positions=torch.empty(2,device='meta')
    with torch.no_grad(),pytest.raises(ValueError,match='Text RoPE'):
        ops.rope(values,positions,hp)


@pytest.mark.parametrize('sections',[(1,1,0),(1.,1.,0.,0.),(-1,2,0,0),(2**31,0,0,0)])
def test_rope_section_metadata_guards(api,sections):
    torch,ops,hp,_=api
    bad=SimpleNamespace(rope_sections=sections,rope_dim=hp.rope_dim,rope_freq_base=hp.rope_freq_base)
    with torch.no_grad(),pytest.raises(ValueError,match='four bounded'):
        ops.rope(torch.ones(2,2,8),torch.zeros(2),bad)


@pytest.mark.parametrize('field,value', [('general.architecture','other'),
    ('qwen4exp.context_length',0),('qwen4exp.context_length',2**31),
    ('qwen4exp.context_length',4096.),('qwen4exp.rope.scaling.type','linear'),
    ('qwen4exp.rope.scaling.factor',2.)])
def test_metadata_architecture_context_and_scaling_overrides_rejected(api,field,value):
    _,_,_,metadata=api
    from ngramma_runtime.native_attention import RopeConfig
    modified=dict(metadata);modified[field]=value
    with pytest.raises(ValueError):
        RopeConfig.from_metadata(modified)


@pytest.mark.parametrize('kind',['shape','nan','positive_inf','all_masked'])
def test_softmax_mask_guards(api,kind):
    torch,ops,_,_=api
    scores=torch.zeros(2,3,8)
    mask=torch.zeros(3,8)
    if kind=='shape': mask=mask[:2]
    elif kind=='nan': mask[0,0]=float('nan')
    elif kind=='positive_inf': mask[0,0]=float('inf')
    elif kind=='all_masked': mask[1]=float('-inf')
    with torch.no_grad(),pytest.raises(ValueError):
        ops.softmax(scores,mask,.5)


@pytest.mark.parametrize('scale',[float('nan'),float('inf')])
def test_softmax_nonfinite_scale_rejected(api,scale):
    torch,ops,_,_=api
    with torch.no_grad(),pytest.raises(RuntimeError,match='scale must be finite'):
        ops.softmax(torch.zeros(2,3,8),torch.zeros(3,8),scale)


def test_softmax_causal_padding_through_wrapper(api):
    torch,ops,_,_=api
    scores=torch.arange(48,dtype=torch.float32).reshape(2,3,8)/20
    mask=torch.full((3,8),float('-inf'))
    for token in range(3):mask[token,:token+1]=0
    with torch.no_grad():
        output=ops.softmax(scores,mask,.25)
        expected=torch.softmax(scores*.25+mask,dim=-1)
    torch.testing.assert_close(output,expected,rtol=1e-6,atol=1e-7)
    assert torch.count_nonzero(output[...,3:])==0


def test_library_build_identity_mismatch_rejected(api,tmp_path):
    _,ops,_,_=api
    from ngramma_runtime.native_attention import AttentionOps
    source=Path(os.environ['NGRAMMA_FORWARD_LIBRARY']).resolve()
    linked=tmp_path/'test-library.so'
    # A copied library avoids resolve() following a symlink away from its record.
    linked.write_bytes(source.read_bytes())
    Path(str(linked)+'.build.json').write_text(json.dumps({'library_sha256':'0'*64}))
    with pytest.raises(ValueError,match='does not match its build record'):
        AttentionOps(linked,rope_config=ops.rope_config)

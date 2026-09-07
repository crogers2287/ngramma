"""Model-free report contract, offline rendering, and command-line behavior."""
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys

import pytest
from ngramma_runtime.response_report import render_html,validate_report


@pytest.fixture
def report():
    path=Path(__file__).resolve().parents[1]/'examples/memory-edit-workbench/demo.json'
    return json.loads(path.read_text())


def test_demo_contract_is_valid_and_not_mutated(report):
    original=copy.deepcopy(report)
    assert validate_report(report) is report
    text=render_html(report)
    assert 'SYNTHETIC DEMO' in text and 'NOT MODEL EVIDENCE' in text
    assert 'absolute RMS difference' in text and 'not a relative error' in text
    assert report==original


@pytest.mark.parametrize('field,value',[('schema','wrong'),('evidence_kind','unknown'),('title','')])
def test_top_level_contract(report,field,value):
    report[field]=value
    with pytest.raises(ValueError):validate_report(report)


@pytest.mark.parametrize('field,value',[('epsilon',True),('epsilon',float('inf')),
    ('epsilon',float('nan')),('ple_output_rms',-1),('ple_output_rms',float('nan')),
    ('key_changed_count',1.5),('value_changed_count',True),('value_changed_count',-1),
    ('engine_logit_margin_delta',float('inf'))])
def test_invalid_point_values_rejected(report,field,value):
    report['points'][0][field]=value
    with pytest.raises(ValueError):validate_report(report)


def test_missing_and_duplicate_zero_baseline_rejected(report):
    report['points']=[point for point in report['points'] if point['epsilon']!=0]
    with pytest.raises(ValueError,match='baseline'):validate_report(report)
    report['points'].append(copy.deepcopy(report['points'][0]))
    with pytest.raises(ValueError,match='unique'):validate_report(report)


@pytest.mark.parametrize('field',['ple_output_rms','key_changed_count','value_changed_count','engine_logit_margin_delta'])
def test_nonzero_change_at_baseline_is_rejected(report,field):
    point=next(point for point in report['points'] if point['epsilon']==0)
    point[field]=1
    with pytest.raises(ValueError,match='exact zero'):validate_report(report)


def test_measured_and_optional_engine_observations(report):
    report['evidence_kind']='measured'
    report['points'][0]['engine_logit_margin_delta']=-.25
    report['points'][-1]['engine_logit_margin_delta']=.5
    report['points'][3]['engine_logit_margin_delta']=0
    report['points'][1]['engine_logit_margin_delta']=None
    report['provenance']={'deliberately_extra':'accepted but not authenticated'}
    text=render_html(report)
    assert 'DOES NOT ESTABLISH IMPROVED CAPABILITY' in text
    assert 'series-engine_logit_margin_delta' in text
    assert 'Not supplied' in text


class Elements(HTMLParser):
    def __init__(self):
        super().__init__();self.tags=[];self.scripts=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs);self.tags.append((tag,attrs))
        if tag=='script':self.scripts.append(attrs)


def test_offline_accessible_table_toggles_and_escaped_strings(report):
    report['title']='</title><script>alert(1)</script>'
    report['experiment']['notes']='<img src="https://example.test/steal">'
    text=render_html(report)
    parsed=Elements();parsed.feed(text)
    assert len(parsed.scripts)==1 and 'src' not in parsed.scripts[0]
    assert not any(tag in ('img','iframe','link') for tag,_ in parsed.tags)
    assert sum(tag=='table' for tag,_ in parsed.tags)==1
    assert sum(tag=='td' for tag,_ in parsed.tags)==len(report['points'])*5
    assert all(attrs.get('scope')=='col' for tag,attrs in parsed.tags if tag=='th')
    assert sum(tag=='input' and 'data-series' in attrs for tag,attrs in parsed.tags)==3
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in text
    assert "connect-src 'none'" in text


def test_single_baseline_and_large_signed_dynamic_range_render(report):
    zero=next(point for point in report['points'] if point['epsilon']==0)
    single=copy.deepcopy(report);single['points']=[zero]
    assert '<svg' in render_html(single)
    report['points'][0]['epsilon']=-1e300
    report['points'][-1]['epsilon']=1e300
    assert 'evenly spaced observations' in render_html(report)


def test_stdlib_only_command_creates_html(tmp_path):
    root=Path(__file__).resolve().parents[1]
    target=tmp_path/'report.html'
    code="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('ngramma_runtime.response_report',run_name='__main__')"
    result=subprocess.run([sys.executable,'-S','-c',code,str(root/'src'),
        str(root/'examples/memory-edit-workbench/demo.json'),'--html',str(target)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert target.read_text().startswith('<!doctype html>')


def test_optional_quantizer_byte_counts_display_separate_table_and_plot(report):
    for point in report['points']:
        count=0 if point['epsilon']==0 else 2
        point.update(activation_changed_bytes=count,activation_scale_changed_bytes=count,
                     activation_code_changed_bytes=0)
    rendered=render_html(report)
    parsed=Elements();parsed.feed(rendered)
    assert sum(tag=='table' for tag,_ in parsed.tags)==2
    assert 'Scale-only changes can alter dequantized values' in rendered
    assert 'series-activation_scale_changed_bytes' in rendered
    assert 'series-activation_code_changed_bytes' in rendered


@pytest.mark.parametrize('key',['activation_changed_bytes','activation_scale_changed_bytes','activation_code_changed_bytes'])
@pytest.mark.parametrize('value',[-1,True,1.5,None])
def test_quantizer_byte_counts_require_nonnegative_integers(report,key,value):
    report['points'][0][key]=value
    with pytest.raises(ValueError):validate_report(report)


def test_quantizer_byte_total_must_equal_scale_plus_code(report):
    report['points'][0].update(activation_changed_bytes=3,activation_scale_changed_bytes=2,
                               activation_code_changed_bytes=2)
    with pytest.raises(ValueError,match='equal scale plus code'):validate_report(report)


def test_quantizer_byte_baseline_and_partial_measurements(report):
    report['points'][0]['activation_scale_changed_bytes']=1
    text=render_html(report)
    assert 'series-activation_scale_changed_bytes' in text
    assert 'series-activation_changed_bytes' not in text
    zero=next(point for point in report['points'] if point['epsilon']==0)
    zero['activation_scale_changed_bytes']=1
    with pytest.raises(ValueError,match='exact zero activation'):validate_report(report)

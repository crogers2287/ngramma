"""Synthetic response checks and a Python fake engine; no model/native library loads."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/005-behavior/run_batch.py'
spec = importlib.util.spec_from_file_location('behavior_batch_for_tests', PATH)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


@pytest.fixture
def pair():
    job = {'tokens': [1, 2], 'generate': {'max_new_tokens': 4, 'context_tokens': 128,
                                       'compact': True, 'top_k': 2, 'score_tokens': [3]}}
    response = {
        'schema': 'ngramma.greedy-generation/v1', 'ok': True, 'greedy': True,
        'fresh_state': True, 'model_reused': True, 'grammar': None,
        'max_new_tokens': 4, 'prompt_tokens': [1, 2], 'generated_tokens': [3, 9],
        'generated_count': 2, 'text': 'A', 'text_bytes_hex': '41',
        'stop_reason': 'eog', 'stopped_on_eog': True, 'eog_token': 9,
        'effective_new_token_budget': 4, 'context_tokens': 128, 'vocab_size': 10,
        'compact': True, 'temperature': 0, 'tie_break': 'lowest_token_id',
        'first_step': {'greedy_token_id': 3, 'top_logits': [
            {'token_id': 3, 'logit': 2.0}, {'token_id': 4, 'logit': 1.0}],
            'requested_logits': [{'token_id': 3, 'logit': 2.0}]},
    }
    return job, response


def test_valid_generation_and_tokenization(pair):
    batch.validate_response(*pair)
    batch.validate_response({'tokenize_only': True}, {'tokens': [1, 2]})


@pytest.mark.parametrize('field,value', [
    ('ok', False), ('schema', 'wrong'), ('greedy', False), ('fresh_state', False),
    ('grammar', 'A|B'), ('max_new_tokens', 5), ('prompt_tokens', [1, 3]),
    ('generated_tokens', []), ('stop_reason', 'unknown'), ('generated_count', 1),
    ('text', None), ('text_bytes_hex', '42'),
])
def test_existing_response_guards(pair, field, value):
    job, response = pair
    response[field] = value
    with pytest.raises(ValueError):
        batch.validate_response(job, response)


def test_budget_exceeded(pair):
    job, response = pair
    response.update(generated_tokens=[3]*5, generated_count=5)
    with pytest.raises(ValueError, match='budget'):
        batch.validate_response(job, response)


@pytest.mark.parametrize('response', [{}, {'tokens': []}, {'tokens': '1'}])
def test_missing_tokenization(response):
    with pytest.raises(ValueError):
        batch.validate_response({'tokenize_only': True}, response)


@pytest.mark.parametrize('field,value', [
    ('generated_tokens', [True, 9]), ('generated_tokens', [10, 9]),
    ('eog_token', 8), ('stopped_on_eog', False), ('first_step', None),
    ('effective_new_token_budget', 1), ('context_tokens', 64), ('greedy', 1),
    ('first_step', {'greedy_token_id': 3, 'top_logits': [{'token_id': 3, 'logit': float('nan')}], 'requested_logits': []}),
])
def test_review_gap_rejects_tampered_generation(pair, field, value):
    job, response = pair
    response[field] = value
    with pytest.raises((ValueError, TypeError)):
        batch.validate_response(job, response)


def test_incomplete_utf8_text_is_valid(pair):
    job, response = pair
    response.update(text='\ufffd', text_bytes_hex='e2')
    batch.validate_response(job, response)


def fake_args(tmp_path, pair, mode='success'):
    job, response = pair
    engine = tmp_path / 'fake-engine'
    engine.write_text(f'#!{sys.executable}\n' +
        'import json, sys, time\n' +
        f'mode={mode!r}\nresponse={response!r}\n' +
        "if mode=='bad-ready':\n print(json.dumps({'ready':False}),flush=True); sys.exit(0)\n" +
        "print('fake engine diagnostic',flush=True)\nprint(json.dumps({'ready':True,'vocab_size':10}),flush=True)\n" +
        'for line in sys.stdin:\n' +
        " if mode=='exit': sys.exit(0)\n" +
        " if mode=='hang': time.sleep(3)\n" +
        " if mode=='error': print(json.dumps({'error':'synthetic engine error'}),flush=True); continue\n" +
        " if mode=='mutate':\n  with open(sys.argv[0], 'a') as f: f.write('\\n# changed\\n')\n" +
        ' print(json.dumps(response),flush=True)\n')
    engine.chmod(0o755)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'identity_sha256': 'synthetic-not-a-measurement',
                                    'shards': [{'path': 'never-opened.gguf'}]}))
    jobs = tmp_path / 'jobs.json'
    jobs.write_text(json.dumps([{'id': 'first', **job}, {'id': 'repeat', **copy.deepcopy(job)}]))
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    return SimpleNamespace(manifest=manifest, jobs=jobs, lens=engine, runtime=runtime,
                           overlay=None, timeout=1 if mode=='hang' else 10,
                           local=tmp_path / 'local', output=tmp_path / 'result.json')


@pytest.fixture
def healthy_resources(monkeypatch):
    monkeypatch.setattr(batch, 'child_resources', lambda pid: {
        'rss_bytes': 10 << 20, 'peak_rss_bytes': 12 << 20, 'available_bytes': 100 << 30})


def test_fake_engine_repeated_jobs_and_complete_record(tmp_path, pair, healthy_resources):
    args = fake_args(tmp_path, pair)
    batch.run(args)
    record = json.loads(args.output.read_text())
    assert record['status'] == 'complete'
    assert record['inputs_unchanged'] is True
    assert [j['id'] for j in record['jobs']] == ['first', 'repeat']
    assert record['jobs'][0]['response'] == record['jobs'][1]['response']
    assert not args.output.with_suffix('.json.pending').exists()
    assert 'fake engine diagnostic' in (args.local / 'responses.jsonl').read_text()


@pytest.mark.parametrize('mode,error', [('bad-ready', RuntimeError), ('exit', RuntimeError),
                                      ('error', ValueError), ('hang', TimeoutError),
                                      ('mutate', ValueError)])
def test_fake_engine_failure_record(tmp_path, pair, healthy_resources, mode, error):
    args = fake_args(tmp_path, pair, mode)
    with pytest.raises(error):
        batch.run(args)
    record = json.loads(args.output.read_text())
    assert record['status'] == 'failed'
    assert record['error'].startswith(error.__name__ + ':')
    assert 'inputs_unchanged' not in record


def test_memory_limit_preserves_failure_record(tmp_path, pair, monkeypatch):
    args = fake_args(tmp_path, pair)
    monkeypatch.setattr(batch, 'child_resources', lambda pid: {
        'rss_bytes': 81 << 30, 'peak_rss_bytes': 81 << 30, 'available_bytes': 100 << 30})
    with pytest.raises(MemoryError):
        batch.run(args)
    assert json.loads(args.output.read_text())['status'] == 'failed'


def test_duplicate_ids_rejected_before_launch(tmp_path, pair, monkeypatch):
    args = fake_args(tmp_path, pair)
    jobs = json.loads(args.jobs.read_text())
    jobs[1]['id'] = jobs[0]['id']
    args.jobs.write_text(json.dumps(jobs))
    monkeypatch.setattr(batch.subprocess, 'Popen', lambda *a, **k: pytest.fail('Must not launch'))
    with pytest.raises(ValueError, match='Duplicate'):
        batch.run(args)

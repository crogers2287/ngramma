#!/usr/bin/env python3
"""Execute only the predeclared finite candidates and replay real generations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/005-behavior'))
from prepare_jobs import jobs
from run_batch import validate_response
from score_results import grade_completion
from tasks import generate_tasks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def assess(record, baseline, candidate, expected_jobs):
    if record.get('status') != 'complete' or record.get('inputs_unchanged') is not True:
        raise ValueError('Incomplete candidate execution')
    if record['overlay_sha256'] != candidate['overlay_sha256']:
        raise ValueError('Candidate overlay identity mismatch')
    if record['identity_sha256'] != baseline['identity_sha256'] or record['runtime_sha256'] != baseline['runtime_sha256'] or record['lens_sha256'] != baseline['lens_sha256']:
        raise ValueError('Candidate model/runtime differs from baseline')
    for key in ('device', 'threads', 'context', 'batch', 'microbatch', 'flash_attention',
                'cache_type', 'repack', 'fresh_state_per_job', 'sampling'):
        if record['profile'].get(key) != baseline['profile'].get(key):
            raise ValueError('Candidate inference profile differs from baseline')
    if record['jobs_sha256'] != hashlib.sha256((json.dumps(expected_jobs, indent=2)+'\n').encode()).hexdigest():
        raise ValueError('Candidate task requests changed')
    if len(record['jobs']) != len(expected_jobs):
        raise ValueError('Candidate has missing/extra tasks')
    task_map = {t.id: t for t in generate_tasks()+generate_tasks('controls')}
    original = {j['id']: j['response'] for j in baseline['jobs'] if j['id'] in task_map}
    grades, losses, compliance_losses, rescues = [], [], [], []
    for actual, expected in zip(record['jobs'], expected_jobs):
        if actual['id'] != expected['id'] or actual['request'] != {k: v for k, v in expected.items() if k != 'id'}:
            raise ValueError('Candidate job differs from fixed input')
        validate_response(actual['request'], actual['response'])
        name = actual['id']
        if name not in task_map:
            continue
        task = task_map[name]
        if actual['response']['prompt_tokens'] != original[name]['prompt_tokens']:
            raise ValueError('Candidate prompt tokens differ from baseline')
        before = grade_completion(task, original[name])
        after = grade_completion(task, actual['response'])
        if before['success'] and not after['success']:
            losses.append(name)
        if before['well_formed'] and before['terminated'] and not (after['well_formed'] and after['terminated']):
            compliance_losses.append(name)
        if task.split == 'dev' and not before['success'] and after['success']:
            rescues.append(name)
        grades.append({'id': name, 'split': task.split, 'family': task.family,
                       'baseline_success': before['success'], 'candidate_success': after['success'],
                       'text': actual['response']['text'], 'error_kind': after['error_kind'],
                       'stop_reason': actual['response']['stop_reason']})
    score = sum(g['candidate_success'] for g in grades if g['split'] == 'dev')
    return {'label': candidate['label'], 'epsilon': candidate['epsilon'],
            'overlay_sha256': candidate['overlay_sha256'], 'development_correct': score,
            'controls_correct': sum(g['candidate_success'] for g in grades if g['split'] == 'controls'),
            'development_rescues': rescues, 'baseline_correct_losses': losses,
            'new_compliance_losses': compliance_losses,
            'eligible': len(rescues) >= 2 and not losses and not compliance_losses,
            'seconds': record['seconds'], 'tasks': grades}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'runtime', 'lens', 'verification-library', 'local'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--limit', type=int, default=12, help='Run a prefix of the fixed budget; never add candidates')
    a = p.parse_args()
    if not 1 <= a.limit <= 12:
        raise ValueError('Limit must be 1..12')
    exp = Path(__file__).resolve().parent
    recipes_path = exp/'candidates.json'
    recipes = json.loads(recipes_path.read_text())
    baseline = json.loads((ROOT/'experiments/005-behavior/discovery.json').read_text())
    pinned = {str(path): sha(path) for path in (Path(__file__), recipes_path, a.lens, a.verification_library)}
    zero = json.loads((exp/'zero-validation.json').read_text())
    if zero.get('passed') is not True or zero.get('verification_library_sha256') != sha(a.verification_library):
        raise ValueError('Require qualified zero-overlay control')
    candidates = [x for x in recipes['candidates'] if x['label'] != 'zero']
    records = []
    a.local.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates[:a.limit]):
        if index > 0:
            witness = json.loads((exp/'first-candidate-lookup.json').read_text())
            if witness.get('actual_equals_intended_bytes') is not True or witness.get('changed_values_from_original', 0) < 1 or witness.get('overlay_sha256') != candidates[0]['overlay_sha256']:
                raise ValueError('Require measured nonzero multirow lookup qualification')
        label = candidate['label']
        overlay = a.local/'candidates'/label/'rows.fml'
        if sha(overlay) != candidate['overlay_sha256']:
            raise ValueError('Prepared candidate bytes changed')
        requested = jobs(controls=True)
        if index == 0:
            witness = {**requested[2], 'id': 'lookup-witness',
                       'generate': {**requested[2]['generate'], 'compact': False},
                       'capture': ['ple_embd'], 'output_dir': str((a.local/'first-candidate-witness').resolve())}
            requested.append(witness)
        job_path = a.local/(label+'-jobs.json')
        if job_path.exists():
            if json.loads(job_path.read_text()) != requested:
                raise ValueError('Stored candidate job list changed')
        else:
            atomic(job_path, requested)
        output = exp/(label+'.json')
        if not output.exists():
            command = [sys.executable, str(ROOT/'experiments/005-behavior/run_batch.py'),
                       '--manifest', str(a.manifest), '--runtime', str(a.runtime),
                       '--lens', str(a.lens), '--verification-library', str(a.verification_library),
                       '--overlay', str(overlay), '--jobs', str(job_path),
                       '--local', str(a.local/(label+'-run')), '--output', str(output)]
            print('Running candidate '+str(index+1)+'/'+str(len(candidates))+': '+label, flush=True)
            with (a.local/(label+'.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=930)
        record = json.loads(output.read_text())
        if record['profile'].get('verification_library_sha256') != sha(a.verification_library):
            raise ValueError('Candidate verifier differs from qualified control')
        assessed = assess(record, baseline, candidate, requested)
        assessed['record_sha256'] = sha(output)
        records.append(assessed)
        if any(sha(Path(name)) != value for name, value in pinned.items()):
            raise ValueError('Search source/recipe/runtime changed during execution')
        report = {'schema': 'ngramma.instruction-search/v1', 'complete': len(records) == len(candidates),
                  'candidate_recipe_sha256': sha(recipes_path), 'results': records,
                  'winner': None, 'heldout_evaluated': False}
        if report['complete']:
            def rank(item):
                direction = next(i for i, name in enumerate(('rademacher', 'alternating', 'hint_contrast')) if item['label'].startswith(name))
                return (-item['development_correct'], abs(item['epsilon']), direction, int(item['epsilon'] > 0))
            eligible = sorted((r for r in records if r['eligible']), key=rank)
            report['winner'] = eligible[0]['label'] if eligible else None
        atomic(exp/'search-results.json', report)
        print(json.dumps({k: v for k, v in assessed.items() if k != 'tasks'}), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run bounded fresh-state jobs through one isolated, CPU-only model process.

No server, grammar, answer constraint, prefix cache reuse, or teacher is used.
The overlay is fixed for the process and authenticated by the native loader.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def child_resources(pid):
    status = Path(f'/proc/{pid}/status').read_text()
    memory = Path('/proc/meminfo').read_text()
    def value(text, name):
        return next(int(line.split()[1]) * 1024 for line in text.splitlines()
                    if line.startswith(name + ':'))
    return {'rss_bytes': value(status, 'VmRSS'),
            'peak_rss_bytes': value(status, 'VmHWM'),
            'available_bytes': value(memory, 'MemAvailable')}


def validate_response(job, response):
    def integer(value, low, high):
        return type(value) is int and low <= value <= high

    if job.get('tokenize_only'):
        if not isinstance(response.get('tokens'), list) or not response['tokens']:
            raise ValueError('Missing tokenization response')
        if not all(integer(t, 0, 2**31-1) for t in response['tokens']):
            raise ValueError('Invalid tokenizer token ID')
        return
    if response.get('ok') is not True:
        raise ValueError('Engine did not explicitly report success')
    if 'generate' in job:
        for key, expected in (('schema', 'ngramma.greedy-generation/v1'),
                              ('greedy', True), ('fresh_state', True),
                              ('grammar', None), ('max_new_tokens', job['generate']['max_new_tokens'])):
            if key not in response or type(response[key]) is not type(expected) or response[key] != expected:
                raise ValueError('Generation response mismatch: ' + key)
        if not response.get('prompt_tokens') or not response.get('generated_tokens'):
            raise ValueError('Missing generation token evidence')
        if 'tokens' in job and response['prompt_tokens'] != job['tokens']:
            raise ValueError('Prompt token evidence differs from request')
        if response.get('stop_reason') not in ('eog', 'context_limit', 'max_new_tokens'):
            raise ValueError('Missing termination evidence')
        if response.get('generated_count') != len(response['generated_tokens']):
            raise ValueError('Generation count mismatch')
        if len(response['generated_tokens']) > job['generate']['max_new_tokens']:
            raise ValueError('Generation exceeded its budget')
        if not isinstance(response.get('text'), str):
            raise ValueError('Missing generated text')
        raw_text = bytes.fromhex(response.get('text_bytes_hex', ''))
        if raw_text.decode('utf-8', errors='replace') != response['text']:
            raise ValueError('Generated text byte mismatch')
        nv = response.get('vocab_size')
        if not integer(nv, 1, 2**31-1):
            raise ValueError('Invalid vocabulary size')
        for key in ('prompt_tokens', 'generated_tokens'):
            if not isinstance(response[key], list) or not all(integer(t, 0, nv-1) for t in response[key]):
                raise ValueError('Invalid token ID in ' + key)
        context = job['generate'].get('context_tokens', 128)
        if type(response.get('context_tokens')) is not int or response['context_tokens'] != context:
            raise ValueError('Context mismatch')
        budget = min(job['generate']['max_new_tokens'], context - len(response['prompt_tokens']))
        if budget < 1 or type(response.get('effective_new_token_budget')) is not int or response['effective_new_token_budget'] != budget:
            raise ValueError('Effective generation budget mismatch')
        count = len(response['generated_tokens'])
        if not integer(response.get('generated_count'), 1, budget):
            raise ValueError('Invalid generation count')
        ended = response['stop_reason'] == 'eog'
        if response.get('stopped_on_eog') is not ended:
            raise ValueError('EOG flag mismatch')
        if ended:
            if not integer(response.get('eog_token'), 0, nv-1) or response['eog_token'] != response['generated_tokens'][-1]:
                raise ValueError('EOG token mismatch')
        elif response.get('eog_token') is not None or count != budget:
            raise ValueError('Non-EOG termination mismatch')
        if not ended:
            expected_stop = 'context_limit' if budget < job['generate']['max_new_tokens'] else 'max_new_tokens'
            if response['stop_reason'] != expected_stop:
                raise ValueError('Termination limit mismatch')
        if response.get('compact') is not job['generate'].get('compact', True):
            raise ValueError('Compact mode mismatch')
        if response.get('model_reused') is not True or response.get('temperature') != 0 or response.get('tie_break') != 'lowest_token_id':
            raise ValueError('Decoder metadata mismatch')
        first = response.get('first_step')
        if not isinstance(first, dict) or first.get('greedy_token_id') != response['generated_tokens'][0]:
            raise ValueError('First greedy token mismatch')
        for key in ('top_logits', 'requested_logits'):
            scores = first.get(key)
            if not isinstance(scores, list):
                raise ValueError('Missing first-step scores')
            for score in scores:
                if not isinstance(score, dict) or not integer(score.get('token_id'), 0, nv-1) or type(score.get('logit')) not in (int, float) or not math.isfinite(score['logit']):
                    raise ValueError('Invalid first-step score')
        top = first['top_logits']
        if len(top) != job['generate'].get('top_k', min(5, nv)) or top != sorted(top, key=lambda x: (-x['logit'], x['token_id'])):
            raise ValueError('Top-logit ordering/count mismatch')
        if top and top[0]['token_id'] != response['generated_tokens'][0]:
            raise ValueError('Greedy token differs from maximum')
        if [x['token_id'] for x in first['requested_logits']] != job['generate'].get('score_tokens', []):
            raise ValueError('Requested score IDs mismatch')


def run(args):
    identity = json.loads(args.manifest.read_text())
    jobs = json.loads(args.jobs.read_text())
    if not isinstance(jobs, list) or not jobs or len(jobs) > 128:
        raise ValueError('Require 1..128 jobs')
    ids = [j['id'] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate job IDs')
    if not 1 <= args.timeout <= 900:
        raise ValueError('Process timeout must be 1..900 seconds')
    if args.output.exists():
        raise ValueError('Use a new result path')
    args.local.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime.resolve()
    source_paths = [Path(__file__).resolve(), args.jobs.resolve(), args.lens.resolve(),
                    args.manifest.resolve()]
    if args.overlay:
        source_paths.append(args.overlay.resolve())
    before = {str(p): sha(p) for p in source_paths}
    libraries = {p.name: sha(p) for p in sorted(runtime.glob('*.so'))}
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['LD_LIBRARY_PATH'] = str(runtime)
    for key in ('FLASH_MEMORY_OVERLAY', 'FLASH_MEMORY_TRACE'):
        env.pop(key, None)
    if args.overlay:
        env['FLASH_MEMORY_OVERLAY'] = str(args.overlay.resolve())
    command = [str(args.lens.resolve()), '-m', identity['shards'][0]['path'],
               '--device', 'none', '--fit', 'off', '-ngl', '0', '-c', '128',
               '-b', '32', '-ub', '32', '-t', '8', '-tb', '8', '-fa', 'off',
               '-ctk', 'f32', '-ctv', 'f32', '--ngram-on-disk',
               '--ngram-io-threads', '4', '--ngram-cache', '64']
    record = {'schema': 'ngramma.behavior-batch/v1', 'status': 'running',
              'identity_sha256': identity['identity_sha256'],
              'overlay_sha256': sha(args.overlay) if args.overlay else None,
              'lens_sha256': sha(args.lens), 'jobs_sha256': sha(args.jobs),
              'driver_sha256': sha(Path(__file__)), 'runtime_sha256': libraries,
              'profile': {'device': 'CPU', 'threads': 8, 'context': 128,
                          'batch': 32, 'microbatch': 32, 'flash_attention': False,
                          'cache_type': 'f32', 'repack': True,
                          'fresh_state_per_job': True, 'sampling': 'unconstrained greedy',
                          'maximum_rss_gib': 80, 'minimum_available_gib': 24},
              'peak_rss_bytes': 0, 'minimum_available_bytes': None, 'jobs': []}
    start = time.monotonic()
    process = None
    selector = selectors.DefaultSelector()
    buffered = bytearray()
    try:
        with (args.local / 'stderr.log').open('wb') as stderr, \
                (args.local / 'responses.jsonl').open('w') as responses:
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=stderr,
                                       env=env, start_new_session=True)
            selector.register(process.stdout, selectors.EVENT_READ)

            def receive():
                while True:
                    if time.monotonic() - start > args.timeout:
                        raise TimeoutError('Model process exceeded fixed wall-clock budget')
                    if process.poll() is None:
                        use = child_resources(process.pid)
                        record['peak_rss_bytes'] = max(record['peak_rss_bytes'], use['peak_rss_bytes'])
                        old = record['minimum_available_bytes']
                        record['minimum_available_bytes'] = min(old, use['available_bytes']) if old else use['available_bytes']
                        if use['rss_bytes'] > 80 * (1 << 30) or use['available_bytes'] < 24 * (1 << 30):
                            raise MemoryError('Model RSS or host reserve limit exceeded')
                    if b'\n' in buffered:
                        line, _, remainder = buffered.partition(b'\n')
                        buffered[:] = remainder
                        text = line.decode('utf-8')
                        responses.write(text + '\n'); responses.flush()
                        if not text.startswith('{'):
                            continue
                        return json.loads(text)
                    if selector.select(timeout=0.25):
                        data = os.read(process.stdout.fileno(), 65536)
                        if not data:
                            raise RuntimeError('Engine exited before completing its response')
                        buffered.extend(data)
                        if len(buffered) > 32 << 20:
                            raise ValueError('Engine response exceeded 32 MiB limit')

            ready = receive()
            if ready.get('ready') is not True:
                raise RuntimeError('Engine did not become ready')
            record['ready'] = ready
            record['load_seconds'] = time.monotonic() - start
            print(json.dumps({'event': 'ready', 'seconds': record['load_seconds']}), flush=True)
            for index, item in enumerate(jobs):
                job = {k: v for k, v in item.items() if k != 'id'}
                began = time.monotonic()
                process.stdin.write((json.dumps(job) + '\n').encode())
                process.stdin.flush()
                response = receive()
                record['jobs'].append({'id': item['id'], 'request': job,
                                       'response': response, 'seconds': time.monotonic() - began})
                atomic(args.local / 'progress.json', record)
                print(json.dumps({'event': 'job', 'index': index, 'id': item['id'],
                                  'seconds': time.monotonic() - began,
                                  'ok': response.get('ok'),
                                  'text': response.get('text'),
                                  'stop_reason': response.get('stop_reason')}), flush=True)
                validate_response(job, response)
            process.stdin.close()
            process.wait(timeout=min(30, max(1, args.timeout - (time.monotonic() - start))))
            if process.returncode:
                raise RuntimeError('Engine exited with error')
        if any(sha(Path(name)) != value for name, value in before.items()):
            raise ValueError('Input, binary, identity, or driver changed during execution')
        if any(sha(runtime / name) != value for name, value in libraries.items()):
            raise ValueError('Runtime libraries changed during execution')
        record['status'] = 'complete'
        record['inputs_unchanged'] = True
    except BaseException as exc:
        record['status'] = 'failed'
        record['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        selector.close()
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        record['seconds'] = time.monotonic() - start
        atomic(args.output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('jobs', 'manifest', 'runtime', 'lens', 'local', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--timeout', type=int, default=900)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Capture CPU reference tensors in a new local directory, without service changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lens', type=Path, help='Optional locally built diagnostic lens')
    p.add_argument('--overlay', type=Path, help='Explicit experimental rows.fml; engine authenticates model shards')
    p.add_argument('--no-repack', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--capture', action='append', default=[])
    p.add_argument('--chunk-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--timeout', type=int, default=900)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = json.loads(args.manifest.read_text())
    tokens = json.loads(args.tokens.read_text())
    runtime = args.runtime.resolve()
    binary = args.lens.resolve() if args.lens else runtime/'flash-memory-lens'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['LD_LIBRARY_PATH'] = str(runtime)
    env.pop('FLASH_MEMORY_OVERLAY', None)
    env.pop('FLASH_MEMORY_TRACE', None)
    overlay_hash = None
    if args.overlay:
        overlay_path = args.overlay.resolve(strict=True)
        overlay_hash = hashlib.sha256(overlay_path.read_bytes()).hexdigest()
        env['FLASH_MEMORY_OVERLAY'] = str(overlay_path)
    cmd = [str(binary), '-m', identity['shards'][0]['path'], '--device', 'none',
           '--fit', 'off', '-ngl', '0', '-c', '128', '-b', '32', '-ub', '32',
           '-t', str(args.threads), '-tb', str(args.threads), '-fa', 'off',
           '-ctk', 'f32', '-ctv', 'f32', '--ngram-on-disk', '--ngram-io-threads', '4',
           '--ngram-cache', '64']
    if args.no_repack: cmd.append('--no-repack')
    if args.verbose: cmd.append('--verbose')
    job = {'tokens':tokens, 'chunk_size':args.chunk_size, 'output_dir':str(args.output.resolve()), 'capture':args.capture}
    start = time.monotonic()
    with (args.output/'stderr.log').open('w') as err:
        result = subprocess.run(cmd, input=json.dumps(job)+'\n', text=True,
                                stdout=subprocess.PIPE, stderr=err, env=env, timeout=args.timeout)
    (args.output/'responses.jsonl').write_text(result.stdout)
    records = [json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')]
    if result.returncode or not records or records[-1].get('ok') is not True:
        raise RuntimeError('Reference capture failed; inspect local stderr and response logs')
    if args.overlay and hashlib.sha256(overlay_path.read_bytes()).hexdigest() != overlay_hash:
        raise RuntimeError('Overlay changed during engine execution')
    summary = {'tokens':tokens, 'capture_prefixes':args.capture, 'chunk_size':args.chunk_size,
               'threads':args.threads, 'device':'CPU', 'cache_type':'f32', 'repack':not args.no_repack,
               'flash_attention':False, 'context':128, 'batch':32, 'microbatch':32,
               'rope_overrides':False,
               'overlay_sha256':overlay_hash,
               'capture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'seconds':time.monotonic()-start,
               'identity_sha256':identity['identity_sha256'],
               'lens_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
               'tensors':len(records[-1]['tensors']),
               'logits_sha256':hashlib.sha256((args.output/'logits.f32').read_bytes()).hexdigest(),
               'metadata_sha256':hashlib.sha256((args.output/'tensors.json').read_bytes()).hexdigest()}
    (args.output/'capture-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()

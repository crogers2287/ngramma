#!/usr/bin/env python3
"""Compile a local activation encoding GGML probe; never build/start an engine.

Requires matching reconstructed engine headers and existing CPU/base libraries.
This is an operator diagnostic build, not a qualified portable full engine build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-source', type=Path, default=os.environ.get('NGRAMMA_ENGINE_SOURCE'))
    parser.add_argument('--runtime', type=Path, default=os.environ.get('NGRAMMA_RUNTIME'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'), help='Compiler executable (one path, no shell flags)')
    args = parser.parse_args()
    if args.engine_source is None or args.runtime is None:
        parser.error('Supply --engine-source and --runtime, or NGRAMMA_ENGINE_SOURCE and NGRAMMA_RUNTIME')
    engine = args.engine_source.expanduser().resolve()
    runtime = args.runtime.expanduser().resolve()
    include = engine / 'ggml' / 'include'
    source = Path(__file__).resolve().parents[1] / 'src/ngramma_runtime/native/activation_probe.cpp'
    for path in (source, include / 'ggml.h', include / 'ggml-cpu.h', runtime / 'libggml-base.so', runtime / 'libggml-cpu.so'):
        if not path.is_file():
            parser.error(f'Required matching local dependency is missing: {path}')
    compiler = shutil.which(args.cxx)
    if compiler is None:
        parser.error(f'C++ compiler not found: {args.cxx}')
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the destination and replace only on success.
    with tempfile.TemporaryDirectory(prefix='.ngramma-activation-', dir=output.parent) as temporary:
        pending = Path(temporary) / output.name
        command = [compiler, '-std=c++17', '-O2', '-fPIC', '-shared', '-pthread',
                   '-I', str(include), str(source), '-L', str(runtime),
                   '-lggml-cpu', '-lggml-base', '-Wl,-z,defs',
                   '-Xlinker', '-rpath', '-Xlinker', str(runtime), '-o', str(pending)]
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        def sha256(path):
            with path.open('rb') as stream:
                result = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    result.update(chunk)
                return result.hexdigest()

        # Stable labels preserve compiler arguments without exposing host paths.
        replacements = {str(include): '<ENGINE_SOURCE>/ggml/include',
                        str(source): '<REPOSITORY>/src/ngramma_runtime/native/activation_probe.cpp',
                        str(runtime): '<RUNTIME>', str(pending): '<OUTPUT>',
                        compiler: Path(compiler).name}
        version = subprocess.run([compiler, '--version'], check=True, capture_output=True, text=True).stdout
        record = {
            'schema_version': 1,
            'library_sha256': sha256(pending),
            'sources': {'activation_probe.cpp': sha256(source),
                        'build_activation_probe.py': sha256(Path(__file__).resolve())},
            'headers': {path.name: sha256(path) for path in sorted(include.glob('*.h'))},
            'linked_libraries': {name: sha256(runtime / name) for name in ('libggml-cpu.so', 'libggml-base.so')},
            'compiler': {'executable': Path(compiler).name, 'version': version.strip()},
            'command': [replacements.get(value, value) for value in command],
            'qualification': 'Compiled ordinary Q8_0 activation encoding replay; no model-parity or gradient qualification',
        }
        # Record is hashed against the finished library. Consumers must compare
        # library_sha256 before trusting it; the two replacements are not atomic
        # as a pair if a process is interrupted between them.
        pending_record = Path(temporary) / 'build.json'
        pending_record.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(pending, output)
        os.replace(pending_record, Path(str(output) + '.build.json'))
    print(f'Built activation encoding probe: {output}')
    print('No model loaded; numerical model parity and gradients remain unqualified.')


if __name__ == '__main__':
    main()

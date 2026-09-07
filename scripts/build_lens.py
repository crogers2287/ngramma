#!/usr/bin/env python3
"""Build the diagnostic lens against matching prebuilt libraries, without weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-source', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cxx', default='c++')
    args = p.parse_args()
    source = Path(__file__).resolve().parents[1]/'src/ngramma_runtime/native/lens.cpp'
    engine = args.engine_source.resolve()
    runtime = args.runtime.resolve()
    headers = ['common', 'include', 'ggml/include', 'vendor/nlohmann']
    for name in ('libllama.so', 'libllama-common.so', 'libggml.so', 'libggml-base.so'):
        if not (runtime/name).is_file():
            p.error(f'Matching runtime library missing: {name}')
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.lens-build-') as directory:
        pending = Path(directory)/'lens'
        command = [args.cxx, '-std=c++17', '-O2', str(source),
            *['-I'+str(engine/name) for name in headers], '-L'+str(runtime),
            '-lllama-common', '-lllama', '-lggml', '-lggml-base',
            '-Xlinker', '-rpath', '-Xlinker', str(runtime), '-o', str(pending)]
        subprocess.run(command, check=True)
        os.replace(pending, output)
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    record = {'binary_sha256': digest(output), 'source_sha256': digest(source),
              'build_script_sha256': digest(Path(__file__)),
              'compiler': subprocess.check_output([args.cxx, '--version'], text=True).splitlines()[0],
              'runtime_sha256': {name:digest(runtime/name) for name in ('libllama.so','libllama-common.so','libggml.so','libggml-base.so')}}
    output.with_suffix(output.suffix+'.build.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Build optional full-shard parallel verification for the pinned Linux engine."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-source', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'src/ngramma_runtime/native/parallel_verify.cpp'
    header = root / 'reference/engine/memory-overlay.h'
    symbol = '_ZNK12flash_memory7overlay12verify_modelERKSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS7_EE'
    if symbol + '@plt' not in subprocess.check_output(['objdump', '-d', str(a.runtime / 'libllama.so')], text=True):
        raise ValueError('Runtime does not expose the qualified verification call site')
    if a.output.exists():
        raise ValueError('Use a new library path')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', str(source),
                    '-I' + str(header.parent), '-I' + str(a.engine_source / 'vendor/nlohmann'),
                    '-lcrypto', '-pthread', '-o', str(a.output)], check=True)
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    record = {'binary_sha256': sha(a.output), 'source_sha256': sha(source),
              'archived_overlay_header_sha256': sha(header),
              'engine_json_header_sha256': sha(a.engine_source / 'vendor/nlohmann/json.hpp'),
              'runtime_llama_sha256': sha(a.runtime / 'libllama.so'),
              'interposed_symbol': symbol, 'maximum_hash_workers': 3,
              'full_file_sha256': True, 'authentication_cache': False,
              'compiler': subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]}
    a.output.with_suffix(a.output.suffix + '.build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()

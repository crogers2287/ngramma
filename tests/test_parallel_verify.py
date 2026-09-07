"""Full-file authentication checks on synthetic files, never model weights."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def verifier(tmp_path_factory):
    source = os.environ.get('NGRAMMA_ENGINE_SOURCE')
    if not source or not shutil.which('c++'):
        pytest.skip('Configure pinned engine headers and a C++ compiler')
    directory = tmp_path_factory.mktemp('parallel-verify')
    cpp = directory/'main.cpp'
    cpp.write_text('#include "' + str(ROOT/'src/ngramma_runtime/native/parallel_verify.cpp') + '"\n' + r'''
int main() {
    flash_memory::json input; std::cin >> input;
    try {
        flash_memory::overlay value; value.header=input.at("header");
        ngramma::verify_parallel(&value,input.at("paths").get<std::vector<std::string>>());
        std::cout << "{\"ok\":true}";
    } catch(const std::exception & e) {
        std::cout << flash_memory::json({{"ok",false},{"error",e.what()}}).dump();
    }
}
''')
    executable = directory/'check'
    subprocess.run(['c++', '-std=c++17', '-O2', str(cpp),
                    '-I'+str(ROOT/'reference/engine'), '-I'+str(Path(source)/'vendor/nlohmann'),
                    '-lcrypto', '-pthread', '-o', str(executable)], check=True, capture_output=True)

    def run(request):
        out = subprocess.run([str(executable)], input=json.dumps(request), text=True,
                             capture_output=True, check=True)
        markers = [json.loads(line.split(' ', 1)[1]) for line in out.stderr.splitlines()
                   if line.startswith('NGRAMMA_MODEL_VERIFIED ')]
        return json.loads(out.stdout), markers
    return run


def files(tmp_path, count=3):
    shards = []
    for i in range(count):
        path = tmp_path/f'shard-{i}.bin'
        data = (f'complete synthetic shard {i}:'.encode()+bytes(range(256)))*1024
        path.write_bytes(data)
        shards.append({'path': str(path), 'bytes': len(data),
                       'sha256': hashlib.sha256(data).hexdigest()})
    return {'header': {'model_identity': {'shards': shards}},
            'paths': [s['path'] for s in shards]}


def test_complete_hashes_and_bounded_groups(verifier, tmp_path):
    request = files(tmp_path, 7)
    result, markers = verifier(request)
    assert result['ok']
    assert len(markers) == 1
    assert markers[0]['shards'] == 7 and markers[0]['workers'] == 3
    assert markers[0]['bytes'] == sum(s['bytes'] for s in request['header']['model_identity']['shards'])
    assert markers[0]['full_file_sha256'] is True and markers[0]['cache_reused'] is False


@pytest.mark.parametrize('index', [0, 1, 2])
def test_same_size_corruption_in_every_shard_rejected(verifier, tmp_path, index):
    request = files(tmp_path)
    path = Path(request['paths'][index])
    with path.open('r+b') as stream:
        stream.seek(path.stat().st_size-1)
        stream.write(b'!')
    result, markers = verifier(request)
    assert not result['ok'] and not markers


@pytest.mark.parametrize('change', ['count', 'size', 'digest', 'path', 'missing'])
def test_identity_failures_rejected(verifier, tmp_path, change):
    request = files(tmp_path)
    shard = request['header']['model_identity']['shards'][1]
    if change == 'count': request['paths'].pop()
    elif change == 'size': shard['bytes'] += 1
    elif change == 'digest': shard['sha256'] = '0'*64
    elif change == 'path':
        other = tmp_path/'identical-copy.bin'
        other.write_bytes(Path(shard['path']).read_bytes())
        shard['path'] = str(other)
    elif change == 'missing': Path(shard['path']).unlink()
    result, markers = verifier(request)
    assert not result['ok'] and not markers


def test_repeated_path_is_rehashed_after_mutation(verifier, tmp_path):
    request = files(tmp_path)
    assert verifier(request)[0]['ok']
    path = Path(request['paths'][2])
    path.write_bytes(b'X'*path.stat().st_size)
    assert not verifier(request)[0]['ok']


def test_canonical_alias_matches_original_loader_rule(verifier, tmp_path):
    request = files(tmp_path)
    alias = tmp_path/'alias.bin'
    alias.symlink_to(request['paths'][0])
    request['paths'][0] = str(alias)
    assert verifier(request)[0]['ok']

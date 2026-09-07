"""Dependency/path checks only: no checkpoint, native library, or GPU is loaded."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from ngramma_runtime.environment import RuntimeConfig, RuntimeDependencyError


def source_tree(tmp_path):
    engine = tmp_path / 'engine'
    (engine / 'gguf-py/gguf').mkdir(parents=True)
    (engine / 'gguf-py/gguf/__init__.py').touch()
    vendor = tmp_path / 'vendor'
    (vendor / 'engraft').mkdir(parents=True)
    (vendor / 'engraft/__init__.py').touch()
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / 'libflash-memory-dequant.so').touch()
    return engine, vendor, runtime


def test_explicit_paths_and_env(tmp_path):
    engine, vendor, runtime = source_tree(tmp_path)
    cfg = RuntimeConfig.from_env({'NGRAMMA_ENGINE_SOURCE': str(engine), 'NGRAMMA_ENGRAFT_SOURCE': str(vendor / 'engraft'), 'NGRAMMA_RUNTIME': str(runtime)})
    assert cfg.validate() is cfg
    assert cfg.gguf_root == engine / 'gguf-py'
    assert cfg.engraft_root == vendor
    assert cfg.bridge_path == runtime / 'libflash-memory-dequant.so'


def test_direct_bridge_override(tmp_path):
    engine, vendor, runtime = source_tree(tmp_path)
    bridge = tmp_path / 'custom.so'
    bridge.touch()
    cfg = RuntimeConfig(engine, vendor, runtime, bridge)
    assert cfg.validate().bridge_path == bridge


def test_missing_engine_is_actionable():
    with pytest.raises(RuntimeDependencyError, match='NGRAMMA_ENGINE_SOURCE'):
        RuntimeConfig.from_env({}).validate()


def test_bridge_is_optional_for_table(tmp_path):
    engine, vendor, _ = source_tree(tmp_path)
    cfg = RuntimeConfig(engine, vendor)
    assert cfg.validate(require_bridge=False) is cfg
    with pytest.raises(RuntimeDependencyError, match='NGRAMMA_RUNTIME'):
        cfg.validate()


def test_missing_source_and_library_fail(tmp_path):
    engine, vendor, runtime = source_tree(tmp_path)
    with pytest.raises(RuntimeDependencyError, match='ENGRAFT package missing'):
        RuntimeConfig(engine, tmp_path / 'absent', runtime).validate()
    with pytest.raises(RuntimeDependencyError, match='bridge missing'):
        RuntimeConfig(engine, vendor, runtime, tmp_path / 'absent.so').validate()


def test_safe_imports_without_numerical_dependencies():
    source = Path(__file__).resolve().parents[1] / 'src'
    code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
before = list(sys.path)
import ngramma_runtime
from ngramma_runtime import environment, artifacts, table, weights, sequence, resources, activation_reference
assert sys.path == before
assert not any(x in sys.modules for x in ('numpy', 'torch', 'gguf', 'engraft'))
'''
    result = subprocess.run([sys.executable, '-S', '-c', code, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_wrong_preimported_gguf_fails(tmp_path, monkeypatch):
    from ngramma_runtime.environment import worker_imports, configure
    import ngramma_runtime.environment as env
    import types
    engine, vendor, runtime = source_tree(tmp_path)
    monkeypatch.setattr(env, '_config', None)
    monkeypatch.setattr(env, '_prepared', None)
    configure(RuntimeConfig(engine, vendor, runtime))
    monkeypatch.setitem(sys.modules, 'gguf', types.SimpleNamespace(__file__='/unrelated/gguf/__init__.py'))
    with pytest.raises(RuntimeDependencyError, match='fresh process'):
        worker_imports()


def test_reconfiguration_after_import_fails(tmp_path, monkeypatch):
    import ngramma_runtime.environment as env
    first = RuntimeConfig(tmp_path / 'engine1')
    monkeypatch.setattr(env, '_prepared', first)
    with pytest.raises(RuntimeDependencyError, match='fresh process'):
        env.configure(RuntimeConfig(tmp_path / 'engine2'))


def test_resource_limits_reject_nonfinite_or_negative():
    from ngramma_runtime.resources import check_budget
    for limit in (float('inf'), float('nan'), 0, -1):
        with pytest.raises(ValueError):
            check_budget(limit, 0)

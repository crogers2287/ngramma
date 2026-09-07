"""Explicit, process-local paths for the pinned research dependencies."""
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys


class RuntimeDependencyError(RuntimeError):
    """A configured local dependency is unavailable or conflicts with imports."""


@dataclass(frozen=True)
class RuntimeConfig:
    engine_source: Path | None = None
    engraft_source: Path | None = None
    runtime: Path | None = None
    dequant_library: Path | None = None

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        return cls(**{name: Path(env[key]).expanduser().resolve() if env.get(key) else None
                      for name, key in (("engine_source", "NGRAMMA_ENGINE_SOURCE"),
                                        ("engraft_source", "NGRAMMA_ENGRAFT_SOURCE"),
                                        ("runtime", "NGRAMMA_RUNTIME"),
                                        ("dequant_library", "NGRAMMA_DEQUANT_LIBRARY"))})

    @property
    def gguf_root(self):
        if self.engine_source is None:
            raise RuntimeDependencyError("Set NGRAMMA_ENGINE_SOURCE to the pinned engine checkout containing gguf-py.")
        return Path(self.engine_source).expanduser().resolve() / "gguf-py"

    @property
    def engraft_root(self):
        if self.engraft_source is None:
            return None
        source = Path(self.engraft_source).expanduser().resolve()
        return source.parent if source.name == "engraft" and (source / "__init__.py").is_file() else source

    @property
    def bridge_path(self):
        if self.dequant_library is not None:
            return Path(self.dequant_library).expanduser().resolve()
        if self.runtime is not None:
            return Path(self.runtime).expanduser().resolve() / "libflash-memory-dequant.so"
        raise RuntimeDependencyError("Set NGRAMMA_RUNTIME or NGRAMMA_DEQUANT_LIBRARY to the matching local dequantization bridge.")

    def validate(self, *, require_bridge=True):
        if not (self.gguf_root / "gguf" / "__init__.py").is_file():
            raise RuntimeDependencyError(f"Pinned GGUF reader missing: {self.gguf_root / 'gguf'}. Check NGRAMMA_ENGINE_SOURCE.")
        if self.engraft_root is not None and not (self.engraft_root / "engraft" / "__init__.py").is_file():
            raise RuntimeDependencyError(f"ENGRAFT package missing under {self.engraft_root}. Check NGRAMMA_ENGRAFT_SOURCE.")
        if require_bridge and not self.bridge_path.is_file():
            raise RuntimeDependencyError(f"Dequantization bridge missing: {self.bridge_path}. Build the matching engine bridge separately.")
        return self


_config = None
_prepared = None


def configure(value):
    """Configure before importing model classes; changing loaded sources is forbidden."""
    global _config
    if not isinstance(value, RuntimeConfig):
        raise TypeError("configure expects RuntimeConfig")
    if _prepared is not None and value != _prepared:
        raise RuntimeDependencyError("Runtime dependencies are already loaded; configure new paths in a fresh process.")
    _config = value
    return value


def config():
    value = _config if _config is not None else RuntimeConfig.from_env()
    if _prepared is not None and value != _prepared:
        raise RuntimeDependencyError("Runtime configuration changed after imports; use a fresh process.")
    return value


def _check_origin(name, root):
    module = sys.modules.get(name)
    if module is not None:
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(root.resolve()):
            raise RuntimeDependencyError(f"{name} already imported from another source; use a fresh process with pinned paths configured first.")


def worker_imports():
    """Load only explicitly configured GGUF and optionally external ENGRAFT."""
    global _prepared
    cfg = config().validate(require_bridge=False)
    roots = [("gguf", cfg.gguf_root)]
    if cfg.engraft_root is not None:
        roots.append(("engraft", cfg.engraft_root))
    for name, root in roots:
        _check_origin(name, root)
    for _, root in reversed(roots):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    try:
        for name in ("numpy", "gguf", "engraft"):
            importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeDependencyError("Missing research dependency. Install the research extra and supply pinned GGUF/ENGRAFT sources (NGRAMMA_ENGINE_SOURCE, NGRAMMA_ENGRAFT_SOURCE). " + str(exc)) from exc
    _prepared = cfg
    return cfg

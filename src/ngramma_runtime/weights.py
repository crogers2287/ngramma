"""Lazy access to optional local research dependencies."""
from importlib import import_module
from .environment import RuntimeDependencyError

__all__ = ['EngineWeights']

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    try:
        value = getattr(import_module("._weights", __package__), name)
    except ImportError as exc:
        raise RuntimeDependencyError("Optional numerical dependency unavailable; install ngramma-runtime[research] and configure the pinned local engine/ENGRAFT sources: " + str(exc)) from exc
    globals()[name] = value
    return value

"""Local, forward-only research harness; numerical parity remains unqualified."""
from .environment import RuntimeConfig, RuntimeDependencyError, configure

__all__ = ["RuntimeConfig", "RuntimeDependencyError", "configure"]

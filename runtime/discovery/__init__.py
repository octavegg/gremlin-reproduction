"""Public exports, loaded lazily so process ownership needs no JAX runtime."""

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    'EnvironmentBundle': ('gremlin_rl.environment.sampler', 'EnvironmentBundle'),
    'sample_environment': ('gremlin_rl.environment.sampler', 'sample_environment'),
}

__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value

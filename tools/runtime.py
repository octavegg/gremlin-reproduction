"""Load one frozen runtime under its original Python package name."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_runtime(panel):
    if panel not in ('human', 'generated', 'discovery'):
        raise ValueError('unknown runtime')
    package = ROOT/'runtime'/panel
    if panel == 'generated':  # Historical drivers also use this physical path.
        package = package/'src/gremlin_rl'
    entry = package/'__init__.py'
    existing = sys.modules.get('gremlin_rl')
    if existing is not None:
        if Path(existing.__file__).resolve() != entry.resolve():
            raise RuntimeError('use a fresh process for a different frozen runtime')
        return package
    spec = importlib.util.spec_from_file_location(
        'gremlin_rl', entry, submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['gremlin_rl'] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules['gremlin_rl']
        raise
    return package

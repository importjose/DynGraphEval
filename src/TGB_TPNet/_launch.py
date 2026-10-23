"""
Entry-point wrapper used by DynGraphEval train scripts instead of calling
train_link_prediction.py directly.

Problem: uv/pip editable installs (-e .) register a custom MetaPathFinder in
sys.meta_path. MetaPathFinders run before sys.path and PYTHONPATH are
consulted, so 'import models' resolves to DynGraphEval's models/ package
instead of TGB_TPNet's models/ directory.

Fix: replace sys.meta_path with only the three standard Python finders
(BuiltinImporter, FrozenImporter, PathFinder), which removes any editable
install hook regardless of its type name. Also clear sys.modules cache for
'models' in case it was already imported. Then run train_link_prediction.py
via runpy, which preserves __file__, __name__, and sys.argv correctly.
"""
import sys
import os
import runpy
from importlib.machinery import PathFinder
from importlib._bootstrap import BuiltinImporter, FrozenImporter

# Replace sys.meta_path with only the standard finders.
# This removes any editable install hooks regardless of their type name.
sys.meta_path[:] = [BuiltinImporter, FrozenImporter, PathFinder]

# Ensure TGB_TPNet's directory is first in sys.path
_here = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _here:
    sys.path.insert(0, _here)

# Clear any cached import of models/utils that may have resolved to DynGraphEval
for key in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
    del sys.modules[key]

runpy.run_path(
    os.path.join(_here, "train_link_prediction.py"),
    run_name="__main__",
)

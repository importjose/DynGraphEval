"""
Entry-point wrapper used by DynGraphEval train scripts instead of calling
train_link_prediction.py directly.

Problem: uv/pip editable installs (-e .) register a MetaPathFinder in
sys.meta_path. MetaPathFinders run before sys.path and PYTHONPATH are
consulted, so 'import models' resolves to DynGraphEval's models/ package
instead of TGB_TPNet's models/ directory — even when PYTHONPATH points here.

Fix: strip Editable MetaPathFinders, insert this directory first in
sys.path, then run train_link_prediction.py via runpy (which preserves
__file__, __name__, and sys.argv correctly).
"""
import sys
import os
import runpy

# Remove editable install MetaPathFinders
sys.meta_path = [f for f in sys.meta_path if "Editable" not in type(f).__name__]

# Ensure TGB_TPNet's directory is first in sys.path
_here = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _here:
    sys.path.insert(0, _here)

runpy.run_path(
    os.path.join(_here, "train_link_prediction.py"),
    run_name="__main__",
)

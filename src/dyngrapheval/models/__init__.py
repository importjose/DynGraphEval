"""
Model registry for DynGraphEval.

Import any model directly:
    from dyngrapheval.models import TGN
    from dyngrapheval.models import FederatedTGN
    from dyngrapheval.models import FedLink

Or import all at once:
    from dyngrapheval.models import TGN, FederatedTGN, FedLink
"""

from .tgn import TGN
from .fl_tgn import FederatedTGN
from .fedlink import FedLink

__all__ = ["TGN", "FederatedTGN", "FedLink"]

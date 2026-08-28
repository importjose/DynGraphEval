"""
Abstract base class for all models in DynGraphEval.

Every model (TGN, FederatedTGN, FedLink) must implement exactly three methods:

    1. load_checkpoint(path)  — load saved weights from disk
    2. warmup(train_data, val_data)  — replay edges to build internal state
    3. evaluate(eval_data, neg_sampler, split_mode, evaluator) -> float  — return MRR

The evaluator calls these in order:
    model.load_checkpoint(path)
    model.warmup(train_data, val_data)
    mrr = model.evaluate(test_data, neg_sampler, 'test', evaluator)

To add a new model: subclass BaseModel and implement the three abstract methods.
"""

from abc import ABC, abstractmethod
from torch_geometric.data import TemporalData


class BaseModel(ABC):
    """
    Abstract base class for temporal graph models in DynGraphEval.

    Defines the minimal interface the Evaluator needs to run any model.
    Concrete subclasses live in dyngrapheval/models/.
    """

    @abstractmethod
    def load_checkpoint(self, path) -> None:
        """
        Load saved model weights from disk.

        Parameters
        ----------
        path : str or list[str]
            For single models (TGN): a single file path string.
            For federated models (FederatedTGN, FedLink): a list of paths,
            one per client.
        """
        ...

    @abstractmethod
    def warmup(self, train_data: TemporalData, val_data: TemporalData) -> None:
        """
        Replay train+val edges to build internal state before test evaluation.

        For TGN-based models this populates the LastNeighborLoader cache so
        that test-time GNN calls have access to a rich temporal neighborhood.

        For stateless models (FedLink) this is a no-op — the method must still
        exist so the Evaluator can call it uniformly.

        Must be called once after load_checkpoint() and again before each
        additional evaluation pass (the Evaluator handles this automatically).

        Parameters
        ----------
        train_data : TemporalData
        val_data   : TemporalData
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        eval_data: TemporalData,
        neg_sampler,
        split_mode: str,
        evaluator,
    ) -> float:
        """
        Run evaluation and return mean MRR over all test edges.

        Parameters
        ----------
        eval_data   : TemporalData  — edges to evaluate (test set or partition)
        neg_sampler : object with .query_batch(src, dst, t, split_mode) -> list
                      Can be TGB's NegativeEdgeSampler or our NegativeSampler.
        split_mode  : 'val' or 'test'
        evaluator   : TGB evaluator with .eval(input_dict) -> {'mrr': float}

        Returns
        -------
        float : mean MRR across all positive edges
        """
        ...

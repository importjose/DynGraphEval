"""
Recency negative sampling for temporal graph evaluation.

Why recency negatives?
----------------------
TGB's standard 'hist_rnd' negatives are sampled from a source node's *entire*
training history — some of those links may be months or years old. A static
snapshot model that memorizes the graph structure can rank these just as well
as a continuous model, because "this link has existed historically" is enough
information.

Recency negatives are harder: for each positive edge (u, v, t), we sample
destinations that u visited just *before* t — in the window [t - W, t).
To correctly rank v above these candidates, the model must understand that
u has *moved on* from those recently-visited pages, which requires genuine
temporal reasoning. Static models cannot make this distinction.

Classes
-------
RecencyNegativeGenerator
    Generates and saves recency negatives to disk as .pkl files.
    Run once per dataset split; results are reloaded on subsequent runs.

NegativeSampler
    Loads pre-generated negatives and serves them batch-by-batch.
    Drop-in compatible with TGB's NegativeEdgeSampler interface.
"""

import os
import numpy as np
from tqdm import tqdm
from torch_geometric.data import TemporalData
from tgb.utils.utils import save_pkl, load_pkl


class RecencyNegativeGenerator:
    """
    Generate recency negatives for val/test evaluation and save to disk.

    For each positive edge (u, v, t):
        - Collect destinations that u visited in [t - W, t) from training history
        - If fewer than num_neg candidates exist, fill remainder with random dests
        - Exclude v and any other positive destinations at the same (t, u) timestamp

    The window W can be:
        - A fixed float (same window for all sources)
        - None (adaptive): per-source median inter-event time.
          Sparse users get a wider window; dense users get a tighter one.
          This keeps the difficulty calibrated relative to each user's activity rate.

    Output is saved as a .pkl dict: {(src, dst, t): neg_dst_array}.
    If the file already exists it is reused without regenerating.

    Parameters
    ----------
    dataset_name  : str   used in the output filename
    first_dst_id  : int   smallest valid destination node ID
    last_dst_id   : int   largest valid destination node ID
    num_neg       : int   number of negatives per positive edge (default 100)
    seed          : int   random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        dataset_name: str,
        first_dst_id: int,
        last_dst_id: int,
        num_neg: int = 100,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.first_dst_id = first_dst_id
        self.last_dst_id  = last_dst_id
        self.num_neg      = num_neg
        np.random.seed(seed)
        # All valid destination node IDs (used for random fallback sampling)
        self._all_dst = np.arange(first_dst_id, last_dst_id + 1)

    def generate(
        self,
        historical_data: TemporalData,
        eval_data: TemporalData,
        split_mode: str,
        save_dir: str,
        window: float = None,
    ) -> str:
        """
        Generate recency negatives for eval_data and save to disk.

        Parameters
        ----------
        historical_data : TemporalData  — training edges (builds source history)
        eval_data       : TemporalData  — val or test edges to generate negatives for
        split_mode      : 'val' or 'test'
        save_dir        : str  — directory to save the .pkl file
        window          : float or None  — recency window in timestamp units.
                          None = adaptive per-source median inter-event time.

        Returns
        -------
        str : path to the saved .pkl file
        """
        assert split_mode in ("val", "test"), "split_mode must be 'val' or 'test'"
        os.makedirs(save_dir, exist_ok=True)

        # Build a descriptive filename so different configs don't collide
        window_tag = "adaptive" if window is None else str(int(window))
        filename = os.path.join(
            save_dir,
            f"{self.dataset_name}_{split_mode}_recency_{window_tag}_ns.pkl",
        )

        if os.path.exists(filename):
            print(f"[RecencyNegGen] Reusing cached negatives: {filename}")
            return filename

        print(
            f"[RecencyNegGen] Generating recency negatives "
            f"({split_mode}, window={window_tag})..."
        )

        # ── Step 1: Build per-source sorted history from training data ────────
        # src_history[u] = list of (t, dst) tuples, sorted by t ascending
        hist_src = historical_data.src.cpu().numpy()
        hist_dst = historical_data.dst.cpu().numpy()
        hist_t   = historical_data.t.cpu().numpy()

        src_history: dict[int, list] = {}
        for s, d, t in zip(hist_src, hist_dst, hist_t):
            src_history.setdefault(int(s), []).append((float(t), int(d)))
        for s in src_history:
            src_history[s].sort(key=lambda x: x[0])  # sort by timestamp

        # ── Step 2: Compute adaptive window per source (if window=None) ───────
        src_adaptive_window = {}
        if window is None:
            for s, events in src_history.items():
                if len(events) < 2:
                    # Only one interaction: use infinite window (take all history)
                    src_adaptive_window[s] = float("inf")
                else:
                    ts = np.array([e[0] for e in events])
                    # Median gap between consecutive events for this source
                    src_adaptive_window[s] = float(np.median(np.diff(ts)))

        # ── Step 3: Build conflict dict (avoid sampling other true positives) ─
        # At time t, source u may have multiple positive destinations.
        # We exclude all of them from the negative pool.
        eval_src = eval_data.src.cpu().numpy()
        eval_dst = eval_data.dst.cpu().numpy()
        eval_t   = eval_data.t.cpu().numpy()

        conflict: dict[tuple, set] = {}
        for s, d, t in zip(eval_src, eval_dst, eval_t):
            key = (int(t), int(s))
            conflict.setdefault(key, set()).add(int(d))

        # ── Step 4: For each eval edge, sample recency negatives ──────────────
        evaluation_set = {}

        for pos_s, pos_d, pos_t in tqdm(
            zip(eval_src, eval_dst, eval_t),
            total=len(eval_src),
            desc="Generating recency negatives",
        ):
            pos_s, pos_d, pos_t = int(pos_s), int(pos_d), float(pos_t)
            # Destinations to exclude (the true positives at this timestamp)
            forbidden = conflict.get((int(pos_t), pos_s), set())

            # Determine window for this source
            w = src_adaptive_window.get(pos_s, float("inf")) if window is None else window

            # Walk backwards through this source's history to find recency neighbors
            recency_dsts = set()
            if pos_s in src_history:
                for t_h, d_h in reversed(src_history[pos_s]):
                    if t_h >= pos_t:
                        continue        # skip edges at or after pos_t
                    if t_h < pos_t - w:
                        break           # outside the recency window, stop early
                    if d_h not in forbidden:
                        recency_dsts.add(d_h)

            recency_arr = np.array(list(recency_dsts))

            # Sample up to num_neg from the recency pool
            if len(recency_arr) >= self.num_neg:
                neg_dst_arr = np.random.choice(recency_arr, self.num_neg, replace=False)
            else:
                # Fill the remainder with random destinations (same fallback as TGB)
                n_random = self.num_neg - len(recency_arr)
                invalid  = forbidden | recency_dsts
                rnd_pool = np.setdiff1d(self._all_dst, np.array(list(invalid)))
                neg_rnd  = np.random.choice(
                    rnd_pool,
                    min(n_random, len(rnd_pool)),
                    replace=False,
                )
                neg_dst_arr = np.concatenate([recency_arr, neg_rnd])

            evaluation_set[(pos_s, pos_d, int(pos_t))] = neg_dst_arr

        save_pkl(evaluation_set, filename)
        print(f"[RecencyNegGen] Saved to {filename}")
        return filename


class NegativeSampler:
    """
    Load pre-generated negatives from disk and serve them batch-by-batch.

    Interface is compatible with TGB's NegativeEdgeSampler:
        sampler.load_eval_set(path, split_mode='test')
        neg_list = sampler.query_batch(pos_src, pos_dst, pos_t, split_mode='test')

    This makes it a drop-in replacement in any evaluation loop that
    already handles TGB negatives.

    Parameters
    ----------
    strategy : str  — label used in log messages (e.g., 'recency')
    """

    def __init__(self, strategy: str = "recency"):
        self.strategy = strategy
        # eval_set[split_mode] = dict: (src, dst, t) -> np.array of neg dsts
        self.eval_set: dict = {}

    def load_eval_set(self, path: str, split_mode: str = "test") -> None:
        """
        Load a .pkl file produced by RecencyNegativeGenerator.generate().

        Parameters
        ----------
        path       : str   — path to the .pkl file
        split_mode : str   — 'val' or 'test'
        """
        assert split_mode in ("val", "test")
        self.eval_set[split_mode] = load_pkl(path)
        print(
            f"[NegativeSampler] Loaded {split_mode} negatives "
            f"(strategy={self.strategy}) from {path}"
        )

    def query_batch(
        self,
        pos_src,
        pos_dst,
        pos_timestamp,
        split_mode: str = "test",
    ) -> list:
        """
        Return negative destinations for a batch of positive edges.

        Parameters
        ----------
        pos_src       : Tensor or array of source node IDs
        pos_dst       : Tensor or array of destination node IDs
        pos_timestamp : Tensor or array of edge timestamps
        split_mode    : 'val' or 'test'

        Returns
        -------
        list[list[int]] : one list of negative dest IDs per positive edge
        """
        import torch

        assert split_mode in ("val", "test")
        if split_mode not in self.eval_set:
            raise ValueError(
                f"No evaluation set loaded for split '{split_mode}'. "
                "Call load_eval_set() first."
            )

        # Convert tensors to numpy for dict lookup
        if isinstance(pos_src, torch.Tensor):
            pos_src = pos_src.detach().cpu().numpy()
        if isinstance(pos_dst, torch.Tensor):
            pos_dst = pos_dst.detach().cpu().numpy()
        if isinstance(pos_timestamp, torch.Tensor):
            pos_timestamp = pos_timestamp.detach().cpu().numpy()

        neg_samples = []
        for s, d, t in zip(pos_src, pos_dst, pos_timestamp):
            key = (int(s), int(d), int(t))
            if key not in self.eval_set[split_mode]:
                raise KeyError(
                    f"Edge {key} not found in the {split_mode} evaluation set. "
                    "Make sure negatives were generated for this exact data split."
                )
            neg_samples.append([int(x) for x in self.eval_set[split_mode][key]])

        return neg_samples

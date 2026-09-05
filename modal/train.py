"""
Modal training app for TGN on tgbl-wiki.

Run from the DynGraphEval root directory:
    modal run modal/train.py
    modal run modal/train.py --dataset tgbl-wiki --epochs 50 --patience 5

Persistence:
    Datasets and checkpoints are stored in a Modal Volume ('dyngrapheval-data').
    If training is interrupted, resume checkpoints (saved after each epoch)
    let you pick back up from the last completed epoch automatically.

Volume layout:
    /data/datasets/      TGB datasets (downloaded once, reused across runs)
    /data/checkpoints/   best.pkl + resume_epochN.pkl per run
"""

import os
import sys
import subprocess
import time

import modal

# ── Image: all Python dependencies ───────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torch-geometric==2.6.1",
        "py-tgb>=2.2",
        "numpy>=2.0",
        "pandas>=2.0",
        "tqdm>=4.65",
        "pyyaml>=6.0",
    )
)

# ── Persistent volume ─────────────────────────────────────────────────────────
volume     = modal.Volume.from_name("dyngrapheval-data", create_if_missing=True)
VOLUME_PATH     = "/data"
DATASETS_DIR    = f"{VOLUME_PATH}/datasets"
CHECKPOINTS_DIR = f"{VOLUME_PATH}/checkpoints"

# ── Repo mount: copies local DynGraphEval code into the container ─────────────
# Run `modal run modal/train.py` from the DynGraphEval root directory.
repo_mount = modal.Mount.from_local_dir(
    local_path=".",
    remote_path="/repo",
    condition=lambda p: not any(
        p.startswith(seg) for seg in [".git/", "__pycache__/", "checkpoints/", "datasets/", "neg_cache/"]
    ),
)

app = modal.App("dyngrapheval-train")


@app.cls(
    image=image,
    gpu="A10",
    timeout=86400,      # 24-hour hard limit
    volumes={VOLUME_PATH: volume},
    mounts=[repo_mount],
)
class TrainJob:
    """
    Modal class wrapping the TGN training subprocess.

    @modal.enter()  runs once when the container starts
    @modal.exit()   runs when the container shuts down (graceful or not)
                    — commits the volume so epoch checkpoints are preserved
    @modal.method() the main training logic
    """

    @modal.enter()
    def setup(self):
        os.makedirs(DATASETS_DIR,    exist_ok=True)
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        sys.path.insert(0, "/repo")
        self._log("Container ready.")

    @modal.exit()
    def teardown(self):
        """
        Commit the volume so any checkpoint writes are durable.

        Called on both normal exit and Modal-initiated interruption (SIGTERM).
        If the container is killed with SIGKILL this will not run — but resume
        checkpoints are flushed to the volume at the end of each epoch, so at
        most one epoch of progress is lost.
        """
        self._log("Committing volume...")
        volume.commit()
        self._log("Volume committed. Goodbye.")

    @modal.method()
    def run(
        self,
        dataset:       str   = "tgbl-wiki",
        epochs:        int   = 50,
        patience:      int   = 5,
        batch_size:    int   = 200,
        seed:          int   = 0,
        num_layers:    int   = 2,
        num_heads:     int   = 2,
        output_dim:    int   = 100,
        time_feat_dim: int   = 100,
        num_neighbors: int   = 20,
        dropout:       float = 0.1,
        lr:            float = 0.0001,
    ):
        """
        Launch training via models/tgn/train.py.

        - Datasets are cached in the Modal Volume; subsequent runs skip download.
        - Resume checkpoints are written to the volume after every epoch.
        - The best checkpoint (by val MRR) is also copied to the volume.
        """
        self._log(f"Training TGN on {dataset} | epochs={epochs} patience={patience} seed={seed}")

        cmd = [
            sys.executable, "-u", "/repo/models/tgn/train.py",
            "--dataset",       dataset,
            "--epochs",        str(epochs),
            "--patience",      str(patience),
            "--batch_size",    str(batch_size),
            "--seed",          str(seed),
            "--num_layers",    str(num_layers),
            "--num_heads",     str(num_heads),
            "--output_dim",    str(output_dim),
            "--time_feat_dim", str(time_feat_dim),
            "--num_neighbors", str(num_neighbors),
            "--dropout",       str(dropout),
            "--lr",            str(lr),
            "--repo_dir",        "/tmp/TGB_TPNet",
            "--datasets_cache",  DATASETS_DIR,
            "--checkpoints_dir", CHECKPOINTS_DIR,
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True,
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Training subprocess exited with code {proc.returncode}")
        except Exception as e:
            self._log(f"ERROR: {e}")
            raise

        self._log("Training complete.")
        # Commit here too so the final checkpoint is durable immediately
        volume.commit()
        self._log("Volume committed.")

    def _log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@app.local_entrypoint()
def main(
    dataset:    str = "tgbl-wiki",
    epochs:     int = 50,
    patience:   int = 5,
    batch_size: int = 200,
    seed:       int = 0,
):
    """
    CLI entrypoint.

    Examples:
        modal run modal/train.py
        modal run modal/train.py --dataset tgbl-wiki --epochs 100 --seed 1
    """
    job = TrainJob()
    job.run.remote(
        dataset=dataset,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
    )

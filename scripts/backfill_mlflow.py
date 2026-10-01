"""
Backfill existing eval result JSONs into MLflow.

Usage:
    python scripts/backfill_mlflow.py
    python scripts/backfill_mlflow.py --results-dir results --tracking-uri mlruns

Each JSON in results/ becomes one MLflow run in the 'dyngrapheval-eval' experiment.
The run start time is parsed from the filename timestamp (YYYYMMDD_HHMMSS).
"""

import argparse
import datetime
import json
import os
import sys

import mlflow


def _flatten_results(results: dict) -> dict:
    flat = {
        "standard_mrr":         results.get("standard_mrr"),
        "standard_mrr_return":  results.get("standard_mrr_return"),
        "standard_mrr_explore": results.get("standard_mrr_explore"),
        "recency_mrr":          results.get("recency_mrr"),
        "n_scored":             results.get("n_scored"),
    }
    for k, v in results.get("recency_k_curve", {}).items():
        flat[f"recency_{k}"] = v
    for k, v in results.get("recency_return", {}).items():
        flat[f"recency_return_n" if k == "n" else f"recency_return_{k}"] = v
    for k, v in results.get("recency_explore", {}).items():
        flat[f"recency_explore_n" if k == "n" else f"recency_explore_{k}"] = v
    return {k: v for k, v in flat.items() if v is not None}


def _parse_filename(name: str):
    """
    Parse 'tgn_tgbl-wiki_20260911_174748' into (model, dataset, datetime).
    Filename format: {model}_{dataset}_{YYYYMMDD}_{HHMMSS}.json
    Works even when dataset contains hyphens.
    """
    stem = name.replace(".json", "")
    # Last two underscore-separated tokens are date and time
    parts = stem.rsplit("_", 2)
    if len(parts) == 3:
        model_dataset, date_str, time_str = parts
        # model is everything before the first underscore in model_dataset
        model, _, dataset = model_dataset.partition("_")
        try:
            ts = datetime.datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return model, dataset, ts
        except ValueError:
            pass
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",  default="results",
                        help="Directory containing eval JSON files (default: results/)")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                        help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    parser.add_argument("--experiment",   default="dyngrapheval-eval",
                        help="MLflow experiment name")
    parser.add_argument("--default-seed", type=int, default=0,
                        help="Seed to use when not inferable from filename (default: 0)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f"ERROR: results dir not found: {args.results_dir}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    jsons = sorted(f for f in os.listdir(args.results_dir) if f.endswith(".json"))
    if not jsons:
        sys.exit(f"No JSON files found in {args.results_dir}")

    print(f"Backfilling {len(jsons)} run(s) into experiment '{args.experiment}' "
          f"at tracking URI '{args.tracking_uri}'")

    for fname in jsons:
        fpath = os.path.join(args.results_dir, fname)
        with open(fpath) as f:
            results = json.load(f)

        model   = results.get("model")
        dataset = results.get("dataset")
        _, _, ts = _parse_filename(fname)

        run_name = fname.replace(".json", "")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "model":      model or "unknown",
                "dataset":    dataset or "unknown",
                "backfilled": "true",
                "source_file": fname,
            })
            mlflow.log_params({
                "model":                    model,
                "dataset":                  dataset,
                "seed":                     args.default_seed,
                # Training defaults from modal/train.py used for all existing runs
                "epochs":                   50,
                "patience":                 5,
                "batch_size":               200,
                "num_layers":               2,
                "num_heads":                2,
                "output_dim":               100,
                "time_feat_dim":            100,
                "num_neighbors":            20,
                "dropout":                  0.1,
                "lr":                       0.0001,
                "sample_neighbor_strategy": "recent",
            })
            mlflow.log_metrics(_flatten_results(results))
            mlflow.log_artifact(fpath, artifact_path="results")
            if ts:
                mlflow.set_tag("original_timestamp", ts.isoformat())

        print(f"  logged: {fname}")

    print(f"\nDone. Run `mlflow ui --backend-store-uri {args.tracking_uri}` to view.")
    print("      (or just `mlflow ui` if mlflow.db is in the current directory)")


if __name__ == "__main__":
    main()

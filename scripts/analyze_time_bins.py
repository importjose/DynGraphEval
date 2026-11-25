"""Measure age-bin occupancy from the actual temporal edge stream.

This is a DATA-ONLY diagnostic: no checkpoint or GPU is needed. For each positive
edge at time t, query the last K neighbors of both endpoints at times strictly
before t (same first-hop/recent sampler rule as TGN/TGAT). Messages use the
previous interaction of each endpoint as a PROXY for TGN's memory-update age;
TGN's deferred batch message updates mean these proxy ages are NOT exact
TimeEncoder.forward inputs. Recursive/negative-candidate histories are omitted.

Run on Athene:
  .venv/bin/python scripts/analyze_time_bins.py \
      --datasets-dir /mnt/beegfs/home/delgadoj2020/dyngrapheval-data/datasets \
      --output /mnt/beegfs/home/delgadoj2020/dyngrapheval-data/bin_occupancy/tgbl-wiki.json
"""

import argparse
import json
from array import array
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WIDTHS = {"hour": 3600, "day": 86400, "week": 604800,
          "month": 2592000}  # fixed 30 days; meaningful labels only for seconds
SPLITS = ("train", "val", "test")


def accumulate_ages(src, dst, times, splits, num_neighbors=20):
    """Return first-hop valid-neighbor and previous-event ages per split.

    Histories are updated AFTER all events at a timestamp are queried. This
    reproduces the neighbor sampler's strict t_historical < t cutoff; events
    with equal timestamps cannot see each other. Insertion order breaks ties
    among earlier equal-time events, as in the stable history sort.
    """
    src, dst, times, splits = map(np.asarray, (src, dst, times, splits))
    if not (len(src) == len(dst) == len(times) == len(splits)):
        raise ValueError("src, dst, times, splits must have the same length")
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be positive")
    if np.any(np.diff(times.astype(np.float64)) < 0):
        raise ValueError("events must be sorted by nondecreasing timestamp")
    if not np.all(np.isin(splits, SPLITS)):
        raise ValueError("splits must contain only train/val/test")

    histories = defaultdict(lambda: deque(maxlen=num_neighbors))
    last_seen = {}
    neighbor_ages = {s: array("d") for s in SPLITS}
    message_proxy_ages = {s: array("d") for s in SPLITS}
    event_counts = {s: 0 for s in SPLITS}
    neighbor_slots = {s: 0 for s in SPLITS}
    no_prior_interaction = {s: 0 for s in SPLITS}
    i = 0
    while i < len(times):
        j = i + 1
        while j < len(times) and times[j] == times[i]:
            j += 1
        t = float(times[i])
        for k in range(i, j):
            split = str(splits[k])
            event_counts[split] += 1
            neighbor_slots[split] += 2 * num_neighbors
            for node in (int(src[k]), int(dst[k])):
                # Each endpoint's first-hop neighbors in the recent sampler.
                neighbor_ages[split].extend(t - historical_t for historical_t in histories[node])
                # Only a proxy: real memory updates depend on batch/message timing.
                message_proxy_ages[split].append(t - last_seen.get(node, 0.0))
                no_prior_interaction[split] += node not in last_seen
        for k in range(i, j):
            for node in (int(src[k]), int(dst[k])):
                histories[node].append(t)
                last_seen[node] = t
        i = j
    return {"neighbor_ages": neighbor_ages, "message_proxy_ages": message_proxy_ages,
            "event_counts": event_counts, "neighbor_slots": neighbor_slots,
            "no_prior_interaction": no_prior_interaction}


def summarize_ages(ages, widths=WIDTHS):
    a = np.asarray(ages, dtype=np.float64)
    if np.any(a < 0):
        raise ValueError("negative elapsed time: historical cutoff violated")
    report = {"count": int(a.size),
              "age_quantiles_input_units": (
                  {str(q): float(v) for q, v in zip(
                      (0, .1, .25, .5, .75, .9, .99, 1),
                      np.quantile(a, (0, .1, .25, .5, .75, .9, .99, 1)))}
                  if a.size else {})}
    for label, width in widths.items():
        bins = np.floor(a / width).astype(np.int64)
        n = max(len(bins), 1)
        counts = {"bin0": int(np.count_nonzero(bins == 0)),
                  "bin1": int(np.count_nonzero(bins == 1)),
                  "bin2_to_9": int(np.count_nonzero((bins >= 2) & (bins <= 9))),
                  "bin10_plus": int(np.count_nonzero(bins >= 10))}
        report[label] = {"width_input_units": width, "counts": counts,
                         "fraction": {key: value / n for key, value in counts.items()},
                         "distinct_occupied_bins": int(np.unique(bins).size)}
    return report


def analyze(dataset, num_neighbors=20):
    data = dataset.get_TemporalData()
    src = data.src.cpu().numpy()
    dst = data.dst.cpu().numpy()
    times = data.t.cpu().numpy().astype(np.float64)
    masks = {s: np.asarray(getattr(dataset, f"{s}_mask").cpu(), dtype=bool)
             for s in SPLITS}
    if any(len(mask) != len(times) for mask in masks.values()):
        raise ValueError("split mask length differs from event stream")
    split_total = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(split_total == 1):
        raise ValueError("each event must belong to exactly one split")
    split_labels = np.full(len(times), "", dtype="<U5")
    for s, mask in masks.items():
        split_labels[mask] = s
    observed = accumulate_ages(src, dst, times, split_labels, num_neighbors)
    result = {"dataset": dataset.name, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "timestamp_unit": "UNVERIFIED: widths are hours/days only if data.t is seconds",
              "timestamp_range": [float(times.min()), float(times.max())],
              "num_neighbors": num_neighbors, "first_hop_scope": "positive src+dst; strict earlier times; no recursive or negative queries",
              "memory_scope": "previous endpoint interaction; PROXY, not TGN memory-bank last-update time",
              "splits": {}}
    for s in SPLITS:
        result["splits"][s] = {
            "positive_events": observed["event_counts"][s],
            "neighbor_slots_including_padding": observed["neighbor_slots"][s],
            "neighbor_valid_fraction": (len(observed["neighbor_ages"][s]) /
                                        max(observed["neighbor_slots"][s], 1)),
            "neighbor_age": summarize_ages(observed["neighbor_ages"][s]),
            "memory_message_proxy_age": summarize_ages(observed["message_proxy_ages"][s]),
            "endpoints_without_prior_interaction": observed["no_prior_interaction"][s],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tgbl-wiki")
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-neighbors", type=int, default=20)
    args = parser.parse_args()
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    dataset = PyGLinkPropPredDataset(name=args.dataset, root=args.datasets_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = analyze(dataset, args.num_neighbors)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}")
    for split, metrics in result["splits"].items():
        print(f"{split}: {metrics['positive_events']} positives; "
              f"valid neighbors {metrics['neighbor_valid_fraction']:.1%}")
        for label in WIDTHS:
            age = metrics["neighbor_age"][label]
            msg = metrics["memory_message_proxy_age"][label]
            print(f"  {label:5s}  neighbor bin0={age['fraction']['bin0']:.1%} "
                  f"bin1={age['fraction']['bin1']:.1%}; "
                  f"message proxy bin0={msg['fraction']['bin0']:.1%}")


if __name__ == "__main__":
    main()

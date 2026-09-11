"""
DynGraphEval results viewer.

Reads all JSONs from results/ and displays an interactive dashboard.

Usage:
    streamlit run viewer.py

Dependencies:
    pip install streamlit plotly
"""

import json
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

K_VALS = [10, 20, 50, 100, 500, 999]
K_KEYS = [f"k{k}" for k in K_VALS]
K_LABELS = [f"K={k}" for k in K_VALS]

# Display order and colors per model.
# EdgeBank is shown last as the memorization floor reference.
MODEL_META = {
    "graphmixer": {"label": "GraphMixer", "color": "#16a34a"},
    "tgn":        {"label": "TGN",        "color": "#3b82f6"},
    "tgat":       {"label": "TGAT",       "color": "#ea580c"},
    "tpnet":      {"label": "TPNet",      "color": "#7c3aed"},
    "fl_tgn":     {"label": "FL-TGN",     "color": "#0891b2"},
    "fedlink":    {"label": "FedLink",    "color": "#be185d"},
    "edgebank":   {"label": "EdgeBank",   "color": "#6b7280"},
}

def model_label(m: str) -> str:
    return MODEL_META.get(m, {}).get("label", m)

def model_color(m: str) -> str:
    return MODEL_META.get(m, {}).get("color", "#999999")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> list[dict]:
    """
    Load all JSON files from results_dir.

    Handles both the old schema (standard_mrr + recency_mrr only) and the
    full schema (+ recency_k_curve, recency_return, recency_explore).
    Returns a list of dicts, one per file, with None-filled missing fields.
    """
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                raw = json.load(f)
        except Exception:
            continue

        # Normalise to full schema — old files lack k-curve and return/explore
        r = {
            "file":                 os.path.basename(path),
            "model":                raw.get("model", "unknown"),
            "dataset":              raw.get("dataset", "unknown"),
            "standard_mrr":         raw.get("standard_mrr"),
            "standard_mrr_return":  raw.get("standard_mrr_return"),
            "standard_mrr_explore": raw.get("standard_mrr_explore"),
            "recency_mrr":          raw.get("recency_mrr"),
            "n_scored":             raw.get("n_scored"),
        }

        # K-curve (all edges)
        kc = raw.get("recency_k_curve") or {}
        for k in K_KEYS:
            r[f"recency_{k}"] = kc.get(k)

        # Return subset
        ret = raw.get("recency_return") or {}
        r["recency_return_mrr"] = ret.get("mrr")
        r["recency_return_n"]   = ret.get("n")
        for k in K_KEYS:
            r[f"recency_return_{k}"] = ret.get(k)

        # Explore subset
        exp = raw.get("recency_explore") or {}
        r["recency_explore_mrr"] = exp.get("mrr")
        r["recency_explore_n"]   = exp.get("n")
        for k in K_KEYS:
            r[f"recency_explore_{k}"] = exp.get(k)

        records.append(r)

    return records


def best_run_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, dataset), pick the run with the highest recency_mrr.
    Prefer runs that have the full schema (recency_return_mrr populated) so
    that old-format files don't shadow newer ones with the same recency_mrr.
    """
    df = df.copy()
    # full_schema=1 for runs that have return/explore data, 0 otherwise
    df["_full"] = df["recency_return_mrr"].notna().astype(int)
    df_sorted = df.sort_values(["model", "dataset", "recency_mrr", "_full"],
                               ascending=[True, True, False, False])
    result = df_sorted.groupby(["model", "dataset"]).first().reset_index()
    return result.drop(columns=["_full"])


def mean_run_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, dataset), average all numeric columns across runs.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    agg = df.groupby(["model", "dataset"])[numeric_cols].mean().reset_index()
    agg["file"] = "mean of " + df.groupby(["model", "dataset"])["file"].count().astype(str).reset_index()["file"] + " runs"
    return agg


# ── Plotly helpers ────────────────────────────────────────────────────────────

def bar_fig(
    models: list[str],
    datasets: dict[str, list],   # {"Dataset A": [values...]}
    title: str,
    yrange: list = None,
    ylabel: str = "",
    barmode: str = "group",
    floor_model: str = "edgebank",
) -> go.Figure:
    """Grouped bar chart with one bar group per model."""
    labels   = [model_label(m) for m in models]
    colors   = [model_color(m) for m in models]
    patterns = ["" if m != floor_model else "/" for m in models]

    fig = go.Figure()
    for ds_name, values in datasets.items():
        fig.add_trace(go.Bar(
            name=ds_name,
            x=labels,
            y=values,
            marker_color=colors,
            marker_pattern_shape=patterns,
            text=[f"{v:.3f}" if v is not None else "—" for v in values],
            textposition="outside",
            textfont_size=11,
        ))

    fig.update_layout(
        title=title,
        barmode=barmode,
        yaxis=dict(range=yrange, title=ylabel, gridcolor="#eee"),
        xaxis=dict(gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=50, r=20),
        height=360,
    )
    return fig


def line_fig(
    series: list[dict],   # [{"name": str, "model": str, "y": list, "dash": str}]
    x_labels: list[str],
    title: str,
    yrange: list = None,
) -> go.Figure:
    fig = go.Figure()
    for s in series:
        fig.add_trace(go.Scatter(
            name=s["name"],
            x=x_labels,
            y=s["y"],
            mode="lines+markers",
            line=dict(color=model_color(s["model"]), dash=s.get("dash", "solid"), width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=title,
        yaxis=dict(range=yrange, gridcolor="#eee"),
        xaxis=dict(gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=50, r=20),
        height=420,
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DynGraphEval", layout="wide", page_icon="📊")

st.title("DynGraphEval")
st.caption("Temporal graph link prediction — evaluation dashboard")

# Load data
all_records = load_results(RESULTS_DIR)
if not all_records:
    st.error(f"No JSON files found in `{RESULTS_DIR}`. Run an evaluation first.")
    st.stop()

df_all = pd.DataFrame(all_records)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    datasets_available = sorted(df_all["dataset"].unique())
    dataset = st.selectbox("Dataset", datasets_available, index=0)

    df_ds = df_all[df_all["dataset"] == dataset].copy()

    models_available = sorted(
        df_ds["model"].unique(),
        key=lambda m: list(MODEL_META.keys()).index(m) if m in MODEL_META else 99,
    )
    selected_models = st.multiselect(
        "Models",
        options=models_available,
        default=models_available,
        format_func=model_label,
    )

    run_mode = st.radio(
        "When multiple runs exist",
        ["Best run (highest Recency MRR)", "Mean across runs", "All runs individually"],
        index=0,
    )

    st.divider()
    n_files = len(df_ds)
    st.caption(f"{n_files} result file(s) loaded for {dataset}")
    with st.expander("Files"):
        for f in sorted(df_ds["file"].tolist()):
            st.caption(f)

# Filter to selected models
df_ds = df_ds[df_ds["model"].isin(selected_models)]

if df_ds.empty:
    st.warning("No results match the current filters.")
    st.stop()

# Apply run aggregation
if run_mode == "Best run (highest Recency MRR)":
    df_plot = best_run_per_model(df_ds)
elif run_mode == "Mean across runs":
    df_plot = mean_run_per_model(df_ds)
else:
    df_plot = df_ds.copy()

# Sort models for display (EdgeBank last as floor reference)
ordered_models = [m for m in list(MODEL_META.keys()) if m in df_plot["model"].values]
remaining = [m for m in df_plot["model"].unique() if m not in ordered_models]
ordered_models = ordered_models + remaining

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_ret_exp, tab_kcurve, tab_raw = st.tabs([
    "Overview", "Return vs Explore", "K-curve", "Raw data"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tab_overview:

    # ── Summary table ─────────────────────────────────────────────────────
    st.subheader("Summary")

    def fmt(v, bold=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        s = f"{v:.3f}"
        return f"**{s}**" if bold else s

    rows = []
    for m in ordered_models:
        sub = df_plot[df_plot["model"] == m]
        if sub.empty:
            continue
        row = sub.iloc[0]
        n_runs = len(df_ds[df_ds["model"] == m])
        rows.append({
            "Model":          model_label(m),
            "Standard MRR":   fmt(row.get("standard_mrr")),
            "Recency MRR":    fmt(row.get("recency_mrr")),
            "Return MRR":     fmt(row.get("recency_return_mrr")),
            "Explore MRR":    fmt(row.get("recency_explore_mrr")),
            "R/E Gap":        fmt((row.get("recency_return_mrr") or 0) - (row.get("recency_explore_mrr") or 0)),
            "n scored":       int(row["n_scored"]) if row.get("n_scored") and not pd.isna(row["n_scored"]) else "—",
            "Runs":           n_runs,
        })

    st.dataframe(
        pd.DataFrame(rows).set_index("Model"),
        use_container_width=True,
    )

    st.divider()

    # ── Chart 1: Standard vs Recency inversion ────────────────────────────
    st.subheader("Standard MRR vs Recency MRR")
    st.caption(
        "The model that wins Standard MRR loses Recency MRR. "
        "EdgeBank (gray, hatched) is the memorization floor."
    )

    std_vals     = [df_plot[df_plot["model"] == m]["standard_mrr"].iloc[0] if not df_plot[df_plot["model"] == m].empty else None for m in ordered_models]
    recency_vals = [df_plot[df_plot["model"] == m]["recency_mrr"].iloc[0]  if not df_plot[df_plot["model"] == m].empty else None for m in ordered_models]

    fig_inv = go.Figure()
    labels = [model_label(m) for m in ordered_models]
    colors = [model_color(m) for m in ordered_models]
    patterns = ["/" if m == "edgebank" else "" for m in ordered_models]

    for metric, vals, opacity, name in [
        ("Standard MRR",  std_vals,     1.0, "Standard MRR"),
        ("Recency MRR",   recency_vals, 0.45, "Recency MRR"),
    ]:
        fig_inv.add_trace(go.Bar(
            name=name,
            x=labels,
            y=vals,
            marker_color=[model_color(m) for m in ordered_models],
            marker_opacity=opacity,
            marker_pattern_shape=patterns,
            text=[f"{v:.3f}" if v is not None else "—" for v in vals],
            textposition="outside",
            textfont_size=10,
        ))

    fig_inv.update_layout(
        barmode="group",
        yaxis=dict(range=[0.25, 1.0], gridcolor="#eee", title="MRR"),
        xaxis=dict(gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=50, r=20),
        height=380,
    )
    st.plotly_chart(fig_inv, use_container_width=True)

    # ── Chart 2: Memorization floor (gap above EdgeBank) ─────────────────
    eb_row = df_plot[df_plot["model"] == "edgebank"]
    if not eb_row.empty:
        st.divider()
        st.subheader("Advantage over EdgeBank (memorization floor)")
        st.caption(
            "How far above pure memorization each model sits. "
            "The Recency gap is 4–8× larger than the Standard gap — "
            "Recency MRR amplifies differences in temporal reasoning."
        )

        eb_std     = eb_row.iloc[0]["standard_mrr"] or 0
        eb_recency = eb_row.iloc[0]["recency_mrr"]  or 0

        neural_models = [m for m in ordered_models if m != "edgebank"]
        gap_std     = []
        gap_recency = []
        ratios      = []
        for m in neural_models:
            sub = df_plot[df_plot["model"] == m]
            if sub.empty:
                gap_std.append(None); gap_recency.append(None); ratios.append(None)
                continue
            gs = (sub.iloc[0]["standard_mrr"] or 0) - eb_std
            gr = (sub.iloc[0]["recency_mrr"]  or 0) - eb_recency
            gap_std.append(round(gs, 4))
            gap_recency.append(round(gr, 4))
            ratios.append(round(gr / gs, 1) if gs > 0 else None)

        fig_floor = go.Figure()
        neural_labels  = [model_label(m) for m in neural_models]
        neural_colors  = [model_color(m) for m in neural_models]

        for vals, opacity, name in [
            (gap_std,     1.0,  "Standard MRR gap"),
            (gap_recency, 0.45, "Recency MRR gap"),
        ]:
            fig_floor.add_trace(go.Bar(
                name=name,
                x=neural_labels,
                y=vals,
                marker_color=neural_colors,
                marker_opacity=opacity,
                text=[f"+{v:.3f}" if v is not None else "—" for v in vals],
                textposition="outside",
                textfont_size=11,
            ))

        # Ratio annotations
        for i, (lbl, ratio) in enumerate(zip(neural_labels, ratios)):
            if ratio:
                fig_floor.add_annotation(
                    x=lbl, y=max(gap_recency[i] or 0, 0) + 0.03,
                    text=f"{ratio}×",
                    showarrow=False,
                    font=dict(size=11, color="#333"),
                )

        fig_floor.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 0.6], gridcolor="#eee", title="Δ MRR above EdgeBank"),
            xaxis=dict(gridcolor="#eee"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=40, l=50, r=20),
            height=360,
        )
        st.plotly_chart(fig_floor, use_container_width=True)
    else:
        st.info("Run EdgeBank evaluation to see the memorization floor chart.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RETURN VS EXPLORE
# ════════════════════════════════════════════════════════════════════════════

with tab_ret_exp:

    st.subheader("Return vs Explore MRR")
    st.caption(
        "**Return** edges: source has visited this destination before (tests recall). "
        "**Explore** edges: first-time visit (tests generalization). "
        "A large gap indicates the model relies on memorization over structure."
    )

    # Recency Return/Explore chart
    ret_vals = [df_plot[df_plot["model"] == m]["recency_return_mrr"].iloc[0] if not df_plot[df_plot["model"] == m].empty else None for m in ordered_models]
    exp_vals = [df_plot[df_plot["model"] == m]["recency_explore_mrr"].iloc[0] if not df_plot[df_plot["model"] == m].empty else None for m in ordered_models]

    fig_re = go.Figure()
    labels  = [model_label(m) for m in ordered_models]
    colors  = [model_color(m) for m in ordered_models]
    patterns = ["/" if m == "edgebank" else "" for m in ordered_models]

    for vals, opacity, name in [(ret_vals, 1.0, "Return MRR"), (exp_vals, 0.4, "Explore MRR")]:
        fig_re.add_trace(go.Bar(
            name=name,
            x=labels,
            y=vals,
            marker_color=colors,
            marker_opacity=opacity,
            marker_pattern_shape=patterns,
            text=[f"{v:.3f}" if v is not None else "—" for v in vals],
            textposition="outside",
            textfont_size=10,
        ))

    fig_re.update_layout(
        barmode="group",
        yaxis=dict(range=[0.2, 1.0], gridcolor="#eee", title="Recency MRR"),
        xaxis=dict(gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=50, r=20),
        height=380,
    )
    st.plotly_chart(fig_re, use_container_width=True)

    # Gap table
    st.subheader("Return / Explore gap (Recency MRR)")
    st.caption("Sorted by gap descending. Larger gap = more dependent on memorization.")

    gap_rows = []
    for m in ordered_models:
        sub = df_plot[df_plot["model"] == m]
        if sub.empty:
            continue
        row = sub.iloc[0]
        ret = row.get("recency_return_mrr")
        exp = row.get("recency_explore_mrr")
        gap = (ret - exp) if (ret is not None and exp is not None) else None
        gap_rows.append({
            "Model":       model_label(m),
            "Return MRR":  f"{ret:.3f}" if ret else "—",
            "Explore MRR": f"{exp:.3f}" if exp else "—",
            "Gap":         f"{gap:.3f}" if gap is not None else "—",
        })

    gap_rows.sort(key=lambda r: float(r["Gap"]) if r["Gap"] != "—" else -1, reverse=True)
    st.dataframe(pd.DataFrame(gap_rows).set_index("Model"), use_container_width=True)

    # Standard MRR Return/Explore if available
    std_ret_available = any(
        df_plot[df_plot["model"] == m]["standard_mrr_return"].iloc[0] is not None
        for m in ordered_models
        if not df_plot[df_plot["model"] == m].empty
        and not pd.isna(df_plot[df_plot["model"] == m]["standard_mrr_return"].iloc[0] or float("nan"))
    )

    if std_ret_available:
        st.divider()
        st.subheader("Standard MRR — Return vs Explore")
        std_ret_vals = [df_plot[df_plot["model"] == m]["standard_mrr_return"].iloc[0] if not df_plot[df_plot["model"] == m].empty else None for m in ordered_models]
        std_exp_vals = [df_plot[df_plot["model"] == m]["standard_mrr_explore"].iloc[0] if not df_plot[df_plot["model"] == m].empty else None for m in ordered_models]

        fig_std_re = go.Figure()
        for vals, opacity, name in [(std_ret_vals, 1.0, "Return MRR (Standard)"), (std_exp_vals, 0.4, "Explore MRR (Standard)")]:
            fig_std_re.add_trace(go.Bar(
                name=name,
                x=labels,
                y=vals,
                marker_color=colors,
                marker_opacity=opacity,
                marker_pattern_shape=patterns,
                text=[f"{v:.3f}" if v is not None else "—" for v in vals],
                textposition="outside",
                textfont_size=10,
            ))

        fig_std_re.update_layout(
            barmode="group",
            yaxis=dict(range=[0.2, 1.0], gridcolor="#eee", title="Standard MRR"),
            xaxis=dict(gridcolor="#eee"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=40, l=50, r=20),
            height=360,
        )
        st.plotly_chart(fig_std_re, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — K-CURVE
# ════════════════════════════════════════════════════════════════════════════

with tab_kcurve:

    st.subheader("Recency MRR K-curve")
    st.caption(
        "Recency MRR at increasing negative pool depths, computed from a single inference pass. "
        "A steep drop indicates the model's temporal memory degrades quickly."
    )

    col_all, col_ret, col_exp = st.columns(3)
    show_all     = col_all.checkbox("All edges",     value=True)
    show_return  = col_ret.checkbox("Return edges",  value=False)
    show_explore = col_exp.checkbox("Explore edges", value=False)

    series = []
    for m in ordered_models:
        sub = df_plot[df_plot["model"] == m]
        if sub.empty:
            continue
        row = sub.iloc[0]

        if show_all:
            y = [row.get(f"recency_{k}") for k in K_KEYS]
            if any(v is not None for v in y):
                series.append({"name": f"{model_label(m)} — all", "model": m, "y": y, "dash": "solid"})

        if show_return:
            y = [row.get(f"recency_return_{k}") for k in K_KEYS]
            if any(v is not None for v in y):
                series.append({"name": f"{model_label(m)} — return", "model": m, "y": y, "dash": "dash"})

        if show_explore:
            y = [row.get(f"recency_explore_{k}") for k in K_KEYS]
            if any(v is not None for v in y):
                series.append({"name": f"{model_label(m)} — explore", "model": m, "y": y, "dash": "dot"})

    if series:
        all_y = [v for s in series for v in s["y"] if v is not None]
        y_min = max(0.0, min(all_y) - 0.05) if all_y else 0.0
        y_max = min(1.0, max(all_y) + 0.05) if all_y else 1.0

        fig_kc = line_fig(series, K_LABELS, "K-curve: Recency MRR vs negative pool depth", yrange=[y_min, y_max])
        st.plotly_chart(fig_kc, use_container_width=True)
    else:
        st.info("Select at least one edge subset above.")

    # K-curve table
    st.subheader("K-curve values")
    kc_rows = []
    for m in ordered_models:
        sub = df_plot[df_plot["model"] == m]
        if sub.empty:
            continue
        row = sub.iloc[0]
        vals = [row.get(f"recency_{k}") for k in K_KEYS]
        if all(v is None for v in vals):
            continue
        drop = None
        if vals[0] is not None and vals[-1] is not None:
            drop = round(vals[-1] - vals[0], 3)
        kc_rows.append({
            "Model": model_label(m),
            **{f"K={k}": f"{v:.3f}" if v is not None else "—" for k, v in zip(K_VALS, vals)},
            "Drop (K=10→999)": f"{drop:+.3f}" if drop is not None else "—",
        })

    st.dataframe(pd.DataFrame(kc_rows).set_index("Model"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAW DATA
# ════════════════════════════════════════════════════════════════════════════

with tab_raw:
    st.subheader("All loaded results")
    st.caption("One row per JSON file. Use the sidebar to filter by dataset and model.")

    display_cols = [
        "file", "model", "dataset", "standard_mrr", "recency_mrr",
        "recency_return_mrr", "recency_explore_mrr", "n_scored",
    ]
    st.dataframe(
        df_ds[display_cols].sort_values(["model", "file"]).reset_index(drop=True),
        use_container_width=True,
    )

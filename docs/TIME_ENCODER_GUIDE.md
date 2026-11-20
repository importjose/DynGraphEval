# Time encoding in DynGraphEval: a researcher's guide

**Scope.** This guide explains the cosine time encoder actually used by the TGN, TGAT, and GraphMixer implementations in this repository. It distinguishes the encoder from temporal sampling, memory, and the evaluation protocol. It also flags an important train/evaluation mismatch in the existing *NoEnc* ablations. Read this alongside `models/tgn/tpnet_components.py`, especially `TimeEncoder`, `GraphAttentionEmbedding`, and `MemoryModel`.

## 1. The problem it solves

A temporal edge is an event `(source u, destination v, time t, edge features x)`. To score a candidate link at time `t`, a model can inspect earlier events at times `t_i < t`. Node/edge identities say **what happened**; the age `Δ_i = t - t_i` says **when**. If the scoring network receives no age information, two otherwise identical histories with different elapsed times can be indistinguishable. A scalar age can be supplied directly, but nonlinear, multiscale features make it easier for a downstream network to distinguish short from long intervals, without requiring it to learn every temporal transformation from scratch.

There are several *different* sources of temporal information here:

- **Time encoding:** transforms an elapsed-time scalar into a vector.
- **Historical-neighbor sampling:** selects only events strictly before the query time; the default `recent` strategy keeps the latest `num_neighbors` (`NeighborSampler` in `models/tgn/tpnet_components.py`). This is an *architectural recency prior even if the encoder is removed*.
- **Node memory (TGN):** maintains state from previous interactions; also temporal, but not synonymous with its time encoder.
- **Sequence/order and edge features:** the selected history, edge attributes, and their aggregation may retain useful temporal information without explicit time features.
- **Calendar time:** hour of day, day of week, and seasonality require a separate choice of representation. Encoding elapsed time alone is invariant to shifting all timestamps by the same amount and cannot reveal time of day by itself.

The encoder is a **representation**, not a decay function and not a guarantee of temporal reasoning.

## 2. Exact implementation and shapes

`models/tgn/tpnet_components.py:TimeEncoder` (also in `src/TGB_TPNet/models/modules.py`) computes, for input `timestamps` of shape `[B, L]`, an output of shape `[B, L, d]`:

```python
self.w = nn.Linear(1, time_dim)
self.w.weight = nn.Parameter(
    (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)))
    .reshape(time_dim, -1)
)
self.w.bias = nn.Parameter(torch.zeros(time_dim))
# forward: torch.cos(self.w(timestamps.unsqueeze(2)))
```

Mathematically, for feature `j = 0, …, d-1`,

\[
\phi_j(\Delta)=\cos(\omega_j\Delta+b_j),\qquad
\omega_j^{(0)}=10^{-9j/(d-1)},\quad b_j^{(0)}=0\quad(d>1).
\]

`d=100` is the training-script default (`models/tgn/train.py`, `models/tgat/train.py`, `models/graphmixer/train.py`). There are **2d scalar encoder parameters**, weights and biases; when fixed, both have `requires_grad=False`. Initialization spans angular frequencies from `1` to `10^-9` radians per *input timestamp unit*, logarithmically spaced. A frequency `ω` has period `P=2π/|ω|` **input units**, not `1/ω` and not necessarily seconds: verify the dataset's timestamp units before interpreting a period. With `d=100`, adjacent frequencies differ by approximately `10^(9/99) ≈ 1.233`. At `Δ=0`, every coordinate is initially `1`. This is a learnable cosine basis initialized at geometrically spaced scales; unlike classical fixed Fourier features, TGN/TGAT can move the frequencies and phases during training. Fixed GraphMixer retains the same initialized basis. For `d=1`, the `linspace` initialization simply yields frequency `1`; use the general formula only for `d>1`.

**Example, two frequencies:** if `ω=(1,0.01)`, `b=(0,0)`, then `φ(0)=(1,1)`, `φ(10)=(cos 10, cos 0.1)`, and `φ(100)=(cos 100, cos 1)`. The slower component changes little at short intervals; the faster one varies rapidly. None of these coordinates means “recent” by itself.

## 3. Why cosine? Intuition and the math

Imagine a bank of clocks: fast hands distinguish nearby events and slow hands distinguish events separated by large intervals. Concatenating them provides a multiscale description of an age. Log spacing spreads a limited number of coordinates over many orders of magnitude rather than concentrating all resolution at one scale. A downstream MLP or attention layer can combine these signals with node/edge features to *learn* a useful dependence on time.

Useful properties and their qualifications:

1. **Bounded features:** each coordinate lies in `[-1,1]`, avoiding a raw timestamp's unbounded magnitude. This does *not* guarantee stable gradients with respect to frequency when `Δ` is large.
2. **Smoothness:** `∂φ_j/∂Δ = -ω_j sin(ω_jΔ+b_j)`. At initialization (`b=0`), `∂φ_j/∂Δ` is zero exactly at `Δ=0`; near zero, `cos(ωΔ) ≈ 1 - (ωΔ)^2/2`. Small ages can be difficult to distinguish with cosine-only, zero-phase features.
3. **Trainable parameters:** `∂φ_j/∂ω_j = -Δ sin(ω_jΔ+b_j)` and `∂φ_j/∂b_j = -sin(ω_jΔ+b_j)`. Large input scales can produce highly oscillatory optimization; a fixed encoder avoids optimizing these parameters but downstream weights still learn how to use its output.
4. **Not monotone:** `cos` repeats. A large age can map to the same value as a small one on one coordinate. Multiple frequencies may resolve ambiguities over a finite observed range, but there is no globally monotonic age representation or guaranteed unique mapping. If explicit monotone decay is desired, model it separately (e.g. `exp(-λΔ)`, with `λ≥0`).
5. **Kernel viewpoint:** at initialization and zero phase, the feature inner product is `φ(Δ)·φ(Δ') = Σ_j cos(ω_jΔ)cos(ω_jΔ') = ½Σ_j[cos(ω_j(Δ-Δ'))+cos(ω_j(Δ+Δ'))]`. Thus this *cosine-only* representation is **not a strictly shift-invariant Fourier kernel**: the sum-of-ages term remains. Paired sine/cosine features at each frequency instead yield `cos(ω(Δ-Δ'))` via the angle-difference identity. Here the encoder is used as input to trained layers, not as a prescribed kernel.

**Units and numerical precision matter.** Multiplying ages by `s` gives `cos((sω_j)Δ+b_j)`: for *fixed* weights, it shifts the effective periods by `1/s`. For trainable weights, the function class can in principle undo this by learning `ω'_j=ω_j/s`, so scaling is primarily an initialization/optimization/precision intervention, not a clean restriction to one “temporal resolution.” Converting seconds to days (`s=1/86400`) does not round timestamps into days or create a daily-bin representation. Very large floating-point phases can lose precision; very small phases make many coordinates nearly constant. Check the *distribution of actual input deltas* against `|sω_jΔ|`, not just the unit label.

## 4. Where the encoding enters each model

### TGAT: time-conditioned neighbor attention

`models/tgat/tgat_components.py` calls the shared `TimeEncoder` with `parameter_requires_grad=True`. For a query node at time `t`, its time feature is `φ(0)`; for a sampled historical neighbor observed at `t_i`, the feature is `φ(t-t_i)`. In `models/tgn/tpnet_components.py:MultiHeadAttention`, roughly,

\[
q=W_Q[\text{query node state}\;\Vert\;\phi(0)],\quad
k_i=W_K[\text{neighbor state}\;\Vert\;\text{edge features}_i\;\Vert\;\phi(t-t_i)],
\]
\[
a_i=\operatorname{softmax}_i(q^\top k_i/\sqrt{d_h}),\quad
h=\sum_i a_i W_V[\text{neighbor state}\;\Vert\;\text{edge features}_i\;\Vert\;\phi(t-t_i)].
\]

There are multiple heads and further residual, normalization, and merge layers. **The encoder does not directly assign higher weights to recent neighbors**: learned projections and context determine attention. TGAT has no persistent memory bank, but recursive historical-neighbor lookup and the `recent` sampler still carry temporal structure.

### TGN: attention **and** memory messages

`models/tgn/tpnet_components.py:MemoryModel` uses the **same trainable encoder instance** in two paths:

1. `GraphAttentionEmbedding` uses `φ(0)` for query nodes and `φ(t-t_i)` for their historical neighbors, similarly to TGAT.
2. `compute_new_node_raw_messages` computes a separate age since the node memory's last update, `δ_u=t-last_update(u)`, and concatenates `[source memory, destination memory, φ(δ_u), edge features]`. A GRU updates node memory from selected stored messages. Source and destination directions compute their own intervals.

These are **different notions of time**: age of a sampled historical edge versus time since a particular node's memory update. Removing the TGN encoder affects both paths; freezing it preserves both. Memory update order and neighbor selection remain temporal under either ablation. The model computes embeddings before storing the current positive event's new messages (see `compute_src_dst_node_temporal_embeddings`).

### GraphMixer: fixed time features, learned mixing

`models/graphmixer/graphmixer_components.py` sets `parameter_requires_grad=False`, then combines `φ(t-t_i)` with edge features and applies a learned projection, MLP-Mixer layers, and pooling. Padding positions' time encodings are explicitly zero-masked. GraphMixer's separate node-encoder branch aggregates historical node features and does **not** receive these time encodings. “Fixed encoder” does **not** mean a time-independent model or a frozen downstream network.

### Other variants and avoidable conflations

- `models/tpnet/tpnet_components.py` applies `log(Δ+1)` **before** the fixed cosine encoder, so its actual feature is `cos(ω log(Δ+1)+b)`. It is not directly comparable to an unlogged age encoder.
- `models/components.py` contains a *different* PyG-style `TimeEncoder` with `nn.Linear(1, out_channels)` and default PyTorch initialization, used in a separate TGN/FederatedTGN path. Do not transfer the geometric-initialization claim to that class.
- The vendored `src/TGB_TPNet/models/NAT.py:TimeEncode` is another implementation. Always identify the exact class and checkpoint/wrapper before comparing encoders.

## 5. Reading the repo's ablations correctly

| Intervention | What changes | What remains |
|---|---|---|
| `tgn_fixed_enc`, `tgat_fixed_enc` | Training disables gradients for encoder weight **and** bias (`models/*_fixed_enc/patches/train_link_prediction.py`). | Geometric cosine features, neighbor sampling, downstream trained weights, and TGN memory remain. |
| `tgn_no_enc`, `tgat_no_enc` | Training monkey-patches vendored `TimeEncoder.forward` to return zeros (`models/*_no_enc/patches/train_link_prediction.py`). | Time-based neighbor lookup/order and (for TGN) memory persist. **Evaluation currently does not reapply this zero-forward patch; see warning below.** |
| `*_time_{hour,day,week,month}` | Training and `hpc/eval.py` multiply **encoder input** by `1/3600`, `1/86400`, `1/604800`, or `1/2592000`. | Other model inputs and the underlying event timestamps are not rescaled by this patch. No quantization is performed. |
| TGN vs TGAT vs GraphMixer | Architecture, aggregation, and memory differ *along with* encoder settings. | They are **not** a controlled comparison of learnable versus fixed time encoding; use within-architecture variants for that question. |

**Critical reproducibility warning:** `hpc/eval.py` maps `tgn_no_enc` and `tgat_no_enc` to ordinary TGN/TGAT wrappers and has a scaling patch, but **no zero-encoder evaluation patch**. The wrapper's normal `TimeEncoder.forward` is used after loading the *NoEnc* checkpoint. During zeroed training the encoder receives no useful gradient, so evaluation likely supplies its initialized cosine features to a network trained on zeros. The recorded `results/*_no_enc_*.json` therefore **must not be interpreted as a clean zero-encoder-at-test result** without confirming the exact execution path. Fix by applying the same zero-forward intervention for warmup *and* scoring, or by implementing a configurable encoder mode in the wrapper and testing the mode end-to-end. Merely clearing stored outputs before scoring is insufficient for TGN: warmup messages and memory states must also be recomputed. `modal/eval.py` also only enumerates the original model names in its dispatch, not the ablation variants. These observations come from code inspection, not an audit of remote run logs.

## 6. What the current numbers do and do not show

The committed `results/*.json` on `tgbl-wiki` report approximately:

| Model | Standard MRR | Recency MRR |
|---|---:|---:|
| TGN | 0.6010 | 0.7892 |
| TGN+FixedEnc | 0.6922 | 0.8036 |
| TGAT | 0.5680 | 0.7729 |
| TGAT+FixedEnc | 0.5863 | 0.7744 |
| GraphMixer (fixed) | 0.5492 | 0.8208 |
| TGN+NoEnc (evaluation caveat above) | 0.1856 | 0.6682 |
| TGAT+NoEnc (evaluation caveat above) | 0.5601 | 0.7731 |

These support **the narrow observation** that the fixed-initialization encoder is competitive with a trained encoder in the recorded within-model comparisons. They do **not** establish that *all* time encoding is useless, that the trained encoder necessarily learns a recency preference, or that the TGN memory bank alone caused a particular metric difference. Results are from a single dataset; several JSONs have `seed: null`, while the NoEnc files say `seed: 0`. Verify matching checkpoints, train/validation negative sampling, hyperparameters, and evaluation settings before making causal claims. Inspect learned `ω,b` and controlled counterfactual predictions rather than inferring the mechanism solely from MRR.

**Metric distinction:** Standard MRR uses the TGB evaluation negatives; Recency MRR ranks the positive against a source's most recently visited **distinct** destinations before that event, excluding contemporaneous positives (`evaluate/negative_sampler.py`). There may be fewer than `K` available candidates. Changes in MRR reflect candidate-set composition as well as model behavior; a high Recency MRR does not by itself prove the encoder modeled elapsed time well.

## 7. A rigorous research program

1. **Verify the measurement pipeline first.** Record the exact training and eval encoder modes, checkpoint path/hash, seed, dataset timestamp units, and actual `Δ` distributions. Add an assertion/log at warmup and scoring that the intended mode is active. Unit-test zeros at both stages for NoEnc. Keep the input shape/dtype/device intact.
2. **Matched interventions, within each architecture:** train from multiple seeds with learned cosine, fixed cosine, always-zero features, a learned scalar `log1p(Δ)`, and (optionally) fixed sine/cosine pairs or monotone decays. Keep feature dimensions and downstream parameter budgets matched (e.g., project scalar features to `d` dimensions) or explicitly report the mismatch. Apply identical transformations at train, validation, warmup, and test.
3. **Control other time channels.** Compare `recent` and uniform historical sampling separately; for TGN, independently intervene on the message-path and attention-path encoder. Preserve causal neighbor cutoff. This helps distinguish memory, sampling, and encoding contributions.
4. **Check scaling versus real temporal resolution.** Evaluate `sΔ` with train/eval consistency; also try true binning/quantization, which is a different experiment. Consider `log1p(Δ/c)` using a dataset-derived scale `c` (computed on training data only), then examine performance across short/medium/long age bins.
5. **Measure mechanisms, not just headline MRR.** Plot histograms of training and test ages; compare initialized/final frequency magnitudes, implied periods, bias phases, encoder gradients, and output variance per dimension over observed ages. Sweep only a candidate edge's historical age in a controlled synthetic/history-preserving test and plot link-score changes; look for oscillation rather than assuming monotonic decay. Inspect attention conditioned on age while controlling node/edge identity; attention weight alone is not a causal attribution.
6. **Report uncertainty and stratification.** Multiple seeds with paired confidence intervals, Return/Explore and activity-level strata, and matched negative candidate sets. Separate pure feature-removal effects from retraining effects (zero at inference only is an out-of-distribution stress test, not a substitute for retraining).

### Thought experiments to test your understanding

- If every timestamp is shifted by `+C`, what changes? For this relative-time encoder nothing, provided histories/order and numeric precision remain unchanged; a calendar-time feature would change.
- If `Δ` doubles, must the attention weight fall? No. Cosine is periodic and the attention projections are learned.
- If you freeze the encoder, is time removed? No: fixed `φ(Δ)` still varies with `Δ` and downstream layers still learn.
- If TGN's message encoder is zeroed but attention's remains active, is that “NoEnc”? No; the paths carry distinct elapsed-time signals.
- If performance is unchanged on TGAT with a correctly implemented zero encoder, does that prove timestamps are irrelevant? No: temporal neighbor selection and ordering remain, and the conclusion is conditional on dataset/task/evaluation.

## 8. Further reading and code map

- Xu et al., **Inductive Representation Learning on Temporal Graphs** (TGAT, ICLR 2020): functional time encoding and temporal attention; compare its theoretical construction with this implementation's specific cosine-only features.
- Rossi et al., **Temporal Graph Networks for Deep Learning on Dynamic Graphs** (2020): messages, memory updates, temporal embeddings.
- Cong et al., **Do We Really Need Complicated Model Architectures for Temporal Networks?** (GraphMixer, ICLR 2023): fixed temporal features and mixer aggregation.
- `models/tgn/tpnet_components.py`: encoder, attention, neighbor sampler, both TGN time paths.
- `models/tgat/tgat_components.py`, `models/graphmixer/graphmixer_components.py`: model-specific use.
- `models/*_fixed_enc/patches/train_link_prediction.py`, `models/*_no_enc/patches/train_link_prediction.py`, `models/*_time_scale/patches/train_link_prediction.py`: training interventions.
- `hpc/eval.py`, `models/tgn/model.py`, `models/tgat/model.py`, `evaluate/negative_sampler.py`, `results/`: evaluation implementation and evidence.

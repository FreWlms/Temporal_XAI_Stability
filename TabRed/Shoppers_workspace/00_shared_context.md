# Shared context — Temporal Stability of XAI (master's thesis pipeline)

> This file describes the **intended state** of the thesis pipeline and methodology.
> If notebook code contradicts this context, flag the contradiction to the user
> before reconstructing, patching, or silently adapting missing pieces.

Repository: https://github.com/FreWlms/Temporal_XAI_Stability/tree/main

---

## 1. What this project is

A master's thesis on the **temporal stability of feature-attribution explanations**
for tabular machine-learning models under rolling retraining.

The project studies how explanations change when models are retrained over time.
For each adjacent pair of retrained models (A, B), the pipeline:

1. trains R replicas of each model version on its own training window;
2. evaluates both versions on the same future evaluation slice;
3. selects an operationally relevant flagged set of instances to explain;
4. computes per-replica explanations on that flagged set;
5. measures **local dynamic drift** of explanations between windows;
6. compares dynamic drift against **within-window baseline instability** (separate
   for A and B);
7. contextualises the result with covariate drift, target drift, predictive
   performance, global SHAP drift, and within-explainer stochasticity.

The empirical work is based on **TabReD-style preprocessing** of three datasets.
Implementation 1 is Kaggle's *Acquire Valued Shoppers Challenge* using TabReD's
`ecom-offers` recipe; implementations 2 and 3 are work in progress.

The thesis text lives in `Thesis_Latex.md`. Methodology is the abstract framework;
the implementation chapter (currently §6.2 for Shoppers) records dataset-specific
parameter choices only.

---

## 2. Dataset/implementation status

### Implementation 1 — Acquire Valued Shoppers Challenge

Status: implemented end-to-end.

- Source: Kaggle *Acquire Valued Shoppers Challenge*, preprocessed in the spirit of
  TabReD's `ecom-offers.py`. Notebook 00 documents the deviations from the literal
  TabReD recipe (notably a strict pre-offer-date filter on transaction aggregations,
  which closes a leakage hole present in the literal recipe).
- Task: binary classification — predict whether a customer becomes a repeat buyer
  after receiving a personalised offer.
- Temporal key: offer date.
- Time-step granularity: ISO calendar week.
- Rolling-window setup: `L = 5`, `S = 2`, `H = 2`, `R = 8`, `K_FRAC = 0.10`.
- `R = 8` replicas per window for the final run.
- During validation/debugging, notebooks may be run with smaller `R`, currently `R = 2`.
- Feature matrix: 119 columns total — 113 numeric aggregation/offer features +
  6 binary indicator features.
- Note: roughly 41% of training-cohort customers have no historical transactions
  in the source log. They receive zero-valued numeric aggregates and the
  corresponding `never_bought_*` indicators set to one. This is correct
  behaviour, but it concentrates a large mass of the empirical distribution at
  zero for behavioural-aggregate features and should be kept in mind when
  interpreting Wasserstein-based covariate-drift values.

### Implementations 2 and 3

Not started. Choice of datasets and preprocessing scope is still open.

---

## 3. Why TabReD matters

1. **Dataset suitability.** TabReD was designed to make tabular ML evaluation more
   representative of real-world deployment (temporal structure, realistic feature
   engineering, industry-style prediction tasks). This matches the thesis goal of
   studying explanation stability under rolling retraining.
2. **Preprocessing foundation.** TabReD provides dataset-specific preprocessing
   recipes. The thesis leans on these recipes rather than presenting every join and
   aggregation as original methodology.

The thesis does **not** adopt TabReD's static train/validation/test split as the
evaluation protocol. TabReD preprocessing produces fixed-width feature matrices;
the temporal rolling-retraining protocol is defined by this thesis.

---

## 4. Workspace layout

Colab, with Drive mounted.

```text
/content/drive/MyDrive/Thesis/Shoppers_workspace/
├── data/
│   ├── raw/                             # Kaggle CSV dumps
│   ├── processed/                       # X.parquet, Y.npy, feature_names.json
│   ├── windows/                         # window_config.json
│   ├── models/{model_type}/pair_XX/     # trained replicas + predictions.npz
│   ├── shap/{model_type}/pair_XX/       # SHAP attributions (xgboost / mlp_plr)
│   ├── lime/{model_type}/pair_XX/       # LIME attributions (xgboost / mlp_plr)
│   └── results/
│       ├── {model_type}/                # per-model-type drift_metrics.csv
│       └── combined/                    # combined CSV, dashboards, LR coef figure
└── notebooks live outside Drive; run on Colab
```

`{model_type}` is one of `{'xgboost', 'logreg', 'mlp_plr'}`. The workspace path is
Shoppers-specific; future implementations should sit in sibling workspaces (e.g.
`Dataset2_workspace`).

---

## 5. Notebooks and outputs

| # | Notebook | Runtime | Main output |
|---|----------|---------|-------------|
| 00 | `00_data_preprocessing.ipynb` | CPU | `X.parquet`, `Y.npy`, `feature_names.json` |
| 01 | `01_rolling_window_setup.ipynb` | CPU | `window_config.json` |
| 02 | `02_training_replicas.ipynb` | CPU | XGBoost or LR replicas + `predictions.npz` |
| 02b | `02b_training_replicas_mlp_plr.ipynb` | GPU | MLP-PLR replicas + `predictions.npz` |
| 03 | `03_shap_explanations.ipynb` | CPU (xgboost) / GPU (mlp_plr) | SHAP attributions |
| 03b | `03b_lime_explanations.ipynb` | CPU (xgboost) / GPU (mlp_plr) | LIME attributions |
| 04 | `04_drift_analysis.ipynb` | CPU | drift/context metrics, dashboards |

Typical Shoppers run order:

```text
00 → 01
   → 02 (MODEL_TYPE='xgboost')
   → 02 (MODEL_TYPE='logreg')      # produces replica + full-window LR coefs
   → 02b                            # MLP-PLR
   → 03  (MODEL_TYPE='xgboost')
   → 03  (MODEL_TYPE='mlp_plr')    # uses the same 200-instance subsample as LIME
   → 03b (MODEL_TYPE='xgboost')
   → 03b (MODEL_TYPE='mlp_plr')
   → 04
```

LR is **not** explained with SHAP or LIME — running notebook 03 or 03b with
`MODEL_TYPE='logreg'` raises by design (see §8).

Notebooks 02, 02b, 03, 03b implement skip-if-current logic: each pair writes a
`run_params.json` fingerprint, and re-running only recomputes pairs whose
fingerprint changed (window/tuning/model/explainer settings, flagged-set hashes,
explainer-package version, etc.).

---

## 6. Data shapes and feature metadata

For Shoppers:

```python
X = pd.read_parquet(processed/X.parquet).values.astype(np.float32)  # (N, 119)
Y = np.load(processed/Y.npy).astype(np.int8)                        # (N,)
```

`feature_names.json`:
- `"all"`: 119 feature names in column order;
- `"num"`: 113 numeric feature names;
- `"bin"`: 6 binary indicator feature names.

Numeric and binary columns are **not** assumed contiguous. Always derive positions
by name:

```python
feature_names = feature_names_json["all"]
num_col_idx = [feature_names.index(fn) for fn in feature_names_json["num"]]
bin_col_idx = [i for i in range(len(feature_names)) if i not in set(num_col_idx)]
```

Downstream code (notebooks 02b/03/04) relies on `feature_names.json["all"]` as the
single source of truth for column ordering — never hard-code positions.

---

## 7. `window_config.json` schema

```python
config = {
    "parameters": {
        "L": int,        # training-window length (in time steps)
        "S": int,        # step / shift between consecutive A windows
        "H": int,        # gap between training-window end and eval-slice start
        "R": int,        # replicas per model version
        "K_FRAC": float, # fraction of eval instances flagged for explanation
        ...
    },
    "pairs": [
        {
            "pair_id": int,
            "idx_A": [int, ...],        # global row indices into X/Y
            "idx_B": [int, ...],
            "idx_eval": [int, ...],
            "step_label_A": str, "step_label_B": str,
            "eval_start_label": str, "eval_end_label": str,
            "n_train_A": int, "n_train_B": int, "n_eval": int,
        },
        ...
    ],
}
```

A and B windows in a pair are **non-overlapping** and chronologically ordered (A
then B). The eval slice is shared by A and B and sits H steps after window B.
Pairs are mutually independent: each pair has its own A, B, eval triple. Indices
within each window are sorted chronologically (notebook 01's `get_indices`),
which is what TimeSeriesSplit-based hyperparameter tuning relies on inside
notebook 02 / 02b.

---

## 8. Models and explainers

### Model families (Shoppers)

1. **Logistic regression — transparent reference.**
   - Trained with a `Pipeline(StandardScaler, LogisticRegression)` per replica.
   - Coefficient artefacts per pair, all saved by notebook 02:
     - `coef_{A,B}.npy` of shape `(R, p)` — **raw replica coefficients in each
       replica's own training-standardised basis**. Stored for traceability;
       not consumed directly by notebook 04.
     - `coef_{A,B}_ref.npy` of shape `(R, p)` — **replica coefficients
       reprojected into the common pair-level reference basis** (numeric
       features per reference-window σ; binary features per raw 0/1).
       Used by notebook 04 for **within-window coefficient instability**.
     - `coef_{A,B}_full_ref.npy` of shape `(p,)` — single deterministic LR fit
       on the full training window (no bootstrap), reprojected into the same
       reference basis. Drives both the A-vs-B coefficient figure and the
       `coef_dyn_cos` / `coef_dyn_rbo` row columns in notebook 04.
     - `full_model_{A,B}.joblib` — fitted full-window LR pipelines (saved so
       coefficients can be re-anchored if the reference basis ever changes).
   - LR is **not** treated as a post-hoc explanation drift stream. No SHAP, no
     LIME, no cross-window drift on replica coefficients.

2. **XGBoost — strong tree-based baseline.**
   - Explained with `shap.TreeExplainer(model_output='raw')` on log-odds.
   - Deterministic given the fitted ensemble.

3. **MLP-PLR — neural tabular model.**
   - PLR (Periodic → Linear → ReLU) embeddings per numeric feature plus an MLP
     backbone over concatenated embedded numeric + raw binary features.
   - Explained with `shap.GradientExplainer` on the pre-sigmoid logit, with a
     per-replica background of 100 instances sampled from the replica's training
     window.
   - Stochastic; expected to show non-zero run-to-run variation.

### Explainer streams

- `coef`: LR only. Contributes `base_cos_A/B` and `base_rbo_A/B` (within-window
  replica-coefficient instability in the common reference basis) plus
  `coef_dyn_cos` / `coef_dyn_rbo` (cosine / RBO between the two full-window
  reference-basis coefficient vectors) to `drift_metrics`. The dynamic cross-window
  number complements the full-window A-vs-B coefficient figure rather than
  replacing it.
- `shap`: XGBoost (TreeSHAP) and MLP-PLR (GradientExplainer). LR raises.
- `lime`: XGBoost and MLP-PLR. LR raises.

### Output-scale caveat

TreeSHAP and GradientExplainer attributions are produced on the model's raw
output scale (log-odds / pre-sigmoid logit). LIME's local surrogate is fit to
positive-class probabilities. Absolute attribution magnitudes are therefore not
directly comparable across explainer families. Drift comparisons across
explainers use cosine and RBO, both scale-invariant.

---

## 9. Metric / reporting design

The thesis reports a **diagnostic profile**, not a single score.

For each (dataset, model, explainer, window pair):

- `delta_X` — covariate drift (mean feature-wise Wasserstein in a reference frame);
- `delta_Y` — target drift (`|p_+_A − p_+_B|`);
- `pr_auc_A`, `pr_auc_B`, `delta_perf = pr_auc_B − pr_auc_A` (positive = newer
  model better on the shared eval slice);
- `loc_cos`, `loc_rbo` — local dynamic explanation drift (median over flagged
  instances of mean over R×R replica pairs of cosine / RBO distance);
- `base_cos_A`, `base_cos_B`, `base_rbo_A`, `base_rbo_B` — within-window
  baseline instability, **separate** for A and B (median over flagged
  instances of median over `C(R,2)` replica pairs);
- `global_rbo` — global attribution drift, **SHAP only** (`1 − RBO` of mean-|φ|
  feature rankings); NaN for LIME and LR by design;
- `coef_dyn_cos`, `coef_dyn_rbo` — full-window LR-only cross-window
  coefficient gap, in the common reference basis. NaN for SHAP / LIME rows.
- `stoch_cos_A`, `stoch_cos_B`, `stoch_cos`, `stoch_rbo_A`, `stoch_rbo_B`,
  `stoch_rbo` — within-explainer stochasticity (§15), computed in notebook 04
  as median pairwise cosine / RBO distance over `C(Q,2)` repeated explainer
  runs per side. The unsuffixed `stoch_cos` / `stoch_rbo` is the conservative
  max over A and B, used as a single visual marker on dashboards.
  `stoch_n_runs` records `Q`; `explainer_is_deterministic` flags TreeSHAP rows
  (`Q = 1`, both metrics → 0).

### Removed designs (do not regress)

- **No drift ratio.** Quantities like `loc_cos / base_*` or any pooled-baseline
  denominator are not part of the methodology. Report drift, baseline, and
  context side by side.
- **No pooled `sigma_*` baseline.** Always report A and B separately.
- **No global LIME drift.** LIME's local surrogate weights are not aggregated
  into a global ranking.
- **No cross-window drift on LR replica coefficients.** Replica coefficients
  ride a per-replica scaler basis; cross-window distances on them conflate
  model change with basis change. Use `coef_dyn_cos` / `coef_dyn_rbo` (which
  compare full-window single fits in the common reference basis) and the
  full-window coefficient figure instead.

---

## 10. Reference-frame and background decisions

Sensitive area. Do not silently change.

### Covariate drift (notebook 04)

Both A and B numeric features are transformed by a `reference_scaler` (the
`StandardScaler` saved per pair under `data/models/{mt}/pair_XX/`, fit on window
A's numeric columns by the training notebook). Binary features pass through
unchanged. Feature-wise Wasserstein distances are then averaged into `delta_X`.

This reference frame is a **diagnostic device for covariate drift only**. It is
not the same as any model's training-time scaling and must not be reused as a
LIME or SHAP scaler.

### Explanation comparisons — end-to-end framing

Explanation drift is interpreted **operationally**: same feature identities,
same explained instances within a (model, explainer, window pair) comparison,
attribution outputs produced by the explainer configuration that pairs with the
trained replica.

For SHAP:
- TreeSHAP runs without a background (model_output='raw').
- GradientExplainer uses a per-replica background of 100 instances sampled
  from the replica's training window. Numeric features are scaled by the
  replica's own `StandardScaler` before explanation; binary features pass
  through. SHAP attributions therefore live in the replica's standardised
  space.

For LIME:
- The perturbation basis is per-window: the explainer for A replicas is built
  from `X[idx_A]` raw, the explainer for B replicas from `X[idx_B]` raw.
- The flagged subsample is shared across A and B within a pair so attributions
  remain instance-wise comparable across the cross-window comparison.
- All `predict_proba` callables receive **raw** input. The replica's own
  pipeline (XGBoost: raw passthrough; MLP-PLR: a callable that applies the
  replica scaler before forwarding) handles whatever scaling the model needs.
  LIME never pre-scales.
- The 6 binary indicators are declared as `categorical_features` so LIME
  samples them from the empirical distribution. `discretize_continuous=False`
  keeps numeric perturbation continuous.

### MLP-PLR SHAP and LIME share the same subsample

XGBoost SHAP runs on the full flagged set. MLP-PLR SHAP runs on a 200-instance
**rank-spread subsample** of the flagged set, constructed deterministically
from `(p_hat_A, p_hat_B)` with seed `SEED_BASE + pid * 100`. LIME uses the same
construction with the same seed and the same `N_LIME_SUBSAMPLE = 200`, so the
two explainers operate on byte-identical instance subsets per pair —
independently of which notebook ran first. The MLP-PLR SHAP subsample is saved
as `mlp_shap_subsample_idx.npy`; LIME's as `lime_flagged_idx.npy`.

---

## 11. `predictions.npz` — universal per-pair output

Saved by training notebooks under `data/models/{model_type}/pair_{pid:02d}/predictions.npz`.

| Key | Shape | Meaning |
|---|---:|---|
| `preds_A`, `preds_B` | `(R, n_eval)` | per-replica positive-class probabilities |
| `p_hat_A`, `p_hat_B` | `(n_eval,)` | replica-averaged probabilities |
| `flagged_idx` | `(n_flagged,)` | local positions within `idx_eval`, selected as the top `⌈K_FRAC · n_eval⌉` by `max(p_hat_A, p_hat_B)` ; saved in ascending local-index order |
| `Y_eval` | `(n_eval,)` | labels for the evaluation slice |
| `pr_auc_A`, `pr_auc_B`, `roc_auc_A`, `roc_auc_B` | scalar | aggregate metrics |
| `per_replica_pr_auc_A`, `per_replica_pr_auc_B` | `(R,)` | per-replica PR-AUCs |

The flagged set is model-specific (it depends on each model's predictions).
Within a (model, window pair, explainer) comparison, both versions A and B are
explained on the same flagged instances.

---

## 12. Per-model file conventions

Under `data/models/{mt}/pair_{pid:02d}/`:

| File | xgboost | logreg | mlp_plr |
|---|---|---|---|
| `replicas_{A,B}/model_r{r}.joblib` | XGBClassifier | sklearn Pipeline | bundle `{state_dict, arch_config, scaler}` |
| `replicas_{A,B}/training_log_r{r}.csv` | yes | no | yes (per-epoch loss + val PR-AUC) |
| `replicas_{A,B}/seeds_r{r}.json` | yes | yes | yes |
| `hparams_{A,B}.json` | yes | yes | yes |
| `reference_scaler.joblib` | yes | yes | yes |
| `predictions.npz` | yes | yes | yes |
| `coef_{A,B}.npy` | no | `(R, p)`, raw replica coefs in own basis | no |
| `coef_{A,B}_ref.npy` | no | `(R, p)`, replica coefs in reference basis | no |
| `coef_{A,B}_full_ref.npy` | no | `(p,)`, full-window coefs in reference basis | no |
| `full_model_{A,B}.joblib` | no | yes (full-window LR Pipeline) | no |
| `run_params.json` | yes | yes | yes |

Under `data/shap/{mt}/pair_{pid:02d}/` (xgboost / mlp_plr only):

| File | xgboost | mlp_plr |
|---|---|---|
| `shap_{A,B}.npy` | `(R, n_flagged, p)` | `(R, n_explained, p)` |
| `expected_values_{A,B}.npy` | `(R,)`, base value | `(R,)`, mean logit on background (diagnostic only) |
| `mlp_shap_subsample_idx.npy` | absent | `(n_explained,)` positions within `flagged_idx` |
| `shap_stoch_{A,B}.npy` | `(1, n_explained, p)`, deterministic | `(Q, n_explained, p)`, `Q = N_STOCH_RUNS` |
| `stochasticity.json` | `stoch_raw_runs_v1` run descriptor (see below) | same |
| `run_params.json` | yes | yes |

Under `data/lime/{mt}/pair_{pid:02d}/` (xgboost / mlp_plr only):

| File | Meaning |
|---|---|
| `lime_{A,B}.npy` | `(R, n_subset, p)` |
| `lime_flagged_idx.npy` | `(n_subset,)` positions within `flagged_idx` |
| `lime_stoch_{A,B}.npy` | `(Q, n_explained, p)`, `Q = N_STOCH_RUNS` |
| `stochasticity.json` | `stoch_raw_runs_v1` run descriptor (see below) |
| `run_params.json` | yes |

`stochasticity.json` schema (`schema_version = 'stoch_raw_runs_v1'`,
identical for SHAP and LIME):

```text
{
  schema_version,
  n_runs,                     # Q on disk
  run_seeds,                  # [] for TreeSHAP (Q=1); list of length Q otherwise
  replica_selection,          # 'central_prediction_on_explained_instances'
  replica_A, replica_B,       # central replica indices per side
  n_explained, n_features,
  explained_local_idx_sha1,   # SHA-1 of the explained local indices (int64)
  is_deterministic_runs       # True for TreeSHAP, False otherwise
}
```

This is a **run descriptor only**. All cosine / RBO aggregation is performed
by notebook 04 from the raw `*_stoch_{A,B}.npy` tensors; nothing aggregated is
stored on disk.

Under `data/results/`:

| Path | Meaning |
|---|---|
| `{mt}/drift_metrics.csv` | per-model-type drift table (one row per (pair, explainer)) |
| `combined/drift_metrics_combined.csv` | union over model types |
| `combined/drift_dashboard_{shap,lime}.png` | temporal diagnostic dashboard |
| `combined/lr_reference.png` | LR transparent reference figure: full-window A-vs-B coefficient trajectories + replica-noise panel |
| `{mt}/feature_importance_over_time_shap.png` | SHAP per-feature-over-time plot |

---

## 13. Seeding and replica generation

Convention with `SEED_BASE = 42`, pair id `pid`, replica index `r`:

| Source | A window | B window |
|---|---|---|
| bootstrap seed | `SEED_BASE + pid*10_000 + r*2` | `SEED_BASE + pid*10_000 + 5_000 + r*2` |
| model seed | `SEED_BASE + pid*10_000 + r*2 + 1` | `SEED_BASE + pid*10_000 + 5_000 + r*2 + 1` |
| Optuna study seed | `SEED_BASE + pid*10 + 1` | `SEED_BASE + pid*10 + 2` |
| MLP-PLR SHAP background | `SEED_BASE + pid*10_000 + (0 or 5_000) + r` | same formula, B uses 5_000 offset |
| LIME / MLP-SHAP subsample | `SEED_BASE + pid*100` | (shared across A/B per pair) |
| LIME explainer (A/B) | `SEED_BASE + pid*100 + 1 / + 2` | — |

Keep this aligned across model families. Replica `r` of pair `pid` sees the
same bootstrap sample across XGBoost, LR, and MLP-PLR.

Replicas use **class-stratified bootstrap** sampling to preserve the class
ratio of the window. For XGBoost and MLP-PLR, the chronologically last 15% of
each training window is held out as a fixed early-stopping validation tail
before bootstrapping; replicas bootstrap from the remaining portion. LR has no
early stopping and bootstraps from the full window.

GPU training (MLP-PLR) is not bit-reproducible even with fixed seeds. Low-level
cuDNN nondeterminism is accepted as part of retraining variability.

---

## 14. Shared helpers and code dependencies

### Helpers that must stay aligned across notebooks

- `stratified_bootstrap(idx, Y, seed)` — defined in 02 and 02b, byte-identical.
- `compute_flagged_topk(p_hat_A, p_hat_B, k_frac)` — defined in 02 and 02b,
  byte-identical. Uses `⌈k_frac · n⌉`.
- `build_subsample_idx(flagged, p_hat_A, p_hat_B, n_subsample, seed)` — defined
  in 03 (MLP-PLR branch) and 03b, byte-identical. This is what guarantees
  MLP-SHAP and LIME share their subsample.
- `PLREmbedding`, `MLPPLR` — defined in 02b, mirrored in 03 and 03b. Any
  architecture change in 02b must be propagated to both explanation notebooks.

### Notebook 04 metric helpers

- `cosine_distance(u, v)` and `rbo_distance(u, v)` — both-zero → `0.0`,
  exactly-one-zero → `np.nan` (excluded from `nanmedian` aggregations).
- `instance_dynamic_drift(phi_A, phi_B, dist_fn)` — `nanmean` over R×R
  cross-window replica pairs.
- `instance_baseline_instability(phi, dist_fn)` — `nanmedian` over `C(R,2)`
  same-window replica pairs.
- `rbo_global_drift(phi_bar_A, phi_bar_B)` — global SHAP-only drift.

The package `rbo` is a hard dependency for notebook 04; install with `pip install rbo`.

---

## 15. Within-explainer stochasticity diagnostic

Purpose: measure variability caused purely by the explainer process while
holding the trained model and explained instances fixed. This is a **separate**
sanity diagnostic — it does not replace cosine/RBO drift metrics.

Procedure:
- per side (A, B) pick the **central replica** — the replica whose predictions
  on the explained instances minimise MSE against the replica-averaged
  prediction `p_hat`;
- run the explainer `Q = N_STOCH_RUNS = 5` times on the same instances with
  fixed instance order, fixed feature order, fixed background, fixed
  hyperparameters; vary only the explainer's RNG (`STOCH_RUN_SEEDS = [1000,
  1001, 1002, 1003, 1004]`);
- save the raw `(Q, n_explained, p)` attribution tensor as
  `shap_stoch_{A,B}.npy` / `lime_stoch_{A,B}.npy`;
- save the run descriptor `stochasticity.json` (§12 schema). No aggregated
  numbers are stored — notebook 04 reads the raw tensors and computes
  median pairwise cosine / RBO distance over `C(Q,2)` run pairs per instance,
  then median over instances.

TreeSHAP has no RNG knob; it is treated as deterministic and saved with
`Q = 1`. Notebook 04 reports `stoch_cos = stoch_rbo = 0` for that row.

Expected behaviour:
- XGBoost + TreeSHAP: deterministic by construction.
- MLP-PLR + GradientExplainer: mildly stochastic.
- LIME: more stochastic, mostly from perturbation sampling and local-surrogate
  fitting.
- LR coefficient solvers: deterministic (no diagnostic; LR rows carry NaN in
  every `stoch_*` column).

---

## 16. Implementation choices worth not regressing

These are decisions that took back-and-forth to settle. Future agents and reviewers
should respect them unless the user explicitly reverses one.

- **Pre-offer-date filter on aggregations** (notebook 00): aggregations include
  only transactions with `date_diff ≥ 0` relative to the offer date. This
  strengthens the literal `ecom-offers.py` recipe, which does not enforce a
  strict pre-temporal-key bound.
- **Flagged-set size** uses `⌈k · |E_{A,B}|⌉` — methodology specifies ceil,
  code matches.
- **MLP-PLR SHAP subsample = 200, identical to LIME** (§10). Don't fall back
  to the full flagged set without a user instruction.
- **LIME `discretize_continuous=False`**. With heavy-tailed numeric features
  (e.g. spend aggregates) this can amplify LIME stochasticity for those
  features; report it as a known characteristic if the discussion goes there.
- **LR is a transparent reference, not a post-hoc XAI stream.** No SHAP, no
  LIME, no cross-window coefficient drift; just within-window coefficient
  instability + the full-window coefficient figure.
- **Per-replica scalers for LR and MLP-PLR.** Each replica fits its own
  `StandardScaler`. Saved coefficients (LR) and saved attributions (MLP-PLR)
  live in that replica's standardised basis. Don't claim a fixed reference
  basis; the methodology frames stability as end-to-end pipeline behaviour.
- **Per-pair `run_params.json` fingerprints** drive skip-if-current logic.
  Editing them changes recompute behaviour. Adding a knob to a notebook should
  also add it to the corresponding `run_params` dict.

---

## 17. Open work

1. **Implementations 2 and 3 — TabReD datasets.** Choose two further TabReD
   datasets and adapt notebook 00 (preprocessing) and where needed notebook 01
   (windowing) accordingly. Most downstream notebooks should not need
   structural changes — they consume `X.parquet`, `Y.npy`, `feature_names.json`,
   and `window_config.json`.
2. **Cross-dataset reporting.** Once 2 and 3 exist, decide whether the thesis
   reports per-dataset diagnostic profiles only, or also a meta-comparison
   across datasets.
3. **Sensitivity checks** if time permits: LIME `NUM_SAMPLES` sweep at one
   pair; MLP-PLR SHAP background size sweep at one pair. Currently sized as
   computational compromises (NUM_SAMPLES=1000, BG=100); if the thesis is
   going to claim stability at these sizes, sensitivity evidence helps.

---

## 18. Gotchas worth knowing

- `predictions.npz` is written with `np.savez_compressed`. If an old file is
  missing newer keys (e.g. `roc_auc_*`), the training notebook detects it and
  re-runs; manual deletion is rarely needed.
- For binary classifiers, LIME returns positive-class explanations under key
  `1` in `as_map()`. `extract_lime_vector` raises if absent. With flagged
  instances (top-K%) this should not trigger; if it does, that's a real signal
  worth investigating.
- SHAP `expected_value` can be scalar or list-like depending on SHAP/XGBoost
  versions; the XGBoost branch normalises with `if isinstance(..., list)`.
- LIME (`extract_lime_vector`) requests `num_features = n_features` so the
  returned attribution covers all features; features LIME drops as zero
  remain zero in the output vector.
- `drift_metrics_combined.csv` rows for LR carry NaN in `loc_*` and `global_rbo`
  by design but **do** populate `coef_dyn_cos` / `coef_dyn_rbo` and the
  `base_*` columns. The post-hoc dashboards (`drift_dashboard_*.png`) filter
  LR rows out; `lr_reference.png` is the LR-specific cross-window view.
- Notebook outputs may lag behind the intended methodology described here.
  This file is the intended target unless the user says otherwise.

---

## 19. Hard constraints for any AI/agent working on this pipeline

1. **Never execute notebook cells or project code on behalf of the user.**
   The user runs everything manually.
2. **Edit only notebooks/files named in the user's task.**
3. **Do not create new notebooks unless explicitly asked.**
4. **Preserve working code paths unless the requested change requires otherwise.**
5. **If actual notebook state contradicts this context, stop and ask the user.**
6. **Do not silently reconstruct missing prior work.**
7. **When editing code, minimise refactoring.** Prefer targeted changes over
   style rewrites.
8. **When done, summarise changes cell by cell or file by file.** Do not paste
   entire notebook cells unless the user asks.
9. **For notebook architecture changes, keep training and explanation notebooks
   compatible.** `PLREmbedding`, `MLPPLR`, `stratified_bootstrap`,
   `compute_flagged_topk`, and `build_subsample_idx` exist in multiple
   notebooks and must stay byte-identical.
10. **Do not write thesis claims about fixed SHAP/LIME reference sets.** The
    methodology explicitly frames explanation drift as end-to-end (per-replica
    scalers for LR/MLP-PLR; per-window LIME basis); don't reframe it as if
    there were a single shared reference frame.

---

## 20. Pipeline state awareness

The uploaded/current notebooks should match the following state. If you are
an AI reading this and see a contradiction between this file and the actual
notebook code, flag it to the user. Do not reconcile silently.

- `00_data_preprocessing.ipynb`: Shoppers preprocessing, TabReD-inspired with
  pre-offer-date filter on aggregations.
- `01_rolling_window_setup.ipynb`: Shoppers weekly windows, `L=5, S=2, H=2,
  R=8, K_FRAC=0.10`.
- `02_training_replicas.ipynb`: XGBoost or LR. LR additionally writes
  `coef_*_ref.npy` (per-replica coefficients in the common reference basis)
  and fits a deterministic full-window LR per pair, saved as
  `coef_*_full_ref.npy` plus `full_model_*.joblib`.
- `02b_training_replicas_mlp_plr.ipynb`: MLP-PLR with PLR numeric embeddings,
  per-replica scalers, chronological 15% early-stopping tail.
- `03_shap_explanations.ipynb`: TreeSHAP for XGBoost on the full flagged set;
  GradientExplainer for MLP-PLR on a 200-instance subsample shared with LIME;
  raises for `MODEL_TYPE='logreg'`.
- `03b_lime_explanations.ipynb`: LIME for XGBoost and MLP-PLR with per-window
  raw training basis, raw inputs throughout, binary features as categorical;
  raises for `MODEL_TYPE='logreg'`.
- `04_drift_analysis.ipynb`: temporal diagnostic profile — separate A and B
  baselines, SHAP-only global drift, per-explainer stochasticity from
  `*_stoch_{A,B}.npy` tensors, LR contributes within-window coefficient
  instability plus `coef_dyn_*` (full-window cross-window coefficient gap)
  and is excluded from the post-hoc explanation-drift dashboards. Renders the
  dashboards, the LR transparent-reference figure (`lr_reference.png`), and
  the per-feature SHAP-over-time plot.

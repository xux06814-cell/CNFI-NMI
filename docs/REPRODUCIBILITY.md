# Experiment settings

## Install and prepare

Python 3.10+ is supported. Install the maintained implementation and run its
formula and API checks:

```bash
python -m pip install -e ".[experiments,test]"
pytest
python scripts/download_public_data.py --output data/raw
python scripts/prepare_public_data.py --raw-dir data/raw --output data/processed
```

Archive hashes and preprocessing are defined in [data/README.md](../data/README.md).
The manuscript's exact medical cleaning and the traffic input-count difference
are recorded there alongside the actual processed dimensions.

## Full public-data protocol

Use a new result root for each independent run:

```bash
python scripts/run_paper_reproduction.py traffic --data-dir data/processed --output results/my-run
python scripts/run_paper_reproduction.py bank --data-dir data/processed --output results/my-run
python scripts/run_paper_reproduction.py cardio --data-dir data/processed --output results/my-run
```

Each command trains models on the supplied data. It does not load the bundled
reference measurements. `--resume` is explicit and checks the input, source,
thread count and environment contract before reusing the same run's records.

| Setting | Value |
|---|---|
| Selector | `graph_mode="paper"`; raw formulas in [method mapping](METHODOLOGY.md) |
| Support / confidence | .01 / .1 |
| CNFI bins | 5 equal-frequency, fitted on outer training data |
| Kappa / theta | 1 / 0; manuscript prior vectors |
| Outer evaluation | 5 stratified folds, shuffle, seed 42 |
| Inner LightGBM search | 3 stratified folds, seed 42 + outer fold |
| Learning rates / depths | .01, .05, .1 / 3, 5, 7 |
| Trees / classifier seed | 200 / 42 |
| Threads | 2 by default, configurable |
| Sensitivity | alpha 0:.1:1; top 12 fused features; all three datasets |
| Component ablations | top 4 features; traffic dataset |
| Classification threshold | .5; positive label 1 |
| Uncertainty | Mean and sample SD, ddof=1, over five outer folds |
| Network export | First outer training fold; alpha .5 fused node scores |

Tree count, exact grid, fold seeds and standalone component definitions are
implementation choices completing settings omitted from the manuscript.
The manuscript reports an older Python environment; executed library versions
are saved with each run.

The selector uses original category identities, bins only its CNFI branch,
and computes NMI on original observations. Table 2's Hour, Month and vehicle/
lane counts are numeric; SA2/3/4 area codes and Day are categorical. Medical
category codes are declared as categories. After selection, classifier encoding
and median filling are learned from outer training data; unseen categories
receive −1. Selection precedes inner tuning, so the inner search is not fully
nested. Each outer test fold is held out. The alpha sweep is exploratory;
choosing alpha from it requires a separate final evaluation.

## Component definitions

All CNFI component rows use the same feature nodes, qualifying rule pool and
maximum feature-pair aggregation. The manuscript does not specify the separate
Multi and Dynamic rows completely. This run completes them as follows:

| Row | Edge weights | Ranking |
|---|---|---|
| PGI (shared feature graph) | Confidence | PGI 3-WLR algebra |
| + Multi weight | Raw metrics weighted by prior midpoint [.25,.20,.25,.15,.15] | Same PGI ranking |
| + Dynamic thresholding | Confidence × (.10 + .30 × lambda) | Same PGI ranking |
| + Eigen-Rank | Confidence | Raw manuscript Eigen-Rank |
| Full CNFI | Complete adaptive raw metric fusion | Raw manuscript Eigen-Rank |
| SDMI | Empirical-state SDMI scores | Descending scores |
| NMI | Geometric empirical NMI | Descending scores |

The PGI algebra is adapted from the [authors' PGI-SDMI implementation](https://github.com/FengYinLeo/PGI-SDMI/tree/e09a449b8d90b25e9cd19da289dbb8324216c48d)
to this common feature graph; it is a component control, rather than the
original PGI transaction-to-node benchmark pipeline. No extra Lift≥1 filter is
introduced. The Dynamic row isolates the confidence coefficient in the paper's
prior interpolation. The midpoint removes rule-specific modulation in Multi.
Full CNFI is the unsupervised branch for this table, without NMI fusion.
The upstream PGI-SDMI [MIT notice](third-party/PGI-SDMI-LICENSE) is retained.

For SDMI, let `g_b=P(x_b,y)*ln(P(x_b,y)/(P(x_b)*P(y)))` across s observed
states. The implemented per-class contribution is
`sqrt(2*s*sum(g_b²)-2*sum(g_b)²)/(2*s)`, summed over target classes; zero
joint probabilities contribute zero. This computes the pairwise-difference
formula without allocating an s×s array.

## Outputs and other runners

Each dataset directory contains per-fold metrics and selected features,
mean/SD CSVs, exact train/test indices, rule tables, formula diagnostics,
adjacency, NMI, fused scores, source hashes and protocol metadata. The
[measured run](../reproduction/public-data-20261116/README.md) includes actual
results and their comparison with the manuscript. Plotting instructions are in
[FIGURES_AND_SENSITIVITY.md](FIGURES_AND_SENSITIVITY.md).

`scripts/run_experiment.py` remains a general five-fold, alpha=.5 runner. It
requires `--csv`, `--target`, `--positive-label` and supports `--drop` and
`--categorical`. `scripts/run_alpha_sensitivity.py` remains a lighter sweep
with fixed classifier settings (100 trees, rate .1, depth 3 by default);
it is distinct from the full grid-searched public-data protocol above.

# Figure 5 sensitivity and Figure 6 network visualization

The scripts produce **exploratory** results from a user-supplied CSV. They are
not a claim that the numerical results in the manuscript have been replicated.
The exact folds, model grid, feature preprocessing and data provenance must be
independently verified before comparison. No data or previously reported
numbers are bundled.

Install `python -m pip install -e ".[experiments]"`. Run the Python script once
for each dataset, for example:

```bash
python scripts/run_alpha_sensitivity.py --csv data/raw/traffic.csv --target x --positive-label 1 --features 12 --output "matlab/Traffic Dataset"
python scripts/run_alpha_sensitivity.py --csv data/raw/bank.csv --target y --positive-label yes --features 12 --output "matlab/Financial Dataset"
python scripts/run_alpha_sensitivity.py --csv data/raw/cardio.csv --target cardio --positive-label 1 --features 12 --drop id --output "matlab/Medical Dateset"
```

Examples are schematic: remove `--drop id` if the actual file
has no such column. Set the target, positive label, delimiters and dropped
identifiers from the actual dataset. The manuscript defines traffic target as
logical OR of `Emergency service(s)` and `Emergency services` and removes both
source fields from predictors; its table specifies 86 input features. Do not
use `Is Major Incident` as the target for the reported experiment. The exact
processed dataset in the present audit has 81,970 rows; the manuscript reports
85,611. The cause is unresolved, so this file is not a verified exact paper
artifact and is not redistributed here. Using an unknown target to approximate
paper figures is inappropriate.

Each outer training fold fits the repository's full `CNFINMISelector` once.
The script recombines its two full rank scores over alpha 0, 0.1, ..., 1.0;
models use fixed CLI classifier parameters (defaults: 100 trees, learning rate
0.1, max depth 3), with encoding fitted on training data only. Fold indices are
stratified, shuffled, seeded. No per-alpha hyperparameter selection is done.
Results are not claims that the paper used those fixed hyperparameters. Reusing
outer test folds to choose alpha and then presenting the same scores as an
independent test would be biased. Use a nested procedure or a separate test set
for a confirmatory comparison. Variability is sample SD across outer folds, not
a confidence interval. Integer-coded categorical columns are not automatically
identified: use `--categorical job marital education default housing loan contact month poutcome`
for the encoded bank file so these remain categories rather than numerical bins.
This declaration is recorded in the output metadata.

Each output folder contains `sensitivity.json` (actual folds, metrics, selected
features, data SHA-256, settings and package versions), `sensitivity_stats.csv`
(11 means and sample SDs), `adj_directed.csv`, `adj.csv`, `nmi.csv`, `rank.csv`,
and `feature_aliases.csv`. The network is fitted on one training fold, default
fold 1; `adj.csv` uses the maximum of two directed edge weights solely for the
undirected chord diagram. MATLAB will stop rather than invent random data if
any required file is absent. The graph reflects one fold, not a full-data graph.

Place the three folders below the `matlab/` directory, then run MATLAB R2019a or
later from any directory:

```matlab
run('matlab/Fig5_sensitivity.m')
run('matlab/Fig6_SharedLegend.m')
```

Fig 5 displays mean +/- sample SD, clipped to the feasible [0,1] range. Fig 6
reads the actual supplied adjacency, NMI and fused rank score; the chord layout
prunes weak edges, so it is a visualization rather than the full graph. The
complete directed adjacency is preserved separately. Each script writes PNG
and PDF output under `matlab/`. The adapted Figure 6 MATLAB template was
originally provided by the authors, and its no-data random fallback has been
removed. MATLAB aliases are recorded in `feature_aliases.csv`.

The existing `scripts/run_experiment.py` performs feature selection and encoding
on the entire outer training fold before inner CV; its outer test fold is held
out, but its inner validation sees information from the full outer training fold.
It is therefore not fully nested. Fix by putting all preprocessing and feature
selection inside the estimator fitted separately by each inner split before
using its inner scores for hyperparameter assessment. This figure script instead
uses fixed classifier parameters and does not perform inner tuning.

Original source files, licenses, the exact 85,611-to-81,970 row exclusion,
fold assignments, and complete hyperparameter grids are still needed for a
credible numerical reproduction. The dataset README records the target and feature count from manuscript Table 2
and section 4.1.1. Original medical and financial literature results are not
same-data/same-classifier baseline runs.

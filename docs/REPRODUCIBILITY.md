# Reproducibility protocol

## Environment

Use Python 3.10 or newer. A clean environment is recommended:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiments,test]"
pytest
```

The manuscript reports Python 3.8, PyTorch, scikit-learn, LightGBM, five-fold
cross-validation, five equal-frequency bins, minimum support 0.01, minimum
confidence 0.1, and alpha 0.5. Python 3.8 is preserved as historical metadata;
the maintained package targets supported Python releases from 3.10 onward.

## Leakage control

Feature encoding, CNFI-NMI fitting, and model tuning must be learned from the
training fold only. `scripts/run_experiment.py` implements a stratified outer
five-fold evaluation and a three-fold inner grid search. Test-fold data are
never used to choose features or classifier hyperparameters.

Example:

```bash
python scripts/run_experiment.py \
  --csv data/raw/bank-full.csv \
  --target y \
  --features 12 \
  --output results/bank.json
```

## What can and cannot currently be reproduced

The algorithm, equations, fixed priors, synthetic example, and leakage-safe
evaluation protocol are executable and tested. The exact numerical figures in
the paper are not claimed as reproduced by this repository revision because
the exact raw dataset archives, checksums, preprocessing scripts, fold seeds,
and complete LightGBM grids were not embedded in the supplied manuscript or
the original repository. Publishing guessed substitutes would weaken, rather
than improve, scientific reproducibility.

To make the numerical claims fully reproducible, the authors should add the
items listed in `data/README.md`, then commit immutable result JSON files and a
script that regenerates every table and figure from those files.

## Reporting checklist

- Record package versions and operating system.
- Record raw-data URLs, access dates, licenses, and SHA-256 hashes.
- Fix and report all random seeds.
- Preserve per-fold selected features, model parameters, and metrics.
- Report mean and sample standard deviation over the five outer folds.
- Separate exploratory parameter tuning from final confirmatory evaluation.
- Do not compare newly generated metrics with the paper unless the dataset and
  preprocessing hashes match.

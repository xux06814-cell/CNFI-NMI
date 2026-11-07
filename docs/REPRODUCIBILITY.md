# Experiment settings

## Installation and environment

The maintained package requires Python 3.10 or newer. The manuscript reports Python 3.8, PyTorch and scikit-learn (p. 10); this is a maintained reimplementation, not an exact reconstruction of that environment. The selector does not require PyTorch.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[experiments,test]"
pytest
```

## Inputs and outer evaluation

Supply your processed CSV, target and exact positive-class label:

```bash
python scripts/run_experiment.py --csv path/to/processed.csv --target target --positive-label 1 --features 12 --seed 42 --output results/evaluation.json
```

Set `--sep`, `--drop` and `--categorical` for your schema. For example, declare integer-coded Bank Marketing categories:

```text
--categorical job marital education default housing loan contact month poutcome
```

Exclude identifiers and target-source columns. Missing targets are rejected. The positive label is required so precision, recall, F1, specificity and AUC consistently refer to the intended event.

The runner performs five stratified outer folds. Within each outer training fold, the selector sees original observations and category identities. CNFI alone discretizes continuous values; NMI uses original values. After selection, training-derived median filling and categorical encoding prepare selected columns for LightGBM; unknown test categories receive -1. An all-missing numeric training column receives a classifier-side fill of 0.

Preprocessing and feature selection precede three-fold inner LightGBM search; each outer test fold remains held out. Inner tuning is therefore not fully nested.

| Setting | Value | Status |
|---|---|---|
| Graph mode | `normalized` | Paper Eqs. 7-22; default and explicit runner setting |
| Support / confidence | .01 / .1 | Paper p. 10 |
| CNFI bins | 5 equal-frequency | Paper p. 10 |
| Alpha | .5 | Paper Eq. 31 |
| Kappa / theta, priors | Table 1 | Fixed across datasets |
| k_max | 12 by runner default; configurable | Paper gives 12 as an example; Table 5 uses 4 |
| Learning-rate grid | .01, .05, .1 | Repository grid within reported range |
| Maximum-depth grid | 3, 5, 7 | Repository grid within reported range |
| Trees | 200 | Repository choice; not specified in paper |
| Seed / inner folds | 42 / 3 | Repository choices; not specified in paper |
| Threads | 2; `--threads` configurable | Runtime choice, not a paper setting |

Output JSON records graph mode, selector parameters, input SHA-256, positive label, schema declarations, software versions, positional train/test indices, per-fold feature selections, classifier parameters, metrics, means and sample SD (`ddof=1`). Save the code commit and exact command alongside it.

## Scope of reproduction

Formula tests and small CLI fixtures check the implementation. Processed benchmark inputs, original fold assignments, the complete original grid and per-fold result archives are not bundled, so the reported scores have not been independently reproduced.

The main runner evaluates CNFI-NMI. It does not recreate PGI-SDMI, SDMI, MFS, PSO or Table 5 component ablations. Alpha=0/1 in the sweep covers pure NMI/CNFI endpoints, but not those other methods. Their full specifications and measured outputs are required before claiming table/figure reproduction.

See [dataset provenance](../data/README.md) and [figure/sweep instructions](FIGURES_AND_SENSITIVITY.md).

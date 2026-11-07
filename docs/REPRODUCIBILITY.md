# Experiment settings

## Installation

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[experiments,test]"
pytest
```

## Inputs and evaluation

Supply your processed comma-separated CSV and target column:

```bash
python scripts/run_experiment.py --csv path/to/processed.csv --target target --features 12 --seed 42 --output results/evaluation.json
```

The script uses stratified five-fold outer evaluation and three-fold inner
LightGBM grid search. Encoding and feature selection are fitted on the entire
outer training fold before inner search; inner validation is not fully nested.
The outer test fold is held out.

| Setting | Value |
|---|---|
| Support / confidence | 0.01 / 0.1 |
| CNFI bins | 5 equal-frequency bins for continuous inputs |
| Alpha | 0.5 |
| Kappa / theta | 1.0 / 0.0 |
| Learning-rate grid | 0.01, 0.05, 0.1 |
| Maximum-depth grid | 3, 5, 7 |
| Trees | 200 |
| Default seed | 42 |

The runner saves per-fold feature selections, classifier parameters and metrics,
plus the mean and sample standard deviation across folds. Its grid and seed
are the repository defaults; the manuscript's complete grid and original fold
assignments are not available. Exact numerical reproduction remains unverified.

See [dataset inputs](../data/README.md) for target definitions, and
[figure instructions](FIGURES_AND_SENSITIVITY.md) for the separate alpha sweep.
Record the input hash, environment versions and command alongside each run.
The manuscript's medical and financial literature comparators retain their
original datasets and classifiers.

## Evaluation status and graph modes

The authors confirm that the revised normalized formulation has been evaluated.
Earlier repository statements describing it as unevaluated are outdated.

The selector exposes two graph modes:

| Mode | Rule metrics and graph operator |
|---|---|
| `normalized` | Min-max scaled rule metrics; column-stochastic augmented operator and lazy power iteration, as described in the revised manuscript |
| `benchmark` (API default) | Raw rule metrics and the retained augmented spectral implementation |

Select the evaluated normalized formulation explicitly when constructing the
selector:

```python
selector = CNFINMISelector(
    n_features_to_select=12,
    graph_mode="normalized",
)
```

The current experiment and alpha-sensitivity scripts instantiate the selector
with its default graph mode. To run the normalized formulation through those
scripts, set `graph_mode="normalized"` in their selector construction. Record
the graph mode, code revision, processed-data hash, fold assignments and metrics
with each run. The numerical-reproduction limitations described above still
apply to the materials currently distributed in this repository.

## Tests

Unit tests use fixed fixtures to check equations and API behavior. They do not
produce dataset benchmark results.

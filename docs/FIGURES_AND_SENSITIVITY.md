# Sensitivity analysis and MATLAB figures

## Generate your results

Install `python -m pip install -e ".[experiments]"`. Run the sweep on each
of your processed datasets:

```bash
python scripts/run_alpha_sensitivity.py --csv path/to/traffic.csv --target x --positive-label 1 --features 12 --output "results/Traffic Dataset"
python scripts/run_alpha_sensitivity.py --csv path/to/bank.csv --target y --positive-label 1 --features 12 --output "results/Financial Dataset"
python scripts/run_alpha_sensitivity.py --csv path/to/cardio.csv --target cardio --positive-label 1 --features 12 --output "results/Medical Dateset"
```

Set `--target`, `--positive-label` and `--sep` to match your CSV. Use
`--drop` to exclude identifiers and target-source columns, and
`--categorical` to declare integer-coded categories. For encoded Bank Marketing:

```text
--categorical job marital education default housing loan contact month poutcome
```

Traffic target construction and reported dataset dimensions are in
[dataset inputs](../data/README.md). Each run reads the supplied CSV and writes
its own outputs; no previous experiment results are supplied.

## Sweep settings and outputs

The script fits both full feature rankings on each outer training fold and
scans alpha from 0 to 1 in steps of 0.1. Defaults are five stratified folds,
seed 42, 100 LightGBM trees, learning rate 0.1 and maximum depth 3.
Classifier parameters stay fixed during the sweep. The test-fold sweep is
exploratory; use separate validation to select alpha for a final evaluation.

| File | Content |
|---|---|
| `sensitivity.json` | Per-fold metrics, selected features, input SHA-256, settings and versions |
| `sensitivity_stats.csv` | Mean and sample SD at each alpha |
| `adj_directed.csv` | Directed network from the selected training fold, default fold 1 |
| `adj.csv` | Maximum of the two directed weights for undirected plotting |
| `nmi.csv`, `rank.csv` | Feature relevance and fused scores |
| `feature_aliases.csv` | Original feature names and MATLAB aliases |

## Plot your result directory

Under the result root, use the three dataset folder names shown above.
The spelling `Medical Dateset` is retained for template compatibility.
In MATLAB R2019a or later, explicitly select the root containing your outputs:

```matlab
setenv('CNFI_RESULTS_DIR', 'C:/your-project/results')
run('matlab/Fig5_sensitivity.m')
run('matlab/Fig6_SharedLegend.m')
```

Figure 5 shows means with sample SD shading, clipped to [0,1].
Figure 6 uses the supplied network and feature scores, prunes weak edges for
display, and normalizes node encodings within each dataset. Full directed
adjacency remains in its CSV. Both scripts write PNG and PDF files to the
selected result root and require all input files to be present.

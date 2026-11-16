# Sensitivity analysis and MATLAB figures

## Generate your results

For the grid-searched public-data sensitivity and traffic ablation run, use
`scripts/run_paper_reproduction.py` as shown in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The commands below retain the lighter,
fixed-classifier sweep for user-supplied CSVs.

Install `python -m pip install -e ".[experiments]"`. Run the sweep on each
of your processed datasets:

```bash
python scripts/run_alpha_sensitivity.py --csv path/to/traffic.csv --target target --positive-label 1 --features 12 --output "results/Traffic Dataset"
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
its own outputs.

## Sweep settings and outputs

The script uses the `paper` graph mode and fits both full feature rankings on each outer training fold and
scans alpha from 0 to 1 in steps of 0.1. Defaults are five stratified folds,
seed 42, 100 LightGBM trees, learning rate 0.1 and maximum depth 3.
Classifier parameters stay fixed during the sweep. The test-fold sweep is
exploratory; use separate validation to select alpha for a final evaluation.

| File | Content |
|---|---|
| `sensitivity.json` | Per-fold metrics, selected features, fold indices, graph mode, input SHA-256, settings and versions |
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
repo = 'C:/your-project/CNFI-NMI';
setenv('CNFI_RESULTS_DIR', fullfile(repo, 'results', 'my-run'));
run(fullfile(repo, 'matlab', 'Fig5_sensitivity.m'));
run(fullfile(repo, 'matlab', 'Fig6_SharedLegend.m'));
run(fullfile(repo, 'matlab', 'Table5_ablation.m')); % Full traffic ablation run
```

Figure 5 shows means with sample SD shading, clipped to [0,1].
Figure 6 uses the supplied network and feature scores, prunes weak edges for
display, and normalizes node encodings within each dataset. Full directed
adjacency remains in its CSV. Both scripts write PNG and PDF files to the
selected result root and require all input files to be present.

Figure 6 retains normal-weight panel captions. In its medical panel the four
features discussed in the manuscript (`ap_hi`, `ap_lo`, `bp_category`,
`bp_category_encoded`) have bold labels. This annotation does not change scores,
edge weights or feature selection. `Table5_ablation.m` plots the actual seven
component rows with sample-SD error bars.

The [measured public-data bundle](../reproduction/public-data-20261116/README.md)
contains the three exported PNG/PDF figures and their exact plotting CSVs.
`docs/assets/feature-networks.png` remains the manuscript's reference image.

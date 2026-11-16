# Dataset inputs

Download original-source archives, then generate processed CSVs locally:

```bash
python scripts/download_public_data.py --output data/raw
python scripts/prepare_public_data.py --raw-dir data/raw --output data/processed
```

Use `--datasets bank` (or `traffic cardio`) to select a subset. Both scripts
check the archive SHA-256 against the pinned release. If a source requires
authentication, obtain that archive through its dataset page and save it as
`traffic.zip`, `bank.zip` or `cardio.zip` in the raw directory. A changed archive
must be reviewed before changing its expected hash.

| Domain | Source | Manuscript | This public-data preparation |
|---|---|---|---|
| Traffic | [SydneyIncidents authors](https://github.com/Future-Mobility-Lab/SydneyIncidents/tree/dac71f40900bdd136e1fb1958b07dd70b4d4b2e3), `merged-unlabeled.zip` | 86 inputs; row count unspecified | 85,611 rows, 85 inputs |
| Financial | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), complete `bank-full.csv` | 45,211 rows, 16 inputs | 45,211 rows, 16 inputs |
| Medical | [Kaggle Cardiovascular Disease](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset), `cardio_train.csv` | 68,205 rows, 15 inputs | 68,613 rows, 15 inputs |

The financial schema matches the reported data dimensions. Traffic's 87 source
columns become 85 predictors after combining two source columns into one target
and excluding both sources. Medical cleaning bounds are not fully specified in
the manuscript; the explicit reconstruction below produces 68,613 rows.

## Processing

Traffic: strip whitespace from column names; construct binary `target` as the
OR of `Emergency service(s)` and `Emergency services`; remove both source
columns. Retain author-published rows, original categories and missing
predictors. Missing-value handling is fitted within outer training folds.

Bank: read the full `bank-full.csv` from nested `bank.zip`, with semicolon
separator; map `y=yes` to 1 and `no` to 0. Preserve `unknown` as a category.
The smaller sampled `bank.csv` is not used. The processed table is also
provided in [bank_processed.csv.zip](bank_processed.csv.zip).

Medical: remove `id`; retain `80≤ap_hi≤250`, `40≤ap_lo≤200`,
`ap_hi≥ap_lo`, `120≤height≤220`, `30≤weight≤200`. Keep the original `age`
and add `age_years=floor(age/365.25)`, `bmi=weight/(height/100)^2`.
Assign `bp_category` in this order: stage2 if `ap_hi≥140` or `ap_lo≥90`;
stage1 if `ap_hi≥130` or `ap_lo≥80`; elevated if `ap_hi≥120`; otherwise
normal. Add its duplicate integer representation `bp_category_encoded`
(normal=0, elevated=1, stage1=2, stage2=3). The target is `cardio`.

These medical bounds and category definitions are reconstruction choices,
fixed before evaluation. See the generated `data_provenance.json` for actual
dimensions, class counts and processed file hashes.

## Archive SHA-256

```text
traffic f2d8fb9840916c9b1a81e27b3049f4b0b2e6cbccd74622157001a21692d705b5
bank    e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4
cardio  b78ff607480810c3e27de3e11d52c4962a3a81dbb3d69365e1cf3394a67c2b23
```

## Attribution and redistribution

The bundled processed Bank table is derived from UCI Bank Marketing, by
S. Moro, P. Rita and P. Cortez, [DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306).
The dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The only change to the original table is binary target encoding as described
above; retain this attribution when redistributing it.

Traffic and medical row-level tables are obtained from their source pages;
this repository provides the downloader, preprocessing code, hashes and
aggregate experiment outputs. Follow each source's dataset usage terms.

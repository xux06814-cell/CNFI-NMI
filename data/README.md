# Dataset inputs

Place your processed CSV files under `data/raw/` or supply an absolute path.
Each file contains predictors and one binary target column. Exclude identifiers
and columns used directly to construct the target.

| Domain | Dataset in the manuscript | Target | Reported dimensions |
|---|---|---|---|
| Traffic | Sydney Traffic Incident Dataset, Jan. 2017–July 2022 | OR of `Emergency service(s)` and `Emergency services` | 85,611 rows, 86 inputs |
| Medical | Cardiovascular Disease Dataset | `cardio` | 68,205 rows, 15 inputs after excluding `id` and the target |
| Financial | UCI Bank Marketing | `y` | 45,211 rows, 16 inputs |

For traffic, remove both emergency-service source columns after constructing
the target. Use the target column name present in your processed file.

Bank Marketing source: https://archive.ics.uci.edu/dataset/222/bank+marketing

For each dataset, record its source URL, access date, usage terms, file SHA-256,
preprocessing steps, final dimensions and class counts. The source archives and
complete preprocessing pipeline for the manuscript are not bundled.

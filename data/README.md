# Dataset provenance and local layout

The manuscript evaluates three public datasets, but their raw files are not
redistributed here because the repository does not currently contain the exact
source archives, licenses, or checksums used by the authors. Place authorized
copies under `data/raw/`; this directory is ignored by Git.

| Domain | Manuscript description | Expected target | Reported dimensions |
|---|---|---|---|
| Transportation | Sydney Traffic Incident Dataset, Jan. 2017-Jul. 2022 | `emergency service` (OR of `Emergency service(s)` and `Emergency services`) | 86 inputs plus target |
| Medical | Cardiovascular Disease Dataset | dataset-specific binary label | 68,205 rows, 15 features after preprocessing |
| Financial | UCI Bank Marketing | `y` | 45,211 rows, 16 features |

Before publishing benchmark numbers, record for every dataset:

1. the authoritative download URL and access date;
2. the raw archive SHA-256 checksum;
3. every inclusion, exclusion, missing-value, and encoding rule;
4. the final row/column counts and class distribution;
5. the exact command and software environment used to produce each result.

The Bank Marketing dataset is described by the UCI Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/222/bank+marketing>. The two other exact
artifacts must be confirmed by the authors before automated download scripts
are added; similarly named public datasets are not necessarily equivalent.

The manuscript describes 85,611 traffic incidents; a locally found processed
file contains 81,970 rows. Its exact exclusion procedure has not been verified.
Neither file should be presented as the exact manuscript input without further
provenance and benchmark verification.

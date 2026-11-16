"""Check recorded folds, mean/SD tables, figure inputs and published file hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'auc']
DATASETS = ['Traffic Dataset', 'Financial Dataset', 'Medical Dateset']


def check_table(root, stem, group, summary_name, expected_rows):
    frame = pd.read_json(root / f'{stem}_folds.json')
    assert len(frame) == expected_rows
    if group == 'alpha':
        frame['alpha'] = frame.alpha.round(1)
    assert not frame.duplicated(['fold', group]).any()
    assert set(frame.fold) == set(range(1, 6))
    assert np.isfinite(frame[METRICS]).all().all()
    assert frame[METRICS].ge(0).all().all() and frame[METRICS].le(1).all().all()
    assert frame.groupby(group).size().eq(5).all()
    stats = pd.read_csv(root / summary_name).set_index(group)
    if group == 'alpha':
        stats.index = stats.index.round(1)
    computed = frame.groupby(group)[METRICS]
    for metric in METRICS:
        np.testing.assert_allclose(stats[metric], computed.mean()[metric].reindex(stats.index), atol=5e-9, rtol=0)
        np.testing.assert_allclose(stats[metric + '_sd'], computed.std(ddof=1)[metric].reindex(stats.index), atol=5e-9, rtol=0)
    if group == 'alpha':
        np.testing.assert_allclose(sorted(frame.alpha.unique()), np.linspace(0, 1, 11))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    for folder in DATASETS:
        root = args.root / folder
        check_table(root, 'sensitivity', 'alpha', 'sensitivity_stats.csv', 55)
        directed = pd.read_csv(root / 'adj_directed.csv', index_col=0)
        display = pd.read_csv(root / 'adj.csv', index_col=0)
        assert list(directed.index) == list(directed.columns)
        np.testing.assert_allclose(display, np.maximum(directed, directed.T))
        assert set(directed.index) == set(pd.read_csv(root / 'nmi.csv').Feature)
        assert set(directed.index) == set(pd.read_csv(root / 'rank.csv').Feature)
        for fold in range(1, 6):
            indices = pd.read_csv(root / f'indices_fold{fold}.csv')
            train = set(indices.train.dropna().astype(int))
            test = set(indices.test.dropna().astype(int))
            assert not train & test
            assert train | test == set(range(len(train) + len(test)))
            audit = json.loads((root / f'formula_audit_fold{fold}.json').read_text())
            assert audit['full_cnfi']['converged']
    check_table(args.root / DATASETS[0], 'ablation', 'method', 'ablation_summary.csv', 35)
    count = 0
    for line in (args.root / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        assert hashlib.sha256((args.root / relative).read_bytes()).hexdigest() == expected, relative
        count += 1
    print(f'Passed: 165 sensitivity records, 35 ablation records, three networks, {count} file hashes.')


if __name__ == '__main__':
    main()

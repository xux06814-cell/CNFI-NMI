"""Prepare the three public archives using the documented reconstruction schema."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from download_public_data import SOURCES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(dataset, path):
    with zipfile.ZipFile(path) as archive:
        if dataset == 'traffic':
            frame = pd.read_csv(archive.open('merged-unlabeled.csv'), low_memory=False)
            frame.columns = frame.columns.str.strip()
            cols = ['Emergency service(s)', 'Emergency services']
            if not frame[cols].isin([0, 1]).all().all():
                raise ValueError('Expected binary emergency-service fields')
            frame['target'] = frame[cols].max(axis=1).astype(int)
            return frame.drop(columns=cols), 'target'
        if dataset == 'bank':
            with zipfile.ZipFile(io.BytesIO(archive.read('bank.zip'))) as nested:
                frame = pd.read_csv(nested.open('bank-full.csv'), sep=';')
            if not frame.y.isin(['yes', 'no']).all():
                raise ValueError('Unexpected Bank target labels')
            frame['y'] = (frame.y == 'yes').astype(int)
            return frame, 'y'
        frame = pd.read_csv(archive.open('cardio_train.csv'), sep=';')
    keep = (frame.ap_hi.between(80, 250) & frame.ap_lo.between(40, 200)
            & (frame.ap_hi >= frame.ap_lo) & frame.height.between(120, 220)
            & frame.weight.between(30, 200))
    frame = frame.loc[keep].drop(columns='id').copy()
    frame['age_years'] = np.floor(frame.age / 365.25).astype(int)
    frame['bmi'] = frame.weight / (frame.height / 100)**2
    code = np.select([(frame.ap_hi >= 140) | (frame.ap_lo >= 90),
                      (frame.ap_hi >= 130) | (frame.ap_lo >= 80),
                      frame.ap_hi >= 120], [3, 2, 1], default=0)
    frame['bp_category'] = np.asarray(['normal', 'elevated', 'stage1', 'stage2'])[code]
    frame['bp_category_encoded'] = code
    return frame, 'cardio'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for dataset in args.datasets:
        source = SOURCES[dataset]
        path = args.raw_dir / f'{dataset}.zip'
        if digest(path) != source['sha256']:
            raise ValueError(f'{dataset}: archive differs from the documented release')
        frame, target = prepare(dataset, path)
        output = args.output / f'{dataset}.csv'
        frame.to_csv(output, index=False)
        provenance[dataset] = {**source, 'processed_sha256': digest(output),
                               'rows': len(frame), 'inputs': frame.shape[1] - 1,
                               'target': target, 'positive_label': 1,
                               'class_counts': {str(k): int(v) for k, v in frame[target].value_counts().items()},
                               'processing_definition': 'data/README.md'}
        print(dataset, provenance[dataset]['rows'], 'rows,', provenance[dataset]['inputs'], 'inputs')
    (args.output / 'data_provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

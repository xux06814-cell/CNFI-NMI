"""Download pinned public-source archives; verify their SHA-256 before use."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
import zipfile

SOURCES = {
    'traffic': {
        'url': 'https://raw.githubusercontent.com/Future-Mobility-Lab/SydneyIncidents/dac71f40900bdd136e1fb1958b07dd70b4d4b2e3/data/merged-unlabeled.zip',
        'source_page': 'https://github.com/Future-Mobility-Lab/SydneyIncidents',
        'sha256': 'f2d8fb9840916c9b1a81e27b3049f4b0b2e6cbccd74622157001a21692d705b5',
    },
    'bank': {
        'url': 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',
        'source_page': 'https://archive.ics.uci.edu/dataset/222/bank+marketing',
        'sha256': 'e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4',
    },
    'cardio': {
        'url': 'https://www.kaggle.com/api/v1/datasets/download/sulianova/cardiovascular-disease-dataset',
        'source_page': 'https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset',
        'sha256': 'b78ff607480810c3e27de3e11d52c4962a3a81dbb3d69365e1cf3394a67c2b23',
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        source = SOURCES[dataset]
        path = args.output / f'{dataset}.zip'
        if not path.exists():
            partial = path.with_suffix('.zip.part')
            request = Request(source['url'], headers={'User-Agent': 'CNFI-NMI-reproduction/0.1'})
            with urlopen(request, timeout=120) as response, partial.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != source['sha256']:
                raise ValueError(f'{dataset}: downloaded content has a different SHA-256; inspect the source release')
            if not zipfile.is_zipfile(partial):
                raise ValueError(f'{dataset}: expected a ZIP archive')
            partial.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f'{dataset}: existing archive has a different SHA-256')
        print(dataset, source['sha256'])
    (args.output / 'download_sources.json').write_text(
        json.dumps({key: SOURCES[key] for key in args.datasets}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

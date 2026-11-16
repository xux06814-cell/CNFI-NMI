"""Public-data runner interface and an independent SDMI algebra check."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('lightgbm')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_paper_reproduction import sdmi_order


def test_sdmi_vectorization_matches_explicit_pairwise_differences():
    frame = pd.DataFrame({'a': list('aaabbbcc'), 'b': list('abababab'),
                          'constant': ['x'] * 8})
    target = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    scores = {}
    for col in frame:
        states = list(pd.unique(frame[col]))
        score = 0.
        for label in [0, 1]:
            terms = []
            for state in states:
                joint = np.mean((frame[col] == state) & (target == label))
                marginal = np.mean(frame[col] == state) * np.mean(target == label)
                terms.append(joint * np.log(joint / marginal) if joint else 0.)
            squared = sum((a - b)**2 for a in terms for b in terms)
            score += np.sqrt(squared) / (2 * len(states))
        scores[col] = score
    expected = sorted(frame, key=lambda c: -scores[c])
    assert sdmi_order(frame, target) == expected
    assert scores['constant'] == 0


def test_public_runner_help_describes_fresh_output_and_explicit_resume():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_paper_reproduction.py'),
                             '--help'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--data-dir' in result.stdout and '--output' in result.stdout
    assert '--resume' in result.stdout


def test_public_runner_rejects_nonempty_output_before_reading_data(tmp_path):
    output = tmp_path / 'run' / 'Financial Dataset'
    output.mkdir(parents=True)
    (output / 'sensitivity_folds.json').write_text('[]')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_paper_reproduction.py'),
                             'bank', '--data-dir', str(tmp_path / 'no-input'),
                             '--output', str(tmp_path / 'run')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'Output already contains files' in result.stderr
    assert (output / 'sensitivity_folds.json').read_text() == '[]'

"""Small CLI smoke checks; fixtures are not manuscript benchmark data."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def csv_path(tmp_path):
    rows = np.arange(60)
    frame = pd.DataFrame({
        "continuous": rows / 10,
        "category_code": rows % 8,
        "partner": (rows // 2) % 2,
        "target": np.where(rows % 2, "disease", "healthy"),
    })
    path = tmp_path / "fixture.csv"
    frame.to_csv(path, index=False)
    return path


def run_cli(script, *args):
    result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_five_fold_cli_records_paper_mode_and_disjoint_splits(csv_path, tmp_path):
    output = tmp_path / "evaluation.json"
    run_cli("scripts/run_experiment.py", "--csv", csv_path,
            "--target", "target", "--positive-label", "disease", "--features", 2,
            "--categorical", "category_code", "--threads", 1, "--output", output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["graph_mode"] == "normalized"
    assert result["selector_parameters"]["graph_mode"] == "normalized"
    assert result["positive_label"] == "disease"
    held_out = []
    for fold in result["folds"]:
        assert not set(fold["train_indices"]) & set(fold["test_indices"])
        held_out.extend(fold["test_indices"])
        assert len(fold["selected_features"]) == 2
        assert 0 <= fold["auc"] <= 1
    assert sorted(held_out) == list(range(60))


def test_sweep_exports_directed_graph_and_paper_metadata(csv_path, tmp_path):
    output = tmp_path / "sweep"
    run_cli("scripts/run_alpha_sensitivity.py", "--csv", csv_path,
            "--target", "target", "--positive-label", "disease", "--features", 2,
            "--categorical", "category_code", "--n-estimators", 2,
            "--threads", 1, "--output", output)
    result = json.loads((output / "sensitivity.json").read_text(encoding="utf-8"))
    assert result["graph_mode"] == "normalized"
    assert len(result["records"]) == 55
    assert len(result["split_indices"]) == 5
    stats = pd.read_csv(output / "sensitivity_stats.csv")
    np.testing.assert_allclose(stats.alpha, np.linspace(0, 1, 11))
    for metric in ["accuracy", "precision", "recall", "f1", "specificity", "auc"]:
        group = pd.DataFrame(result["records"]).groupby("alpha")[metric]
        np.testing.assert_allclose(stats[metric], group.mean())
        np.testing.assert_allclose(stats[metric + "_sd"], group.std(ddof=1))
    directed = pd.read_csv(output / "adj_directed.csv", index_col=0)
    undirected = pd.read_csv(output / "adj.csv", index_col=0)
    np.testing.assert_allclose(undirected, np.maximum(directed, directed.T))


def test_quickstart_uses_the_paper_defaults(csv_path):
    result = run_cli("examples/quickstart.py", "--csv", csv_path,
                     "--target", "target", "--features", 2,
                     "--categorical", "category_code")
    assert "Transformed shape: (60, 2)" in result.stdout


def test_evaluation_rejects_missing_targets(csv_path, tmp_path):
    frame = pd.read_csv(csv_path)
    frame.loc[0, "target"] = None
    frame.to_csv(csv_path, index=False)
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_experiment.py"), "--csv", str(csv_path),
        "--target", "target", "--positive-label", "disease",
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "target contains missing values" in result.stderr

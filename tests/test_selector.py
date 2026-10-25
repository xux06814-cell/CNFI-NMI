import numpy as np
import pandas as pd
import pytest

from cnfi_nmi import CNFINMISelector


def _sample_data(n=120):
    rng = np.random.default_rng(7)
    signal = rng.integers(0, 2, n)
    partner = np.where(rng.random(n) < 0.9, signal, 1 - signal)
    return (
        pd.DataFrame(
            {
                "signal": signal,
                "partner": partner,
                "noise_a": rng.normal(size=n),
                "noise_b": rng.integers(0, 5, n),
            }
        ),
        pd.Series(signal, name="target"),
    )


def test_rank_normalization_matches_equations_27_and_28():
    scores = CNFINMISelector._rank_scores(["a", "b", "c"])
    assert scores == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3), "c": 0.0}


def test_fit_transform_exposes_auditable_outputs():
    X, y = _sample_data()
    selector = CNFINMISelector(
        n_features_to_select=2,
        min_support=0.05,
        min_confidence=0.1,
    )
    transformed = selector.fit_transform(X, y)

    assert transformed.shape == (len(X), 2)
    assert list(transformed.columns) == selector.selected_features_
    assert set(selector.graph_.nodes) == set(X.columns)
    assert set(selector.ranking_) == set(X.columns)
    assert selector.get_support().sum() == 2
    assert selector.rules_["weight"].notna().all()

    rule = selector.rules_.iloc[0]
    expected_lambda = 1.0 / (1.0 + np.exp(-np.log2(rule["lift"])))
    assert rule["lambda"] == pytest.approx(expected_lambda)
    strong = np.array([0.40, 0.20, 0.10, 0.20, 0.10])
    weak = np.array([0.10, 0.20, 0.40, 0.10, 0.20])
    metrics = rule[
        ["confidence", "lift", "support", "conviction", "leverage"]
    ].to_numpy(float)
    expected_weight = (
        expected_lambda * strong + (1 - expected_lambda) * weak
    ) @ metrics
    assert rule["weight"] == pytest.approx(expected_weight)


def test_geometric_nmi_is_one_for_identical_nonconstant_labels():
    selector = CNFINMISelector()
    values = pd.Series([0, 0, 1, 1, 2, 2])
    assert selector._geometric_nmi(values, values) == pytest.approx(1.0)


def test_target_column_compatibility_path():
    X, y = _sample_data()
    combined = X.assign(outcome=y)
    selector = CNFINMISelector(
        target_col="outcome",
        n_features_to_select=2,
    ).fit(combined)
    assert "outcome" not in selector.feature_names_in_


def test_invalid_cnfi_size_is_rejected():
    X, y = _sample_data()
    with pytest.raises(ValueError, match="cnfi_features"):
        CNFINMISelector(cnfi_features=10).fit(X, y)



@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_both_branches_rank_all_inputs_before_final_selection(alpha):
    X, y = _sample_data()
    selector = CNFINMISelector(n_features_to_select=2, alpha=alpha).fit(X, y)
    names = set(X.columns)
    assert set(selector.cnfi_features_) == names
    assert set(selector.nmi_features_) == names
    assert set(selector.cnfi_scores_) == names
    assert set(selector.nmi_scores_) == names
    assert set(selector.nmi_values_) == names
    assert len(selector.selected_features_) == 2
    expected = {
        name: alpha * selector.cnfi_scores_[name]
        + (1 - alpha) * selector.nmi_scores_[name]
        for name in X.columns
    }
    assert selector.feature_scores_ == pytest.approx(expected)
    if alpha == 0:
        assert selector.ranking_ == selector.nmi_features_
    if alpha == 1:
        assert selector.ranking_ == selector.cnfi_features_


def test_fusion_matches_features_not_list_positions(monkeypatch):
    X, y = _sample_data()
    # Opposing rankings exercise alignment and the full-list denominators.
    cnfi = dict(zip(X.columns, [4.0, 3.0, 2.0, 1.0]))
    nmi = dict(zip(X.columns, [0.1, 0.2, 0.3, 0.4]))
    monkeypatch.setattr(
        CNFINMISelector, "_fit_cnfi", lambda self, data: (None, None, cnfi)
    )
    monkeypatch.setattr(
        CNFINMISelector, "_geometric_nmi", lambda self, column, target: nmi[column.name]
    )
    selector = CNFINMISelector(n_features_to_select=2, alpha=0.25).fit(X, y)
    assert selector.feature_scores_ == pytest.approx(
        dict(zip(X.columns, [0.1875, 0.3125, 0.4375, 0.5625]))
    )
    assert selector.selected_features_ == ["noise_b", "noise_a"]


def test_partial_cnfi_list_is_rejected_and_full_size_is_compatible():
    X, y = _sample_data()
    with pytest.raises(ValueError, match="both branches rank all inputs"):
        CNFINMISelector(n_features_to_select=2, cnfi_features=2).fit(X, y)
    full = CNFINMISelector(n_features_to_select=2, cnfi_features=X.shape[1]).fit(X, y)
    default = CNFINMISelector(n_features_to_select=2).fit(X, y)
    assert full.feature_scores_ == default.feature_scores_
    assert full.selected_features_ == default.selected_features_

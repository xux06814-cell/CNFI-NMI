import numpy as np
import pandas as pd
import pytest

from cnfi_nmi import CNFINMISelector


def _sample_data(n=120):
    # Fixed API fixture; not an experimental dataset.
    rows = np.array([
        [0, 0, 0.1, 0], [0, 0, 0.5, 1], [0, 0, 0.9, 2],
        [0, 0, 0.3, 3], [0, 1, 0.7, 4], [0, 0, 0.2, 0],
        [1, 1, 0.4, 1], [1, 1, 0.8, 2], [1, 1, 0.6, 3],
        [1, 0, 1.0, 4], [1, 1, 0.0, 0], [1, 1, 1.1, 1],
    ])
    data = np.tile(rows, (int(np.ceil(n / len(rows))), 1))[:n]
    X = pd.DataFrame(data, columns=["signal", "partner", "noise_a", "noise_b"])
    return X, pd.Series(data[:, 0].astype(int), name="target")


def test_rank_normalization_matches_equations_29_and_30():
    scores = CNFINMISelector._rank_scores(["a", "b", "c"])
    assert scores == {"a": 1.0, "b": 0.5, "c": 0.0}


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
    columns = ["confidence", "lift", "support", "conviction", "leverage"]
    metrics = selector.rules_[columns].to_numpy(float)
    metrics = (metrics - metrics.min(axis=0)) / (
        metrics.max(axis=0) - metrics.min(axis=0) + 1e-9
    )
    expected_weight = (
        expected_lambda * strong + (1 - expected_lambda) * weak
    ) @ metrics[0]
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
        dict(zip(X.columns, [0.25, 5 / 12, 7 / 12, 0.75]))
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


def test_single_feature_rank_is_zero():
    assert CNFINMISelector._rank_scores(["only"]) == {"only": 0.0}


def test_paper_mode_weights_and_empty_graph():
    X, y = _sample_data()
    selector = CNFINMISelector(n_features_to_select=2, graph_mode="normalized").fit(X, y)
    assert selector.rules_["weight"].between(0, 1).all()
    assert np.isfinite(list(selector.feature_scores_.values())).all()
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b"])
    assert selector._eigen_rank(graph) == {"a": 0.0, "b": 0.0}


def test_paper_mode_handles_dangling_node():
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edge("a", "b", weight=0.5)
    selector = CNFINMISelector(graph_mode="normalized")
    scores = selector._eigen_rank(graph)
    assert all(np.isfinite(list(scores.values())))
    assert all(v >= 0 for v in scores.values())


def test_normalized_operator_is_stochastic_with_dangling_nodes():
    selector = CNFINMISelector(graph_mode="normalized")
    adjacency = np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    operator = selector._normalized_operator(adjacency)
    assert np.all(operator >= 0)
    np.testing.assert_allclose(operator.sum(axis=0), 1.0)
    np.testing.assert_allclose(operator[:3, 0], [0, 0.425, 0.425])
    np.testing.assert_allclose(operator[:3, 1], 0.85 * operator[:3, 3])

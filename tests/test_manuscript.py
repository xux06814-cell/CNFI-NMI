"""Independent small-data checks against the supplied manuscript, Eqs. 1-31.

These fixtures test calculations, not the paper's reported benchmark scores.
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cnfi_nmi import CNFINMISelector
from cnfi_nmi.selector import METRIC_ORDER


def test_defaults_return_full_paper_ranking():
    X = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    selector = CNFINMISelector().fit(X, X.a)
    assert selector.graph_mode == "normalized"
    assert selector.n_features_to_select is None
    assert set(selector.selected_features_) == set(X.columns)
    assert set(selector.cnfi_values_) == set(X.columns)


def test_rules_against_empirical_probabilities_and_max_aggregation():
    X = pd.DataFrame({
        "a": list("000000111111"),
        "b": list("000001111100"),
        "c": list("012012012012"),
    })
    selector = CNFINMISelector().fit(X, X.a)
    expected_metrics = []
    for rule in selector.rules_.itertuples():
        def indicator(item):
            feature_id, value = item.split("=", 1)
            return X.iloc[:, int(feature_id[1:])].eq(value).to_numpy()
        left, right = indicator(rule.antecedent_item), indicator(rule.consequent_item)
        pa, pb, joint = left.mean(), right.mean(), (left & right).mean()
        confidence = joint / pa
        expected_metrics.append([
            confidence, joint / (pa * pb), joint,
            (1 - pb) / (1 - confidence + 1e-9), joint - pa * pb,
        ])
    metrics = np.asarray(expected_metrics)
    np.testing.assert_allclose(selector.rules_[list(METRIC_ORDER)], metrics)
    scaled = (metrics - metrics.min(0)) / (np.ptp(metrics, axis=0) + 1e-9)
    modulation = 1 / (1 + np.exp(-np.log2(metrics[:, 1])))
    priors = (modulation[:, None] * np.array([.4, .2, .1, .2, .1])
              + (1 - modulation[:, None]) * np.array([.1, .2, .4, .1, .2]))
    weights = (scaled * priors).sum(1)
    np.testing.assert_allclose(selector.rules_["lambda"], modulation)
    np.testing.assert_allclose(selector.rules_["weight"], weights)
    grouped = selector.rules_.groupby(["source", "target"])["weight"].max()
    expected_edges = {(src, dst): w for (src, dst), w in grouped.items() if w > 0}
    assert nx.get_edge_attributes(selector.graph_, "weight") == pytest.approx(expected_edges)


@pytest.mark.parametrize("rho", [.6, .85])
@pytest.mark.parametrize("tolerance", [1e-3, 1e-6])
def test_eigen_rank_against_explicit_matrix_and_unweighted_pagerank(rho, tolerance, monkeypatch):
    # Asymmetric, unequal-weight network with a dangling and an isolated node.
    A = np.array([[0, .2, .7, 0], [.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    graph = nx.from_numpy_array(A, create_using=nx.DiGraph)
    selector = CNFINMISelector(rho=rho, tol=tolerance)
    q = (np.array([1., 1., 1., 0.]) + 1e-9) / (3 + 4e-9)
    Q = np.zeros((5, 5))
    Q[:4, 0] = rho * np.array([0, 2/9, 7/9, 0])
    Q[:4, 1] = rho * np.array([1, 0, 0, 0])
    Q[:4, 2:4] = rho * q[:, None]
    Q[:4, 4] = q
    Q[4, :4] = 1 - rho
    np.testing.assert_allclose(selector._normalized_operator(A), Q, atol=1e-15)
    # Reference Eqs. 20-21, with the manuscript's L1 stopping condition.
    vector = np.full(5, .2)
    for _ in range(1000):
        updated = (vector + Q @ vector) / 2
        updated /= np.abs(updated).sum()
        difference = np.abs(updated - vector).sum()
        vector = updated
        if difference < tolerance:
            break
    # Solve the independent PageRank linear system, with uniform dangling mass.
    P = (A > 0).astype(float)
    P[0] /= 2
    P[2:] = .25
    pr = np.linalg.solve(np.eye(4) - rho * P.T, np.full(4, (1-rho)/4))
    strength = A.sum(1)
    def norm(v):
        return (v - v.min()) / (np.ptp(v) + 1e-9)
    expected = vector[:4] + .5 * (norm(pr) + norm(strength)) * vector[4]
    # Isolate the Q stopping rule from the library's PageRank approximation,
    # and verify that Eq. 16 requests the same rho and an unweighted graph.
    def exact_pagerank(graph, *, alpha, weight, **kwargs):
        assert alpha == rho
        assert weight is None
        return dict(enumerate(pr))
    monkeypatch.setattr(nx, "pagerank", exact_pagerank)
    np.testing.assert_allclose(list(selector._eigen_rank(graph).values()), expected,
                               rtol=0, atol=1e-14)


def test_equal_rule_metrics_drop_all_edges_and_score_zero():
    X = pd.DataFrame({"a": ["same"] * 10, "b": ["same"] * 10})
    selector = CNFINMISelector().fit(X, [0, 1] * 5)
    assert selector.rules_["weight"].eq(0).all()
    assert selector.graph_.number_of_edges() == 0
    assert selector.cnfi_values_ == {"a": 0., "b": 0.}


def test_original_continuous_values_use_geometric_nmi():
    # Six unique observations determine the binary target: I=H(Y)=1,
    # H(X)=log2(6), so NMI=1/sqrt(log2(6)), not arithmetic NMI or binned NMI.
    X = pd.DataFrame({"continuous": np.arange(6, dtype=float), "constant": [1] * 6})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    selector = CNFINMISelector(n_bins=2).fit(X, y)
    assert selector.nmi_values_["continuous"] == pytest.approx(1 / np.sqrt(np.log2(6)))
    assert selector.nmi_values_["constant"] == 0
    assert selector._geometric_nmi(X.continuous, pd.Series([1] * 6)) == 0


def test_historical_mode_is_rejected():
    X = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="graph_mode must be normalized"):
        CNFINMISelector(graph_mode="benchmark").fit(X, X.a)


def test_paper_mode_rejects_nonpaper_edge_aggregation():
    X = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    with pytest.raises(ValueError, match="requires max edge aggregation"):
        CNFINMISelector(edge_aggregation="mean").fit(X, X.a)

"""Independent checks against compiled-manuscript raw Eqs. 1-29.

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
    assert selector.graph_mode == "paper"
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
    modulation = 1 / (1 + np.exp(-np.log2(metrics[:, 1])))
    priors = (modulation[:, None] * np.array([.4, .2, .1, .2, .1])
              + (1 - modulation[:, None]) * np.array([.1, .2, .4, .1, .2]))
    weights = (metrics * priors).sum(1)
    np.testing.assert_allclose(selector.rules_["lambda"], modulation)
    np.testing.assert_allclose(selector.rules_["weight"], weights)
    grouped = selector.rules_.groupby(["source", "target"])["weight"].max()
    expected_edges = {(src, dst): w for (src, dst), w in grouped.items() if w > 0}
    assert nx.get_edge_attributes(selector.graph_, "weight") == pytest.approx(expected_edges)


@pytest.mark.parametrize("rho", [.6, .85])
@pytest.mark.parametrize("tolerance", [1e-3, 1e-6])
def test_legacy_eigen_rank_against_stochastic_matrix(rho, tolerance, monkeypatch):
    # Asymmetric, unequal-weight network with a dangling and an isolated node.
    A = np.array([[0, .2, .7, 0], [.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    graph = nx.from_numpy_array(A, create_using=nx.DiGraph)
    selector = CNFINMISelector(graph_mode="normalized", rho=rho, tol=tolerance)
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


def test_legacy_equal_rule_metrics_drop_all_edges_and_score_zero():
    X = pd.DataFrame({"a": ["same"] * 10, "b": ["same"] * 10})
    selector = CNFINMISelector(graph_mode="normalized").fit(X, [0, 1] * 5)
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
    with pytest.raises(ValueError, match="graph_mode must be paper or normalized"):
        CNFINMISelector(graph_mode="benchmark").fit(X, X.a)


def test_paper_mode_rejects_nonpaper_edge_aggregation():
    X = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    with pytest.raises(ValueError, match="requires max edge aggregation"):
        CNFINMISelector(edge_aggregation="mean").fit(X, X.a)


def test_raw_paper_equal_metrics_are_not_scaled_to_zero():
    X = pd.DataFrame({'a': ['same'] * 10, 'b': ['same'] * 10})
    selector = CNFINMISelector().fit(X, [0, 1] * 5)
    # conf=lift=support=1, conviction=leverage=0; lambda=.5.
    np.testing.assert_allclose(selector.rules_['weight'], .25 + .20 + .25)
    assert selector.graph_.number_of_edges() == 2


def test_raw_paper_rank_against_direct_eigendecomposition():
    from cnfi_nmi.selector import paper_eigen_rank
    A = np.array([[0., .2, .7, 0], [.5, 0, .3, 0], [.1, 0, 0, 0], [0, 0, 0, 0]])
    graph = nx.from_numpy_array(A, create_using=nx.DiGraph)
    degree = (A > 0).sum(0)
    Q = np.zeros((5, 5))
    Q[:4, :4] = A
    Q[:4, 4] = 1
    Q[4, :4] = degree / (degree.sum() + 1e-9)
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    v = eigenvectors[:, np.argmax(eigenvalues.real)].real
    v *= np.sign(v.sum())
    v /= np.linalg.norm(v, 2)
    # Independent linear solve for unweighted PageRank, uniform dangling mass.
    transition = (A > 0).astype(float)
    for row in range(4):
        transition[row] = transition[row] / transition[row].sum() if transition[row].sum() else .25
    pr = np.linalg.solve(np.eye(4) - .85 * transition.T, np.full(4, .15 / 4))
    def norm(values):
        return (values - values.min()) / (np.ptp(values) + 1e-9)
    expected = v[:4] + .5 * (norm(pr) + norm(A.sum(1))) * v[4]
    observed, audit = paper_eigen_rank(graph, tol=1e-9)
    np.testing.assert_allclose(list(observed.values()), expected, atol=1e-5, rtol=0)
    assert audit['converged']
    assert audit['vector_l2_norm'] == pytest.approx(1.)


def test_legacy_mode_retains_previous_rank_denominator():
    assert CNFINMISelector(graph_mode='normalized')._rank_scores(['a', 'b', 'c']) == {'a': 1., 'b': .5, 'c': 0.}


def test_raw_periodic_operator_reports_nonconvergence():
    from cnfi_nmi.selector import paper_eigen_rank
    # Unequal two-cycle: direct power iteration oscillates, unlike a lazy
    # operator. Do not silently change the paper's matrix to make this pass.
    graph = nx.DiGraph()
    graph.add_edge('a', 'b', weight=2e8)
    graph.add_edge('b', 'a', weight=1e8)
    with pytest.raises(ArithmeticError, match='not converged'):
        paper_eigen_rank(graph, max_iter=50)

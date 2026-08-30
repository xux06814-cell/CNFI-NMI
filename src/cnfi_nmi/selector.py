"""CNFI-NMI, implemented from Eqs. (1)-(29) of BDMA-2026-0010."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils.validation import check_is_fitted

MetricOrder = tuple[str, str, str, str, str]
METRIC_ORDER: MetricOrder = (
    "confidence",
    "lift",
    "support",
    "conviction",
    "leverage",
)


class CNFINMISelector(TransformerMixin, BaseEstimator):
    """Select tabular features by interaction structure and target relevance.

    Parameters follow the accepted manuscript. ``cnfi_features`` is the paper's
    :math:`k_f`, while ``n_features_to_select`` is the optional :math:`k_max`.
    The implementation keeps feature selection inside ``fit`` so it can be used
    safely in a scikit-learn cross-validation pipeline.

    Parameters
    ----------
    n_features_to_select:
        Number of features returned by ``transform``. ``None`` keeps the full
        fused ranking.
    cnfi_features:
        Number of features assigned to the CNFI list. ``None`` uses half of the
        input dimensions (rounded up), an explicit software default because the
        manuscript defines :math:`k_f` as user supplied.
    alpha:
        Fusion balance between CNFI and NMI. The manuscript default is 0.5.
    min_support, min_confidence:
        FP-Growth and association-rule thresholds.
    n_bins:
        Number of equal-frequency bins used only by CNFI.
    kappa, theta:
        Logistic modulation parameters from Eq. (8).
    edge_aggregation:
        Deterministic reduction when several bin-level rules map to the same
        feature pair. The manuscript does not specify this reduction; ``max``
        retains the strongest rule and is the documented repository default.
    target_col:
        Optional compatibility convenience. When ``y`` is omitted in ``fit``,
        this column is removed from ``X`` and used as the target.
    """

    def __init__(
        self,
        n_features_to_select: int | None = 10,
        cnfi_features: int | None = None,
        *,
        alpha: float = 0.5,
        min_support: float = 0.01,
        min_confidence: float = 0.1,
        n_bins: int = 5,
        kappa: float = 1.0,
        theta: float = 0.0,
        strong_prior: tuple[float, float, float, float, float] = (
            0.40,
            0.20,
            0.10,
            0.20,
            0.10,
        ),
        weak_prior: tuple[float, float, float, float, float] = (
            0.10,
            0.20,
            0.40,
            0.10,
            0.20,
        ),
        edge_aggregation: Literal["max", "mean", "sum"] = "max",
        epsilon: float = 1e-9,
        max_iter: int = 1000,
        tol: float = 1e-6,
        target_col: str | None = None,
    ) -> None:
        self.n_features_to_select = n_features_to_select
        self.cnfi_features = cnfi_features
        self.alpha = alpha
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.n_bins = n_bins
        self.kappa = kappa
        self.theta = theta
        self.strong_prior = strong_prior
        self.weak_prior = weak_prior
        self.edge_aggregation = edge_aggregation
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.target_col = target_col

    def fit(self, X: pd.DataFrame, y: Iterable[object] | None = None):
        """Fit the three-stage selector and return ``self``."""
        X_features, target = self._validate_input(X, y)
        self.feature_names_in_ = np.asarray(X_features.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)

        self._validate_parameters(self.n_features_in_)
        kf = self.cnfi_features
        if kf is None:
            kf = max(1, math.ceil(self.n_features_in_ / 2))

        discretized = self._equal_frequency_binning(X_features)
        self.graph_, self.rules_, raw_cnfi = self._fit_cnfi(discretized)

        cnfi_order = self._descending_order(raw_cnfi, self.feature_names_in_)
        self.cnfi_features_ = cnfi_order[:kf]
        self.cnfi_scores_ = self._rank_scores(self.cnfi_features_)

        remaining = [
            name for name in self.feature_names_in_ if name not in self.cnfi_scores_
        ]
        self.nmi_values_ = {
            name: self._geometric_nmi(X_features[name], target) for name in remaining
        }
        nmi_order = self._descending_order(self.nmi_values_, remaining)
        self.nmi_features_ = nmi_order
        self.nmi_scores_ = self._rank_scores(nmi_order)

        self.feature_scores_ = {
            name: (
                self.alpha * self.cnfi_scores_.get(name, 0.0)
                + (1.0 - self.alpha) * self.nmi_scores_.get(name, 0.0)
            )
            for name in self.feature_names_in_
        }
        self.ranking_ = self._descending_order(
            self.feature_scores_, self.feature_names_in_
        )
        keep = self.n_features_to_select
        self.selected_features_ = (
            self.ranking_ if keep is None else self.ranking_[:keep]
        )
        self.support_ = np.asarray(
            [name in self.selected_features_ for name in self.feature_names_in_],
            dtype=bool,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return selected columns ordered by the fused rank."""
        check_is_fitted(self, "selected_features_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CNFINMISelector.transform expects a pandas DataFrame.")
        missing = [name for name in self.feature_names_in_ if name not in X.columns]
        if missing:
            raise ValueError(f"Input is missing fitted features: {missing}")
        return X.loc[:, self.selected_features_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return a scikit-learn-compatible feature-selection mask or indices."""
        check_is_fitted(self, "support_")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return selected feature names in fused-rank order."""
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)

    def _validate_input(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CNFINMISelector.fit expects a pandas DataFrame.")
        if X.empty or X.shape[1] == 0:
            raise ValueError("X must contain at least one row and one feature.")
        if X.columns.has_duplicates:
            raise ValueError("Feature names must be unique.")

        features = X.copy()
        if y is None:
            if self.target_col is None or self.target_col not in features.columns:
                raise ValueError("Pass y or provide a target_col present in X.")
            target = features.pop(self.target_col)
        else:
            values = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
            if values.ndim != 1 or len(values) != len(features):
                raise ValueError("X and y must have the same number of rows.")
            target = pd.Series(values, index=features.index, name="target")
            if self.target_col is not None and self.target_col in features.columns:
                features = features.drop(columns=[self.target_col])
        if len(target) != len(features):
            raise ValueError("X and y must have the same number of rows.")
        if target.isna().any():
            raise ValueError("The target contains missing values.")
        return features, target

    def _validate_parameters(self, n_features: int) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if not 0.0 < self.min_support <= 1.0:
            raise ValueError("min_support must be in (0, 1].")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if self.kappa <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("kappa and epsilon must be positive.")
        if self.max_iter < 1 or self.tol <= 0.0:
            raise ValueError("max_iter and tol must be positive.")
        if self.edge_aggregation not in {"max", "mean", "sum"}:
            raise ValueError("edge_aggregation must be max, mean, or sum.")
        if self.cnfi_features is not None and not 1 <= self.cnfi_features <= n_features:
            raise ValueError(
                "cnfi_features must be between 1 and the number of features."
            )
        if self.n_features_to_select is not None and not (
            1 <= self.n_features_to_select <= n_features
        ):
            raise ValueError(
                "n_features_to_select must be between 1 and the number of features."
            )
        for label, values in (
            ("strong_prior", self.strong_prior),
            ("weak_prior", self.weak_prior),
        ):
            if len(values) != len(METRIC_ORDER) or any(value < 0 for value in values):
                raise ValueError(f"{label} must contain five non-negative values.")
            if not np.isclose(sum(values), 1.0):
                raise ValueError(f"{label} must sum to 1.")

    def _equal_frequency_binning(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        for name in X.columns:
            column = X[name]
            if (
                pd.api.types.is_numeric_dtype(column)
                and column.nunique(dropna=True) > self.n_bins
            ):
                binned = pd.qcut(column, q=self.n_bins, labels=False, duplicates="drop")
                result[name] = binned.astype("Int64").astype("string").fillna("<NA>")
            else:
                result[name] = column.astype("string").fillna("<NA>")
        return result

    def _fit_cnfi(self, X: pd.DataFrame):
        names = list(X.columns)
        item_to_feature: dict[str, str] = {}
        transactions: list[list[str]] = []
        for _, row in X.iterrows():
            transaction = []
            for index, name in enumerate(names):
                item = f"f{index}={row[name]}"
                item_to_feature[item] = name
                transaction.append(item)
            transactions.append(transaction)

        encoder = TransactionEncoder()
        encoded = encoder.fit(transactions).transform(transactions)
        one_hot = pd.DataFrame(encoded, columns=encoder.columns_)
        frequent = fpgrowth(
            one_hot,
            min_support=self.min_support,
            use_colnames=True,
            max_len=2,
        )

        graph = nx.DiGraph()
        graph.add_nodes_from(names)
        empty = pd.DataFrame(
            columns=[*METRIC_ORDER, "lambda", "weight", "source", "target"]
        )
        if frequent.empty or not (frequent["itemsets"].map(len) >= 2).any():
            warnings.warn(
                "No qualifying pairwise itemsets; CNFI scores are tied at zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            return graph, empty, {name: 0.0 for name in names}

        rules = association_rules(
            frequent,
            metric="confidence",
            min_threshold=self.min_confidence,
        )
        records = []
        for _, rule in rules.iterrows():
            if len(rule["antecedents"]) != 1 or len(rule["consequents"]) != 1:
                continue
            antecedent = next(iter(rule["antecedents"]))
            consequent = next(iter(rule["consequents"]))
            source = item_to_feature[antecedent]
            target = item_to_feature[consequent]
            if source == target:
                continue

            confidence = float(rule["confidence"])
            lift = float(rule["lift"])
            support = float(rule["support"])
            leverage = float(rule["leverage"])
            conviction = (1.0 - float(rule["consequent support"])) / (
                1.0 - confidence + self.epsilon
            )
            log_lift = math.log2(max(lift, self.epsilon))
            z = np.clip(self.kappa * (log_lift - self.theta), -709.0, 709.0)
            modulation = float(1.0 / (1.0 + math.exp(-z)))
            adaptive = modulation * np.asarray(self.strong_prior, dtype=float) + (
                1.0 - modulation
            ) * np.asarray(self.weak_prior, dtype=float)
            metrics = np.asarray(
                [confidence, lift, support, conviction, leverage], dtype=float
            )
            weight = float(adaptive @ metrics)
            records.append(
                {
                    "confidence": confidence,
                    "lift": lift,
                    "support": support,
                    "conviction": conviction,
                    "leverage": leverage,
                    "lambda": modulation,
                    "weight": weight,
                    "source": source,
                    "target": target,
                }
            )

        rule_table = pd.DataFrame.from_records(records, columns=empty.columns)
        if rule_table.empty:
            warnings.warn(
                "No cross-feature rules survived filtering; CNFI scores are tied at zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            return graph, rule_table, {name: 0.0 for name in names}

        grouped = rule_table.groupby(["source", "target"], sort=False)["weight"]
        edge_weights = getattr(grouped, self.edge_aggregation)()
        for (source, target), weight in edge_weights.items():
            graph.add_edge(source, target, weight=float(weight))
        return graph, rule_table, self._eigen_rank(graph)

    def _eigen_rank(self, graph: nx.DiGraph) -> dict[str, float]:
        nodes = list(graph.nodes)
        if graph.number_of_edges() == 0:
            return {name: 0.0 for name in nodes}

        adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")
        in_degree = (adjacency > 0.0).sum(axis=0).astype(float)
        jump = in_degree / (in_degree.sum() + self.epsilon)
        n = len(nodes)
        extended = np.zeros((n + 1, n + 1), dtype=float)
        extended[:n, :n] = adjacency
        extended[:n, n] = 1.0
        extended[n, :n] = jump

        eigenvector = np.ones(n + 1, dtype=float) / (n + 1)
        for _ in range(self.max_iter):
            updated = extended @ eigenvector
            norm = np.linalg.norm(updated)
            if norm <= self.epsilon:
                break
            updated /= norm
            if np.linalg.norm(updated - eigenvector) < self.tol:
                eigenvector = updated
                break
            eigenvector = updated

        pagerank_map = nx.pagerank(graph, alpha=0.85, weight=None)
        pagerank = np.asarray([pagerank_map[name] for name in nodes], dtype=float)
        out_strength = adjacency.sum(axis=1)
        influence = 0.5 * self._minmax(pagerank) + 0.5 * self._minmax(out_strength)
        global_influence = eigenvector[n]
        return {
            name: float(eigenvector[index] + influence[index] * global_influence)
            for index, name in enumerate(nodes)
        }

    def _minmax(self, values: np.ndarray) -> np.ndarray:
        return (values - values.min()) / (values.max() - values.min() + self.epsilon)

    def _geometric_nmi(self, feature: pd.Series, target: pd.Series) -> float:
        feature_codes, _ = pd.factorize(feature.astype("string").fillna("<NA>"))
        target_codes, _ = pd.factorize(target.astype("string").fillna("<NA>"))
        if self._entropy(feature_codes) <= self.epsilon:
            return 0.0
        if self._entropy(target_codes) <= self.epsilon:
            return 0.0
        return float(
            normalized_mutual_info_score(
                target_codes,
                feature_codes,
                average_method="geometric",
            )
        )

    @staticmethod
    def _entropy(codes: np.ndarray) -> float:
        _, counts = np.unique(codes, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-(probabilities * np.log2(probabilities)).sum())

    @staticmethod
    def _rank_scores(order: Iterable[str]) -> dict[str, float]:
        ordered = list(order)
        size = len(ordered)
        if size == 0:
            return {}
        return {
            name: (size - rank) / size for rank, name in enumerate(ordered, start=1)
        }

    @staticmethod
    def _descending_order(
        scores: dict[str, float], stable_order: Iterable[str]
    ) -> list[str]:
        positions = {name: index for index, name in enumerate(stable_order)}
        return sorted(scores, key=lambda name: (-scores[name], positions[name]))

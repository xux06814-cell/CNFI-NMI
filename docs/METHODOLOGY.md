# Method-to-code correspondence

This document maps the accepted manuscript (BDMA-2026-0010) to the reference
implementation. Equation numbers refer to the manuscript, which is not
redistributed in this repository.

## Stage 1 - Complex Network Feature Interaction

Continuous features are equal-frequency discretized into five bins for
FP-Growth. Pairwise rules must satisfy `min_support = 0.01` and
`min_confidence = 0.1` by default. Every rule is evaluated using Confidence,
Lift, Support, Conviction, and Leverage (Eqs. 1-5).

The modulation signal uses the manuscript's base-2 PMI proxy:

$$
\lambda(R_{ij}) = \frac{1}{1 + e^{-\kappa(\log_2 \mathrm{Lift}(R_{ij})-\theta)}}.
$$

The manuscript calls this validated mechanism **dynamic thresholding**. It
introduces no additional variable: $\theta$ is the logistic transition center,
$\kappa$ controls its steepness, and the rule-specific $\lambda(R_{ij})$
changes smoothly with Lift. The FP-Growth support and confidence thresholds
remain fixed filters. The ablation label “Dynamic thres.” refers to this
mechanism.

With metric order `[Confidence, Lift, Support, Conviction, Leverage]`, the
repository defaults exactly match Table 1:

| Parameter | Value |
|---|---|
| $\kappa$ | 1.0 |
| $\theta$ | 0.0 |
| $w_{strong}$ | `[0.40, 0.20, 0.10, 0.20, 0.10]` |
| $w_{weak}$ | `[0.10, 0.20, 0.40, 0.10, 0.20]` |

The per-rule metric weight is
$\omega=\lambda w_{strong}+(1-\lambda)w_{weak}$, and the edge weight is
$W(R_{ij})=\sum_m \omega_mV_m$ (Eqs. 7-9). The feature graph includes every
input feature, including isolated nodes, as required by $V=F$.

Eigen-Rank is calculated from the extended matrix in Eq. 13. PageRank uses the
unweighted transition in Eq. 14, out-strength uses Eq. 15, both are min-max
normalized with $\epsilon_0=10^{-9}$, and the final score follows Eq. 20.

## Stage 2 - Normalized Mutual Information

NMI is computed on the original, non-binned feature column, as specified in the
experimental protocol. Values are treated as empirical discrete outcomes. The
implementation uses geometric normalization, as specified in the revised manuscript:

$$
\mathrm{NMI}(X;Y)=\frac{I(X;Y)}{\sqrt{H(X)H(Y)}}.
$$

A constant feature or constant target receives 0 because the mathematical
expression has a zero denominator and carries no usable ranking information.
Empirical NMI can be inflated for high-cardinality features.

## Stage 3 - Rank fusion

Both CNFI and NMI independently rank every input feature. With $d$ input
features, both list lengths are $N_{cnfi}=N_{nmi}=d$; there is no intermediate
CNFI truncation or remaining-feature branch. For each full list of size $N$,
the rank score is:

$$
R(F_i)=\frac{N-\mathrm{rank}(i)}{N-1}\quad (N>1).
$$

The best feature scores 1 and the last scores 0. For a single feature the score is 0. This rescales both original branch scores equally and preserves the fused ranking.
For each feature, the final score is
$S_{final}=\alpha R_{CNFI}+(1-\alpha)R_{NMI}$. Scores are matched
by feature name before weighting. At $\alpha=1$ the ranking follows CNFI; at
$\alpha=0$ it follows NMI across all inputs. Ties retain input-column order.
Only the final fused ranking is truncated by `n_features_to_select` ($k_{max}$).

The old `cnfi_features` argument is retained for API compatibility but accepts
only `None` or $d$. It no longer creates an independently truncated branch.

## Explicit implementation convention

FP-Growth produces multiple bin-level rules that may collapse to the same raw
feature pair. The manuscript defines one adjacency entry $A_{ij}$ per directed feature
pair by taking the maximum rule weight. This is the implementation default
(`edge_aggregation="max"`); `mean` and `sum` remain optional alternatives for
sensitivity analysis.


## Evaluated normalized formulation

Set `graph_mode="normalized"` to use the revised manuscript's normalized
formulation, which has been evaluated as confirmed by the authors. Each of the
five rule metrics is min-max scaled across qualifying
training rules (a constant metric contributes zero), while the modulation
coefficient still uses raw Lift. Rule weights are the adaptive weighted sum
of scaled metrics, and feature-pair edges use the maximum. Zero-weight edges
are omitted from the scaled graph. A smoothed in-degree distribution supplies
dangling transitions and the auxiliary column of a column-stochastic matrix.
The stationary vector is estimated by lazy power iteration, `(I + Q)/2`;
`rho` defaults to 0.85 and must lie strictly between 0 and 1. Empty graphs
produce tied zero scores.

The default `graph_mode="benchmark"` retains raw rule metrics and its augmented
spectral operator. The default mode does not indicate the evaluation status of
the normalized formulation. Configure the graph mode explicitly for each
experiment and record it with the results; see
[evaluation status and graph modes](REPRODUCIBILITY.md#evaluation-status-and-graph-modes).

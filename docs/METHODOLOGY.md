# Method-to-code correspondence

This document maps the accepted manuscript (BDMA-2026-0010) to the reference
implementation. Equation numbers refer to the manuscript PDF in `paper/`.

## Stage 1 - Complex Network Feature Interaction

Continuous features are equal-frequency discretized into five bins for
FP-Growth. Pairwise rules must satisfy `min_support = 0.01` and
`min_confidence = 0.1` by default. Every rule is evaluated using Confidence,
Lift, Support, Conviction, and Leverage (Eqs. 1-5).

The modulation signal uses the manuscript's base-2 PMI proxy:

$$
\lambda(R_{ij}) = \frac{1}{1 + e^{-\kappa(\log_2 \mathrm{Lift}(R_{ij})-\theta)}}.
$$

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
implementation uses geometric normalization, matching Eq. 26:

$$
\mathrm{NMI}(X;Y)=\frac{I(X;Y)}{\sqrt{H(X)H(Y)}}.
$$

A constant feature or constant target receives 0 because the mathematical
expression has a zero denominator and carries no usable ranking information.

## Stage 3 - Rank fusion

The top $k_f$ CNFI features and the remaining NMI-ranked features are disjoint.
For a list of size $N$, the repository implements the manuscript literally:

$$
R(F_i)=\frac{N-\mathrm{rank}(i)}{N}.
$$

Consequently, the best feature scores $(N-1)/N$ and the last feature scores 0.
The final score is $S_{final}=\alpha R_{CNFI}+(1-\alpha)R_{NMI}$ (Eq. 29).

## Explicit implementation convention

FP-Growth produces multiple bin-level rules that may collapse to the same raw
feature pair. The manuscript defines one adjacency entry $A_{ij}$ but does not
state how duplicate state-level rules are reduced. This implementation uses
the maximum rule weight by default (`edge_aggregation="max"`), with `mean` and
`sum` available for sensitivity analysis. This convention is disclosed rather
than presented as a claim from the paper.

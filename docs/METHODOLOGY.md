# Method-to-code correspondence

The default `graph_mode="paper"` follows the supplied compiled manuscript
`BDMA_2026_0010_编译后.pdf`, SHA-256
`8da1c637d9f0dabe0ce0ff561bf74e7eda6f03a024eca3981a62c962b0771191`.
The main equations are implemented in `src/cnfi_nmi/selector.py`; shared input,
binning, NMI and sklearn utilities are in `normalized_selector.py`.

## CNFI

Only training predictors enter the rule miner. Numeric columns with more than
five observed values are equal-frequency binned with `pandas.qcut`; duplicate
boundaries are dropped. Other values retain their category identity. Declare
integer-coded categories as string/category dtype. Missing predictors form an
explicit state. The target never enters rule mining.

Transactions use feature-qualified value items. `mine_pairs` counts singleton
and pair supports by sparse matrix multiplication. This is exactly equivalent
to FP-Growth with `max_len=2`: an infrequent singleton cannot belong to a
frequent pair. Default support/confidence filters are .01/.1. Only rules
between distinct features are retained. The published formula check compares
these supports with FP-Growth on real Bank observations.

Metrics are ordered `[Confidence, Lift, Support, Conviction, Leverage]`.
Conviction uses `(1-P(consequent))/(1-confidence+epsilon)`, `epsilon=1e-9`.
The compiled manuscript's weighted sum uses **raw metric values**:

```text
lambda = sigmoid(kappa * (log2(Lift) - theta))
w_strong = [.40, .20, .10, .20, .10]
w_weak   = [.10, .20, .40, .10, .20]
w_rule   = lambda * w_strong + (1-lambda) * w_weak
edge_rule_weight = dot(w_rule, raw_metrics)
```

Defaults are `kappa=1`, `theta=0`. Dynamic thresholding names this logistic
transition; support and confidence filters stay fixed. Every input feature is
a node. Multiple value-level rules for a directed feature pair contribute their
maximum positive weight.

For adjacency `A[i,j]` from feature i to feature j, the raw Eigen-Rank equations
are:

```text
q = in_degree / (sum(in_degree) + epsilon)
Q = [[A, ones(d,1)], [q.T, 0]]
L = ones(d+1)/(d+1)
repeat: L_new = Q @ L; L_new /= ||L_new||_2
stop when ||L_new-L||_2 < 1e-6
mu = .5 * minmax(unweighted_PageRank(rho=.85))
     + .5 * minmax(weighted_out_strength)
SER = L[:d] + mu * L[d]
```

Min-max denominators use additive epsilon. The raw operator has no stochastic
normalization, lazy step or damping block. `rho` applies to the separate
PageRank term. The iteration limit is 10,000; a nonconvergent operator raises
`ArithmeticError`. A periodic-graph regression test covers this case. An empty
positive-edge graph receives zero raw CNFI scores.

Audit attributes are `rules_`, `graph_`, `cnfi_values_` (raw SER),
`mining_audit_` and `eigen_audit_`. `cnfi_scores_` contains rank scores.

## NMI and rank fusion

All original, non-binned feature columns are evaluated independently of CNFI.
Observed continuous values are treated as empirical discrete states:

```text
NMI(X,Y) = I(X;Y) / sqrt(H(X)*H(Y))
```

A constant feature or target receives zero. Empirical NMI can favor features
with many distinct states; inspect `nmi_values_` when interpreting the ranking.

Both branches rank all d inputs. With one-based rank, the compiled manuscript's
Eqs. 27–28 use `R=(d-rank)/d`. Fusion is
`alpha*R_CNFI+(1-alpha)*R_NMI`, matched by feature identity, default alpha=.5.
Ties retain input-column order. Only the final fused ranking is truncated by
`n_features_to_select`; `None` keeps the full list. The compatibility parameter
`cnfi_features` accepts only `None` or d.

## Earlier normalized mode

`graph_mode="normalized"` explicitly retains the earlier implementation:
per-metric min-max scaling, epsilon-smoothed q, a column-stochastic augmented
operator, lazy L1 iteration and rank denominator d−1. It is kept for existing
callers. The experiment scripts use `paper` mode.

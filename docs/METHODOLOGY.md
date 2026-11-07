# Method-to-code correspondence

Equation numbers refer to the 20-page BDMA-2026-0010 accepted manuscript. The selector uses `graph_mode="normalized"`.

## Stage 1: CNFI (Eqs. 1-22; pp. 4-7)

Only training features enter FP-Growth; the target is excluded. Numeric columns with more than five distinct values are equal-frequency binned with `pandas.qcut`, with duplicate boundaries dropped. Smaller numeric alphabets are retained as categories. Declare integer-coded categories as string/category dtype so their codes are not binned. Missing feature values form an explicit category; the benchmark tables report no missing values, so this is an implementation convention for other inputs.

Transactions contain feature-qualified value items. FP-Growth uses `max_len=2`, consistent with the scope stated in Section 5.2 (p. 17). Default support/confidence filters are .01/.1. Each qualifying single-item to single-item rule between distinct features has the metrics in Eqs. 1-5, ordered as `[Confidence, Lift, Support, Conviction, Leverage]`.

Conviction is `(1-P(consequent))/(1-confidence+epsilon)`, including when confidence is below 1, with `epsilon=1e-9`. Min-max scale each metric across all qualifying rules (Eq. 7), using the same additive epsilon. A constant metric contributes zero. The rule-specific modulation uses **raw** Lift (Eqs. 6, 9):

$$\lambda(R)=\frac{1}{1+\exp[-\kappa(\log_2\mathrm{Lift}(R)-\theta)]}.$$

The priors match Table 1 (p. 11): `kappa=1`, `theta=0`, `strong_prior=[.40,.20,.10,.20,.10]`, `weak_prior=[.10,.20,.40,.10,.20]`. Eq. 10 interpolates these priors; Eq. 8 sums normalized metrics with the resulting weights. "Dynamic thresholding" names this logistic transition; it does not change the support/confidence filters.

Every input feature is a graph node, including isolated features. Multiple value-level rules for a directed feature pair contribute their **maximum** weight. Zero-weight edges are omitted (p. 6). For adjacency `A[i,j]` from i to j:

- Eqs. 12-13: `q=(in_degree+epsilon)/(sum(in_degree)+d*epsilon)`.
- Eq. 14: each source column of T is its normalized outgoing weights; a dangling source uses q.
- Eq. 15: `Q=[[rho*T,q],[(1-rho)*ones(1,d),0]]`, with `rho=.85`.
- Eq. 16: PageRank uses the **unweighted** directed graph, the same rho, uniform teleportation, and uniform dangling-mass redistribution.
- Eqs. 17-19: influence is the equal mixture of min-max normalized PageRank and weighted out-strength, with additive epsilon.
- Eqs. 20-21: initialize uniformly, apply `(I+Q)/2`, normalize by L1, and stop at L1 iterate difference below `1e-6`, or 1000 updates.
- Eq. 22: `SER[i]=L[i]+influence[i]*L[d]`. An empty positive-edge graph assigns zero to every raw CNFI score (p. 6).

Audit outputs are `rules_` (value-item identities, raw/normalized metrics, lambda and weights), `graph_`, and `cnfi_values_` (raw SER). `cnfi_scores_` contains normalized **rank** scores, not raw SER.

## Stage 2: NMI (Eqs. 23-28; pp. 7-8)

All original, non-binned feature columns are evaluated independently of CNFI. Observed continuous values are empirical discrete categories. NMI uses geometric normalization:

$$\mathrm{NMI}(X;Y)=\frac{I(X;Y)}{\sqrt{H(X)H(Y)}}.$$

A constant feature or target receives zero. Empirical NMI can be inflated for high-cardinality features; the implementation follows the paper definition rather than substituting a continuous MI estimator. Inspect `nmi_values_`.

## Stage 3: rank fusion (Eqs. 29-31; pp. 8-9)

Both branches rank **all d inputs**. For each list, `R=(d-rank)/(d-1)` with one-based rank, or zero if d=1. Match by feature identity and compute `S_final=alpha*R_CNFI+(1-alpha)*R_NMI`, default alpha=.5. Ties retain input-column order, a deterministic convention where the paper does not specify tie handling. Only the fused list is truncated by `n_features_to_select`. Its default `None` implements Algorithm 1's omitted optional k_max.

The compatibility parameter `cnfi_features` accepts only `None` or d; intermediate branch truncation is rejected.


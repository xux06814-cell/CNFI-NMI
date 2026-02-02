import numpy as np
import pandas as pd
import networkx as nx
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from typing import List, Dict, Tuple, Optional

class CNFI_NMI_Selector(BaseEstimator, TransformerMixin):
    """
    CNFI-NMI: A Robust Feature Selection Framework.
    
    Implements the methodology described in:
    "CNFI-NMI: A Robust Feature Selection Framework Integrating Complex Network 
    Interactions and Normalized Mutual Information" (BDMA-2026-0010).
    
    Attributes:
        selected_features_ (List[str]): Final list of selected features.
        feature_scores_ (Dict[str, float]): Combined scores for all features.
    """

    def __init__(self, 
                 target_col: str, 
                 n_features: int = 10,
                 alpha: float = 0.5,
                 min_support: float = 0.01, 
                 min_conf: float = 0.1,
                 n_bins: int = 5,
                 sigmoid_k: float = 1.0,
                 sigmoid_theta: float = 0.0):
        """
        Args:
            target_col: Name of the target variable column.
            n_features: Number of features to select (top-k).
            alpha: Fusion weight (default 0.5).
            min_support: FP-Growth support threshold[cite: 636].
            min_conf: FP-Growth confidence threshold[cite: 636].
            n_bins: Number of bins for discretization (default 5).
            sigmoid_k: Slope parameter for adaptive weighting (Eq. 7).
            sigmoid_theta: Center parameter for adaptive weighting (Eq. 7).
        """
        self.target_col = target_col
        self.n_features = n_features
        self.alpha = alpha
        self.min_support = min_support
        self.min_conf = min_conf
        self.n_bins = n_bins
        self.sigmoid_k = sigmoid_k
        self.sigmoid_theta = sigmoid_theta
        
        # Define weight configurations based on paper [cite: 443-446]
        # w_strong favors Confidence (Reliability)
        self.w_strong = {
            'confidence': 0.40, 'lift': 0.20, 'support': 0.10, 
            'conviction': 0.20, 'leverage': 0.10
        }
        # w_weak favors Support (Robustness/Backbone)
        self.w_weak = {
            'confidence': 0.10, 'lift': 0.20, 'support': 0.40, 
            'conviction': 0.10, 'leverage': 0.20
        }

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Executes the three-stage feature selection process.
        """
        # --- Preprocessing: Equal-Frequency Discretization  ---
        # Note: We combine X and y for rule mining context
        data = X.copy()
        if y is not None:
            data[self.target_col] = y
        
        df_binned = self._discretize(data)
        
        # --- Stage 1: CNFI (Complex Network Feature Interaction) ---
        cnfi_scores, cnfi_selected = self._stage_1_cnfi(df_binned)
        
        # --- Stage 2: NMI (Normalized Mutual Information) ---
        # Only calculated for features NOT selected in Stage 1 [cite: 517]
        remain_features = [f for f in X.columns if f not in cnfi_selected]
        nmi_scores = self._stage_2_nmi(df_binned, remain_features)
        
        # --- Stage 3: Fusion [cite: 600-603] ---
        self.feature_scores_ = self._stage_3_fusion(cnfi_scores, nmi_scores, X.columns)
        
        # Select Top-K
        sorted_feats = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
        self.selected_features_ = [f for f, s in sorted_feats[:self.n_features]]
        
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.selected_features_]

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Equal-Frequency Binning (5 bins) as per.
        """
        df_disc = df.copy()
        for col in df.columns:
            if col == self.target_col: 
                continue # Target often treated as categorical
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > self.n_bins:
                try:
                    # qcut for equal frequency
                    df_disc[col] = pd.qcut(df[col], q=self.n_bins, labels=False, duplicates='drop')
                except ValueError:
                    # Fallback to cut if data is skewed/sparse
                    df_disc[col] = pd.cut(df[col], bins=self.n_bins, labels=False)
            else:
                # Ensure categorical/integer encoding
                df_disc[col] = df_disc[col].astype('category').cat.codes
        return df_disc

    def _stage_1_cnfi(self, df: pd.DataFrame) -> Tuple[Dict[str, float], List[str]]:
        """
        Mines association rules and computes Eigen-Rank.
        """
        # 1. Transaction Generation
        tx = []
        features_only = df.drop(columns=[self.target_col])
        for idx, row in features_only.iterrows():
            # Create itemset: "FeatureName::BinValue"
            tx.append([f"{col}::{val}" for col, val in row.items()])
            
        # 2. FP-Growth [cite: 292]
        te = TransactionEncoder()
        te_ary = te.fit_transform(tx)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent = fpgrowth(df_encoded, min_support=self.min_support, use_colnames=True, max_len=2)
        if frequent.empty:
            return {}, []
            
        # 3. Rule Generation with all metrics [cite: 295-312]
        rules = association_rules(frequent, metric="confidence", 
                                  min_threshold=self.min_conf, support_only=False)
        
        # 4. Composite Adaptive Weighting [cite: 316-326]
        # Calculate Log(Lift) for PMI proxy
        rules['log_lift'] = np.log(rules['lift'] + 1e-8)
        
        # Calculate lambda (Sigmoid function) - Eq (7)
        # lambda = 1 / (1 + e^(-k * (log_lift - theta)))
        rules['lambda'] = 1 / (1 + np.exp(-self.sigmoid_k * (rules['log_lift'] - self.sigmoid_theta)))
        
        # Calculate final weight - Eq (6) & (8)
        # W = lambda * Score_strong + (1-lambda) * Score_weak
        w_list = []
        for _, r in rules.iterrows():
            score_strong = sum(r[m] * w for m, w in self.w_strong.items() if m in r)
            score_weak = sum(r[m] * w for m, w in self.w_weak.items() if m in r)
            w_final = r['lambda'] * score_strong + (1 - r['lambda']) * score_weak
            w_list.append(w_final)
        
        rules['weight'] = w_list

        # 5. Build Graph & Eigen-Rank [cite: 313, 447]
        G = nx.DiGraph()
        for i, r in rules.iterrows():
            src = list(r['antecedents'])[0].split("::")[0] # Extract raw feature name
            dst = list(r['consequents'])[0].split("::")[0]
            if src != dst:
                G.add_edge(src, dst, weight=r['weight'])
                
        # Calculate Eigen-Rank scores
        er_scores = self._calculate_eigen_rank(G)
        
        # Normalize R_CNFI based on rank [cite: 522]
        sorted_nodes = sorted(er_scores, key=er_scores.get, reverse=True)
        N_cnfi = len(sorted_nodes)
        norm_scores = {node: (N_cnfi - i) / N_cnfi for i, node in enumerate(sorted_nodes)}
        
        # Determine top features (heuristic: top 50% or top-N for next stage exclusion)
        # Paper implies CNFI selects a subset, NMI ranks the rest. 
        # Here we return all scored features but mark top ones.
        return norm_scores, sorted_nodes[:len(sorted_nodes)//2]

    def _calculate_eigen_rank(self, G: nx.DiGraph) -> Dict[str, float]:
        """
        Implements Eigen-Rank centrality [cite: 447-467].
        """
        nodes = list(G.nodes())
        n = len(nodes)
        if n == 0: return {}
        
        # Adjacency Matrix A (Weighted)
        A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')
        
        # In-degree k_in
        k_in = (A > 0).sum(axis=0)
        
        # Extended Transition Matrix M - Eq (11)
        # Last column is p (jump probability based on normalized in-degree)
        p = k_in / (k_in.sum() + 1e-8) # Eq (12)
        
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = 1.0 # Last column (virtual node -> all nodes)
        M[n, :n] = p   # Last row (all nodes -> virtual node)
        
        # Power Iteration
        L = np.ones(n + 1) / (n + 1)
        for _ in range(100):
            L_next = M @ L
            L_next /= np.linalg.norm(L_next) # Normalization Eq (17)
            if np.linalg.norm(L_next - L) < 1e-6:
                L = L_next
                break
            L = L_next
            
        # Influence Vector mu - Eq (15)
        # mu = 0.5 * norm(PR) + 0.5 * norm(S_out)
        pr = np.array(list(nx.pagerank(G, weight='weight').values()))
        s_out = A.sum(axis=1)
        
        # Normalize vectors to [0,1] or unit norm as implied by "norm()"
        def normalize(v): return (v - v.min()) / (v.max() - v.min() + 1e-8)
        
        mu = 0.5 * normalize(pr) + 0.5 * normalize(s_out)
        
        # Final Score - Eq (18): S(Fi) = Li + mu_i * L_{n+1}
        scores = {}
        L_global = L[n] # The component of the virtual node
        for i, node in enumerate(nodes):
            scores[node] = L[i] + mu[i] * L_global
            
        return scores

    def _stage_2_nmi(self, df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
        """
        Calculates NMI for remaining features [cite: 485-517].
        """
        y = df[self.target_col]
        scores = {}
        from sklearn.metrics import normalized_mutual_info_score
        
        for feat in features:
            if feat not in df.columns: continue
            # Use 'arithmetic' average for NMI denominator as standard proxy for Eq 24
            score = normalized_mutual_info_score(y, df[feat], average_method='arithmetic')
            scores[feat] = score
            
        # Normalize R_NMI based on rank [cite: 536]
        sorted_feats = sorted(scores, key=scores.get, reverse=True)
        N_nmi = len(sorted_feats)
        rank_scores = {f: (N_nmi - i) / N_nmi for i, f in enumerate(sorted_feats)}
        return rank_scores

    def _stage_3_fusion(self, cnfi_scores: Dict, nmi_scores: Dict, all_feats: List[str]) -> Dict:
        """
        Fuses scores using parameter alpha[cite: 601].
        """
        final_scores = {}
        for f in all_feats:
            if f == self.target_col: continue
            s_c = cnfi_scores.get(f, 0.0)
            s_n = nmi_scores.get(f, 0.0)
            # Eq (26)
            final_scores[f] = self.alpha * s_c + (1 - self.alpha) * s_n
        return final_scores

# -----------------------------------------------------------
# 使用示例 (Usage Example)
# -----------------------------------------------------------
if __name__ == "__main__":
    # 生成模拟数据
    X_dummy = pd.DataFrame(np.random.rand(200, 20), columns=[f'feat_{i}' for i in range(20)])
    y_dummy = np.random.randint(0, 2, 200)
    
    # 初始化选择器
    selector = CNFI_NMI_Selector(target_col='target', n_features=5, n_bins=5)
    
    # 拟合
    selector.fit(X_dummy, pd.Series(y_dummy, name='target'))
    
    # 输出结果
    print("Selected Features:", selector.selected_features_)
    print("Feature Scores (Top 5):")
    for f in selector.selected_features_:
        print(f"{f}: {selector.feature_scores_[f]:.4f}")

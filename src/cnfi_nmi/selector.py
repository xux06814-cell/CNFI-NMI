"""CNFI-NMI for the compiled manuscript, raw Eqs. (5)-(29).

The previous normalized implementation remains available through the explicit
``graph_mode="normalized"`` compatibility option. The default is ``paper``.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from mlxtend.frequent_patterns import association_rules

from .normalized_selector import CNFINMISelector as _NormalizedSelector, METRIC_ORDER

MANUSCRIPT_SHA256='8da1c637d9f0dabe0ce0ff561bf74e7eda6f03a024eca3981a62c962b0771191'

METRICS=['confidence','lift','support','conviction','leverage']

def mine_pairs(X,min_support):
    """Exact pair supports, equivalent to FP-Growth with max_len=2.

    Filtering infrequent singletons before sparse multiplication changes no
    eligible pair and avoids building a dense matrix of rare street names.
    """
    n=len(X);rows=[];cols=[];items=[];counts=[];mapping={};offset=0
    for i,c in enumerate(X):
        code,values=pd.factorize(X[c].astype('string').fillna('<NA>'),sort=False)
        count=np.bincount(code,minlength=len(values));keep=np.flatnonzero(count/n>=min_support)
        lookup=np.full(len(values),-1,dtype=np.int64);lookup[keep]=np.arange(len(keep))+offset
        ix=lookup[code];good=ix>=0
        rows.append(np.flatnonzero(good));cols.append(ix[good])
        for k in keep:
            item=f'f{i}={values[k]}';items.append(item);mapping[item]=c;counts.append(int(count[k]))
        offset+=len(keep)
    if not items:return pd.DataFrame(columns=['support','itemsets']),mapping
    rr=np.concatenate(rows);cc=np.concatenate(cols)
    matrix=sparse.csr_matrix((np.ones(len(rr),dtype=np.int64),(rr,cc)),shape=(n,len(items)))
    joint=(matrix.T@matrix).toarray()
    result=[(count/n,frozenset([item])) for item,count in zip(items,counts)]
    r,c=np.where(np.triu(joint,1)/n>=min_support)
    result.extend((joint[i,j]/n,frozenset([items[i],items[j]])) for i,j in zip(r,c))
    return pd.DataFrame(result,columns=['support','itemsets']),mapping

def feature_graph(rules,names,weights):
    G=nx.DiGraph();G.add_nodes_from(names)
    table=rules[['source','target']].copy();table['weight']=np.asarray(weights,dtype=float)
    assert np.isfinite(table.weight).all()
    for (source,target),w in table.groupby(['source','target'],sort=False).weight.max().items():
        if w>0:G.add_edge(source,target,weight=float(w))
    return G

def paper_eigen_rank(graph,epsilon=1e-9,rho=.85,tol=1e-6,max_iter=10000):
    names=list(graph);n=len(names)
    if not graph.number_of_edges():return {c:0. for c in names},{'converged':True,'iterations':0,'empty_graph':True}
    A=nx.to_numpy_array(graph,nodelist=names,weight='weight')
    indegree=(A>0).sum(axis=0).astype(float)
    q=indegree/(indegree.sum()+epsilon)  # Eq.12, no additive numerator.
    Q=np.zeros((n+1,n+1));Q[:n,:n]=A;Q[:n,n]=1.;Q[n,:n]=q  # Eq.13.
    v=np.ones(n+1)/(n+1);converged=False
    for iteration in range(1,max_iter+1):
        nxt=Q@v;norm=np.linalg.norm(nxt,2)
        if not norm>0:raise ArithmeticError('Zero norm in the PDF power iteration')
        nxt/=norm;delta=np.linalg.norm(nxt-v,2);v=nxt
        if delta<tol:converged=True;break
    if not converged:raise ArithmeticError(f'PDF power iteration not converged: {delta}')
    pr=nx.pagerank(graph,alpha=rho,weight=None,max_iter=1000,tol=tol)
    pr=np.asarray([pr[c] for c in names]);strength=A.sum(axis=1)
    def minmax(a):return (a-a.min())/(a.max()-a.min()+epsilon)
    mu=.5*minmax(pr)+.5*minmax(strength)
    score=v[:-1]+mu*v[-1]
    eigenvalue=float(v@(Q@v))
    audit={'converged':converged,'iterations':iteration,'iterate_l2_difference':float(delta),'vector_l2_norm':float(np.linalg.norm(v,2)),'eigenvalue':eigenvalue,'relative_eigen_residual':float(np.linalg.norm(Q@v-eigenvalue*v,2)/max(np.linalg.norm(Q@v,2),epsilon)),'global_influence':float(v[-1]),'max_edge_weight':float(A.max()),'q_sum':float(q.sum()),'operator':'Q=[[A,1],[q.T,0]], direct multiplication, L2 norm, no lazy/stochastic operator'}
    assert np.isfinite(score).all() and abs(np.linalg.norm(v)-1)<1e-10
    return dict(zip(names,map(float,score))),audit

def pgi_feature_rank(graph,tol=1e-6,max_iter=1000):
    """Official PGI 3-WLR algebra on the shared PDF feature graph.

    Node representation and rule pool are fixed across the ablation rows.
    Zero-centrality sum receives zero contribution, as a numerical convention.
    """
    names=list(graph);n=len(names)
    if not graph.number_of_edges():return {c:0. for c in names}
    A=nx.to_numpy_array(graph,nodelist=names,weight='weight')
    T=np.zeros((n+1,n+1));T[:n,:n]=A;T[:n,n]=1;T[n,:n]=(A>0).sum(axis=0)
    den=T.sum(axis=1);T=np.divide(T,den[:,None],out=np.zeros_like(T),where=den[:,None]>0)
    v=np.ones(n+1)
    for _ in range(max_iter):
        nxt=T.T@v
        if np.linalg.norm(nxt-v,1)<tol:v=nxt;break
        v=nxt
    strength=A.sum(axis=1)
    bc=np.asarray(list(nx.betweenness_centrality(graph,weight=None).values()))
    extra=.5*(strength/max(strength.sum(),1e-9)+bc/max(bc.sum(),1e-9))
    return dict(zip(names,map(float,v[:-1]+extra*v[-1])))

class CNFINMISelector(_NormalizedSelector):
    """Rank all features and optionally retain the final top-k fusion.

    ``paper`` uses raw rule metrics, Q=[[A,1],[q.T,0]], L2 iteration and
    (d-rank)/d, as specified in the compiled manuscript. ``normalized`` retains
    the earlier metric-scaled, stochastic-operator implementation explicitly.
    Quantile bins are fitted only in ``fit``; NMI uses original observations.
    """

    def __init__(self,n_features_to_select=None,cnfi_features=None,*,alpha=.5,
                 min_support=.01,min_confidence=.1,n_bins=5,kappa=1.,theta=0.,
                 strong_prior=(.40,.20,.10,.20,.10),
                 weak_prior=(.10,.20,.40,.10,.20),edge_aggregation='max',
                 graph_mode='paper',rho=.85,epsilon=1e-9,max_iter=10000,
                 tol=1e-6,target_col=None):
        super().__init__(n_features_to_select=n_features_to_select,
            cnfi_features=cnfi_features,alpha=alpha,min_support=min_support,
            min_confidence=min_confidence,n_bins=n_bins,kappa=kappa,theta=theta,
            strong_prior=strong_prior,weak_prior=weak_prior,
            edge_aggregation=edge_aggregation,graph_mode=graph_mode,rho=rho,
            epsilon=epsilon,max_iter=max_iter,tol=tol,target_col=target_col)

    def _validate_parameters(self,n_features):
        mode=self.graph_mode
        if mode not in {'paper','normalized'}:raise ValueError('graph_mode must be paper or normalized')
        self.graph_mode='normalized'
        try:super()._validate_parameters(n_features)
        finally:self.graph_mode=mode

    def _fit_cnfi(self,X):
        if self.graph_mode=='normalized':return super()._fit_cnfi(X)
        names=list(X);frequent,mapping=mine_pairs(X,self.min_support)
        if frequent.empty or not frequent.itemsets.map(len).eq(2).any():
            graph=nx.DiGraph();graph.add_nodes_from(names)
            empty=pd.DataFrame(columns=[*METRICS,'lambda','weight','source','target','antecedent_item','consequent_item'])
            self.mining_audit_={'frequent_itemsets':len(frequent),'rules':0,'feature_nodes':len(graph),'feature_edges':0,'raw_metric_weighting':True,'same_rule_pool_for_all_ablation_rows':True}
            return graph,empty,self._eigen_rank(graph)
        with np.errstate(invalid='ignore',divide='ignore'):
            raw=association_rules(frequent,metric='confidence',min_threshold=self.min_confidence)
        ant=raw.antecedents.map(lambda x:next(iter(x)));con=raw.consequents.map(lambda x:next(iter(x)))
        rules=raw[['confidence','lift','support','leverage']].copy()
        rules['conviction']=(1-raw['consequent support'])/(1-raw.confidence+self.epsilon)
        rules['source']=ant.map(mapping);rules['target']=con.map(mapping)
        rules['antecedent_item']=ant;rules['consequent_item']=con
        rules=rules.loc[rules.source!=rules.target].reset_index(drop=True)
        z=np.clip(self.kappa*(np.log2(np.maximum(rules.lift.to_numpy(),self.epsilon))-self.theta),-709,709)
        modulation=1/(1+np.exp(-z));rules['lambda']=modulation
        adaptive=modulation[:,None]*np.asarray(self.strong_prior)+(1-modulation[:,None])*np.asarray(self.weak_prior)
        assert np.allclose(adaptive.sum(axis=1),1,atol=1e-14,rtol=0)
        rules['weight']=np.sum(adaptive*rules[METRICS].to_numpy(),axis=1)  # raw Eq.7.
        assert np.isfinite(rules[METRICS+['weight']].to_numpy()).all()
        graph=feature_graph(rules,names,rules.weight)
        self.mining_audit_={'frequent_itemsets':len(frequent),'rules':len(rules),'feature_nodes':len(graph),'feature_edges':graph.number_of_edges(),'raw_metric_weighting':True,'same_rule_pool_for_all_ablation_rows':True}
        return graph,rules,self._eigen_rank(graph)

    def _eigen_rank(self,graph):
        if self.graph_mode=='normalized':return super()._eigen_rank(graph)
        scores,audit=paper_eigen_rank(graph,self.epsilon,self.rho,self.tol,self.max_iter)
        self.eigen_audit_=audit
        return scores

    def _rank_scores(self,order):
        if self.graph_mode=='normalized':return _NormalizedSelector._rank_scores(order)
        names=list(order);d=len(names)
        return {c:(d-rank)/d for rank,c in enumerate(names,1)}  # Eqs.27-28.

def ablation_rankings(selector,names):
    r=selector.rules_;raw=r[METRICS].to_numpy()
    static=(np.asarray(selector.strong_prior)+np.asarray(selector.weak_prior))/2
    # The paper does not define these standalone variants. The selected
    # completion follows its two prior vectors without introducing new values.
    dynamic_conf=r.confidence.to_numpy()*(selector.weak_prior[0]+(selector.strong_prior[0]-selector.weak_prior[0])*r['lambda'].to_numpy())
    Gconf=feature_graph(r,names,r.confidence)
    Gmulti=feature_graph(r,names,raw@static)
    Gdynamic=feature_graph(r,names,dynamic_conf)
    def order(scores):return sorted(names,key=lambda c:-scores[c])
    eigen_scores,audit=paper_eigen_rank(Gconf)
    orders={'PGI (shared feature graph)':order(pgi_feature_rank(Gconf)),'+ Multi weight (paper-prior midpoint)':order(pgi_feature_rank(Gmulti)),'+ Dynamic thres. (paper confidence coefficient)':order(pgi_feature_rank(Gdynamic)),'+ Eigen-Rank (PDF)':order(eigen_scores),'CNFI (full, PDF)':selector.cnfi_features_}
    return orders,{'partial_eigen':audit,'static_weights':static.tolist(),'dynamic_confidence_coefficient':'0.10+0.30*lambda','graph_control':'same feature nodes, same support/confidence-qualified rule pool, max rule aggregation; no extra Lift>=1 filter','PGI_adapter':'3-WLR on feature-level graph, not max-state aggregation'}

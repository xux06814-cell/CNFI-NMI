"""Validate PDF formulas and pair enumeration using real Bank observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from paper_selector import PaperCNFINMISelector,mine_pairs,METRICS
ROOT=Path(__file__).resolve().parent
data=pd.read_csv(ROOT/'processed/bank.csv');y=data.pop('y');X=data.iloc[:,:8]
selector=PaperCNFINMISelector(n_features_to_select=4,graph_mode='paper_raw')
bins=selector._equal_frequency_binning(X)
transactions=[[f'f{i}={v}' for i,v in enumerate(row)] for row in bins.itertuples(index=False,name=None)]
enc=TransactionEncoder();onehot=pd.DataFrame(enc.fit(transactions).transform(transactions),columns=enc.columns_)
old=fpgrowth(onehot,min_support=.01,use_colnames=True,max_len=2)
new,_=mine_pairs(bins,.01);a=dict(zip(old.itemsets,old.support));b=dict(zip(new.itemsets,new.support))
assert a.keys()==b.keys()
diff=max(abs(a[k]-b[k]) for k in a);assert diff==0
selector.fit(X,y);r=selector.rules_;lam=r['lambda'].to_numpy()
w=lam[:,None]*np.asarray(selector.strong_prior)+(1-lam[:,None])*np.asarray(selector.weak_prior)
weight_error=float(np.max(np.abs(np.sum(w*r[METRICS].to_numpy(),axis=1)-r.weight.to_numpy())))
assert weight_error==0
assert selector.cnfi_scores_[selector.cnfi_features_[0]]==7/8
names=list(X);A=nx.to_numpy_array(selector.graph_,nodelist=names);degree=(A>0).sum(axis=0)
q=degree/(degree.sum()+1e-9);Q=np.zeros((9,9));Q[:8,:8]=A;Q[:8,8]=1;Q[8,:8]=q
values,vectors=np.linalg.eig(Q);v=np.real(vectors[:,np.argmax(values.real)]);v*=np.sign(v.sum());v/=np.linalg.norm(v)
pr=nx.pagerank(selector.graph_,alpha=.85,weight=None,max_iter=1000,tol=1e-6);pr=np.asarray([pr[c] for c in names]);s=A.sum(axis=1)
norm=lambda z:(z-z.min())/(z.max()-z.min()+1e-9)
direct=v[:8]+(.5*norm(pr)+.5*norm(s))*v[8]
score_error=float(np.max(np.abs(direct-np.asarray([selector.cnfi_values_[c] for c in names]))))
assert score_error<1e-4
result={'input':'All 45211 freshly downloaded Bank rows, first 8 real predictors; formula validation only, no performance estimate','pairs_fpgrowth_equal':True,'itemsets':len(a),'max_support_error':diff,'raw_weight_error':weight_error,'max_score_difference_from_direct_eigendecomposition':score_error,'rank_denominator':'d, verified top rank 7/8','power_iteration':selector.eigen_audit_,'passed':True}
(ROOT/'formula_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True)

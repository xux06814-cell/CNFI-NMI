"""Experimental alpha sweep using the repository's CNFINMISelector.

Selection and preprocessing are fitted on each outer training fold only.
Classifier settings are fixed and specified by the user, not tuned on test folds.
The resulting test-fold sweep is exploratory and cannot be used to select alpha
and also claim unbiased performance from the same folds.
"""
import argparse, hashlib, json, platform, sys
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from cnfi_nmi import CNFINMISelector


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',type=Path,required=True);ap.add_argument('--target',required=True)
    ap.add_argument('--positive-label',required=True,help='Exact target value for the positive class')
    ap.add_argument('--drop',nargs='*',default=[],help='Identifier or leakage columns to exclude')
    ap.add_argument('--categorical',nargs='*',default=[],help='Integer-coded categorical columns to keep as categories')
    ap.add_argument('--sep',default=',');ap.add_argument('--features',type=int,default=12)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--learning-rate',type=float,default=.1);ap.add_argument('--max-depth',type=int,default=3)
    ap.add_argument('--n-estimators',type=int,default=100);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--network-fold',type=int,default=1)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); df=pd.read_csv(a.csv,sep=a.sep,low_memory=False)
    if a.target not in df: raise ValueError('target column not found')
    yraw=df.pop(a.target)
    if yraw.isna().any():raise ValueError('target has missing values')
    labels=yraw.astype(str); unique=sorted(labels.unique().tolist())
    if len(unique)!=2 or a.positive_label not in unique:raise ValueError(f'expected 2 labels including positive label; got {unique}')
    y=(labels==a.positive_label).astype(int);X=df.drop(columns=a.drop)
    emergency_sources={'Emergency service(s)','Emergency services'} & set(X.columns)
    if emergency_sources:
        raise ValueError(f'Potential target leakage: drop source columns {sorted(emergency_sources)} after constructing the target')
    for col in a.categorical:
        if col not in X:raise ValueError(f'categorical column not found: {col}')
        X[col]=X[col].astype('string')
    if not 1<=a.features<=X.shape[1] or not 1<=a.network_fold<=a.folds:raise ValueError('invalid features or network-fold')
    if y.value_counts().min()<a.folds:raise ValueError('each class must have at least as many samples as folds')
    a.output.mkdir(parents=True,exist_ok=True);records=[]
    cv=StratifiedKFold(n_splits=a.folds,shuffle=True,random_state=a.seed)
    for fold,(tr,te) in enumerate(cv.split(X,y),1):
        train=X.iloc[tr].reset_index(drop=True);test=X.iloc[te].reset_index(drop=True);yt=y.iloc[tr].reset_index(drop=True);yv=y.iloc[te].reset_index(drop=True)
        selector=CNFINMISelector(n_features_to_select=a.features,alpha=.5).fit(train,yt)
        names=list(train.columns); d=len(names)
        if fold==a.network_fold:
            # Encode special characters as ASCII CSV aliases for MATLAB R2019a.
            # Refuse collisions, preserve full mapping separately.
            alias=[str(c).replace('≥','_GE_') for c in names]
            if len(set(alias))!=len(alias) or any(any(ch in c for ch in [',', '\n', '\r', chr(34)]) for c in alias):raise ValueError('MATLAB network export requires unique names without commas or newlines')
            graph=nx.to_numpy_array(selector.graph_,nodelist=names,weight='weight')
            # Original plotting template is undirected; retain original directed matrix as well.
            pd.DataFrame(graph,index=alias,columns=alias).to_csv(a.output/'adj_directed.csv',index_label='Feature')
            pd.DataFrame(np.maximum(graph,graph.T),index=alias,columns=alias).to_csv(a.output/'adj.csv',index_label='Feature')
            pd.DataFrame({'Feature':alias,'NMI':[selector.nmi_values_[c] for c in names]}).to_csv(a.output/'nmi.csv',index=False)
            pd.DataFrame({'Feature':alias,'Score':[selector.feature_scores_[c] for c in names]}).to_csv(a.output/'rank.csv',index=False)
            pd.DataFrame({'original':names,'alias':alias}).to_csv(a.output/'feature_aliases.csv',index=False)
        for alpha in np.linspace(0,1,11):
            scores={c:alpha*selector.cnfi_scores_[c]+(1-alpha)*selector.nmi_scores_[c] for c in names}
            position={c:i for i,c in enumerate(names)};chosen=sorted(names,key=lambda c:(-scores[c],position[c]))[:a.features]
            u=train[chosen].copy();v=test[chosen].copy()
            for c in chosen:
                if not pd.api.types.is_numeric_dtype(u[c]):
                    codes,values=pd.factorize(u[c].astype('string'),sort=True)
                    u[c]=pd.Categorical.from_codes(codes,categories=range(len(values)))
                    v[c]=pd.Categorical.from_codes(pd.Index(values).get_indexer(v[c].astype('string')),categories=range(len(values)))
            clf=LGBMClassifier(learning_rate=a.learning_rate,max_depth=a.max_depth,n_estimators=a.n_estimators,random_state=a.seed,n_jobs=a.threads,verbosity=-1,deterministic=True,force_col_wise=True)
            clf.fit(u,yt);prob=clf.predict_proba(v)[:,1];pred=(prob>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(yv,pred,labels=[0,1]).ravel()
            records.append({'fold':fold,'alpha':round(float(alpha),1),'accuracy':accuracy_score(yv,pred),'precision':precision_score(yv,pred,zero_division=0),'recall':recall_score(yv,pred,zero_division=0),'f1':f1_score(yv,pred,zero_division=0),'specificity':tn/(tn+fp) if tn+fp else 0.,'auc':roc_auc_score(yv,prob),'selected_features':chosen})
        print(f'fold {fold}/{a.folds} done',flush=True)
    stats=pd.DataFrame(records);metrics=['accuracy','precision','recall','f1','specificity','auc']
    group=stats.groupby('alpha')[metrics];means=group.mean();std=group.std(ddof=1).add_suffix('_sd');means.join(std).reset_index().to_csv(a.output/'sensitivity_stats.csv',index=False)
    metadata={'data_sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest(),'rows':len(X),'inputs':d,'target':a.target,'positive_label':a.positive_label,'dropped':a.drop,'categorical':a.categorical,'seed':a.seed,'folds':a.folds,'network_training_fold':a.network_fold,'features':a.features,'model':{'learning_rate':a.learning_rate,'max_depth':a.max_depth,'n_estimators':a.n_estimators,'threshold':.5},'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'networkx':nx.__version__,'lightgbm':lightgbm.__version__},'warning':'Exploratory test-fold sweep; not paper exact, not an independent test after alpha selection. Directed adjacency additionally exported; adj.csv symmetrizes by maximum solely for MATLAB chord visualization.','records':records}
    (a.output/'sensitivity.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
if __name__=='__main__':main()

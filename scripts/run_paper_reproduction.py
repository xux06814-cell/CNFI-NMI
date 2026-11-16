"""Public-data sensitivity and component ablations, trained in the current run."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,hashlib,json,platform,sys
from importlib.metadata import version
import numpy as np
import pandas as pd
import networkx as nx
import sklearn,lightgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from run_experiment import encode_without_leakage
from cnfi_nmi import CNFINMISelector
from cnfi_nmi.selector import MANUSCRIPT_SHA256,ablation_rankings
import cnfi_nmi.selector as selector_module
import cnfi_nmi.normalized_selector as base_module

METRICS=['accuracy','precision','recall','f1','specificity','auc']
FOLDERS={'traffic':'Traffic Dataset','bank':'Financial Dataset','cardio':'Medical Dateset'}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(y,prob):
    pred=(prob>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(zip(METRICS,[accuracy_score(y,pred),precision_score(y,pred,zero_division=0),recall_score(y,pred,zero_division=0),f1_score(y,pred,zero_division=0),tn/(tn+fp),roc_auc_score(y,prob)]))
def sdmi_order(X,y):
    """Empirical-state SDMI with the pairwise-difference sum vectorized exactly."""
    scores={};y=np.asarray(y)
    for c in X:
        inv,values=pd.factorize(X[c].astype('string').fillna('<NA>'));s=len(values)
        pb=np.bincount(inv,minlength=s)/len(y);score=0.
        for label in np.unique(y):
            mask=y==label;pa=mask.mean();pab=np.bincount(inv[mask],minlength=s)/len(y)
            q=np.zeros(s);nz=pab>0;q[nz]=pab[nz]*np.log(pab[nz]/(pa*pb[nz]))
            score+=np.sqrt(max(0.,2*s*np.dot(q,q)-2*q.sum()**2))/(2*s)
        scores[c]=score
    return sorted(X.columns,key=lambda c:-scores[c])
def write_records(records,out,stem,group):
    frame=pd.DataFrame(records);frame.to_json(out/f'{stem}_folds.json',orient='records',indent=2)
    frame.to_csv(out/f'{stem}_folds.csv',index=False)
    g=frame.groupby(group,sort=False)[METRICS]
    summary=g.mean().join(g.std(ddof=1).add_suffix('_sd')).reset_index()
    summary.to_csv(out/('sensitivity_stats.csv' if stem=='sensitivity' else 'ablation_summary.csv'),index=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=FOLDERS)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Root for this run; dataset subfolder is created below it.')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--resume',action='store_true',help='Resume only with an identical input, source and environment contract.')
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    out=args.output/FOLDERS[args.dataset]
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Output already contains files. Choose a fresh --output, or use --resume for the same run.')
    out.mkdir(parents=True,exist_ok=True)
    path=args.data_dir/f'{args.dataset}.csv'
    frame=pd.read_csv(path,low_memory=False);target={'traffic':'target','bank':'y','cardio':'cardio'}[args.dataset]
    y=frame.pop(target);X=frame
    if y.isna().any() or set(y.unique())!={0,1}:raise ValueError('Expected a nonmissing binary 0/1 target')
    if min(y.value_counts())<5:raise ValueError('Each class needs at least five rows')
    contract={'data_sha256':digest(path),'runner_sha256':digest(Path(__file__)),
              'selector_sha256':digest(Path(selector_module.__file__)),
              'base_selector_sha256':digest(Path(base_module.__file__)),
              'encoding_helper_sha256':digest(Path(__file__).with_name('run_experiment.py')),
              'paper_sha256':MANUSCRIPT_SHA256,'threads':args.threads,
              'versions':{'python':platform.python_version(),'numpy':np.__version__,
                          'pandas':pd.__version__,'sklearn':sklearn.__version__,
                          'networkx':nx.__version__,'lightgbm':lightgbm.__version__,
                          'mlxtend':version('mlxtend'),'scipy':version('scipy')}}
    cp=out/'checkpoint_contract.json'
    if cp.exists():
        if not args.resume:raise ValueError('Output already contains a run. Choose a fresh --output, or use --resume for the same run.')
        if json.loads(cp.read_text())!=contract:raise ValueError('Checkpoint contract differs; use a fresh output directory')
    elif any(out.iterdir()):raise ValueError('Existing records lack a data/code contract')
    else:cp.write_text(json.dumps(contract,indent=2),encoding='utf8')
    categories=[]
    if args.dataset=='traffic':categories=['SA2_CODE21','SA3_CODE21','SA4_CODE21','Day']
    if args.dataset=='cardio':categories=['gender','cholesterol','gluc','smoke','alco','active','bp_category_encoded']
    # Table 2 calls Hour, Month and lane/vehicle counts numerical features.
    for c in categories:X[c]=X[c].astype('string')
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    records=json.loads((out/'sensitivity_folds.json').read_text()) if (out/'sensitivity_folds.json').exists() else []
    ablation=json.loads((out/'ablation_folds.json').read_text()) if (out/'ablation_folds.json').exists() else []
    names=list(X)
    selector=CNFINMISelector(n_features_to_select=12,graph_mode='paper')
    for fold,(tr,te) in enumerate(cv.split(X,y),1):
        complete=sum(r['fold']==fold for r in records)==11 and (args.dataset!='traffic' or sum(r['fold']==fold for r in ablation)==7)
        if complete:continue
        print(args.dataset,'fold',fold,'mining PDF rules',flush=True)
        train=X.iloc[tr].reset_index(drop=True);test=X.iloc[te].reset_index(drop=True);yt=y.iloc[tr];yv=y.iloc[te]
        selector=CNFINMISelector(n_features_to_select=12,graph_mode='paper').fit(train,yt)
        names=list(train);audit={'full_cnfi':selector.eigen_audit_,'mining':selector.mining_audit_}
        (out/f'formula_audit_fold{fold}.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
        selector.rules_.to_pickle(out/f'rules_fold{fold}.pkl')
        pd.DataFrame({'train':pd.Series(tr),'test':pd.Series(te)}).to_csv(out/f'indices_fold{fold}.csv',index=False)
        if fold==1:
            aliases=[c.replace('≥','_GE_') for c in names]
            graph=nx.to_numpy_array(selector.graph_,nodelist=names,weight='weight')
            pd.DataFrame(graph,index=aliases,columns=aliases).to_csv(out/'adj_directed.csv',index_label='Feature')
            pd.DataFrame(np.maximum(graph,graph.T),index=aliases,columns=aliases).to_csv(out/'adj.csv',index_label='Feature')
            pd.DataFrame({'Feature':aliases,'NMI':[selector.nmi_values_[c] for c in names]}).to_csv(out/'nmi.csv',index=False)
            pd.DataFrame({'Feature':aliases,'Score':[selector.feature_scores_[c] for c in names]}).to_csv(out/'rank.csv',index=False)
            pd.DataFrame({'original':names,'alias':aliases}).to_csv(out/'feature_aliases.csv',index=False)
        cache={}
        for r in records+ablation:
            if r['fold']==fold:cache[tuple(r['selected_features'])]={k:v for k,v in r.items() if k not in ['fold','alpha','method']}
        def evaluate(chosen):
            key=tuple(chosen)
            if key not in cache:
                u,v=encode_without_leakage(train[chosen],test[chosen])
                u.columns=v.columns=[f'f{i}' for i in range(len(chosen))]
                search=GridSearchCV(LGBMClassifier(random_state=42,n_jobs=args.threads,verbosity=-1,deterministic=True,force_col_wise=True),{'learning_rate':[.01,.05,.1],'max_depth':[3,5,7],'n_estimators':[200]},scoring='roc_auc',cv=StratifiedKFold(3,shuffle=True,random_state=42+fold),n_jobs=1)
                search.fit(u,yt);prob=search.predict_proba(v)[:,1]
                cache[key]={**metrics(yv,prob),'selected_features':chosen,'best_parameters':search.best_params_}
            return cache[key]
        for alpha in np.linspace(0,1,11):
            scores={c:alpha*selector.cnfi_scores_[c]+(1-alpha)*selector.nmi_scores_[c] for c in names}
            chosen=sorted(names,key=lambda c:-scores[c])[:12]
            records=[r for r in records if not(r['fold']==fold and abs(r['alpha']-alpha)<1e-8)]
            records.append({'fold':fold,'alpha':round(float(alpha),1),**evaluate(chosen)})
            write_records(records,out,'sensitivity','alpha')
            print(args.dataset,'fold',fold,'alpha',round(alpha,1),'done',flush=True)
        if args.dataset=='traffic':
            orders,partial_audit=ablation_rankings(selector,names)
            orders.update({'SDMI (formula)':sdmi_order(train,yt),'NMI':selector.nmi_features_})
            audit['ablation']=partial_audit
            (out/f'formula_audit_fold{fold}.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
            for method,order in orders.items():
                ablation=[r for r in ablation if not(r['fold']==fold and r['method']==method)]
                ablation.append({'fold':fold,'method':method,**evaluate(order[:4])})
                write_records(ablation,out,'ablation','method')
                print(args.dataset,'fold',fold,method,'done',flush=True)
    protocol={'completed':True,'dataset':args.dataset,'data_sha256':contract['data_sha256'],
              'code_contract':contract,'rows':len(X),'inputs':len(names),'target':target,
              'positive_label':1,'categorical':categories,'seed':42,'outer_folds':5,
              'inner_folds':3,'features_sweep':12,'features_ablation':4,
              'classifier':'LightGBM 200 trees; learning_rate .01/.05/.1 and max_depth 3/5/7; training-only encoding and inner AUC search',
              'selector':selector.get_params(),'network':'First outer training fold; max symmetrization only for display',
              'uncertainty':'Sample SD across five outer folds',
              'ablation_completion':'Same feature nodes and .01/.1 rule pool, max feature-pair aggregation. PGI 3-WLR adapted to that feature graph. Multi = raw metrics dot prior midpoint [.25,.20,.25,.15,.15]. Dynamic = confidence*(.10+.30*lambda). Eigen changes only ranking to the PDF formula. Standalone variants complete definitions omitted from the manuscript; see docs/REPRODUCIBILITY.md.',
              'data_processing':'See data/README.md and the generated data_provenance.json for source and cleaning definitions.',
              'selection_tuning':'Selection before inner search; outer test held out; alpha sweep exploratory',
              'versions':contract['versions']}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    print(args.dataset,'COMPLETE',flush=True)
if __name__=='__main__':main()

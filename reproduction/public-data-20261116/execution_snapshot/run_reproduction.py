"""New full runs using the supplied PDF, without loading previous results."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,hashlib,json,platform,sys
import numpy as np
import pandas as pd
import networkx as nx
import sklearn,lightgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'repository/src'))
sys.path.insert(0,str(ROOT/'repository/scripts'))
from run_experiment import encode_without_leakage
from paper_selector import PaperCNFINMISelector,ablation_rankings

METRICS=['accuracy','precision','recall','f1','specificity','auc']
FOLDERS={'traffic':'Traffic Dataset','bank':'Financial Dataset','cardio':'Medical Dateset'}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(y,prob):
    pred=(prob>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(zip(METRICS,[accuracy_score(y,pred),precision_score(y,pred,zero_division=0),recall_score(y,pred,zero_division=0),f1_score(y,pred,zero_division=0),tn/(tn+fp),roc_auc_score(y,prob)]))
def sdmi_order(X,y):
    """Same raw empirical-state SDMI and algebra as the preceding run."""
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
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=FOLDERS);args=parser.parse_args()
    out=ROOT/'results'/FOLDERS[args.dataset];out.mkdir(parents=True,exist_ok=True)
    path=ROOT/'processed'/f'{args.dataset}.csv'
    frame=pd.read_csv(path,low_memory=False);target={'traffic':'target','bank':'y','cardio':'cardio'}[args.dataset]
    y=frame.pop(target);X=frame
    contract={'data_sha256':digest(path),'runner_sha256':digest(Path(__file__)),'paper_selector_sha256':digest(ROOT/'paper_selector.py'),'upstream_selector_sha256':digest(ROOT/'repository/src/cnfi_nmi/selector.py'),'paper_sha256':digest(ROOT/'manuscript.pdf')}
    cp=out/'checkpoint_contract.json'
    if cp.exists():
        if json.loads(cp.read_text())!=contract:raise ValueError('Checkpoint contract differs; use a fresh output directory')
    elif (out/'sensitivity_folds.json').exists():raise ValueError('Existing records lack a data/code contract')
    else:cp.write_text(json.dumps(contract,indent=2),encoding='utf8')
    categories=[]
    if args.dataset=='traffic':categories=['SA2_CODE21','SA3_CODE21','SA4_CODE21','Day']
    if args.dataset=='cardio':categories=['gender','cholesterol','gluc','smoke','alco','active','bp_category_encoded']
    # Table 2 calls Hour, Month and lane/vehicle counts numerical features.
    for c in categories:X[c]=X[c].astype('string')
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    records=json.loads((out/'sensitivity_folds.json').read_text()) if (out/'sensitivity_folds.json').exists() else []
    ablation=json.loads((out/'ablation_folds.json').read_text()) if (out/'ablation_folds.json').exists() else []
    for fold,(tr,te) in enumerate(cv.split(X,y),1):
        complete=sum(r['fold']==fold for r in records)==11 and (args.dataset!='traffic' or sum(r['fold']==fold for r in ablation)==7)
        if complete and fold<5:continue
        print(args.dataset,'fold',fold,'mining PDF rules',flush=True)
        train=X.iloc[tr].reset_index(drop=True);test=X.iloc[te].reset_index(drop=True);yt=y.iloc[tr];yv=y.iloc[te]
        selector=PaperCNFINMISelector(n_features_to_select=12,graph_mode='paper_raw').fit(train,yt)
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
                search=GridSearchCV(LGBMClassifier(random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True),{'learning_rate':[.01,.05,.1],'max_depth':[3,5,7],'n_estimators':[200]},scoring='roc_auc',cv=StratifiedKFold(3,shuffle=True,random_state=42+fold),n_jobs=1)
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
    protocol={'completed':True,'dataset':args.dataset,'data_sha256':contract['data_sha256'],'code_contract':contract,'rows':len(X),'inputs':len(names),'target':target,'positive_label':1,'categorical':categories,'seed':42,'outer_folds':5,'inner_folds':3,'features_sweep':12,'features_ablation':4,'classifier':'UNCHANGED: LightGBM 200 trees; learning_rate .01/.05/.1 and max_depth 3/5/7; same training-only encoding and inner AUC search','selector':selector.get_params(),'network':'UNCHANGED: first outer TRAIN fold, max symmetrization only for display','uncertainty':'sample SD across five outer folds','formula_changes':['Raw metric values in Eq.7, no per-metric min-max scaling','q=indegree/(sum(indegree)+1e-9)','Q=[[A,1],[q.T,0]], direct Q multiplication and L2 norm','(d-rank)/d in Eqs.27-28','Hour/Month/lane/vehicle counts numeric per Table2'],'ablation_completion':'User authorized closest interpretation where paper unspecified. Same feature nodes and .01/.1 rule pool, max feature-pair aggregation across methods. PGI 3-WLR on that feature graph. Multi = raw metrics dot prior midpoint [.25,.20,.25,.15,.15]. Dynamic = confidence*(.10+.30*lambda) from Eq.9 confidence coefficient. Eigen changes only ranking to PDF formula. Full uses complete raw PDF formula. This shared feature-graph PGI is an adaptation, not an exact recovery of the authors original PGI transaction preprocessing.','data_processing':'Same freshly downloaded public archives and explicit cleaning as preceding run. Paper does not give exact CVD cleaning bounds or explain traffic 85 vs 86 input discrepancy.','selection_tuning':'UNCHANGED: selection before inner search; outer test held out; alpha sweep exploratory','versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    print(args.dataset,'COMPLETE',flush=True)
if __name__=='__main__':main()

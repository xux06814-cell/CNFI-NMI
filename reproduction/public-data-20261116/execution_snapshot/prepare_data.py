"""Prepare only freshly downloaded primary-source files in this directory."""
from pathlib import Path
import hashlib, io, json, zipfile, subprocess
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'processed';OUT.mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def zipcsv(p,name,sep=','):
    with zipfile.ZipFile(p) as z:return pd.read_csv(z.open(name),sep=sep,low_memory=False)
sources={}
z=ROOT/'traffic_author/data/merged-unlabeled.zip'
t=zipcsv(z,'merged-unlabeled.csv');before=len(t)
t.columns=t.columns.str.strip()
assert t[['Emergency service(s)','Emergency services']].isin([0,1]).all().all()
t['target']=t[['Emergency service(s)','Emergency services']].max(axis=1).astype(int)
t=t.drop(columns=['Emergency service(s)','Emergency services'])
# Preserve rows and original category identities. Missing predictors are handled
# by the selector and training-only classifier imputer, never by global fitting.
t.to_csv(OUT/'traffic.csv',index=False)
sources['traffic']={'url':'https://github.com/Future-Mobility-Lab/SydneyIncidents','file':str(z.relative_to(ROOT)),'sha256':digest(z),'raw_rows':before,'rows':len(t),'inputs':t.shape[1]-1,'steps':['strip column whitespace','target = OR of the two binary emergency-service fields','remove both source fields','retain all author-published rows and missing predictors'],'difference':'85 inputs, not the manuscript 86; no undocumented extra field added'}
with zipfile.ZipFile(ROOT/'bank.zip') as outer:
    with zipfile.ZipFile(io.BytesIO(outer.read('bank.zip'))) as inner:
        b=pd.read_csv(inner.open('bank-full.csv'),sep=';')
b['y']=(b['y']=='yes').astype(int);b.to_csv(OUT/'bank.csv',index=False)
sources['bank']={'url':'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip','sha256':digest(ROOT/'bank.zip'),'rows':len(b),'inputs':b.shape[1]-1,'steps':['read complete bank-full.csv, not bank.csv sample','yes=1, no=0','preserve unknown as explicit category']}
c=zipcsv(ROOT/'cardio.zip','cardio_train.csv',';');before=len(c)
# Explicit reconstruction of unspecified cleaning, fixed before evaluation.
# Bounds are data-validity conventions, not tuned to the manuscript metrics.
keep=c.ap_hi.between(80,250)&c.ap_lo.between(40,200)&(c.ap_hi>=c.ap_lo)&c.height.between(120,220)&c.weight.between(30,200)
c=c.loc[keep].drop(columns='id').copy()
c['age_years']=np.floor(c.age/365.25).astype(int)
c['bmi']=c.weight/(c.height/100)**2
code=np.select([(c.ap_hi>=140)|(c.ap_lo>=90),(c.ap_hi>=130)|(c.ap_lo>=80),(c.ap_hi>=120)], [3,2,1],default=0)
c['bp_category']=np.asarray(['normal','elevated','stage1','stage2'])[code]
c['bp_category_encoded']=code
c.to_csv(OUT/'cardio.csv',index=False)
sources['medical']={'url':'https://www.kaggle.com/api/v1/datasets/download/sulianova/cardiovascular-disease-dataset','sha256':digest(ROOT/'cardio.zip'),'raw_rows':before,'rows':len(c),'inputs':c.shape[1]-1,'steps':['drop id','retain ap_hi 80..250, ap_lo 40..200, ap_hi>=ap_lo, height 120..220, weight 30..200','age_years=floor(age/365.25)','bmi=weight/(height/100)^2','bp category: stage2 if hi>=140 or lo>=90, else stage1 if hi>=130 or lo>=80, else elevated if hi>=120, else normal','duplicate category as integer bp_category_encoded for paper schema'],'difference':'Cleaning bounds and category definitions are explicit reconstruction choices; original full preprocessing is not in repository'}
for name,frame in [('traffic',t),('bank',b),('medical',c)]:
    target={'traffic':'target','bank':'y','medical':'cardio'}[name]
    sources[name]['class_counts']={str(k):int(v) for k,v in frame[target].value_counts().items()}
    p=OUT/({'medical':'cardio'}.get(name,name)+'.csv');sources[name]['processed_sha256']=digest(p)
(ROOT/'data_provenance.json').write_text(json.dumps(sources,indent=2),encoding='utf8')
print(json.dumps(sources,indent=2),flush=True)

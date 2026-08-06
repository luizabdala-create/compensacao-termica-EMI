# -*- coding: utf-8 -*-
"""
PARTE B v2 — classificação nas 7 bandas, usando as MELHORES configs da Fase 8.
Config por banda = moda das configs escolhidas pela CV interna nos folds. Mesmo
protocolo LOTO + negative control. Substitui checkpoints/parteB.csv ao final.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA; from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC; from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,recall_score,
                             precision_score,confusion_matrix)
CKPT=os.path.join(ROOT,"checkpoints","parteB.csv")
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
# moda das configs por (banda,metodo)
bycfg={}
for k,g in bestcfg.items():
    banda,_,m=k.split("|"); bycfg.setdefault((banda,m),[]).append(json.dumps(g,sort_keys=True))
modal={bm:json.loads(Counter(v).most_common(1)[0][0]) for bm,v in bycfg.items()}
BANDS=sorted({b for b,_ in modal.keys()},key=lambda s:int(s.split("-")[0]))
print("bandas:",BANDS)
TAU={"rank":8,"max_shift_frac":0.14,"nsteps":81}

def clf_factory(name,seed=0):
    if name=="logreg": return Pipeline([("sc",StandardScaler()),("c",LogisticRegression(max_iter=5000,class_weight="balanced",random_state=seed))])
    if name=="svm": return Pipeline([("sc",StandardScaler()),("c",SVC(kernel="rbf",class_weight="balanced",random_state=seed))])
    if name=="rfc": return Pipeline([("sc",StandardScaler()),("c",RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=-1,random_state=seed))])
def feats(Y,ref,fs,tr,te):
    R=Y-ref[None,:]
    if fs in ("FS1","FS2"):
        M=[]
        for i in range(len(Y)):
            m=P.all_metrics(Y[i],ref)
            M.append([m["RMSD"],m["CCDM"]] if fs=="FS1" else [m["RMSD"],m["CCDM"],m["RMSE"],m["MAE"],m["NRMSE"],m["CORR"],m["SAM_deg"]])
        M=np.array(M); return M[tr],M[te]
    if fs=="FS3":
        w=max(1,R.shape[1]//500); Rs=np.vstack([P.moving_average(r,51)[::w] for r in R])
        pca=PCA(n_components=min(10,int(tr.sum())-1),random_state=0).fit(Rs[tr]); return pca.transform(Rs[tr]),pca.transform(Rs[te])
    if fs=="FS4":
        F=[]
        for r in R:
            rs=P.moving_average(r,51); idx=np.argsort(-np.abs(rs))[:200]
            F.append([rs.max(),rs.min(),np.ptp(rs),np.std(rs),np.mean(np.abs(rs)),float(np.mean(idx)),float(np.std(idx)),float(np.percentile(np.abs(rs),95)),float(np.sum(rs**2))])
        F=np.array(F); return F[tr],F[te]
def ev(yte,pred,binario):
    d={"accuracy":accuracy_score(yte,pred),"bal_acc":balanced_accuracy_score(yte,pred),"macro_f1":f1_score(yte,pred,average="macro",zero_division=0)}
    if binario:
        cm=confusion_matrix(yte,pred,labels=[0,1])
        d.update({"recall_dano":recall_score(yte,pred,pos_label=1,zero_division=0),"recall_sem_dano":recall_score(yte,pred,pos_label=0,zero_division=0),
                  "precision_dano":precision_score(yte,pred,pos_label=1,zero_division=0),"n_falso_saudavel":int(cm[1,0]),"n_dano_real":int(cm[1].sum()),
                  "taxa_falso_saudavel":float(cm[1,0]/max(cm[1].sum(),1))})
    else:
        f1=f1_score(yte,pred,average=None,labels=[0,1,2],zero_division=0); rc=recall_score(yte,pred,average=None,labels=[0,1,2],zero_division=0)
        for i,c in enumerate([0,1,2]): d[f"f1_D{c}"]=float(f1[i]); d[f"recall_D{c}"]=float(rc[i])
        d["conf"]=json.dumps(confusion_matrix(yte,pred,labels=[0,1,2]).tolist())
    return d
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int); ybin=(y>0).astype(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
rows=[]; t0=time.time()
for banda in BANDS:
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,30.0)
    print(f"\n=== {banda} kHz ===",flush=True)
    for T_test in FOLDS:
        if np.isclose(T_test,30.0): continue
        te=np.isclose(T,T_test); tr=~te; trh=tr&(y==0)
        def cfg(m,default): return modal.get((banda,m),default)
        try:
            M={"Original":X.copy(),
               "Park":P.comp_park(X,ref,f,cfg("Park",{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5}))[0],
               "RF_direct":P.comp_rf(X,T,ref,trh,cfg("RF_direct",{"n_estimators":300,"max_depth":10,"input_decim":8}),"direct")[0],
               "RF_temponly":P.comp_rf(X,T,ref,trh,{"n_estimators":300,"max_depth":10},"temponly")[0],
               "AE":P.comp_ae(X,T,ref,trh,cfg("AE",{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"epochs":450,"patience":55}),30.0,seed=42)[0],
               "TAU_T":P.comp_tauT(X,T,ref,trh,f,TAU)[0]}
        except Exception as e: print("  err comp:",e); continue
        for mname,Y in M.items():
            for fs in ["FS1","FS2","FS3","FS4"]:
                try: Xtr,Xte=feats(Y,ref,fs,tr,te)
                except Exception: continue
                for cname in ["logreg","svm","rfc"]:
                    for task,yy in [("bin",ybin),("multi",y)]:
                        try:
                            c=clf_factory(cname); c.fit(Xtr,yy[tr]); pr=c.predict(Xte)
                            rec={"banda":banda,"T_ref":30.0,"T_test":T_test,"metodo":mname,"feature_set":fs,"clf":cname,"task":task,"controle":"real"}; rec.update(ev(yy[te],pr,task=="bin")); rows.append(rec)
                            rng=np.random.RandomState(0); ysh=yy[tr].copy(); rng.shuffle(ysh)
                            c2=clf_factory(cname); c2.fit(Xtr,ysh); pr2=c2.predict(Xte)
                            rec2={"banda":banda,"T_ref":30.0,"T_test":T_test,"metodo":mname,"feature_set":fs,"clf":cname,"task":task,"controle":"shuffled"}; rec2.update(ev(yy[te],pr2,task=="bin")); rows.append(rec2)
                        except Exception: pass
        pd.DataFrame(rows).to_csv(CKPT,index=False)
        print(f"  T={T_test} ok | {(time.time()-t0)/60:.1f} min | {len(rows)} regs",flush=True)
d=pd.DataFrame(rows); d.to_csv(CKPT,index=False)
d.to_csv(os.path.join(ROOT,"03_dano_binario","parteB_v2_todos.csv"),index=False)
print(f"\n✅ PARTE B v2: {len(d)} regs | {(time.time()-t0)/60:.1f} min",flush=True)
real=d[d.controle=="real"]
for task in ["bin","multi"]:
    s=real[real.task==task]; cols=["bal_acc","macro_f1"]+(["recall_dano","taxa_falso_saudavel"] if task=="bin" else ["f1_D0","f1_D1","f1_D2"])
    print(f"\n--- {task} ---"); print(s.groupby("metodo")[cols].mean().round(4).sort_values("bal_acc",ascending=False).to_string())
    sh=d[(d.task==task)&(d.controle=="shuffled")].groupby("metodo")["bal_acc"].mean().round(3)
    print("negative control bal_acc:",sh.to_dict())

# -*- coding: utf-8 -*-
"""
PARTE B — RECONHECIMENTO DE DANO (binário e multiclasse), out-of-sample
=======================================================================
Protocolo idêntico para todos os compensadores: LOTO externo, classificador treinado
SÓ nas temperaturas de treino, features derivadas da curva compensada.
Inclui NEGATIVE CONTROL (rótulos embaralhados) para detectar leakage.

Feature sets:
  FS1 = RMSD + CCDM
  FS2 = conjunto completo de métricas (RMSD,CCDM,RMSE,MAE,NRMSE,CORR,SAM)
  FS3 = PCA do resíduo (compensada - referência), PCA ajustado SÓ no treino
  FS4 = features de pico (posição/amplitude dos maiores extremos do resíduo)
Classificadores: LogisticRegression (principal), SVM-RBF, RandomForestClassifier.
"""
import os, sys, json, time, traceback, numpy as np, pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, precision_score, confusion_matrix)

OUTB=os.path.join(ROOT,"03_dano_binario"); OUTM=os.path.join(ROOT,"04_dano_multiclasse")
os.makedirs(OUTB,exist_ok=True); os.makedirs(OUTM,exist_ok=True)
CKPT=os.path.join(ROOT,"checkpoints","parteB.csv")

# Bandas escolhidas: as 4 tunadas na Fase 3/3b. T_ref=30 (fixo p/ comparabilidade).
CONFIGS=[(30,40,30.0),(30,60,30.0),(60,70,30.0),(70,80,30.0)]

# Hiperparâmetros TUNADOS por inner CV (Fase 3 e 3b) — por banda.
TUNED={
 "30-40":{"park":{"max_shift_frac":0.05,"nsteps":121,"smooth_win":9},
          "rf":{"n_estimators":250,"max_depth":6,"min_samples_leaf":2,"min_samples_split":4,"max_features":"sqrt","smooth_win":5},
          "ae":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"dropout":0.10,"noise":0.01,"epochs":500,"patience":70}},
 "30-60":{"park":{"max_shift_frac":0.05,"nsteps":121,"smooth_win":9},
          "rf":{"n_estimators":250,"max_depth":10,"min_samples_leaf":2,"min_samples_split":4,"max_features":"sqrt","smooth_win":5},
          "ae":{"n_input":2000,"n_anchors":128,"latent":16,"hidden":128,"lr":2e-3,"dropout":0.10,"noise":0.01,"epochs":500,"patience":70}},
 "60-70":{"park":{"max_shift_frac":0.25,"nsteps":121,"smooth_win":9},
          "rf":{"n_estimators":250,"max_depth":6,"min_samples_leaf":2,"min_samples_split":4,"max_features":"sqrt","smooth_win":5},
          "ae":{"n_input":2000,"n_anchors":128,"latent":16,"hidden":256,"lr":2e-3,"dropout":0.10,"noise":0.01,"epochs":500,"patience":70}},
 "70-80":{"park":{"max_shift_frac":0.15,"nsteps":121,"smooth_win":9},
          "rf":{"n_estimators":250,"max_depth":10,"min_samples_leaf":2,"min_samples_split":4,"max_features":"sqrt","smooth_win":5},
          "ae":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"dropout":0.10,"noise":0.01,"epochs":500,"patience":70}},
}
PARAMS_TAU ={"rank":8,"max_shift_frac":0.14,"nsteps":81}

def clf_factory(name,seed=0):
    if name=="logreg":
        return Pipeline([("sc",StandardScaler()),("c",LogisticRegression(max_iter=5000,class_weight="balanced",random_state=seed))])
    if name=="svm":
        return Pipeline([("sc",StandardScaler()),("c",SVC(kernel="rbf",class_weight="balanced",random_state=seed))])
    if name=="rfc":
        return Pipeline([("sc",StandardScaler()),("c",RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=-1,random_state=seed))])
    raise ValueError(name)

def feats(Y, ref, fs, tr_mask, te_mask):
    """Retorna (Xtr,Xte). PCA ajustado SÓ no treino."""
    R = Y - ref[None,:]
    if fs in ("FS1","FS2"):
        M=[]
        for i in range(len(Y)):
            m=P.all_metrics(Y[i],ref)
            M.append([m["RMSD"],m["CCDM"]] if fs=="FS1" else
                     [m["RMSD"],m["CCDM"],m["RMSE"],m["MAE"],m["NRMSE"],m["CORR"],m["SAM_deg"]])
        M=np.array(M); return M[tr_mask],M[te_mask]
    if fs=="FS3":
        w=max(1,R.shape[1]//500)
        Rs=np.vstack([P.moving_average(r,51)[::w] for r in R])
        pca=PCA(n_components=min(10,tr_mask.sum()-1),random_state=0).fit(Rs[tr_mask])
        return pca.transform(Rs[tr_mask]),pca.transform(Rs[te_mask])
    if fs=="FS4":
        F=[]
        for r in R:
            rs=P.moving_average(r,51)
            idx=np.argsort(-np.abs(rs))[:200]
            F.append([rs.max(),rs.min(),np.ptp(rs),np.std(rs),np.mean(np.abs(rs)),
                      float(np.mean(idx)),float(np.std(idx)),
                      float(np.percentile(np.abs(rs),95)),float(np.sum(rs**2))])
        F=np.array(F); return F[tr_mask],F[te_mask]
    raise ValueError(fs)

def evaluate(yte,pred,binario):
    d={"accuracy":accuracy_score(yte,pred),"bal_acc":balanced_accuracy_score(yte,pred),
       "macro_f1":f1_score(yte,pred,average="macro",zero_division=0)}
    if binario:
        cm=confusion_matrix(yte,pred,labels=[0,1])
        d["recall_dano"]=recall_score(yte,pred,pos_label=1,zero_division=0)
        d["recall_sem_dano"]=recall_score(yte,pred,pos_label=0,zero_division=0)
        d["precision_dano"]=precision_score(yte,pred,pos_label=1,zero_division=0)
        d["n_falso_saudavel"]=int(cm[1,0]); d["n_dano_real"]=int(cm[1].sum())
        d["taxa_falso_saudavel"]=float(cm[1,0]/max(cm[1].sum(),1))
    else:
        f1=f1_score(yte,pred,average=None,labels=[0,1,2],zero_division=0)
        rc=recall_score(yte,pred,average=None,labels=[0,1,2],zero_division=0)
        for i,c in enumerate([0,1,2]): d[f"f1_D{c}"]=float(f1[i]); d[f"recall_D{c}"]=float(rc[i])
        d["conf"]=json.dumps(confusion_matrix(yte,pred,labels=[0,1,2]).tolist())
    return d

df,_,_=P.load_base()
T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int); ybin=(y>0).astype(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]

rows=[]; t0=time.time()
for (lo,hi,T_ref) in CONFIGS:
    fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64)
    ref,n_ref=P.build_reference(df,fc,T_ref)
    if ref is None: continue
    banda=f"{lo}-{hi}"
    print(f"\n=== {banda} kHz | T_ref={T_ref} | {len(fc)} pts ===")
    for T_test in FOLDS:
        if np.isclose(T_test,T_ref): continue          # fold onde a ref contém o teste
        te=np.isclose(T,T_test); tr=~te; trh=tr&(y==0)
        tp=TUNED[banda]
        try:
            M={"Original":(X.copy(),{}),
               "Park":P.comp_park(X,ref,f,tp["park"]),
               "RF_direct":P.comp_rf(X,T,ref,trh,tp["rf"],"direct"),
               "RF_temponly":P.comp_rf(X,T,ref,trh,tp["rf"],"temponly"),
               "AE":P.comp_ae(X,T,ref,trh,tp["ae"],T_ref,seed=42),
               "TAU_T":P.comp_tauT(X,T,ref,trh,f,PARAMS_TAU)}
        except Exception as e:
            print("  erro compensacao:",e); continue
        for mname,(Y,_) in M.items():
            for fs in ["FS1","FS2","FS3","FS4"]:
                try: Xtr,Xte=feats(Y,ref,fs,tr,te)
                except Exception: continue
                for cname in ["logreg","svm","rfc"]:
                    for task,yy in [("bin",ybin),("multi",y)]:
                        try:
                            c=clf_factory(cname); c.fit(Xtr,yy[tr]); pr=c.predict(Xte)
                            rec={"banda":banda,"T_ref":T_ref,"T_test":T_test,"metodo":mname,
                                 "feature_set":fs,"clf":cname,"task":task,"controle":"real"}
                            rec.update(evaluate(yy[te],pr,task=="bin")); rows.append(rec)
                            # NEGATIVE CONTROL: rótulos embaralhados no treino
                            rng=np.random.RandomState(0); ysh=yy[tr].copy(); rng.shuffle(ysh)
                            c2=clf_factory(cname); c2.fit(Xtr,ysh); pr2=c2.predict(Xte)
                            rec2={**{k:rec[k] for k in ["banda","T_ref","T_test","metodo","feature_set","clf","task"]},
                                  "controle":"shuffled"}
                            rec2.update(evaluate(yy[te],pr2,task=="bin")); rows.append(rec2)
                        except Exception: pass
        print(f"  fold T={T_test} ok ({len(rows)} registros, {(time.time()-t0)/60:.1f} min)")
    pd.DataFrame(rows).to_csv(CKPT,index=False)

d=pd.DataFrame(rows)
d.to_csv(os.path.join(ROOT,"03_dano_binario","parteB_todos_resultados.csv"),index=False)
print(f"\n✅ PARTE B: {len(d)} registros em {(time.time()-t0)/60:.1f} min")

for task,out in [("bin",OUTB),("multi",OUTM)]:
    s=d[(d.task==task)&(d.controle=="real")]
    if not len(s): continue
    cols=["bal_acc","macro_f1"]+(["recall_dano","taxa_falso_saudavel"] if task=="bin" else ["f1_D0","f1_D1","f1_D2"])
    g=s.groupby("metodo")[cols].mean().round(4).sort_values("bal_acc",ascending=False)
    print(f"\n--- {task.upper()} (média sobre bandas/T_ref/folds/features/clfs) ---"); print(g.to_string())
    g.to_csv(os.path.join(out,f"resumo_{task}_por_metodo.csv"))
    sh=d[(d.task==task)&(d.controle=="shuffled")].groupby("metodo")["bal_acc"].mean().round(4)
    print(f"  NEGATIVE CONTROL (rótulos embaralhados) bal_acc: {sh.to_dict()}")
    print(f"  -> esperado ~{0.5 if task=='bin' else 0.33:.2f}; valores muito acima indicam LEAKAGE")

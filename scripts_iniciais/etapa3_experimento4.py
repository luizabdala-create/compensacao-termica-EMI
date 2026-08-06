# -*- coding: utf-8 -*-
"""
ETAPA 3 — EXPERIMENTO 4 (métrica (B) decisiva: CLASSIFICAÇÃO de dano out-of-sample)
==================================================================================
A compensação térmica é útil se, além de remover temperatura (A), PRESERVA/REVELA
o dano de forma classificável. Testamos:
  - front-end de compensação: Original | Park | Fisico_curva(V7)
  - classificador: nearest-centroid sobre o resíduo (y_comp - y_ref), suavizado
  - treino do classificador nas TEMPS DE TREINO, teste nas TEMPS DE TESTE (out-of-sample)
Reporta accuracy, balanced accuracy e matriz de confusão por front-end.
Hipótese: Original confunde (temperatura domina o resíduo); Park remove temperatura
e deixa o dano classificável -> compensação AJUDA a detecção (o objetivo real).
"""
import os, sys, json, numpy as np, pandas as pd
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa3_exp4"); os.makedirs(OUT,exist_ok=True)
SPLITS={
 "B_ref30":{"train":[0,10,30,50,70],"test":[-10,20,40,60],"ref":30.0},
 "A_ref20":{"train":[0,10,20,40,60],"test":[-10,30,50,70],"ref":20.0},
}
def load():
    df=pd.read_pickle(BASE).reset_index(drop=True)
    df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
    df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
    fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX); return df[["temperatura_c","falha"]+fcols].copy(),fcols,fHz
def subset(df,temps):
    m=np.zeros(len(df),bool)
    for T in temps: m|=np.isclose(df["temperatura_c"],T)
    return df[m].copy()

def comp_park(X,y_ref,fHz):
    Y=np.zeros_like(X)
    for i in range(len(X)): Y[i]=v7.park_single(X[i],y_ref,fHz)
    return Y
def comp_fisico(X,T,df_tr,fcols,fHz,ref):
    v7._CALIBRATION_CACHE.clear()
    hbt,tt,y_ref,_=v7.get_healthy_references_by_temperature(df_tr,fcols,ref)
    Y=np.zeros_like(X)
    for i in range(len(X)):
        p=v7.get_calibration_params_for_T(float(T[i]),hbt,tt,y_ref,fHz)
        Y[i]=v7.apply_calibration_to_curve(X[i],p,fHz,alpha=1.0)
    return Y

def smooth_resid(Y,y_ref,win=51):
    R=Y-y_ref[None,:]
    return np.vstack([v7.moving_average(r,win) for r in R])

def nc_classify(Rtr,ytr,Rte):
    cents={c:Rtr[ytr==c].mean(0) for c in np.unique(ytr)}
    cs=sorted(cents); C=np.vstack([cents[c] for c in cs])
    preds=[]
    for r in Rte:
        d=np.sqrt(((C-r)**2).sum(1)); preds.append(cs[int(np.argmin(d))])
    return np.array(preds)

def bal_acc(y,p):
    accs=[]
    for c in np.unique(y):
        m=y==c
        if m.sum(): accs.append((p[m]==c).mean())
    return float(np.mean(accs))
def confusion(y,p,classes=(0,1,2)):
    M=np.zeros((3,3),int)
    for yi,pi in zip(y,p): M[int(yi),int(pi)]+=1
    return M

df,fcols,fHz=load(); rep=[]
for name,cfg in SPLITS.items():
    print("="*90); print(f"SPLIT {name} treino={cfg['train']} teste={cfg['test']} ref={cfg['ref']}"); print("="*90)
    df_tr=subset(df,cfg["train"]); df_te=subset(df,cfg["test"])
    v7._CALIBRATION_CACHE.clear(); v7.REF_TEMP=cfg["ref"]
    hbt,tt,y_ref,_=v7.get_healthy_references_by_temperature(df_tr,fcols,cfg["ref"])
    Xtr=df_tr[fcols].to_numpy(np.float64); ytr=df_tr["falha"].to_numpy(int); Ttr=df_tr["temperatura_c"].to_numpy(float)
    Xte=df_te[fcols].to_numpy(np.float64); yte=df_te["falha"].to_numpy(int); Tte=df_te["temperatura_c"].to_numpy(float)
    fronts={}
    fronts["Original"]=(Xtr.copy(),Xte.copy())
    fronts["Park"]=(comp_park(Xtr,y_ref,fHz),comp_park(Xte,y_ref,fHz))
    fronts["Fisico_curva"]=(comp_fisico(Xtr,Ttr,df_tr,fcols,fHz,cfg["ref"]),comp_fisico(Xte,Tte,df_tr,fcols,fHz,cfg["ref"]))
    res={}
    for fn,(Gtr,Gte) in fronts.items():
        Rtr=smooth_resid(Gtr,y_ref); Rte=smooth_resid(Gte,y_ref)
        p=nc_classify(Rtr,ytr,Rte)
        acc=float((p==yte).mean()); ba=bal_acc(yte,p); M=confusion(yte,p)
        res[fn]={"acc":acc,"bal_acc":ba,"confusion":M.tolist()}
        print(f"\n[{fn}]  acc={acc:.3f} | balanced_acc={ba:.3f}")
        print("   matriz confusão (linha=verdadeiro 0/1/2, coluna=predito):")
        for r in M: print("     ",r)
    rep.append({"name":name,"n_test":len(yte),"res":res})
with open(os.path.join(OUT,"exp4_resumo.json"),"w",encoding="utf-8") as f: json.dump(rep,f,indent=2,ensure_ascii=False)
print(f"\n✅ salvo em {os.path.join(OUT,'exp4_resumo.json')}")

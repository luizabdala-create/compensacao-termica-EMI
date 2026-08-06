# -*- coding: utf-8 -*-
"""
ETAPA 5 — EXP 6: versão HONESTA do AE-linear que bate o Park
============================================================
Corrige o leakage de hiperparâmetro: o nº k de componentes térmicas removidas é
escolhido por validação cruzada leave-one-temperature-out DENTRO do treino, e só
então avaliado no teste. Sem olhar o teste para escolher k.

Pipeline (treinado só no saudável para a parte térmica):
  1) registro por shift multiescala a y_ref (não é Park)
  2) R = alinhado - y_ref
  3) PCA do subespaço térmico nas SAUDÁVEIS de treino; remove k comps
  4) classifica R_dano (nearest-centroid euclidiano), centróides do treino
Seleção de k: LOTO-CV no treino (maximiza balanced accuracy média).
Reporta teste no k* e compara com Park.
"""
import os, sys, json, numpy as np, pandas as pd
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa5_exp6"); os.makedirs(OUT,exist_ok=True)
SPLITS={
 "B_ref30":{"train":[0,10,30,50,70],"test":[-10,20,40,60],"ref":30.0},
 "A_ref20":{"train":[0,10,20,40,60],"test":[-10,30,50,70],"ref":20.0},
}
KS=[0,1,2,3,4,5,6,8,10,12]

def load():
    df=pd.read_pickle(BASE).reset_index(drop=True)
    df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
    df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
    fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX); return df[["temperatura_c","falha"]+fcols].copy(),fcols,fHz
def subset(df,temps):
    m=np.zeros(len(df),bool)
    for T in temps: m|=np.isclose(df["temperatura_c"],T)
    return df[m].copy()
def align_to_ref(X,y_ref,fHz):
    Y=np.zeros_like(X)
    for i in range(len(X)):
        tau=v7.estimate_shift_multiscale(X[i],y_ref,fHz,max_frac=0.14,n_coarse=101,n_fine=81,
             prior_tau=0.0,prior_penalty=0.015,min_improvement=0.0,return_info=False)
        Y[i]=v7.shift_interp(X[i],fHz,tau)
    return Y
def smooth_rows(R,win=51): return np.vstack([v7.moving_average(r,win) for r in R])
def nc(Rtr,ytr,Rte):
    cs=sorted(np.unique(ytr)); C=np.vstack([Rtr[ytr==c].mean(0) for c in cs])
    return np.array([cs[int(np.argmin(((C-r)**2).sum(1)))] for r in Rte])
def bal(y,p):
    u=np.unique(y); return float(np.mean([(p[y==c]==c).mean() for c in u]))
def conf(y,p):
    M=np.zeros((3,3),int)
    for yi,pi in zip(y,p): M[int(yi),int(pi)]+=1
    return M

def thermal_remove(Rtr_full,ytr,Rte_full,k):
    """PCA do saudável de treino; remove k comps de treino e teste."""
    Rh=Rtr_full[ytr==0]; mu=Rh.mean(0)
    if k==0: return Rtr_full.copy(),Rte_full.copy()
    _,_,Vt=np.linalg.svd(Rh-mu,full_matrices=False); Vk=Vt[:k]
    Rtr_d=(Rtr_full-mu)-((Rtr_full-mu)@Vk.T)@Vk
    Rte_d=(Rte_full-mu)-((Rte_full-mu)@Vk.T)@Vk
    return Rtr_d,Rte_d

df,fcols,fHz=load(); rep=[]
for name,cfg in SPLITS.items():
    print("="*92); print(f"SPLIT {name} treino={cfg['train']} teste={cfg['test']} ref={cfg['ref']}"); print("="*92)
    df_tr=subset(df,cfg["train"]).reset_index(drop=True); df_te=subset(df,cfg["test"]).reset_index(drop=True)
    v7._CALIBRATION_CACHE.clear(); v7.REF_TEMP=cfg["ref"]
    hbt,tt,y_ref,_=v7.get_healthy_references_by_temperature(df_tr,fcols,cfg["ref"])
    Xtr=df_tr[fcols].to_numpy(np.float64); ytr=df_tr["falha"].to_numpy(int); Ttr=df_tr["temperatura_c"].to_numpy(float)
    Xte=df_te[fcols].to_numpy(np.float64); yte=df_te["falha"].to_numpy(int)

    A_tr=align_to_ref(Xtr,y_ref,fHz); A_te=align_to_ref(Xte,y_ref,fHz)
    Rtr_full=A_tr-y_ref[None,:]; Rte_full=A_te-y_ref[None,:]

    # --- LOTO-CV no TREINO para escolher k (sem olhar teste) ---
    train_temps=sorted(np.unique(Ttr)); cv_scores={k:[] for k in KS}
    for thold in train_temps:
        hold=np.isclose(Ttr,thold); keep=~hold
        if hold.sum()==0 or keep.sum()==0: continue
        Rk_tr=Rtr_full[keep]; yk_tr=ytr[keep]; Rk_ho=Rtr_full[hold]; yk_ho=ytr[hold]
        if len(np.unique(yk_ho))<2:
            pass
        for k in KS:
            Rd_tr,Rd_ho=thermal_remove(Rk_tr,yk_tr,Rk_ho,k)
            p=nc(smooth_rows(Rd_tr),yk_tr,smooth_rows(Rd_ho))
            cv_scores[k].append(bal(yk_ho,p))
    cv_mean={k:float(np.mean(v)) for k,v in cv_scores.items() if len(v)}
    kstar=max(cv_mean,key=cv_mean.get)
    print("CV (treino) balanced-acc por k:")
    for k in KS:
        if k in cv_mean: print(f"    k={k:2d}: CV={cv_mean[k]:.3f}"+("  <= k* escolhido" if k==kstar else ""))

    # --- Park baseline no teste ---
    Pk_tr=np.vstack([v7.park_single(x,y_ref,fHz) for x in Xtr]); Pk_te=np.vstack([v7.park_single(x,y_ref,fHz) for x in Xte])
    park_p=nc(smooth_rows(Pk_tr-y_ref[None,:]),ytr,smooth_rows(Pk_te-y_ref[None,:])); park_ba=bal(yte,park_p)

    # --- teste no k* ---
    Rd_tr,Rd_te=thermal_remove(Rtr_full,ytr,Rte_full,kstar)
    p=nc(smooth_rows(Rd_tr),ytr,smooth_rows(Rd_te)); ae_ba=bal(yte,p)
    print(f"\n>>> TESTE: AE-linear(k*={kstar}) balanced-acc = {ae_ba:.3f}  |  Park = {park_ba:.3f}  |  Δ = {ae_ba-park_ba:+.3f}")
    print(f"    Matriz confusão AE (linha=verdadeiro, coluna=predito):")
    for r in conf(yte,p): print("      ",r)
    rep.append({"name":name,"kstar":int(kstar),"cv_mean":cv_mean,
                "ae_test_balacc":ae_ba,"park_test_balacc":park_ba,"conf_ae":conf(yte,p).tolist()})

with open(os.path.join(OUT,"exp6_honesto.json"),"w",encoding="utf-8") as f: json.dump(rep,f,indent=2,ensure_ascii=False)
print(f"\n✅ salvo em {os.path.join(OUT,'exp6_honesto.json')}")

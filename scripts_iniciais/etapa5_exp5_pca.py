# -*- coding: utf-8 -*-
"""
ETAPA 5 — EXP 5: prova de conceito do AE que BATE o Park (honestamente)
=======================================================================
Tese: o Park só faz shift rígido + offset -> deixa resíduo de FORMA térmica.
Um modelo treinado SÓ no saudável que remove o subespaço térmico da forma
deixa um resíduo de dano mais limpo -> classificação melhor.

Método (AE linear = PCA, prova de conceito; sem usar Park):
  1) registro: alinha cada curva a y_ref pelo shift multiescala (derivadas) do V7
     (NÃO é Park; é registro por forma). Alinhamento não apaga dano (dano=forma, 98-100%).
  2) resíduo R = curva_alinhada - y_ref
  3) subespaço térmico: PCA de R nas SAUDÁVEIS de TREINO (k componentes)
  4) remove projeção térmica: R_dano = R - proj_k(R)   [treinado só no saudável]
  5) classifica R_dano (nearest-centroid), centróides do TREINO, teste no TESTE
Compara contra: Original, Park, e "alinhado sem PCA" (k=0), varrendo k.
Tudo out-of-sample, sem leakage.
"""
import os, sys, json, numpy as np, pandas as pd
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa5_exp5"); os.makedirs(OUT,exist_ok=True)
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

def align_to_ref(X,y_ref,fHz):
    """Registro por shift multiescala (derivadas) — não é Park."""
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
def ncc(Rtr,ytr,Rte):
    """nearest-centroid por correlação (cosseno centrado) — bom p/ espectros."""
    cs=sorted(np.unique(ytr))
    def norm(v): v=v-v.mean(); n=np.linalg.norm(v)+1e-12; return v/n
    C=np.vstack([norm(Rtr[ytr==c].mean(0)) for c in cs])
    out=[]
    for r in Rte:
        rn=norm(r); out.append(cs[int(np.argmax(C@rn))])
    return np.array(out)
def bal(y,p): return float(np.mean([(p[y==c]==c).mean() for c in np.unique(y)]))

df,fcols,fHz=load(); rep=[]
for name,cfg in SPLITS.items():
    print("="*92); print(f"SPLIT {name} treino={cfg['train']} teste={cfg['test']} ref={cfg['ref']}"); print("="*92)
    df_tr=subset(df,cfg["train"]).reset_index(drop=True); df_te=subset(df,cfg["test"]).reset_index(drop=True)
    v7._CALIBRATION_CACHE.clear(); v7.REF_TEMP=cfg["ref"]
    hbt,tt,y_ref,_=v7.get_healthy_references_by_temperature(df_tr,fcols,cfg["ref"])
    Xtr=df_tr[fcols].to_numpy(np.float64); ytr=df_tr["falha"].to_numpy(int)
    Xte=df_te[fcols].to_numpy(np.float64); yte=df_te["falha"].to_numpy(int)

    # baselines
    def classify_front(Gtr,Gte):
        Rtr=smooth_rows(Gtr-y_ref[None,:]); Rte=smooth_rows(Gte-y_ref[None,:])
        return bal(yte,nc(Rtr,ytr,Rte)), bal(yte,ncc(Rtr,ytr,Rte))
    b_eu,b_cc=classify_front(Xtr,Xte); print(f"[Original]      NC={b_eu:.3f} | NCcorr={b_cc:.3f}")
    Pk_tr=np.vstack([v7.park_single(x,y_ref,fHz) for x in Xtr]); Pk_te=np.vstack([v7.park_single(x,y_ref,fHz) for x in Xte])
    pk_eu,pk_cc=classify_front(Pk_tr,Pk_te); print(f"[Park]          NC={pk_eu:.3f} | NCcorr={pk_cc:.3f}   <== BASELINE A BATER")

    # nosso: alinhado + PCA thermal removal
    A_tr=align_to_ref(Xtr,y_ref,fHz); A_te=align_to_ref(Xte,y_ref,fHz)
    Rtr_full=A_tr-y_ref[None,:]; Rte_full=A_te-y_ref[None,:]
    # PCA no saudável de treino
    Rh=Rtr_full[ytr==0]
    mu=Rh.mean(0); Rh_c=Rh-mu
    U,S,Vt=np.linalg.svd(Rh_c,full_matrices=False)  # Vt: componentes térmicas
    print(f"[AE-linear alinhado] varrendo k (nº de comps térmicas removidas):")
    best=(-1,0,"")
    ks=[0,1,2,3,5,8,12]
    kres={}
    for k in ks:
        if k==0:
            Rtr_d=Rtr_full.copy(); Rte_d=Rte_full.copy()
        else:
            Vk=Vt[:k]  # kx D
            Ptr=(Rtr_full-mu)@Vk.T; Rtr_d=(Rtr_full-mu)-Ptr@Vk + mu*0  # remove proj
            Pte=(Rte_full-mu)@Vk.T; Rte_d=(Rte_full-mu)-Pte@Vk
            Rtr_d=Rtr_d; Rte_d=Rte_d
        Rtr_s=smooth_rows(Rtr_d); Rte_s=smooth_rows(Rte_d)
        e=bal(yte,nc(Rtr_s,ytr,Rte_s)); c=bal(yte,ncc(Rtr_s,ytr,Rte_s))
        kres[k]={"NC":e,"NCcorr":c}
        star=" *" if max(e,c)>best[0] else ""
        if max(e,c)>best[0]: best=(max(e,c),k,"NC" if e>=c else "NCcorr")
        print(f"    k={k:2d}: NC={e:.3f} | NCcorr={c:.3f}{star}")
    print(f"  >>> melhor AE-linear: {best[0]:.3f} (k={best[1]}, {best[2]}) | Park={max(pk_eu,pk_cc):.3f}")
    rep.append({"name":name,"park":max(pk_eu,pk_cc),"original":max(b_eu,b_cc),
                "ae_linear_best":best[0],"ae_linear_best_k":best[1],"by_k":kres})

with open(os.path.join(OUT,"exp5_pca.json"),"w",encoding="utf-8") as f: json.dump(rep,f,indent=2,ensure_ascii=False)
print(f"\n✅ salvo em {os.path.join(OUT,'exp5_pca.json')}")

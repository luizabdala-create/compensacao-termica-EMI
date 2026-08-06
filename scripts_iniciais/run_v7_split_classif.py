# -*- coding: utf-8 -*-
"""
Fecha a ETAPA 3: V7 COMPLETO out-of-sample (treina AE só no treino, aplica no teste),
sem leakage, e compara (A) remoção térmica e (B) classificação de dano contra Park.
Front-ends: Original | Park | Fisico | AE_V7_completo (físico+AEres+post+microshift).
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa3_v7_split"); os.makedirs(OUT,exist_ok=True)
v7.OUTPUT_DIR=OUT  # p/ CSVs internos do V7 caírem aqui
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

def fisico_df(df,fcols,fHz,hbt,tt,y_ref):
    X=df[fcols].to_numpy(np.float64); T=df["temperatura_c"].to_numpy(float)
    Y=np.zeros_like(X)
    for i in range(len(X)):
        p=v7.get_calibration_params_for_T(float(T[i]),hbt,tt,y_ref,fHz)
        Y[i]=v7.apply_calibration_to_curve(X[i],p,fHz,alpha=1.0)
    return Y

def ae_blend(df,Y_phys,ae_pack,fcols,fHz,y_ref):
    residual_pred,t_pred=v7.predizer_residual_ae(df,Y_phys,ae_pack)
    n_points=Y_phys.shape[1]; win=v7.odd_window_from_frac(n_points,v7.RESIDUAL_SMOOTH_WIN_FRAC,minimum=7)
    lower=ae_pack["prep"]["residual_lower"]; upper=ae_pack["prep"]["residual_upper"]
    Y=np.zeros_like(Y_phys)
    for i in range(len(Y_phys)):
        r=v7.moving_average(residual_pred[i],win).astype(np.float32); r=np.clip(r,lower,upper)
        w=v7.peak_protection_weight(Y_phys[i])*v7.damage_protection_weight(Y_phys[i],y_ref)
        Y[i]=(Y_phys[i]+v7.AE_RESIDUAL_BLEND*w*r)
    meta=df.drop(columns=fcols).copy()
    dfae=pd.concat([meta,pd.DataFrame(Y,columns=fcols,index=df.index)],axis=1)
    dfae["temperatura_ae_pred"]=t_pred
    return dfae

def smooth_resid(Y,y_ref,win=51): return np.vstack([v7.moving_average(r,win) for r in (Y-y_ref[None,:])])
def nc(Rtr,ytr,Rte):
    cs=sorted(np.unique(ytr)); C=np.vstack([Rtr[ytr==c].mean(0) for c in cs])
    return np.array([cs[int(np.argmin(((C-r)**2).sum(1)))] for r in Rte])
def bal(y,p): return float(np.mean([(p[y==c]==c).mean() for c in np.unique(y)]))
def conf(y,p):
    M=np.zeros((3,3),int)
    for yi,pi in zip(y,p): M[int(yi),int(pi)]+=1
    return M
def rmsd0(Y,y_ref,fal):
    idx=np.where(fal==0)[0]; return float(np.mean([v7.rmsd(Y[i],y_ref) for i in idx]))

df,fcols,fHz=load(); rep=[]
for name,cfg in SPLITS.items():
    print("="*90); print(f"SPLIT {name}"); print("="*90)
    df_tr=subset(df,cfg["train"]).reset_index(drop=True); df_te=subset(df,cfg["test"]).reset_index(drop=True)
    v7._CALIBRATION_CACHE.clear(); v7.REF_TEMP=cfg["ref"]
    hbt,tt,y_ref,_=v7.get_healthy_references_by_temperature(df_tr,fcols,cfg["ref"])
    ytr=df_tr["falha"].to_numpy(int); yte=df_te["falha"].to_numpy(int)

    # treina AE só no treino
    ae_pack=v7.treinar_residual_ae(df_tr,fcols,fHz,y_ref,hbt,tt)

    # físico
    Yph_tr=fisico_df(df_tr,fcols,fHz,hbt,tt,y_ref); Yph_te=fisico_df(df_te,fcols,fHz,hbt,tt,y_ref)
    # AE blend
    dfae_tr=ae_blend(df_tr,Yph_tr,ae_pack,fcols,fHz,y_ref)
    dfae_te=ae_blend(df_te,Yph_te,ae_pack,fcols,fHz,y_ref)
    # post-centering: pack aprendido no TREINO, aplicado a treino e teste
    post_pack=v7.build_post_center_pack(dfae_tr,fcols,fHz,y_ref)
    dfae_tr_p=v7.apply_post_centering(dfae_tr,fcols,fHz,y_ref,post_pack)
    dfae_te_p=v7.apply_post_centering(dfae_te,fcols,fHz,y_ref,post_pack)

    fronts={
      "Original":(df_tr[fcols].to_numpy(np.float64),df_te[fcols].to_numpy(np.float64)),
      "Park":(np.vstack([v7.park_single(x,y_ref,fHz) for x in df_tr[fcols].to_numpy(np.float64)]),
              np.vstack([v7.park_single(x,y_ref,fHz) for x in df_te[fcols].to_numpy(np.float64)])),
      "Fisico":(Yph_tr,Yph_te),
      "AE_V7_completo":(dfae_tr_p[fcols].to_numpy(np.float64),dfae_te_p[fcols].to_numpy(np.float64)),
    }
    res={}
    for fn,(Gtr,Gte) in fronts.items():
        p=nc(smooth_resid(Gtr,y_ref),ytr,smooth_resid(Gte,y_ref))
        res[fn]={"A_rmsd_saudavel":rmsd0(Gte,y_ref,yte),"bal_acc":bal(yte,p),"conf":conf(yte,p).tolist()}
        print(f"[{fn:16s}] (A) RMSD saud={res[fn]['A_rmsd_saudavel']:7.3f} | (B) bal_acc={res[fn]['bal_acc']:.3f} | conf={res[fn]['conf']}")
    rep.append({"name":name,"res":res})

with open(os.path.join(OUT,"v7_split_classif.json"),"w",encoding="utf-8") as f: json.dump(rep,f,indent=2,ensure_ascii=False)
print(f"\n✅ salvo em {os.path.join(OUT,'v7_split_classif.json')}")

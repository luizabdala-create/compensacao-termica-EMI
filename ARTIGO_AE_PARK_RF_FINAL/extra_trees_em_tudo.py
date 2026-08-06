# -*- coding: utf-8 -*-
"""
EXTRA TREES EM TODA A ANÁLISE (decisão: método principal da família 'florestas de árvores').
Recomputa o Extra Trees com o MESMO schema e métricas do fase8 (todas as métricas D0/D1/D2, sep,
healthy_sep, full_order) para cada banda × temperatura, e ADICIONA como método 'ExtraTrees' ao
fase8_tuning_ampliado.csv. Assim, todas as figuras/tabelas/estatísticas que leem esse CSV passam
a incluir o Extra Trees automaticamente (comparação por banda, CD, heatmaps, Pareto, distribuições).
Também estende a varredura de T_ref (tref_full.csv) com o Extra Trees.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
from sklearn.ensemble import ExtraTreesRegressor
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"02_compensacao"); CSV=os.path.join(OUT,"fase8_tuning_ampliado.csv"); T_REF=30.0
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]=="RF_direct"]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else {"n_estimators":300,"max_depth":10,"min_samples_leaf":1,"max_features":"sqrt","smooth_win":5,"input_decim":4}
def comp_et(X,T,ref,mask,params):
    idc=params.get("input_decim",4); Xh=X[mask]; Th=T[mask]
    et=ExtraTreesRegressor(n_estimators=params.get("n_estimators",300),max_depth=params.get("max_depth",10),
        min_samples_leaf=params.get("min_samples_leaf",1),max_features=params.get("max_features","sqrt"),n_jobs=-1,random_state=0)
    et.fit(P._rf_feats(Xh,Th,"direct",idc),ref[None,:]-Xh)
    Y=X+et.predict(P._rf_feats(X,T,"direct",idc)); w=params.get("smooth_win",5)
    if w>1: Y=np.vstack([P.moving_average(yy,w) for yy in Y])
    return Y
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]

# ---------- (1) fase8 + ExtraTrees ----------
old=pd.read_csv(CSV); cols=list(old.columns)
if "ExtraTrees" in old.metodo.unique():
    print("ExtraTrees já presente no fase8 — pulando recomputo",flush=True)
else:
    bands=sorted(old.banda.unique()); rows=[]; t0=time.time()
    for banda in bands:
        lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
        X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF); cf=cfg(banda)
        for T_test in FOLDS:
            if np.isclose(T_test,T_REF): continue
            te=np.isclose(T,T_test); trh=(~te)&(y==0)
            if trh.sum()<8: continue
            Y=comp_et(X,T,ref,trh,cf)
            rec={"banda":banda,"T_test":T_test,"metodo":"ExtraTrees","cv_inner":np.nan,"cfg":json.dumps(cf)}
            pc={}
            for c in [0,1,2]:
                ii=np.where(te&(y==c))[0]
                if len(ii)==0: continue
                M=[P.all_metrics(Y[i],ref) for i in ii]
                for kk in M[0]: rec[f"{kk}_D{c}"]=float(np.mean([mm[kk] for mm in M]))
                pc[c]=rec[f"RMSD_D{c}"]
            rec["alteracao_RMSD"]=float(np.mean([P.rmsd(Y[i],X[i]) for i in np.where(te)[0]]))
            if {0,1,2}<=set(pc):
                rec["sep_RMSD_D1"]=pc[1]-pc[0]; rec["sep_RMSD_D2"]=pc[2]-pc[0]
                hs,fo=P.monotonicity(pc); rec["healthy_sep"]=hs; rec["full_order"]=fo
            rows.append(rec)
        print(f"  ET fase8 {banda} ok | {(time.time()-t0)/60:.1f} min",flush=True)
    new=pd.DataFrame(rows)
    for c in cols:
        if c not in new.columns: new[c]=np.nan
    new=new[cols]
    out=pd.concat([old,new],ignore_index=True)
    out.to_csv(CSV,index=False); out.to_csv(os.path.join(ROOT,"checkpoints","fase8_tuning.csv"),index=False)
    print(f"✅ fase8 + ExtraTrees: {out.shape} | {out.metodo.nunique()} métodos",flush=True)

# ---------- (2) tref_full + ExtraTrees ----------
TF=os.path.join(ROOT,"06_sensibilidade_referencia","tref_full.csv")
if os.path.exists(TF):
    tf=pd.read_csv(TF)
    if "ExtraTrees" in tf.metodo.unique():
        print("ExtraTrees já em tref_full — pulando",flush=True)
    else:
        TREFS=sorted(tf.T_ref.unique()); tbands=sorted(tf.banda.unique()); rows=[]; t0=time.time()
        for banda in tbands:
            lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
            X=df[fc].to_numpy(np.float64); cf=cfg(banda)
            for T_ref in TREFS:
                m0=np.isclose(T,T_ref)&(y==0)
                if m0.sum()<2: continue
                ref=np.median(X[m0],axis=0)
                for T_test in FOLDS:
                    if np.isclose(T_test,T_ref): continue
                    te=np.isclose(T,T_test); trh=(~te)&(y==0)
                    if trh.sum()<8: continue
                    Y=comp_et(X,T,ref,trh,cf); rec={"banda":banda,"T_ref":T_ref,"T_test":T_test,"metodo":"ExtraTrees","dist":abs(T_test-T_ref)}
                    for c in [0,1,2]:
                        ii=np.where(te&(y==c))[0]
                        if len(ii): rec[f"RMSD_D{c}"]=float(np.mean([P.rmsd(Y[i],ref) for i in ii]))
                    i0=np.where(te&(y==0))[0]; rec["CCDM_D0"]=float(np.mean([P.ccdm(Y[i],ref) for i in i0])) if len(i0) else np.nan
                    rows.append(rec)
            print(f"  ET tref {banda} ok | {(time.time()-t0)/60:.1f} min",flush=True)
        tf2=pd.concat([tf,pd.DataFrame(rows)],ignore_index=True); tf2.to_csv(TF,index=False)
        print(f"✅ tref_full + ExtraTrees: {tf2.shape}",flush=True)
print("\n✅ Extra Trees integrado às análises principais",flush=True)

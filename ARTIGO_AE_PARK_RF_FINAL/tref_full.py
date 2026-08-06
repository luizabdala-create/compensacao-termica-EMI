# -*- coding: utf-8 -*-
"""
VARREDURA COMPLETA DE TEMPERATURA DE REFERÊNCIA.
T_ref em {-10,0,10,20,30,40,50,60,70,80} × bandas {40-50, 70-80}, configs tunadas fixas.
Para cada T_ref, LOTO nas demais temperaturas com 3 classes. Diz qual T_ref minimiza
o RMSD saudável e como cada método degrada com a distância à referência.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"06_sensibilidade_referencia"); os.makedirs(OUT,exist_ok=True)
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda,m,default):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else default
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
TREFS=[-10,0,10,20,30,40,50,60,70,80]; BANDS=["40-50","70-80"]
rows=[]; t0=time.time()
for banda in BANDS:
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64)
    cP=cfg(banda,"Park",{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5})
    cR=cfg(banda,"RF_direct",{"n_estimators":300,"max_depth":10,"input_decim":8,"smooth_win":5})
    cA=cfg(banda,"AE",{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"epochs":450,"patience":55})
    print(f"\n=== {banda} ===",flush=True)
    for T_REF in TREFS:
        m0=np.isclose(T,T_REF)&(y==0)
        if m0.sum()<2: print(f"  T_ref={T_REF}: sem saudável"); continue
        ref=np.median(X[m0],axis=0)
        for T_test in FOLDS:
            if np.isclose(T_test,T_REF): continue
            te=np.isclose(T,T_test); trh=(~te)&(y==0)
            comps={"Original":X,"Park":P.comp_park(X,ref,f,cP)[0],
                   "RF_direct":P.comp_rf(X,T,ref,trh,cR,"direct")[0],
                   "AE":P.comp_ae(X,T,ref,trh,cA,float(T_REF),seed=42)[0]}
            for m,Y in comps.items():
                pc={}
                for c in [0,1,2]:
                    ii=np.where(te&(y==c))[0]
                    if len(ii):
                        pc[c]=float(np.mean([P.rmsd(Y[i],ref) for i in ii]))
                rec={"banda":banda,"T_ref":T_REF,"T_test":T_test,"metodo":m,"dist":abs(T_test-T_REF)}
                for c in pc: rec[f"RMSD_D{c}"]=pc[c]
                rec["CCDM_D0"]=float(np.mean([P.ccdm(Y[i],ref) for i in np.where(te&(y==0))[0]])) if (te&(y==0)).sum() else np.nan
                rows.append(rec)
        print(f"  T_ref={T_REF} ok | {(time.time()-t0)/60:.1f} min",flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"tref_full.csv"),index=False)
print(f"\n✅ {len(d)} regs | {(time.time()-t0)/60:.1f} min",flush=True)
print("\n=== RMSD_D0 médio por T_ref (banda 40-50) ===")
for banda in BANDS:
    s=d[d.banda==banda]
    pv=s.pivot_table(index="T_ref",columns="metodo",values="RMSD_D0",aggfunc="mean")[["AE","Park","RF_direct"]].round(3)
    print(f"\n[{banda}]"); print(pv.to_string())
    for m in ["AE","Park","RF_direct"]:
        print(f"  melhor T_ref p/ {m}: {pv[m].idxmin()} ({pv[m].min():.3f}) | pior: {pv[m].idxmax()} ({pv[m].max():.3f})")

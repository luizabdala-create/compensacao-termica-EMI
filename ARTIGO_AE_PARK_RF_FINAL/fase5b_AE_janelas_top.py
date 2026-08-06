# -*- coding: utf-8 -*-
"""AE nas melhores janelas GLOBAIS (fairness: o coarse-to-fine só testou regiões dos
métodos rápidos). Também Park/RF/TAU_T nas mesmas p/ tabela comparável."""
import os,sys,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"07_janelas")
# top janelas globais + janelas fortes de cada método
WINS=[(75,78),(90,93),(60,63),(72,75),(70,80),(45,48),(60,65),(30,33)]
T_REF=30.0
AE={"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"dropout":0.10,"noise":0.01,"epochs":450,"patience":65}
PARK={"max_shift_frac":0.15,"nsteps":121,"smooth_win":9}
RF={"n_estimators":250,"max_depth":10,"min_samples_leaf":2,"min_samples_split":4,"max_features":"sqrt","smooth_win":5}
TAU={"rank":8,"max_shift_frac":0.14,"nsteps":81}
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
rows=[]
for lo,hi in WINS:
    fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
    acc={m:{"r0":[],"hs":[],"s1":[],"s2":[]} for m in ["AE","Park","RF_direct","TAU_T"]}
    for T_test in FOLDS:
        if np.isclose(T_test,T_REF): continue
        te=np.isclose(T,T_test); trh=(~te)&(y==0)
        try:
            comps={"AE":P.comp_ae(X,T,ref,trh,AE,T_REF,seed=42)[0],
                   "Park":P.comp_park(X,ref,f,PARK)[0],
                   "RF_direct":P.comp_rf(X,T,ref,trh,RF,"direct")[0],
                   "TAU_T":P.comp_tauT(X,T,ref,trh,f,TAU)[0]}
        except Exception as e: print("err",lo,hi,e); continue
        for m,Y in comps.items():
            pc={}
            for c in [0,1,2]:
                ii=np.where(te&(y==c))[0]
                if len(ii): pc[c]=float(np.mean([P.rmsd(Y[i],ref) for i in ii]))
            if 0 in pc: acc[m]["r0"].append(pc[0])
            if {0,1,2}<=set(pc):
                acc[m]["hs"].append(P.monotonicity(pc)[0]); acc[m]["s1"].append(pc[1]-pc[0]); acc[m]["s2"].append(pc[2]-pc[0])
    for m,v in acc.items():
        if v["r0"]:
            rows.append({"janela":f"{lo}-{hi}","largura":hi-lo,"metodo":m,
                "RMSD_D0":float(np.mean(v["r0"])),"healthy_sep":float(np.mean(v["hs"])) if v["hs"] else np.nan,
                "sep_D1":float(np.mean(v["s1"])) if v["s1"] else np.nan,"sep_D2":float(np.mean(v["s2"])) if v["s2"] else np.nan})
    print(f"{lo}-{hi}: "+" ".join(f"{m}={np.mean(acc[m]['r0']):.3f}" for m in acc if acc[m]['r0']))
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"janelas_top_4metodos.csv"),index=False)
print("\n"+"="*70); print("MELHORES JANELAS — 4 métodos (RMSD_D0, só healthy_sep=1)"); print("="*70)
piv=d.pivot_table(index="janela",columns="metodo",values="RMSD_D0").round(3)
print(piv.to_string())
print("\nvencedor por janela:")
for j in piv.index:
    print(f"   {j:8s}: {piv.loc[j].idxmin()} ({piv.loc[j].min():.3f})")
print(f"\nmelhor RMSD_D0 global: {d.loc[d.RMSD_D0.idxmin(),'metodo']} em {d.loc[d.RMSD_D0.idxmin(),'janela']} = {d.RMSD_D0.min():.3f}")

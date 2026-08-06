# -*- coding: utf-8 -*-
"""
FASE 4/5 — JANELAS MÓVEIS + JANELAS EM PICOS/VALES (coarse-to-fine)
==================================================================
Coarse: varre janelas móveis de largura {3,5,10} kHz em 30-100 kHz.
Métodos rápidos (Park, RF_direct, TAU_T) — seleção de região por RMSD_D0 e healthy_sep,
avaliada por LOTO nas 10 temperaturas com 3 classes. O AE é caro; roda depois só nas
melhores regiões (fase5, chamada no fim). Janelas em picos detectadas na referência.
"""
import os,sys,time,re,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"07_janelas"); os.makedirs(OUT,exist_ok=True)
CKPT=os.path.join(ROOT,"checkpoints","fase4_janelas.csv")
T_REF=30.0
WIDTHS=[3,5,10]; STEP={3:3,5:5,10:5}   # step coarse
LO,HI=30,100

df,_,_=P.load_base()
T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
PARK={"max_shift_frac":0.10,"nsteps":81,"smooth_win":5}
RF={"n_estimators":200,"max_depth":10,"min_samples_leaf":2,"min_samples_split":4,"max_features":"sqrt","smooth_win":5}
TAU={"rank":8,"max_shift_frac":0.14,"nsteps":61}

def eval_window(lo,hi):
    fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//4000); fc,f=P.band(lo,hi,decim=dec)
    if len(fc)<20: return None
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
    out={m:{"rmsd0":[],"hs":[]} for m in ["Park","RF_direct","TAU_T"]}
    for T_test in FOLDS:
        if np.isclose(T_test,T_REF): continue
        te=np.isclose(T,T_test); tr=~te; trh=tr&(y==0)
        try:
            comps={"Park":P.comp_park(X,ref,f,PARK)[0],
                   "RF_direct":P.comp_rf(X,T,ref,trh,RF,"direct")[0],
                   "TAU_T":P.comp_tauT(X,T,ref,trh,f,TAU)[0]}
        except Exception: continue
        for m,Y in comps.items():
            pc={}
            for c in [0,1,2]:
                ii=np.where(te&(y==c))[0];
                if len(ii): pc[c]=float(np.mean([P.rmsd(Y[i],ref) for i in ii]))
            if 0 in pc: out[m]["rmsd0"].append(pc[0])
            if {0,1,2}<=set(pc):
                hs,_=P.monotonicity(pc); out[m]["hs"].append(hs)
    rows=[]
    for m,v in out.items():
        if v["rmsd0"]:
            rows.append({"lo":lo,"hi":hi,"centro":(lo+hi)/2,"largura":hi-lo,"metodo":m,
                         "RMSD_D0":float(np.mean(v["rmsd0"])),
                         "healthy_sep":float(np.mean(v["hs"])) if v["hs"] else np.nan,
                         "n_folds":len(v["rmsd0"])})
    return rows

t0=time.time(); rows=[]
for w in WIDTHS:
    st=STEP[w]; centers=np.arange(LO,HI-w+0.1,st)
    print(f"\n--- largura {w} kHz: {len(centers)} janelas ---")
    for c0 in centers:
        lo,hi=float(c0),float(c0+w)
        r=eval_window(lo,hi)
        if r: rows.extend(r)
        pd.DataFrame(rows).to_csv(CKPT,index=False)
    print(f"  largura {w} ok | {(time.time()-t0)/60:.1f} min | {len(rows)} registros")

d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"janelas_moveis_coarse.csv"),index=False)
print("\n"+"="*80); print("MELHORES JANELAS por método (menor RMSD_D0, entre as com healthy_sep=1)"); print("="*80)
best_regions=[]
for m in ["Park","RF_direct","TAU_T"]:
    s=d[(d.metodo==m)].copy()
    s_ok=s[s.healthy_sep>=0.999]
    pool=s_ok if len(s_ok) else s
    top=pool.sort_values("RMSD_D0").head(5)
    print(f"\n{m}:")
    print(top[["lo","hi","largura","RMSD_D0","healthy_sep","n_folds"]].to_string(index=False))
    for _,r in top.head(3).iterrows(): best_regions.append((r["lo"],r["hi"]))

# mapa: melhor RMSD_D0 (min entre métodos) por janela
print("\n"+"="*80); print("Top 10 janelas globais (min RMSD_D0 entre métodos)"); print("="*80)
piv=d.pivot_table(index=["lo","hi","largura"],columns="metodo",values="RMSD_D0")
piv["min"]=piv.min(axis=1); piv["best"]=piv[["Park","RF_direct","TAU_T"]].idxmin(axis=1)
print(piv.sort_values("min").head(10).round(3).to_string())
piv.to_csv(os.path.join(OUT,"janelas_pivot.csv"))

# ------- FASE 5: AE nas melhores 3 regiões distintas -------
uniq=[]
for lo,hi in best_regions:
    if not any(abs(lo-a)<2 and abs(hi-b)<2 for a,b in uniq): uniq.append((lo,hi))
uniq=uniq[:3]
print(f"\n=== FASE 5: AE nas melhores regiões {uniq} ===")
AE={"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"dropout":0.10,"noise":0.01,"epochs":400,"patience":60}
aerows=[]
for lo,hi in uniq:
    fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
    r0=[]; hsl=[]
    for T_test in FOLDS:
        if np.isclose(T_test,T_REF): continue
        te=np.isclose(T,T_test); trh=(~te)&(y==0)
        try: Y=P.comp_ae(X,T,ref,trh,AE,T_REF,seed=42)[0]
        except Exception: continue
        pc={}
        for c in [0,1,2]:
            ii=np.where(te&(y==c))[0]
            if len(ii): pc[c]=float(np.mean([P.rmsd(Y[i],ref) for i in ii]))
        if 0 in pc: r0.append(pc[0])
        if {0,1,2}<=set(pc): hsl.append(P.monotonicity(pc)[0])
    aerows.append({"lo":lo,"hi":hi,"metodo":"AE","RMSD_D0":float(np.mean(r0)) if r0 else np.nan,
                   "healthy_sep":float(np.mean(hsl)) if hsl else np.nan})
    print(f"  AE {lo}-{hi}: RMSD_D0={aerows[-1]['RMSD_D0']:.3f} hs={aerows[-1]['healthy_sep']:.3f}")
pd.DataFrame(aerows).to_csv(os.path.join(OUT,"janelas_AE_melhores.csv"),index=False)
print(f"\n✅ FASE 4/5 concluída | {(time.time()-t0)/60:.1f} min")

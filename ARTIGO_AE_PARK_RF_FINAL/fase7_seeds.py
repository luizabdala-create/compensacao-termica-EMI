# -*- coding: utf-8 -*-
"""
FASE 7 — ESTABILIDADE DE SEEDS (finalistas)
===========================================
O AE mudou muito com hiperparâmetros; é obrigatório medir também a variância entre
seeds. AE: seeds 42/123/2026. RF: random_state 0/1/2. Park: determinístico.
4 bandas tunadas × 9 folds LOTO. Hiperparâmetros = os escolhidos pelo inner CV (Fase 3/3b).
"""
import os,sys,time,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"02_compensacao"); CKPT=os.path.join(ROOT,"checkpoints","fase7_seeds.csv")
T_REF=30.0
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
df,_,_=P.load_base()
T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
AE_SEEDS=[42,123,2026]; RF_SEEDS=[0,1,2]
rows=[]; t0=time.time()
for banda,tp in TUNED.items():
    lo,hi=map(int,banda.split("-"))
    fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
    print(f"\n=== {banda} ===")
    for T_test in FOLDS:
        if np.isclose(T_test,T_REF): continue
        te=np.isclose(T,T_test); tr=~te; trh=tr&(y==0)
        jobs=[("Park",0,lambda s: P.comp_park(X,ref,f,tp["park"])[0])]
        jobs+=[("AE",s,lambda s=s: P.comp_ae(X,T,ref,trh,tp["ae"],T_REF,seed=s)[0]) for s in AE_SEEDS]
        jobs+=[("RF_direct",s,lambda s=s: P.comp_rf(X,T,ref,trh,{**tp["rf"],"seed":s},"direct")[0]) for s in RF_SEEDS]
        for mname,seed,fn in jobs:
            try: Y=fn()
            except Exception as e: print("  err",mname,e); continue
            rec={"banda":banda,"T_test":T_test,"metodo":mname,"seed":seed}
            pc={}
            for c in [0,1,2]:
                ii=np.where(te&(y==c))[0]
                if len(ii)==0: continue
                rec[f"RMSD_D{c}"]=float(np.mean([P.rmsd(Y[i],ref) for i in ii]))
                pc[c]=rec[f"RMSD_D{c}"]
            if {0,1,2}<=set(pc):
                hs,_=P.monotonicity(pc); rec["healthy_sep"]=hs
            rows.append(rec)
        pd.DataFrame(rows).to_csv(CKPT,index=False)
        print(f"  T={T_test} ok | {(time.time()-t0)/60:.1f} min")
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"fase7_seeds.csv"),index=False)

print("\n"+"="*80); print("VARIÂNCIA ENTRE SEEDS — RMSD_D0 (média sobre folds, por banda)"); print("="*80)
# média por (banda,metodo,seed) sobre folds; depois média±std entre seeds
per_seed=d.groupby(["banda","metodo","seed"])["RMSD_D0"].mean().reset_index()
agg=per_seed.groupby(["banda","metodo"])["RMSD_D0"].agg(["mean","std","min","max"]).round(4)
print(agg.to_string())
agg.to_csv(os.path.join(OUT,"fase7_seeds_resumo.csv"))
print("\nCoeficiente de variação entre seeds (std/mean) — instabilidade:")
cv=(per_seed.groupby(["banda","metodo"])["RMSD_D0"].std()/
    per_seed.groupby(["banda","metodo"])["RMSD_D0"].mean()).round(4)
for (b,m),v in cv.items():
    if m in ("AE","RF_direct"): print(f"   {b:8s} {m:10s}: CV={v}")
print(f"\n✅ FASE 7: {len(d)} registros | {(time.time()-t0)/60:.1f} min")

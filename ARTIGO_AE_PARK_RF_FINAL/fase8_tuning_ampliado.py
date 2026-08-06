# -*- coding: utf-8 -*-
"""
FASE 8 (enxuta) — TUNING de todos os métodos, nested LOTO, 7 bandas de 10 kHz.
Grades focadas + RF com entrada decimada (input_decim=8) para tratabilidade.
Seleção por INNER CV (2 folds, RMSD saudável). Fold externo intocado. Checkpointed.
"""
import os,sys,time,json,traceback,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"02_compensacao"); CKPT=os.path.join(ROOT,"checkpoints","fase8_tuning.csv")
CFGJSON=os.path.join(ROOT,"checkpoints","fase8_bestcfg.json"); ERRS=os.path.join(ROOT,"logs","fase8_erros.csv")
T_REF=30.0; N_INNER=2
BANDS=[(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,100)]
AE_GRID=[dict(n_input=2000,n_anchors=128,latent=lt,hidden=256,lr=2e-3,dropout=0.10,noise=0.01,
              epochs=450,patience=55,lambda_d1=ld) for lt in [8,16] for ld in [0.3,0.5]]  # 4
RF_GRID=[dict(n_estimators=300,max_depth=dp,min_samples_leaf=lf,min_samples_split=4,
              max_features="sqrt",smooth_win=5,input_decim=8)
         for dp in [10,None] for lf in [1,2]]  # 4
PARK_GRID=[dict(max_shift_frac=s,nsteps=121,smooth_win=w) for s in [0.05,0.10,0.15] for w in [5,9]]  # 6
RF_TEMP={"n_estimators":300,"max_depth":10,"min_samples_leaf":2,"min_samples_split":4,"max_features":"sqrt","smooth_win":5}
TAU={"rank":8,"max_shift_frac":0.14,"nsteps":81}

df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
print(f"bandas={len(BANDS)} folds={len(FOLDS)} | AE={len(AE_GRID)} RF={len(RF_GRID)} Park={len(PARK_GRID)}",flush=True)
rows=[]; bestcfg={}; errs=[]
def save(): pd.DataFrame(rows).to_csv(CKPT,index=False); json.dump(bestcfg,open(CFGJSON,"w"),indent=1)
t0=time.time()
for lo,hi in BANDS:
    banda=f"{lo}-{hi}"; fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
    print(f"\n=== {banda} kHz | {len(fc)} pts ===",flush=True)
    for T_test in FOLDS:
        if np.isclose(T_test,T_REF): continue
        te=np.isclose(T,T_test); tr=~te
        trh=np.array(sorted(np.unique(T[tr&(y==0)]))); inner=trh[::max(1,len(trh)//N_INNER)][:N_INNER]
        def ibest(grid,make):
            best=(np.inf,grid[0])
            for g in grid:
                sc=[]
                for tk in inner:
                    ho=tr&np.isclose(T,tk)&(y==0); keep=tr&~np.isclose(T,tk)&(y==0)
                    if ho.sum()==0 or keep.sum()<8: continue
                    try:
                        Yc=make(g,keep); ii=np.where(ho)[0]; sc.append(np.mean([P.rmsd(Yc[i],ref) for i in ii]))
                    except Exception: sc.append(np.inf)
                mm=float(np.mean(sc)) if sc else np.inf
                if mm<best[0]: best=(mm,g)
            return best
        specs={"AE":(AE_GRID,lambda g,k:P.comp_ae(X,T,ref,k,g,T_REF,seed=42)[0]),
               "RF_direct":(RF_GRID,lambda g,k:P.comp_rf(X,T,ref,k,g,"direct")[0]),
               "Park":(PARK_GRID,lambda g,k:P.comp_park(X,ref,f,g)[0])}
        results={}
        for mname,(grid,make) in specs.items():
            try:
                cv,g=ibest(grid,make); Y=make(g,tr&(y==0)); results[mname]=(Y,g,cv)
            except Exception as e: errs.append({"banda":banda,"T_test":T_test,"metodo":mname,"erro":str(e)})
        # ablation e auxiliar com config fixa
        try: results["RF_temponly"]=(P.comp_rf(X,T,ref,tr&(y==0),RF_TEMP,"temponly")[0],RF_TEMP,np.nan)
        except Exception as e: errs.append({"banda":banda,"T_test":T_test,"metodo":"RF_temponly","erro":str(e)})
        try: results["TAU_T"]=(P.comp_tauT(X,T,ref,tr&(y==0),f,TAU)[0],TAU,np.nan)
        except Exception as e: errs.append({"banda":banda,"T_test":T_test,"metodo":"TAU_T","erro":str(e)})
        for mname,(Y,g,cv) in results.items():
            rec={"banda":banda,"T_test":T_test,"metodo":mname,"cv_inner":cv,"cfg":json.dumps(g)}
            pc={}
            for c in [0,1,2]:
                ii=np.where(te&(y==c))[0]
                if len(ii)==0: continue
                M=[P.all_metrics(Y[i],ref) for i in ii]
                for kk in M[0]: rec[f"{kk}_D{c}"]=float(np.mean([m[kk] for m in M]))
                pc[c]=rec[f"RMSD_D{c}"]
            rec["alteracao_RMSD"]=float(np.mean([P.rmsd(Y[i],X[i]) for i in np.where(te)[0]]))
            if {0,1,2}<=set(pc):
                rec["sep_RMSD_D1"]=pc[1]-pc[0]; rec["sep_RMSD_D2"]=pc[2]-pc[0]
                hs,fo=P.monotonicity(pc); rec["healthy_sep"]=hs; rec["full_order"]=fo
            rows.append(rec); bestcfg[f"{banda}|{T_test}|{mname}"]=g
        save(); print(f"  T={T_test} ok | {(time.time()-t0)/60:.1f} min | {len(rows)} regs",flush=True)
if errs: pd.DataFrame(errs).to_csv(ERRS,index=False)
save(); d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"fase8_tuning_ampliado.csv"),index=False)
print(f"\n✅ FASE 8: {len(d)} regs | {(time.time()-t0)/60:.1f} min | erros={len(errs)}",flush=True)
MJ=["AE","Park","RF_direct","RF_temponly","TAU_T"]
piv=d.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean").round(3)[MJ].reindex([f"{a}-{b}" for a,b in BANDS])
print("\nRMSD_D0 por banda:"); print(piv.to_string(),flush=True)
print("\nvencedor por banda:")
for b in piv.index:
    main=piv.loc[b][["AE","Park","RF_direct"]]; print(f"   {b:8s}: {main.idxmin()} ({main.min():.3f}) | Park={piv.loc[b,'Park']:.3f}")
aewin=int(sum(min(piv.loc[b,'AE'],piv.loc[b,'RF_direct'])<piv.loc[b,'Park'] for b in piv.index))
print(f"\nAE ou RF batem Park: {aewin}/{len(piv)} bandas",flush=True)
hs=d.pivot_table(index="banda",columns="metodo",values="healthy_sep",aggfunc="mean").round(3)[MJ].reindex([f"{a}-{b}" for a,b in BANDS])
print("\nhealthy_sep por banda:"); print(hs.to_string(),flush=True)

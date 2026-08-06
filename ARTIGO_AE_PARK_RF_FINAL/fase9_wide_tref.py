# -*- coding: utf-8 -*-
"""
FASE 9 — (A) BANDAS LARGAS tunadas (wide vs narrow) + (B) VARREDURA DE T_ref.
(A) anexa bandas largas 30-50, 30-70, 30-100 ao fase8_tuning.csv (mesmo protocolo).
(B) varre T_ref em {0,10,20,30,40,50,60,70} na banda 40-50, config fixa tunada.
"""
import os,sys,time,json,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"02_compensacao"); F8=os.path.join(ROOT,"checkpoints","fase8_tuning.csv")
F8CFG=os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")
TREF_OUT=os.path.join(ROOT,"06_sensibilidade_referencia","tref_sweep.csv"); os.makedirs(os.path.dirname(TREF_OUT),exist_ok=True)
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
AE_GRID=[dict(n_input=2000,n_anchors=128,latent=lt,hidden=256,lr=2e-3,dropout=0.10,noise=0.01,epochs=450,patience=55,lambda_d1=ld) for lt in [8,16] for ld in [0.3,0.5]]
RF_GRID=[dict(n_estimators=300,max_depth=dp,min_samples_leaf=lf,min_samples_split=4,max_features="sqrt",smooth_win=5,input_decim=8) for dp in [10,None] for lf in [1,2]]
PARK_GRID=[dict(max_shift_frac=s,nsteps=121,smooth_win=w) for s in [0.05,0.10,0.15] for w in [5,9]]
RF_TEMP={"n_estimators":300,"max_depth":10,"min_samples_leaf":2,"input_decim":8,"smooth_win":5}
TAU={"rank":8,"max_shift_frac":0.14,"nsteps":81}; N_INNER=2

def eval_metrics(Y,ref,te):
    rec={}; pc={}
    for c in [0,1,2]:
        ii=np.where(te&(y==c))[0]
        if len(ii)==0: continue
        M=[P.all_metrics(Y[i],ref) for i in ii]
        for k in M[0]: rec[f"{k}_D{c}"]=float(np.mean([m[k] for m in M]))
        pc[c]=rec[f"RMSD_D{c}"]
    if {0,1,2}<=set(pc):
        rec["sep_RMSD_D1"]=pc[1]-pc[0]; rec["sep_RMSD_D2"]=pc[2]-pc[0]
        hs,fo=P.monotonicity(pc); rec["healthy_sep"]=hs; rec["full_order"]=fo
    return rec

# ============ (A) BANDAS LARGAS (append ao fase8) ============
rows=[]; bestcfg=json.load(open(F8CFG)) if os.path.exists(F8CFG) else {}
t0=time.time()
for lo,hi in [(30,50),(30,70),(30,100)]:
    banda=f"{lo}-{hi}"; fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,30.0)
    print(f"\n=== LARGA {banda} | {len(fc)} pts ===",flush=True)
    for T_test in FOLDS:
        if np.isclose(T_test,30.0): continue
        te=np.isclose(T,T_test); tr=~te; trh=np.array(sorted(np.unique(T[tr&(y==0)]))); inner=trh[::max(1,len(trh)//N_INNER)][:N_INNER]
        def ibest(grid,make):
            best=(np.inf,grid[0])
            for g in grid:
                sc=[]
                for tk in inner:
                    ho=tr&np.isclose(T,tk)&(y==0); keep=tr&~np.isclose(T,tk)&(y==0)
                    if ho.sum()==0 or keep.sum()<8: continue
                    try: Yc=make(g,keep); ii=np.where(ho)[0]; sc.append(np.mean([P.rmsd(Yc[i],ref) for i in ii]))
                    except Exception: sc.append(np.inf)
                mm=float(np.mean(sc)) if sc else np.inf
                if mm<best[0]: best=(mm,g)
            return best
        res={}
        for mname,grid,make in [("AE",AE_GRID,lambda g,k:P.comp_ae(X,T,ref,k,g,30.0,seed=42)[0]),
                                ("RF_direct",RF_GRID,lambda g,k:P.comp_rf(X,T,ref,k,g,"direct")[0]),
                                ("Park",PARK_GRID,lambda g,k:P.comp_park(X,ref,f,g)[0])]:
            try: cv,g=ibest(grid,make); res[mname]=(make(g,tr&(y==0)),g)
            except Exception as e: print("err",mname,e)
        try: res["RF_temponly"]=(P.comp_rf(X,T,ref,tr&(y==0),RF_TEMP,"temponly")[0],RF_TEMP)
        except Exception: pass
        try: res["TAU_T"]=(P.comp_tauT(X,T,ref,tr&(y==0),f,TAU)[0],TAU)
        except Exception: pass
        for mname,(Y,g) in res.items():
            rec={"banda":banda,"T_test":T_test,"metodo":mname,"cv_inner":np.nan,"cfg":json.dumps(g)}
            rec["alteracao_RMSD"]=float(np.mean([P.rmsd(Y[i],X[i]) for i in np.where(te)[0]]))
            rec.update(eval_metrics(Y,ref,te)); rows.append(rec); bestcfg[f"{banda}|{T_test}|{mname}"]=g
        print(f"  T={T_test} ok | {(time.time()-t0)/60:.1f} min",flush=True)
# anexa ao fase8
if os.path.exists(F8):
    old=pd.read_csv(F8); allc=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
else: allc=pd.DataFrame(rows)
allc.to_csv(F8,index=False); allc.to_csv(os.path.join(OUT,"fase8_tuning_ampliado.csv"),index=False)
json.dump(bestcfg,open(F8CFG,"w"),indent=1)
print(f"\n(A) bandas largas anexadas. total linhas={len(allc)}",flush=True)

# ============ (B) VARREDURA DE T_ref (banda 40-50, config fixa tunada) ============
print("\n=== (B) VARREDURA DE T_ref em 40-50 kHz ===",flush=True)
lo,hi=40,50; fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec); X=df[fc].to_numpy(np.float64)
# config fixa tunada (mediana do fase8 para 40-50)
def modal(banda,m,default):
    from collections import Counter
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else default
cfgs={"Park":modal("40-50","Park",{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5}),
      "RF_direct":modal("40-50","RF_direct",{"n_estimators":300,"max_depth":10,"input_decim":8,"smooth_win":5}),
      "AE":modal("40-50","AE",{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"epochs":450,"patience":55})}
trows=[]
for T_REF in [0,10,20,30,40,50,60,70]:
    m0=np.isclose(T,T_REF)&(y==0)
    if m0.sum()<2: print(f"  T_ref={T_REF}: sem saudável, pulado"); continue
    ref=np.median(X[m0],axis=0)
    for T_test in FOLDS:
        if np.isclose(T_test,T_REF): continue
        te=np.isclose(T,T_test); trh=(~te)&(y==0)
        comps={"Park":P.comp_park(X,ref,f,cfgs["Park"])[0],
               "RF_direct":P.comp_rf(X,T,ref,trh,cfgs["RF_direct"],"direct")[0],
               "AE":P.comp_ae(X,T,ref,trh,cfgs["AE"],float(T_REF),seed=42)[0],
               "Original":X}
        for m,Y in comps.items():
            rec={"T_ref":T_REF,"T_test":T_test,"metodo":m}; rec.update(eval_metrics(Y,ref,te)); trows.append(rec)
    print(f"  T_ref={T_REF} ok | {(time.time()-t0)/60:.1f} min",flush=True)
td=pd.DataFrame(trows); td.to_csv(TREF_OUT,index=False)
print("\nVarredura T_ref — RMSD_D0 médio por método × T_ref:")
print(td.groupby(["metodo","T_ref"])["RMSD_D0"].mean().round(3).unstack(0).to_string())
print(f"\n✅ FASE 9 | {(time.time()-t0)/60:.1f} min",flush=True)

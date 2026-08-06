# -*- coding: utf-8 -*-
"""
FASE 2 — SCREENING: bandas x T_ref x LOTO, todos os métodos.
Checkpoint incremental, try/except por configuração, log de erros.
Decimação automática para manter <=10000 pontos por banda (custo constante).
"""
import os, sys, time, json, traceback, numpy as np, pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P

OUT=os.path.join(ROOT,"02_compensacao"); os.makedirs(OUT,exist_ok=True)
CKPT=os.path.join(ROOT,"checkpoints","fase2_master.csv")
ERRS=os.path.join(ROOT,"logs","erros_execucao.csv")
os.makedirs(os.path.dirname(CKPT),exist_ok=True); os.makedirs(os.path.dirname(ERRS),exist_ok=True)

BANDS=[(30,40),(30,50),(30,60),(30,70),(30,80),(30,90),(30,100),
       (40,50),(50,60),(60,70),(70,80),(80,90),(90,100)]
TREFS=[20.0,30.0,40.0]
MAXPTS=10000

PARAMS_PARK={"max_shift_frac":0.10,"nsteps":111,"smooth_win":5}
PARAMS_RF  ={"n_estimators":250,"max_depth":10,"min_samples_leaf":2,"min_samples_split":4,"smooth_win":5}
PARAMS_AE  ={"n_input":1000,"n_anchors":64,"epochs":400,"patience":60,"latent":12,"hidden":128,"lr":2e-3}
PARAMS_TAU ={"rank":8,"max_shift_frac":0.14,"nsteps":81}

df,_,_=P.load_base()
T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
# folds externos: temperaturas com as 3 classes (auditoria confirmou 10)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
print(f"folds LOTO externos ({len(FOLDS)}): {FOLDS}")

done=set()
if os.path.exists(CKPT):
    prev=pd.read_csv(CKPT)
    done={(r.banda,r.T_ref,r.T_test,r.metodo) for r in prev.itertuples()}
    print(f"checkpoint: {len(prev)} linhas já feitas")
rows=[]; errs=[]
def flush():
    if not rows: return
    d=pd.DataFrame(rows)
    d.to_csv(CKPT,mode="a",header=not os.path.exists(CKPT),index=False)
    rows.clear()

t_start=time.time(); n_cfg=0
for (lo,hi) in BANDS:
    fcols,f=P.band(lo,hi,decim=1)
    dec=max(1,len(fcols)//MAXPTS)
    fcols,f=P.band(lo,hi,decim=dec)
    X=df[fcols].to_numpy(np.float64)
    banda=f"{lo}-{hi}"
    print(f"\n=== BANDA {banda} kHz | {len(fcols)} pts (decim={dec}) ===")
    for T_ref in TREFS:
        ref,n_ref=P.build_reference(df,fcols,T_ref)   # PROTOCOLO A: congelada
        if ref is None: continue
        for T_test in FOLDS:
            te=np.isclose(T,T_test); tr=~te; trh=tr&(y==0)
            if trh.sum()<8: continue
            same_as_ref = bool(np.isclose(T_test,T_ref))
            methods={}
            try:
                methods["Original"]=(X.copy(),{})
                methods["Park"]=P.comp_park(X,ref,f,PARAMS_PARK)
                methods["RF_direct"]=P.comp_rf(X,T,ref,trh,PARAMS_RF,mode="direct")
                methods["RF_temponly"]=P.comp_rf(X,T,ref,trh,PARAMS_RF,mode="temponly")
                methods["AE"]=P.comp_ae(X,T,ref,trh,PARAMS_AE,T_ref,seed=42)
                methods["TAU_T"]=P.comp_tauT(X,T,ref,trh,f,PARAMS_TAU)
            except Exception as e:
                errs.append({"banda":banda,"T_ref":T_ref,"T_test":T_test,"erro":str(e),
                             "tb":traceback.format_exc()[-600:]}); continue
            for mname,(Y,info) in methods.items():
                if (banda,T_ref,T_test,mname) in done: continue
                try:
                    rec={"banda":banda,"lo_khz":lo,"hi_khz":hi,"n_pontos":len(fcols),"decim":dec,
                         "T_ref":T_ref,"T_test":T_test,"metodo":mname,
                         "T_test_eh_T_ref":same_as_ref,"n_ref":n_ref,"n_train_healthy":int(trh.sum())}
                    perclass={}
                    for c in [0,1,2]:
                        ii=np.where(te&(y==c))[0]
                        if len(ii)==0: continue
                        M=[P.all_metrics(Y[i],ref) for i in ii]
                        for k in M[0]:
                            rec[f"{k}_D{c}"]=float(np.mean([m[k] for m in M]))
                        rec[f"n_D{c}"]=len(ii)
                        perclass[c]=rec[f"RMSD_D{c}"]
                    # alteração da curva (C7): quanto o método mexeu no sinal original
                    ii_all=np.where(te)[0]
                    rec["alteracao_RMSD"]=float(np.mean([P.rmsd(Y[i],X[i]) for i in ii_all]))
                    if {0,1,2}<=set(perclass):
                        rec["sep_RMSD_D1"]=perclass[1]-perclass[0]
                        rec["sep_RMSD_D2"]=perclass[2]-perclass[0]
                        hs,fo=P.monotonicity(perclass); rec["healthy_sep"]=hs; rec["full_order"]=fo
                        cc={c:rec[f"CCDM_D{c}"] for c in [0,1,2]}
                        hs2,fo2=P.monotonicity(cc); rec["healthy_sep_ccdm"]=hs2; rec["full_order_ccdm"]=fo2
                    rows.append(rec); n_cfg+=1
                except Exception as e:
                    errs.append({"banda":banda,"T_ref":T_ref,"T_test":T_test,"metodo":mname,
                                 "erro":str(e),"tb":traceback.format_exc()[-600:]})
            flush()
        el=time.time()-t_start
        print(f"  T_ref={T_ref}: acumulado {n_cfg} registros | {el/60:.1f} min")
flush()
if errs: pd.DataFrame(errs).to_csv(ERRS,index=False)
print(f"\n✅ FASE 2 concluída: {n_cfg} registros | {(time.time()-t_start)/60:.1f} min | erros={len(errs)}")

# resumo rápido
d=pd.read_csv(CKPT)
d.to_csv(os.path.join(OUT,"master_results_fase2.csv"),index=False)
g=d[~d.T_test_eh_T_ref].groupby("metodo").agg(
    RMSD_D0=("RMSD_D0","mean"),CCDM_D0=("CCDM_D0","mean"),
    sep_D1=("sep_RMSD_D1","mean"),sep_D2=("sep_RMSD_D2","mean"),
    healthy_sep=("healthy_sep","mean"),full_order=("full_order","mean"),n=("RMSD_D0","size")).round(4)
print("\nRESUMO GLOBAL (todas bandas/T_ref/folds, excluindo T_test==T_ref):")
print(g.sort_values("RMSD_D0").to_string())
g.to_csv(os.path.join(OUT,"resumo_global_fase2.csv"))

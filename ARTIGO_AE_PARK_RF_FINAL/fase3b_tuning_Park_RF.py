# -*- coding: utf-8 -*-
"""
FASE 3b — TUNING DE PARK E RF (mesmo protocolo do AE: inner CV, fold externo intocado)
======================================================================================
Necessário para comparação JUSTA: na Fase 3 só o AE foi tunado.
Mesmas bandas, mesmos folds externos, mesma seleção por inner CV (RMSD saudável).
"""
import os,sys,time,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"02_compensacao")
CKPT=os.path.join(ROOT,"checkpoints","fase3b_ParkRF.csv")
BANDS=[(30,40),(30,60),(60,70),(70,80)]; T_REF=30.0
N_INNER=3

GRID_PARK=[{"max_shift_frac":s,"nsteps":121,"smooth_win":w}
           for s in [0.02,0.05,0.10,0.15,0.25] for w in [1,5,9]]
GRID_RF=[{"n_estimators":n,"max_depth":dp,"min_samples_leaf":lf,
          "min_samples_split":4,"max_features":"sqrt","smooth_win":w}
         for n in [250,450] for dp in [6,10,None] for lf in [2,8] for w in [5]]
print(f"grid Park={len(GRID_PARK)} | grid RF={len(GRID_RF)}")

df,_,_=P.load_base()
T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]

rows=[]; t0=time.time()
for (lo,hi) in BANDS:
    fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
    banda=f"{lo}-{hi}"; print(f"\n=== {banda} kHz ===")
    for T_test in FOLDS:
        if np.isclose(T_test,T_REF): continue
        te=np.isclose(T,T_test); tr=~te
        tr_h=np.array(sorted(np.unique(T[tr&(y==0)])))
        inner=tr_h[::max(1,len(tr_h)//N_INNER)][:N_INNER]

        def inner_score(fn):
            sc=[]
            for tk in inner:
                ho=tr&np.isclose(T,tk)&(y==0); keep=tr&~np.isclose(T,tk)&(y==0)
                if ho.sum()==0 or keep.sum()<8: continue
                try:
                    Yc=fn(keep); ii=np.where(ho)[0]
                    sc.append(np.mean([P.rmsd(Yc[i],ref) for i in ii]))
                except Exception: sc.append(np.inf)
            return float(np.mean(sc)) if sc else np.inf

        # ---- PARK (não treina, mas os hiperparâmetros são escolhidos no inner) ----
        bp=(np.inf,GRID_PARK[0])
        for g in GRID_PARK:
            s=inner_score(lambda keep,g=g: P.comp_park(X,ref,f,g)[0])
            if s<bp[0]: bp=(s,g)
        Ypk,_=P.comp_park(X,ref,f,bp[1])
        # ---- RF direct ----
        br=(np.inf,GRID_RF[0])
        for g in GRID_RF:
            s=inner_score(lambda keep,g=g: P.comp_rf(X,T,ref,keep,g,"direct")[0])
            if s<br[0]: br=(s,g)
        Yrf,_=P.comp_rf(X,T,ref,tr&(y==0),br[1],"direct")
        # ---- RF temponly ----
        bt=(np.inf,GRID_RF[0])
        for g in GRID_RF:
            s=inner_score(lambda keep,g=g: P.comp_rf(X,T,ref,keep,g,"temponly")[0])
            if s<bt[0]: bt=(s,g)
        Yrt,_=P.comp_rf(X,T,ref,tr&(y==0),bt[1],"temponly")

        for mname,Y,cfg,cv in [("Park_tuned",Ypk,bp[1],bp[0]),
                               ("RF_direct_tuned",Yrf,br[1],br[0]),
                               ("RF_temponly_tuned",Yrt,bt[1],bt[0])]:
            rec={"banda":banda,"T_ref":T_REF,"T_test":T_test,"metodo":mname,
                 "cv_inner_rmsd":cv,"cfg":str(cfg)}
            pc={}
            for c in [0,1,2]:
                ii=np.where(te&(y==c))[0]
                if len(ii)==0: continue
                rec[f"RMSD_D{c}"]=float(np.mean([P.rmsd(Y[i],ref) for i in ii]))
                rec[f"CCDM_D{c}"]=float(np.mean([P.ccdm(Y[i],ref) for i in ii]))
                pc[c]=rec[f"RMSD_D{c}"]
            if {0,1,2}<=set(pc):
                hs,fo=P.monotonicity(pc); rec["healthy_sep"]=hs; rec["full_order"]=fo
                rec["sep_RMSD_D1"]=pc[1]-pc[0]; rec["sep_RMSD_D2"]=pc[2]-pc[0]
            rows.append(rec)
        pd.DataFrame(rows).to_csv(CKPT,index=False)
        print(f"  T={T_test} ok | {(time.time()-t0)/60:.1f} min")

d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"fase3b_Park_RF_tuned.csv"),index=False)
print(f"\n✅ FASE 3b: {len(d)} registros | {(time.time()-t0)/60:.1f} min")
print("\nPARK/RF COM TUNING por banda:")
print(d.groupby(["banda","metodo"]).agg(RMSD_D0=("RMSD_D0","mean"),CCDM_D0=("CCDM_D0","mean"),
    healthy_sep=("healthy_sep","mean"),sep_D1=("sep_RMSD_D1","mean"),
    sep_D2=("sep_RMSD_D2","mean")).round(4).to_string())

# comparação final justa: todos tunados
ae=pd.read_csv(os.path.join(ROOT,"checkpoints","fase3_AE.csv")); ae["metodo"]="AE_tuned"
allm=pd.concat([ae,d],ignore_index=True)
allm.to_csv(os.path.join(OUT,"fase3_TODOS_tunados.csv"),index=False)
print("\n" + "="*92)
print("COMPARAÇÃO JUSTA — TODOS COM TUNING (inner CV), por banda")
print("="*92)
piv=allm.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean").round(3)
print("RMSD_D0:"); print(piv.to_string())
piv2=allm.pivot_table(index="banda",columns="metodo",values="healthy_sep",aggfunc="mean").round(3)
print("\nhealthy_sep:"); print(piv2.to_string())
piv.to_csv(os.path.join(OUT,"comparacao_justa_RMSD_D0.csv"))
piv2.to_csv(os.path.join(OUT,"comparacao_justa_healthy_sep.csv"))

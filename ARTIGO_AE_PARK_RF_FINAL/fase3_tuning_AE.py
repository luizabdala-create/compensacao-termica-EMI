# -*- coding: utf-8 -*-
"""
FASE 3 — TUNING HONESTO DO AUTOENCODER (nested CV)
==================================================
O AE perdeu no screening com UMA configuração. Antes de concluir qualquer coisa,
ele recebe busca de hiperparâmetros com seleção por INNER CV (só temperaturas de
treino). O fold externo continua intocado.

Bandas: onde o AE foi bem (70-80, 60-70) e onde falhou (30-40, 30-60) — para
descobrir SE o tuning conserta as bandas largas ou se a limitação é estrutural.
"""
import os,sys,time,itertools,traceback,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"02_compensacao"); os.makedirs(OUT,exist_ok=True)
CKPT=os.path.join(ROOT,"checkpoints","fase3_AE.csv")

BANDS=[(30,40),(30,60),(60,70),(70,80)]
T_REF=30.0
GRID=[dict(n_input=ni,n_anchors=na,latent=lt,hidden=hd,lr=lr,dropout=dr,noise=ns,
           epochs=500,patience=70,lambda_d1=0.3)
      for ni,na in [(500,32),(1000,64),(2000,128)]
      for lt in [8,16]
      for hd in [128,256]
      for lr in [2e-3]
      for dr in [0.10]
      for ns in [0.01]]
print(f"grid AE: {len(GRID)} configurações")

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
        # INNER CV: 4 temperaturas de treino deixadas de fora (só saudáveis p/ avaliar)
        tr_h_temps=np.array(sorted(np.unique(T[tr&(y==0)])))
        inner=tr_h_temps[::max(1,len(tr_h_temps)//4)][:4]
        best=(np.inf,None)
        for gi,g in enumerate(GRID):
            sc=[]
            for tk in inner:
                ho=tr&np.isclose(T,tk)&(y==0); keep=tr&~np.isclose(T,tk)&(y==0)
                if ho.sum()==0 or keep.sum()<8: continue
                try:
                    Yc,_=P.comp_ae(X,T,ref,keep,g,T_REF,seed=42)
                    ii=np.where(ho)[0]; sc.append(np.mean([P.rmsd(Yc[i],ref) for i in ii]))
                except Exception: sc.append(np.inf)
            if sc and np.mean(sc)<best[0]: best=(float(np.mean(sc)),gi)
        gstar=GRID[best[1]] if best[1] is not None else GRID[0]
        # avaliação no fold externo (uma única vez)
        try:
            Y,info=P.comp_ae(X,T,ref,tr&(y==0),gstar,T_REF,seed=42)
        except Exception as e:
            print("  erro:",e); continue
        rec={"banda":banda,"T_ref":T_REF,"T_test":T_test,"cv_inner_rmsd":best[0],
             "n_input":gstar["n_input"],"n_anchors":gstar["n_anchors"],
             "latent":gstar["latent"],"hidden":gstar["hidden"],"n_params":info["n_params"]}
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
        rows.append(rec); pd.DataFrame(rows).to_csv(CKPT,index=False)
        print(f"  T={T_test}: RMSD_D0={rec.get('RMSD_D0',np.nan):.3f} "
              f"(cfg n_in={gstar['n_input']} lat={gstar['latent']} hid={gstar['hidden']}) "
              f"| {(time.time()-t0)/60:.1f} min")

d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"fase3_AE_tuned.csv"),index=False)
print(f"\n✅ FASE 3: {len(d)} folds | {(time.time()-t0)/60:.1f} min")
print("\nAE COM TUNING (por banda):")
g=d.groupby("banda").agg(RMSD_D0=("RMSD_D0","mean"),CCDM_D0=("CCDM_D0","mean"),
    healthy_sep=("healthy_sep","mean"),sep_D1=("sep_RMSD_D1","mean"),
    sep_D2=("sep_RMSD_D2","mean"),n=("RMSD_D0","size")).round(4)
print(g.to_string())
print("\nComparação com o AE SEM tuning (Fase 2):")
f2=pd.read_csv(os.path.join(ROOT,"checkpoints","fase2_master.csv"))
f2=f2[(~f2.T_test_eh_T_ref)&(f2.metodo=="AE")&(f2.T_ref==T_REF)]
for b in g.index:
    a=f2[f2.banda==b]["RMSD_D0"].mean()
    print(f"  {b:8s}: sem tuning={a:.3f} -> com tuning={g.loc[b,'RMSD_D0']:.3f} "
          f"({100*(g.loc[b,'RMSD_D0']-a)/a:+.1f}%)")
print("\nconfigurações escolhidas pelo inner CV:")
print(d.groupby(["banda"])[["n_input","latent","hidden","n_params"]].agg(lambda s:s.mode().iloc[0]).to_string())

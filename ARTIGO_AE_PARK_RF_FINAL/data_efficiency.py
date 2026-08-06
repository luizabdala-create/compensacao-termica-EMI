# -*- coding: utf-8 -*-
"""
EFICIÊNCIA DE DADOS (curva de aprendizado): como o RMSD saudável de teste varia com o
NÚMERO de temperaturas de treino (k). LOTO externo; para cada fold, sorteiam-se k temperaturas
saudáveis de treino (sem tocar no teste), treina-se cada método e mede-se o RMSD saudável no fold.
Park não usa dados de treino (só a referência) -> linha ~constante (base de comparação).
Mostra quanta variedade térmica RF e AE precisam. Banda 70-80 kHz (melhor caso do AE).
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas"); os.makedirs(OUT,exist_ok=True)
CK=os.path.join(ROOT,"checkpoints","data_efficiency.csv")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":12.5,"legend.fontsize":9.5,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
CM={"Park":"#2ca02c","RF_direct":"#d62728","AE":"#1f5fd0"}; LB={"Park":"Park","RF_direct":"Random Forest","AE":"Autoencoder"}
T_REF=30.0; BANDA="70-80"; KS=[2,4,6,8,None]; SEEDS=[0,1]
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(m,d):
    ks=[k for k in bestcfg if k.split("|")[0]==BANDA and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
DEF={"Park":{"max_shift_frac":0.15,"nsteps":121,"smooth_win":9},
     "RF_direct":{"n_estimators":300,"max_depth":10,"input_decim":4,"smooth_win":5},
     "AE":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":384,"lr":1e-3,"epochs":550,"patience":75}}
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
lo,hi=map(int,BANDA.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2]) and not np.isclose(t,T_REF)]
htemps=sorted([t for t in np.unique(T[y==0]) if not np.isclose(t,T_REF)])

rows=[]; t0=time.time()
for T_test in FOLDS:
    te=np.isclose(T,T_test); idx0=np.where(te&(y==0))[0]
    avail=[t for t in htemps if not np.isclose(t,T_test)]
    # Park (independe de k)
    Yp=P.comp_park(X,ref,f,cfg("Park",DEF["Park"]))[0]; rmsd_park=np.mean([P.rmsd(Yp[i],ref) for i in idx0])
    for k in KS:
        kk=len(avail) if k is None else k
        if kk>len(avail): continue
        for sd in SEEDS:
            rng=np.random.RandomState(sd)
            sub=avail if k is None else list(rng.choice(avail,k,replace=False))
            trh=np.isin(np.round(T,3),np.round(sub,3))&(y==0)
            if trh.sum()<3: continue
            try:
                Yr=P.comp_rf(X,T,ref,trh,cfg("RF_direct",DEF["RF_direct"]),"direct")[0]
                Ya=P.comp_ae(X,T,ref,trh,cfg("AE",DEF["AE"]),T_REF,seed=42)[0]
            except Exception as e: print("err",T_test,k,e,flush=True); continue
            rows.append({"T_test":T_test,"k":kk,"seed":sd,
                         "RF_direct":float(np.mean([P.rmsd(Yr[i],ref) for i in idx0])),
                         "AE":float(np.mean([P.rmsd(Ya[i],ref) for i in idx0])),
                         "Park":float(rmsd_park)})
            if k is None: break  # k=all não depende de seed
    pd.DataFrame(rows).to_csv(CK,index=False); print(f"  T={T_test} ok | {(time.time()-t0)/60:.1f} min",flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"data_efficiency.csv"),index=False)

g=d.groupby("k")[["AE","RF_direct","Park"]].mean()
print("\n=== RMSD saudável vs nº de temperaturas de treino (70-80 kHz) ===\n",g.round(3).to_string(),flush=True)
fig,ax=plt.subplots(figsize=(8,4.8),dpi=170)
for m in ["Park","RF_direct","AE"]:
    gg=d.groupby("k")[m]; mu=gg.mean(); sd=gg.std()
    ax.plot(mu.index,mu.values,marker="o",lw=2.2,color=CM[m],label=LB[m])
    ax.fill_between(mu.index,mu-sd,mu+sd,color=CM[m],alpha=.13)
ax.set_xlabel("Número de temperaturas de treino (k)"); ax.set_ylabel("RMSD saudável de teste (LOTO)")
ax.set_title(f"Eficiência de dados — {BANDA} kHz (sombra = ±1 desvio-padrão entre folds/sorteios)")
ax.legend(frameon=False); fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figA_data_efficiency.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figA_data_efficiency",flush=True)
print(f"\n✅ eficiência de dados concluída em {(time.time()-t0)/60:.1f} min",flush=True)

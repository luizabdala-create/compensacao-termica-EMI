# -*- coding: utf-8 -*-
"""
ABLATIONS DE ENTRADA (pedido do revisor). Banda 70-80 kHz, LOTO, RMSD saudável.
 RF/Floresta: curve-only | stats-only | temp-only | curve+stats (sem T) | completo.
 AE: completo | sem temperatura (T fixado em Tref) | sem perda de derivada (lambda_d1=0).
Isola de onde vem o desempenho de cada família.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","legend.fontsize":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
T_REF=30.0; BANDA="70-80"
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(m,d):
    ks=[k for k in bestcfg if k.split("|")[0]==BANDA and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
lo,hi=map(int,BANDA.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
from sklearn.ensemble import ExtraTreesRegressor as ET
def feats(Xa,Ta,variant,idc=4):
    Tc=np.asarray(Ta,float).reshape(-1,1); mu=Xa.mean(1,keepdims=True); sd=Xa.std(1,keepdims=True); amp=(Xa.max(1)-Xa.min(1)).reshape(-1,1)
    Xin=Xa[:,::idc]
    return {"curva":Xin,"estatísticas":np.hstack([mu,sd,amp]),"só temperatura":np.hstack([Tc,Tc**2]),
            "curva+estat. (sem T)":np.hstack([Xin,mu,sd,amp]),"completo":np.hstack([Xin,mu,sd,amp,Tc])}[variant]
def comp_forest(variant):
    r=[]
    for tk in FOLDS:
        if np.isclose(tk,T_REF): continue
        te=np.isclose(T,tk); trh=(~te)&(y==0); i0=np.where(te&(y==0))[0]
        if trh.sum()<8: continue
        m=ET(n_estimators=300,max_depth=10,min_samples_leaf=1,max_features="sqrt",n_jobs=-1,random_state=0)
        m.fit(feats(X[trh],T[trh],variant),ref[None,:]-X[trh])
        Y=X+m.predict(feats(X,T,variant)); Y=np.vstack([P.moving_average(yy,5) for yy in Y])
        r.append(np.mean([P.rmsd(Y[i],ref) for i in i0]))
    return float(np.mean(r))
def comp_ae_abl(mode):
    cA=cfg("AE",{"n_input":2000,"n_anchors":128,"latent":8,"hidden":384,"lr":1e-3,"epochs":550,"patience":75})
    if mode=="sem derivada": cA=dict(cA); cA["lambda_d1"]=0.0
    Tin=np.full_like(T,T_REF) if mode=="sem T" else T
    r=[]
    for tk in FOLDS:
        if np.isclose(tk,T_REF): continue
        te=np.isclose(T,tk); trh=(~te)&(y==0); i0=np.where(te&(y==0))[0]
        if trh.sum()<8: continue
        Y=P.comp_ae(X,Tin,ref,trh,cA,T_REF,seed=42)[0]
        r.append(np.mean([P.rmsd(Y[i],ref) for i in i0]))
    return float(np.mean(r))
t0=time.time()
print("=== ABLATIONS FLORESTA (70-80) ===",flush=True); rf={}
for v in ["só temperatura","estatísticas","curva","curva+estat. (sem T)","completo"]:
    rf[v]=comp_forest(v); print(f"  {v:22s}: {rf[v]:.3f} | {(time.time()-t0)/60:.1f}min",flush=True)
print("=== ABLATIONS AUTOENCODER (70-80) ===",flush=True); ae={}
for v in ["completo","sem T","sem derivada"]:
    ae[v]=comp_ae_abl(v); print(f"  {v:14s}: {ae[v]:.3f} | {(time.time()-t0)/60:.1f}min",flush=True)
json.dump({"floresta":rf,"autoencoder":ae},open(os.path.join(OUT,"ablations.json"),"w"),indent=2,ensure_ascii=False)
# figura
fig,axes=plt.subplots(1,2,figsize=(14,4.6),dpi=170)
o1=["só temperatura","estatísticas","curva","curva+estat. (sem T)","completo"]
axes[0].bar(range(len(o1)),[rf[v] for v in o1],color="#d62728",edgecolor="k")
for i,v in enumerate(o1): axes[0].text(i,rf[v]+.02,f"{rf[v]:.2f}",ha="center",fontsize=8)
axes[0].set_xticks(range(len(o1))); axes[0].set_xticklabels(o1,rotation=25,ha="right"); axes[0].set_ylabel("RMSD saudável")
axes[0].set_title("(a) Floresta: ablação das entradas")
o2=["completo","sem T","sem derivada"]
axes[1].bar(range(len(o2)),[ae[v] for v in o2],color="#1f5fd0",edgecolor="k")
for i,v in enumerate(o2): axes[1].text(i,ae[v]+.02,f"{ae[v]:.2f}",ha="center",fontsize=8)
axes[1].set_xticks(range(len(o2))); axes[1].set_xticklabels(o2); axes[1].set_title("(b) Autoencoder: ablação")
fig.suptitle("Ablações de entrada — de onde vem o desempenho de cada família (70–80 kHz, LOTO)",y=1.02,fontsize=13)
fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figR_ablations.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figR_ablations",flush=True)
print(f"\n✅ ablations concluídas em {(time.time()-t0)/60:.1f} min",flush=True)

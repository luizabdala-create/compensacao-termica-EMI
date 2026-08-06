# -*- coding: utf-8 -*-
"""
ANÁLISES DO REVISOR — itens 4 (extrapolação térmica) e 5 (histerese leave-one-direction-out).
 (4) EXTRAPOLAÇÃO: splits BLOQUEADOS (não-LOTO): treinar num intervalo térmico e testar FORA dele.
     A: treino T<=50, teste {60,70,80};  B: treino T>=20, teste {-10,0,10}.  Compara Park/RF/ExtraTrees/
     AE/interpolação. Mostra quando cada método deixa de ser confiável fora da faixa de calibração.
 (5) HISTERESE: usa a coluna 'sentido' (aquecimento/resfriamento). Treina numa direção e testa na outra.
Configs tunadas. RMSD saudável. Extra Trees incluído (item 7).
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from scipy.interpolate import interp1d
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","legend.fontsize":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
T_REF=30.0; BANDS=["30-40","70-80","30-70"]
CM={"Park":"#2ca02c","RF":"#d62728","ExtraTrees":"#ff7f0e","AE":"#1f5fd0","Interp":"#777"}
LBLD={"Park":"Park","RF":"Random Forest","ExtraTrees":"RF otimizado","AE":"Autoencoder","Interp":"Interpolação"}
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda,m,d):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
AED={"n_input":2000,"n_anchors":128,"latent":8,"hidden":384,"lr":1e-3,"epochs":550,"patience":75}
RFD={"n_estimators":300,"max_depth":10,"min_samples_leaf":1,"max_features":"sqrt","smooth_win":5,"input_decim":4}
df,cols_all,meta=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
sentido=df["sentido"].to_numpy() if "sentido" in df.columns else None
def comp_et(X,ref,trh,params):
    idc=params.get("input_decim",4); Xh=X[trh]; Th=T[trh]
    et=ExtraTreesRegressor(n_estimators=params.get("n_estimators",300),max_depth=params.get("max_depth",10),
        min_samples_leaf=params.get("min_samples_leaf",1),max_features=params.get("max_features","sqrt"),n_jobs=-1,random_state=0)
    et.fit(P._rf_feats(Xh,Th,"direct",idc),ref[None,:]-Xh)
    Y=X+et.predict(P._rf_feats(X,T,"direct",idc)); w=params.get("smooth_win",5)
    if w>1: Y=np.vstack([P.moving_average(yy,w) for yy in Y])
    return Y
def comp_interp(X,ref,trh,idx):
    Th=T[trh]; Xh=X[trh]; ts=sorted(np.unique(Th)); Zt=np.array([Xh[np.isclose(Th,t)].mean(0) for t in ts])
    fint=interp1d(ts,Zt,axis=0,kind="linear",bounds_error=False,fill_value=(Zt[0],Zt[-1]))
    return np.array([X[i]-fint(T[i])+ref for i in idx])
def band(b):
    lo,hi=map(int,b.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    return df[fc].to_numpy(np.float64),f,P.build_reference(df,fc,T_REF)[0]
def rmsd_set(Y,ref,idx): return float(np.mean([P.rmsd(Y[i],ref) for i in idx]))
t0=time.time()

# ================= (4) EXTRAPOLAÇÃO =================
SPLITS={"Extrapolar quente\n(treino ≤50, teste 60–80)":(lambda t: t<=50, [60,70,80]),
        "Extrapolar frio\n(treino ≥20, teste −10–10)":(lambda t: t>=20, [-10,0,10])}
rows=[]
for b in BANDS:
    X,f,ref=band(b)
    for sname,(trainmask_fn,test_temps) in SPLITS.items():
        tr=np.array([trainmask_fn(t) for t in T])&(y==0)
        idx=np.where(np.isin(np.round(T,3),test_temps)&(y==0))[0]
        if tr.sum()<8 or len(idx)==0: continue
        comps={"Park":P.comp_park(X,ref,f,cfg(b,"Park",{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5}))[0],
               "RF":P.comp_rf(X,T,ref,tr,cfg(b,"RF_direct",RFD),"direct")[0],
               "ExtraTrees":comp_et(X,ref,tr,cfg(b,"RF_direct",RFD)),
               "AE":P.comp_ae(X,T,ref,tr,cfg(b,"AE",AED),T_REF,seed=42)[0]}
        r={"banda":b,"split":sname.split(chr(10))[0]}
        for m,Y in comps.items(): r[m]=rmsd_set(Y,ref,idx)
        rows.append(r); print(f"  extrap {b} | {r['split']}: "+" ".join(f"{k}={r[k]:.2f}" for k in ['Park','RF','ExtraTrees','AE'])+f" | {(time.time()-t0)/60:.1f}min",flush=True)
de=pd.DataFrame(rows); de.to_csv(os.path.join(OUT,"extrapolacao.csv"),index=False)
# figura extrapolação
fig,axes=plt.subplots(1,2,figsize=(14,4.8),dpi=170)
for ax,sp in zip(axes,de.split.unique()):
    s=de[de.split==sp]; x=np.arange(len(s)); w=.15
    for i,m in enumerate(["Park","RF","ExtraTrees","AE"]):
        ax.bar(x+(i-1.5)*w,s[m],w,color=CM[m],edgecolor="k",lw=.4,label=LBLD[m] if ax is axes[0] else None)
    ax.set_xticks(x); ax.set_xticklabels(s.banda); ax.set_title(sp); ax.set_ylabel("RMSD saudável (bloco de teste)")
axes[0].legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(1.05,-0.12))
fig.suptitle("Extrapolação térmica: desempenho FORA da faixa de temperatura de treino (teste bloqueado)",y=1.02,fontsize=13)
fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figR_extrapolacao.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figR_extrapolacao",flush=True)

# ================= (5) HISTERESE (leave-one-direction-out) =================
if sentido is not None:
    dirs=sorted(pd.unique(sentido)); print("sentidos:",dirs,flush=True)
    hrows=[]
    for b in BANDS:
        X,f,ref=band(b)
        for tr_dir in dirs:
            te_dir=[d for d in dirs if d!=tr_dir]
            tr=(sentido==tr_dir)&(y==0); idx=np.where(np.isin(sentido,te_dir)&(y==0))[0]
            if tr.sum()<8 or len(idx)==0: continue
            comps={"Park":P.comp_park(X,ref,f,cfg(b,"Park",{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5}))[0],
                   "RF":P.comp_rf(X,T,ref,tr,cfg(b,"RF_direct",RFD),"direct")[0],
                   "ExtraTrees":comp_et(X,ref,tr,cfg(b,"RF_direct",RFD)),
                   "AE":P.comp_ae(X,T,ref,tr,cfg(b,"AE",AED),T_REF,seed=42)[0]}
            r={"banda":b,"treino_sentido":str(tr_dir)}
            for m,Y in comps.items(): r[m]=rmsd_set(Y,ref,idx)
            hrows.append(r); print(f"  histerese {b} treino={tr_dir}: "+" ".join(f"{k}={r[k]:.2f}" for k in ['Park','RF','ExtraTrees','AE'])+f" | {(time.time()-t0)/60:.1f}min",flush=True)
    dh=pd.DataFrame(hrows); dh.to_csv(os.path.join(OUT,"histerese_dir.csv"),index=False)
    fig,ax=plt.subplots(figsize=(9,4.6),dpi=170); x=np.arange(len(dh)); w=.2
    lab=[f"{r.banda}\ntreino {r.treino_sentido}" for _,r in dh.iterrows()]
    for i,m in enumerate(["Park","RF","ExtraTrees","AE"]):
        ax.bar(x+(i-1.5)*w,dh[m],w,color=CM[m],edgecolor="k",lw=.4,label=LBLD[m])
    ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=8); ax.set_ylabel("RMSD saudável (direção de teste)")
    ax.set_title("Histerese: treinar numa direção térmica (aquecimento/resfriamento) e testar na outra",pad=8)
    ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.14)); fig.tight_layout()
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figR_histerese_dir.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig: figR_histerese_dir",flush=True)
else:
    print("coluna 'sentido' ausente — histerese pulada",flush=True)
print(f"\n✅ extrapolação e histerese concluídas em {(time.time()-t0)/60:.1f} min",flush=True)

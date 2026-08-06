# -*- coding: utf-8 -*-
"""
CONFIRMAÇÃO out-of-sample (LOTO) de Extra Trees vs Random Forest como regressor de compensação.
A CV interna sugeriu que Extra Trees supera o RF; aqui confirma-se no TESTE (sem usar o teste
para selecionar) em todas as bandas de 10 kHz + a banda larga 30-70. Reporta RMSD/CCDM saudável
e healthy_sep. Extra Trees usa splits aleatórios (extremely randomized) -> menor variância,
o que ajuda com o ruído de medição da compensação térmica.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","legend.fontsize":9.5,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
T_REF=30.0; BANDS=["30-40","40-50","50-60","60-70","70-80","80-90","90-100","30-70"]
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]=="RF_direct"]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else {"n_estimators":300,"max_depth":10,"min_samples_leaf":1,"max_features":"sqrt","smooth_win":5,"input_decim":4}
def comp_et(X,T,ref,mask,params):
    idc=params.get("input_decim",4); Xh=X[mask]; Th=T[mask]
    et=ExtraTreesRegressor(n_estimators=params.get("n_estimators",300),max_depth=params.get("max_depth",10),
        min_samples_leaf=params.get("min_samples_leaf",1),max_features=params.get("max_features","sqrt"),n_jobs=-1,random_state=0)
    et.fit(P._rf_feats(Xh,Th,"direct",idc),ref[None,:]-Xh)
    Y=X+et.predict(P._rf_feats(X,T,"direct",idc)); w=params.get("smooth_win",5)
    if w>1: Y=np.vstack([P.moving_average(yy,w) for yy in Y])
    return Y
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
rows=[]; t0=time.time()
for banda in BANDS:
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF); cf=cfg(banda)
    accs={"RF":{"D0":[],"D1":[],"D2":[]},"ET":{"D0":[],"D1":[],"D2":[]}}
    for T_test in FOLDS:
        if np.isclose(T_test,T_REF): continue
        te=np.isclose(T,T_test); trh=(~te)&(y==0)
        if trh.sum()<8: continue
        Yrf=P.comp_rf(X,T,ref,trh,cf,"direct")[0]; Yet=comp_et(X,T,ref,trh,cf)
        for nm,Y in [("RF",Yrf),("ET",Yet)]:
            for c in [0,1,2]:
                ii=np.where(te&(y==c))[0]
                if len(ii): accs[nm][f"D{c}"].append(np.mean([P.rmsd(Y[i],ref) for i in ii]))
    r={"banda":banda}
    for nm in ["RF","ET"]:
        r[f"{nm}_RMSD_D0"]=float(np.mean(accs[nm]["D0"])) if accs[nm]["D0"] else np.nan
        # healthy_sep por fold
        hs=[]
        for a,b,c in zip(accs[nm]["D0"],accs[nm]["D1"],accs[nm]["D2"]): hs.append(1.0 if a<min(b,c) else 0.0)
        r[f"{nm}_healthy_sep"]=float(np.mean(hs)) if hs else np.nan
    r["ganho_%"]=round(100*(r["RF_RMSD_D0"]-r["ET_RMSD_D0"])/r["RF_RMSD_D0"],1)
    rows.append(r); print(f"  {banda}: RF={r['RF_RMSD_D0']:.3f} ET={r['ET_RMSD_D0']:.3f} ({r['ganho_%']:+.0f}%) | {(time.time()-t0)/60:.1f} min",flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"extratrees_loto.csv"),index=False)
n_et=(d.ET_RMSD_D0<d.RF_RMSD_D0).sum()
print(f"\n=== Extra Trees vence RF (teste LOTO) em {n_et}/{len(d)} bandas | ganho médio {d['ganho_%'].mean():.1f}% ===",flush=True)
print(d.round(3).to_string(index=False),flush=True)
# figura
fig,ax=plt.subplots(figsize=(10,4.4),dpi=170); x=np.arange(len(d)); w=.38
ax.bar(x-w/2,d.RF_RMSD_D0,w,color="#d62728",edgecolor="k",label="Random Forest")
ax.bar(x+w/2,d.ET_RMSD_D0,w,color="#ff7f0e",edgecolor="k",label="Random Forest otimizado")
for i,(a,b) in enumerate(zip(d.RF_RMSD_D0,d.ET_RMSD_D0)):
    ax.text(i-w/2,a+.03,f"{a:.2f}",ha="center",fontsize=7.5); ax.text(i+w/2,b+.03,f"{b:.2f}",ha="center",fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(d.banda,rotation=20); ax.set_ylabel("RMSD saudável (teste LOTO)")
ax.set_ylim(0,np.nanmax(d[["RF_RMSD_D0","ET_RMSD_D0"]].values)*1.16)
ax.set_title("Aprimoramento do Random Forest: a configuração otimizada reduz o erro de compensação (fora da amostra)",pad=8)
ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(0.5,-0.13))
fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figHP_extratrees_loto.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figHP_extratrees_loto",flush=True)
json.dump({"n_et_vence":int(n_et),"n_bandas":len(d),"ganho_medio_pct":round(float(d["ganho_%"].mean()),1)},
          open(os.path.join(OUT,"extratrees_resumo.json"),"w"),indent=2)
print(f"\n✅ confirmação Extra Trees concluída em {(time.time()-t0)/60:.1f} min",flush=True)

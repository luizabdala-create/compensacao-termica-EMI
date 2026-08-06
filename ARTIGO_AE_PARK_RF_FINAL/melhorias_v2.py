# -*- coding: utf-8 -*-
"""
MELHORIAS V2 — tentar extrair mais desempenho, honestamente (seleção por CV interna).
 (1) TUNAR o Extra Trees (grade própria) vs Extra Trees default vs Random Forest.
 (2) ENSEMBLE Autoencoder + Extra Trees (média das compensações) — testar se combinar
     as duas melhores famílias supera cada uma isoladamente.
 (3) SELETOR v2 por banda incluindo AE, Extra Trees (tunado) e Park — escolha por CV interna.
Bandas: 30-40, 60-70, 70-80, 30-70. Tudo confirmado no TESTE LOTO (sem selecionar pelo teste).
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","legend.fontsize":9.5,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
T_REF=30.0; BANDS=["30-40","60-70","70-80","30-70"]
CM={"Park":"#2ca02c","RF":"#d62728","ET":"#ff7f0e","AE":"#1f5fd0","Ensemble":"#6a3d9a","Seletor":"#17becf"}
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda,m,d):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
AE_DEF={"n_input":2000,"n_anchors":128,"latent":8,"hidden":384,"lr":1e-3,"epochs":550,"patience":75}
RF_DEF={"n_estimators":300,"max_depth":10,"min_samples_leaf":1,"max_features":"sqrt","smooth_win":5,"input_decim":4}
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
inner=[t for t in FOLDS if not np.isclose(t,T_REF)][::2]
def bdata(banda):
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    return df[fc].to_numpy(np.float64),f,P.build_reference(df,fc,T_REF)[0]
def comp_et(X,ref,mask,params):
    idc=params.get("input_decim",4); Xh=X[mask]; Th=T[mask]
    et=ExtraTreesRegressor(n_estimators=params.get("n_estimators",300),max_depth=params.get("max_depth",10),
        min_samples_leaf=params.get("min_samples_leaf",1),max_features=params.get("max_features","sqrt"),n_jobs=-1,random_state=0)
    et.fit(P._rf_feats(Xh,Th,"direct",idc),ref[None,:]-Xh)
    Y=X+et.predict(P._rf_feats(X,T,"direct",idc)); w=params.get("smooth_win",5)
    if w>1: Y=np.vstack([P.moving_average(yy,w) for yy in Y])
    return Y
def inner_cv(banda,make):
    X,f,ref=bdata(banda); sc=[]
    for tk in inner:
        ho=np.isclose(T,tk)&(y==0); keep=(~np.isclose(T,tk))&(y==0)
        if ho.sum()==0 or keep.sum()<8: continue
        try: Y=make(X,f,ref,keep); ii=np.where(ho)[0]; sc.append(np.mean([P.rmsd(Y[i],ref) for i in ii]))
        except Exception: sc.append(np.inf)
    return float(np.mean(sc)) if sc else np.inf
def loto(banda,make):
    X,f,ref=bdata(banda); r=[]
    for tk in FOLDS:
        if np.isclose(tk,T_REF): continue
        te=np.isclose(T,tk); trh=(~te)&(y==0); i0=np.where(te&(y==0))[0]
        if trh.sum()<8 or len(i0)==0: continue
        try: Y=make(X,f,ref,trh); r.append(np.mean([P.rmsd(Y[i],ref) for i in i0]))
        except Exception: pass
    return float(np.mean(r)) if r else np.nan
t0=time.time()

# ===== (1) TUNAR EXTRA TREES =====
# grade enxuta (configs baratos: input_decim=4, sem max_depth=None/600 árvores que travavam)
ET_GRID=[dict(n_estimators=300,max_depth=dp,min_samples_leaf=lf,max_features="sqrt",smooth_win=5,input_decim=4)
         for dp in [10,16] for lf in [1,2]]
print(f"=== TUNAR Extra Trees | {len(ET_GRID)} configs ===",flush=True)
et_best={}
for banda in BANDS:
    best=(np.inf,ET_GRID[0])
    for g in ET_GRID:
        cv=inner_cv(banda,lambda X,f,ref,k,gg=g: comp_et(X,ref,k,gg))
        if cv<best[0]: best=(cv,g)
    et_best[banda]=best[1]
    print(f"  {banda}: ET tunado CV={best[0]:.3f}",flush=True)

# ===== (2) confirma LOTO: RF, ET default, ET tunado, AE, Ensemble =====
print("\n=== LOTO: RF / ET_def / ET_tun / AE / Ensemble(AE+ET_tun) ===",flush=True)
rows=[]
for banda in BANDS:
    cE=et_best[banda]; cA=cfg(banda,"AE",AE_DEF); cR=cfg(banda,"RF_direct",RF_DEF)
    def mk_ae(X,f,ref,k): return P.comp_ae(X,T,ref,k,cA,T_REF,seed=42)[0]
    def mk_ett(X,f,ref,k): return comp_et(X,ref,k,cE)
    def mk_ens(X,f,ref,k): return 0.5*(P.comp_ae(X,T,ref,k,cA,T_REF,seed=42)[0]+comp_et(X,ref,k,cE))
    r={"banda":banda,
       "RF":loto(banda,lambda X,f,ref,k: P.comp_rf(X,T,ref,k,cR,"direct")[0]),
       "ET_def":loto(banda,lambda X,f,ref,k: comp_et(X,ref,k,RF_DEF)),
       "ET_tun":loto(banda,mk_ett),
       "AE":loto(banda,mk_ae),
       "Ensemble":loto(banda,mk_ens)}
    rows.append(r); print(f"  {banda}: RF={r['RF']:.2f} ET_def={r['ET_def']:.2f} ET_tun={r['ET_tun']:.2f} AE={r['AE']:.2f} Ens={r['Ensemble']:.2f} | {(time.time()-t0)/60:.1f}min",flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"melhorias_v2_loto.csv"),index=False)

# ===== (3) seletor v2 (por banda, escolha por CV interna entre AE/ET_tun/Park) =====
sel=[]
for banda in BANDS:
    cE=et_best[banda]; cA=cfg(banda,"AE",AE_DEF); cP=cfg(banda,"Park",{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5})
    cv={"AE":inner_cv(banda,lambda X,f,ref,k: P.comp_ae(X,T,ref,k,cA,T_REF,seed=42)[0]),
        "ET":inner_cv(banda,lambda X,f,ref,k,gg=cE: comp_et(X,ref,k,gg)),
        "Park":inner_cv(banda,lambda X,f,ref,k: P.comp_park(X,ref,f,cP)[0])}
    esc=min(cv,key=cv.get); row=d[d.banda==banda].iloc[0]
    teste={"AE":row["AE"],"ET":row["ET_tun"],"Park":loto(banda,lambda X,f,ref,k: P.comp_park(X,ref,f,cP)[0])}
    sel.append({"banda":banda,"escolhido":esc,"rmsd_seletor":teste[esc],"melhor_possivel":min(teste.values())})
    print(f"  seletor {banda}: escolhe {esc} -> {teste[esc]:.2f} (oráculo {min(teste.values()):.2f})",flush=True)
ds=pd.DataFrame(sel); ds.to_csv(os.path.join(OUT,"seletor_v2.csv"),index=False)

# resumo
resumo={m:round(float(d[m].mean()),3) for m in ["RF","ET_def","ET_tun","AE","Ensemble"]}
resumo["Seletor_v2"]=round(float(ds.rmsd_seletor.mean()),3)
json.dump({"media_rmsd":resumo,"et_best":{b:et_best[b] for b in BANDS}},open(os.path.join(OUT,"melhorias_v2_resumo.json"),"w"),indent=2)
print("\n=== MÉDIA RMSD (4 bandas) ===",flush=True)
for k,v in sorted(resumo.items(),key=lambda x:x[1]): print(f"  {k:12s}: {v:.3f}",flush=True)

# figura
fig,ax=plt.subplots(figsize=(11,4.6),dpi=170); x=np.arange(len(d)); methods=["RF","ET_tun","AE","Ensemble"]; w=.2
LBM={"RF":"Random Forest","ET_tun":"Extra Trees (tunado)","AE":"Autoencoder","Ensemble":"Ensemble AE+ET"}
CC={"RF":"#d62728","ET_tun":"#ff7f0e","AE":"#1f5fd0","Ensemble":"#6a3d9a"}
for i,m in enumerate(methods):
    ax.bar(x+(i-1.5)*w,d[m],w,color=CC[m],edgecolor="k",lw=.5,label=LBM[m])
ax.set_xticks(x); ax.set_xticklabels(d.banda); ax.set_ylabel("RMSD saudável (teste LOTO)")
ax.set_ylim(0,np.nanmax(d[methods].values)*1.16)
ax.set_title("Melhorias V2: Extra Trees tunado, Autoencoder e ensemble por banda",pad=8)
ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.12))
fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figHP_melhorias_v2.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figHP_melhorias_v2",flush=True)
print(f"\n✅ melhorias V2 concluídas em {(time.time()-t0)/60:.1f} min",flush=True)

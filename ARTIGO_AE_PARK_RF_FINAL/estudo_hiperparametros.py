# -*- coding: utf-8 -*-
"""
ESTUDO DE HIPERPARÂMETROS (análise de sensibilidade) — AE e RF.
Varre cada hiperparâmetro em torno do melhor atual, medindo RMSD saudável por CV INTERNA
(só temperaturas de treino — NUNCA o teste). Objetivo: (i) mostrar como cada hiperparâmetro
afeta a compensação; (ii) verificar se há config melhor que a atual; (iii) testar um regressor
alternativo (Extra Trees) para a compensação. Banda 70-80 kHz (melhor caso do AE) para o AE;
RF/Extra Trees em 3 bandas. Seleção sempre por CV interna (anti-vazamento).
"""
import os,sys,json,time,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":11.5,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","xtick.labelsize":9.5,"ytick.labelsize":9.5,"legend.fontsize":9,"pdf.fonttype":42,
 "axes.spines.top":False,"axes.spines.right":False})
T_REF=30.0
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
def savef(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n,flush=True)

def band_data(banda):
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF); return X,f,ref

def inner_cv_rmsd(banda,make,inner):
    """RMSD saudável médio por CV interna (temps de treino como held-out interno)."""
    X,f,ref=band_data(banda); sc=[]
    for tk in inner:
        ho=np.isclose(T,tk)&(y==0); keep=(~np.isclose(T,tk))&(y==0)
        if ho.sum()==0 or keep.sum()<8: continue
        try:
            Y=make(X,f,ref,keep); ii=np.where(ho)[0]; sc.append(np.mean([P.rmsd(Y[i],ref) for i in ii]))
        except Exception as e: sc.append(np.inf)
    return float(np.mean(sc)) if sc else np.inf

# ===================== (1) SENSIBILIDADE DO AE (70-80) =====================
BANDA_AE="70-80"
BASE_AE=dict(n_input=2000,n_anchors=128,latent=8,hidden=384,lr=1e-3,dropout=0.10,noise=0.01,epochs=550,patience=75,lambda_d1=0.3,wd=1e-4)
inner_ae=[t for t in FOLDS if not np.isclose(t,T_REF)][::2]
GRID_AE={"n_input":[1000,1500,2000,2500,3333],"latent":[4,8,16,24,32],"hidden":[128,256,384,512],
         "lr":[5e-4,1e-3,2e-3,3e-3],"dropout":[0.0,0.05,0.10,0.20],"lambda_d1":[0.0,0.1,0.3,0.5,1.0],
         "n_anchors":[64,128,256],"wd":[0.0,1e-4,1e-3]}
print(f"=== SENSIBILIDADE AE ({BANDA_AE}) | inner={len(inner_ae)} folds ===",flush=True)
t0=time.time(); rows_ae=[]
base_cv=inner_cv_rmsd(BANDA_AE,lambda X,f,ref,k: P.comp_ae(X,T,ref,k,BASE_AE,T_REF,seed=42)[0],inner_ae)
print(f"  base AE CV={base_cv:.3f}",flush=True)
for hp,vals in GRID_AE.items():
    for v in vals:
        cfg=dict(BASE_AE); cfg[hp]=v
        cv=inner_cv_rmsd(BANDA_AE,lambda X,f,ref,k,c=cfg: P.comp_ae(X,T,ref,k,c,T_REF,seed=42)[0],inner_ae)
        rows_ae.append({"hp":hp,"valor":v,"cv_rmsd":cv,"is_base":v==BASE_AE[hp]})
    print(f"  {hp} ok | {(time.time()-t0)/60:.1f} min",flush=True)
dae=pd.DataFrame(rows_ae); dae.to_csv(os.path.join(OUT,"hp_sensibilidade_ae.csv"),index=False)
best_ae=dae.loc[dae.cv_rmsd.idxmin()]
print(f"  MELHOR AE (1-D): {best_ae.hp}={best_ae.valor} -> CV={best_ae.cv_rmsd:.3f} (base={base_cv:.3f})",flush=True)

fig,axes=plt.subplots(2,4,figsize=(16,7),dpi=160)
for ax,hp in zip(axes.ravel(),GRID_AE.keys()):
    s=dae[dae.hp==hp].sort_values("valor")
    ax.plot(range(len(s)),s.cv_rmsd,marker="o",lw=2,color="#1f5fd0")
    bi=np.where(s.is_base.values)[0]
    if len(bi): ax.plot(bi[0],s.cv_rmsd.values[bi[0]],"o",ms=11,mfc="none",mec="crimson",mew=2,label="atual")
    ax.set_xticks(range(len(s))); ax.set_xticklabels([str(v) for v in s.valor],rotation=30,fontsize=8)
    ax.set_title(hp); ax.set_ylabel("RMSD (CV interna)"); ax.legend(frameon=False,fontsize=8)
fig.suptitle(f"Sensibilidade do Autoencoder a cada hiperparâmetro — {BANDA_AE} kHz (círculo vermelho = valor atual)",y=1.01,fontsize=13)
fig.tight_layout(); savef(fig,"figHP_ae_sensibilidade")

# ===================== (2) SENSIBILIDADE DO RF (70-80) =====================
BANDA_RF="70-80"
BASE_RF=dict(n_estimators=300,max_depth=10,min_samples_leaf=1,min_samples_split=4,max_features="sqrt",smooth_win=5,input_decim=4)
GRID_RF={"n_estimators":[100,200,300,600,900],"max_depth":[6,10,16,24,None],"min_samples_leaf":[1,2,4,8],
         "input_decim":[1,2,4,8,16],"max_features":["sqrt","log2",0.3,0.5,1.0],"smooth_win":[1,5,9,15]}
print(f"\n=== SENSIBILIDADE RF ({BANDA_RF}) ===",flush=True)
rows_rf=[]
def _cmp(v): return v if v is not None else "None"
for hp,vals in GRID_RF.items():
    for v in vals:
        cfg=dict(BASE_RF); cfg[hp]=v
        cv=inner_cv_rmsd(BANDA_RF,lambda X,f,ref,k,c=cfg: P.comp_rf(X,T,ref,k,c,"direct")[0],inner_ae)
        rows_rf.append({"hp":hp,"valor":_cmp(v),"cv_rmsd":cv,"is_base":_cmp(v)==_cmp(BASE_RF[hp])})
    print(f"  {hp} ok | {(time.time()-t0)/60:.1f} min",flush=True)
drf=pd.DataFrame(rows_rf); drf.to_csv(os.path.join(OUT,"hp_sensibilidade_rf.csv"),index=False)
best_rf=drf.loc[drf.cv_rmsd.idxmin()]; print(f"  MELHOR RF (1-D): {best_rf.hp}={best_rf.valor} -> CV={best_rf.cv_rmsd:.3f}",flush=True)
fig,axes=plt.subplots(2,3,figsize=(14,7),dpi=160)
for ax,hp in zip(axes.ravel(),GRID_RF.keys()):
    s=drf[drf.hp==hp]
    ax.plot(range(len(s)),s.cv_rmsd,marker="s",lw=2,color="#d62728")
    bi=np.where(s.is_base.values)[0]
    if len(bi): ax.plot(bi[0],s.cv_rmsd.values[bi[0]],"o",ms=11,mfc="none",mec="k",mew=2,label="atual")
    ax.set_xticks(range(len(s))); ax.set_xticklabels([str(v) for v in s.valor],rotation=30,fontsize=8)
    ax.set_title(hp); ax.set_ylabel("RMSD (CV interna)"); ax.legend(frameon=False,fontsize=8)
fig.suptitle(f"Sensibilidade do Random Forest a cada hiperparâmetro — {BANDA_RF} kHz",y=1.01,fontsize=13)
fig.tight_layout(); savef(fig,"figHP_rf_sensibilidade")

# ===================== (3) REGRESSOR ALTERNATIVO: EXTRA TREES =====================
print("\n=== RF vs Extra Trees (CV interna) ===",flush=True)
def comp_et(X,T_,ref,mask,params):
    idc=params.get("input_decim",4); Xh=X[mask]; Th=T_[mask]
    et=ExtraTreesRegressor(n_estimators=params.get("n_estimators",300),max_depth=params.get("max_depth",10),
        min_samples_leaf=params.get("min_samples_leaf",1),max_features=params.get("max_features","sqrt"),n_jobs=-1,random_state=0)
    et.fit(P._rf_feats(Xh,Th,"direct",idc),ref[None,:]-Xh)
    Y=X+et.predict(P._rf_feats(X,T_,"direct",idc)); w=params.get("smooth_win",5)
    if w>1: Y=np.vstack([P.moving_average(yy,w) for yy in Y])
    return Y
rows_reg=[]
for banda in ["30-40","70-80","30-70"]:
    cvrf=inner_cv_rmsd(banda,lambda X,f,ref,k: P.comp_rf(X,T,ref,k,BASE_RF,"direct")[0],inner_ae)
    cvet=inner_cv_rmsd(banda,lambda X,f,ref,k: comp_et(X,T,ref,k,BASE_RF),inner_ae)
    rows_reg.append({"banda":banda,"RandomForest":round(cvrf,3),"ExtraTrees":round(cvet,3),
                     "melhor":"ExtraTrees" if cvet<cvrf else "RandomForest"})
    print(f"  {banda}: RF={cvrf:.3f} ET={cvet:.3f} -> {'ET' if cvet<cvrf else 'RF'}",flush=True)
dreg=pd.DataFrame(rows_reg); dreg.to_csv(os.path.join(OUT,"regressores_alternativos.csv"),index=False)
fig,ax=plt.subplots(figsize=(8,4),dpi=160); x=np.arange(len(dreg)); w=.35
ax.bar(x-w/2,dreg.RandomForest,w,color="#d62728",edgecolor="k",label="Random Forest")
ax.bar(x+w/2,dreg.ExtraTrees,w,color="#ff7f0e",edgecolor="k",label="RF otimizado")
for i,(a,b) in enumerate(zip(dreg.RandomForest,dreg.ExtraTrees)):
    ax.text(i-w/2,a+.02,f"{a:.2f}",ha="center",fontsize=8); ax.text(i+w/2,b+.02,f"{b:.2f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(dreg.banda); ax.set_ylabel("RMSD (CV interna)")
ax.set_ylim(0,max(dreg[["RandomForest","ExtraTrees"]].values.max()*1.18,1))
ax.set_title("Regressor de compensação: Random Forest básico vs. otimizado",pad=8)
ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(0.5,-0.12))
fig.tight_layout(); savef(fig,"figHP_regressores")

# ===================== SUGESTÃO (só reporta; adoção manual por CV) =====================
sug={"ae_base_cv":round(base_cv,3),"ae_melhor_1d":{"hp":best_ae.hp,"valor":str(best_ae.valor),"cv":round(float(best_ae.cv_rmsd),3),
      "ganho_vs_base":round(base_cv-float(best_ae.cv_rmsd),3)},
     "rf_melhor_1d":{"hp":best_rf.hp,"valor":str(best_rf.valor),"cv":round(float(best_rf.cv_rmsd),3)},
     "extratrees_vence":int((dreg.melhor=="ExtraTrees").sum()),"n_bandas":len(dreg)}
json.dump(sug,open(os.path.join(OUT,"hp_sugestao.json"),"w"),indent=2)
print(f"\n=== SUGESTÃO ===\n{json.dumps(sug,indent=2)}",flush=True)
print(f"\n✅ estudo de hiperparâmetros concluído em {(time.time()-t0)/60:.1f} min",flush=True)

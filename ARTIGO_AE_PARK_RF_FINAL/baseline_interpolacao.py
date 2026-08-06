# -*- coding: utf-8 -*-
"""
BASELINE DE INTERPOLAÇÃO TÉRMICA (pedido do revisor — comparação essencial nº1).
Método transparente, sem treino de ML: para cada frequência f, estima-se a curva saudável
esperada em T por INTERPOLAÇÃO (linear e spline) ao longo da temperatura, usando apenas as
temperaturas saudáveis de TREINO. A compensação remove o desvio térmico saudável e alinha à
referência:  Z_comp(f,T) = Z(f,T) − Z_saud_interp(f,T) + z_ref(f).
Avaliado em LOTO em todas as bandas; comparado a Park, Random Forest e Autoencoder (fase8) e a
Ridge multioutput e PCR. Responde: o ganho vem do aprendizado de máquina ou só da interpolação?
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","legend.fontsize":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
T_REF=30.0; BANDS=["30-40","40-50","50-60","60-70","70-80","80-90","90-100","30-70","30-100"]
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]

def comp_interp(X,ref,trh,T_test,kind):
    """Interpolação térmica ponto-a-ponto (linear/spline) da curva saudável ao longo de T."""
    Th=T[trh]; Xh=X[trh]; ts=sorted(np.unique(Th))
    Zt=np.array([Xh[np.isclose(Th,t)].mean(0) for t in ts])
    k = kind if (kind=="linear" or len(ts)>3) else "linear"
    fint=interp1d(ts,Zt,axis=0,kind=k,bounds_error=False,fill_value=(Zt[0],Zt[-1]))
    est=fint(T_test)                     # curva saudável esperada em T_test
    return est                            # usado por curva abaixo

def comp_ridge(X,ref,trh,alpha=10.0):
    """Ridge multioutput: prediz a correção ref-curva a partir de (T,|T-Tref|)."""
    Th=T[trh]; Xh=X[trh]
    A=np.c_[Th, np.abs(Th-T_REF)]; B=ref[None,:]-Xh
    r=Ridge(alpha=alpha).fit(A,B)
    return r
def comp_pcr(X,ref,trh,n=6,alpha=1.0):
    Th=T[trh]; Xh=X[trh]; A=np.c_[Th,np.abs(Th-T_REF),Th**2]
    B=ref[None,:]-Xh; pca=PCA(n_components=min(n,B.shape[0]-1)).fit(B)
    r=Ridge(alpha=alpha).fit(A,pca.transform(B))
    return r,pca

rows=[]; t0=time.time()
for banda in BANDS:
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
    acc={m:[] for m in ["Interp-linear","Interp-spline","Ridge","PCR"]}
    for T_test in FOLDS:
        if np.isclose(T_test,T_REF): continue
        te=np.isclose(T,T_test); trh=(~te)&(y==0); i0=np.where(te&(y==0))[0]
        if trh.sum()<8 or len(i0)==0: continue
        for kind,nm in [("linear","Interp-linear"),("cubic","Interp-spline")]:
            est=comp_interp(X,ref,trh,T_test,kind)
            Y=X[i0]-est[None,:]+ref[None,:]
            acc[nm].append(np.mean([P.rmsd(Y[j],ref) for j in range(len(i0))]))
        A_te=np.c_[T[i0],np.abs(T[i0]-T_REF)]
        r=comp_ridge(X,ref,trh); Yr=X[i0]+r.predict(A_te)
        acc["Ridge"].append(np.mean([P.rmsd(Yr[j],ref) for j in range(len(i0))]))
        rp,pca=comp_pcr(X,ref,trh); A_te2=np.c_[T[i0],np.abs(T[i0]-T_REF),T[i0]**2]
        Yp=X[i0]+pca.inverse_transform(rp.predict(A_te2))
        acc["PCR"].append(np.mean([P.rmsd(Yp[j],ref) for j in range(len(i0))]))
    r={"banda":banda}
    for m in acc: r[m]=float(np.mean(acc[m])) if acc[m] else np.nan
    rows.append(r); print(f"  {banda}: interp-lin={r['Interp-linear']:.2f} spline={r['Interp-spline']:.2f} Ridge={r['Ridge']:.2f} PCR={r['PCR']:.2f} | {(time.time()-t0)/60:.1f}min",flush=True)
d=pd.DataFrame(rows)

# junta com fase8 (AE/Park/RF)
f8=pd.read_csv(os.path.join(ROOT,"02_compensacao","fase8_tuning_ampliado.csv")); f8=f8[f8.metodo!="TAU_T"]
piv=f8.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean")
d=d.set_index("banda")
for m,col in [("Park","Park"),("RF_direct","RF_direct"),("AE","AE")]:
    d[{"Park":"Park","RF_direct":"Random Forest","AE":"Autoencoder"}[m]]=piv[m] if m in piv.columns else np.nan
d=d.reset_index(); d.to_csv(os.path.join(OUT,"baseline_interpolacao.csv"),index=False)
print("\n=== RMSD saudável (LOTO) — baselines simples vs ML ===",flush=True)
print(d.round(3).to_string(index=False),flush=True)
# quem vence cada banda
methods=["Interp-linear","Interp-spline","Ridge","PCR","Park","Random Forest","Autoencoder"]
print("\n=== média sobre bandas ===",flush=True)
med=d[methods].mean().sort_values()
for m,v in med.items(): print(f"  {m:16s}: {v:.3f}",flush=True)
json.dump({"media":{m:round(float(d[m].mean()),3) for m in methods}},open(os.path.join(OUT,"baseline_interp_resumo.json"),"w"),indent=2)

# figura
fig,ax=plt.subplots(figsize=(12,5),dpi=170); x=np.arange(len(d)); w=.11
CC={"Interp-linear":"#8c8c8c","Interp-spline":"#4d4d4d","Ridge":"#9467bd","PCR":"#17becf","Park":"#2ca02c","Random Forest":"#d62728","Autoencoder":"#1f5fd0"}
for i,m in enumerate(methods):
    ax.bar(x+(i-3)*w,d[m],w,color=CC[m],edgecolor="k",lw=.4,label=m)
ax.set_xticks(x); ax.set_xticklabels(d.banda,rotation=20); ax.set_ylabel("RMSD saudável (teste LOTO)")
ax.set_ylim(0,np.nanmax(d[methods].values)*1.15)
ax.set_title("Baselines transparentes (interpolação térmica, Ridge, PCR) vs. aprendizado de máquina",pad=8)
ax.legend(frameon=False,ncol=7,loc="upper center",bbox_to_anchor=(0.5,-0.12),fontsize=8)
fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figR_baseline_interp.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figR_baseline_interp",flush=True)
print(f"\n✅ baseline de interpolação concluído em {(time.time()-t0)/60:.1f} min",flush=True)

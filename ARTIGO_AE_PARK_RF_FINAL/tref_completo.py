# -*- coding: utf-8 -*-
"""
VARREDURA COMPLETA DE TEMPERATURA DE REFERÊNCIA (T_ref) — todo o processo, várias bandas.
T_ref ∈ {-10,0,...,80} (toda a faixa) × bandas {30-40,40-50,60-70,70-80,30-70}.
Para cada T_ref: referência = mediana das saudáveis @T_ref; LOTO nas demais temperaturas;
mede-se RMSD/CCDM saudável de teste e a degradação com a distância térmica |T-T_ref|.
Reaproveita as bandas já feitas (40-50,70-80 em tref_full.csv) e computa as faltantes.
Configs tunadas. Checkpoint por banda. Gera figuras e explica o padrão.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"06_sensibilidade_referencia"); FIG=os.path.join(ROOT,"10_figuras_artigo"); os.makedirs(OUT,exist_ok=True)
CK=os.path.join(ROOT,"checkpoints","tref_completo.csv")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9.5,"pdf.fonttype":42,
 "axes.spines.top":False,"axes.spines.right":False})
CM={"Park":"#2ca02c","RF_direct":"#d62728","AE":"#1f5fd0","ExtraTrees":"#ff7f0e","Original":"#7f7f7f"}
LB={"Park":"Park","RF_direct":"Random Forest","AE":"Autoencoder","ExtraTrees":"Extra Trees","Original":"Original"}
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda,m,d):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
DEF={"Park":{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5},
     "RF_direct":{"n_estimators":300,"max_depth":10,"input_decim":8,"smooth_win":5},
     "AE":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"epochs":450,"patience":55}}
TREFS=[-10,0,10,20,30,40,50,60,70,80]; ALLB=["30-40","40-50","60-70","70-80","30-70"]; NEW=["30-40","60-70","30-70"]
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]

# reaproveita o existente
prev=pd.read_csv(os.path.join(OUT,"tref_full.csv")) if os.path.exists(os.path.join(OUT,"tref_full.csv")) else pd.DataFrame()
done=set()
if os.path.exists(CK):
    ck=pd.read_csv(CK); prev=pd.concat([prev,ck],ignore_index=True); done=set(ck.banda.unique())
rows=[]; t0=time.time()
for banda in [b for b in NEW if b not in done]:
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64)
    cP=cfg(banda,"Park",DEF["Park"]); cR=cfg(banda,"RF_direct",DEF["RF_direct"]); cA=cfg(banda,"AE",DEF["AE"])
    print(f"\n=== {banda} ({len(fc)} pts) ===",flush=True)
    for T_REF in TREFS:
        m0=np.isclose(T,T_REF)&(y==0)
        if m0.sum()<2: continue
        ref=np.median(X[m0],axis=0)
        for T_test in FOLDS:
            if np.isclose(T_test,T_REF): continue
            te=np.isclose(T,T_test); trh=(~te)&(y==0)
            try:
                comps={"Original":X,"Park":P.comp_park(X,ref,f,cP)[0],
                       "RF_direct":P.comp_rf(X,T,ref,trh,cR,"direct")[0],
                       "AE":P.comp_ae(X,T,ref,trh,cA,float(T_REF),seed=42)[0]}
            except Exception as e: print("  err",banda,T_REF,T_test,e,flush=True); continue
            for m,Y in comps.items():
                rec={"banda":banda,"T_ref":T_REF,"T_test":T_test,"metodo":m,"dist":abs(T_test-T_REF)}
                for c in [0,1,2]:
                    ii=np.where(te&(y==c))[0]
                    if len(ii): rec[f"RMSD_D{c}"]=float(np.mean([P.rmsd(Y[i],ref) for i in ii]))
                i0=np.where(te&(y==0))[0]
                rec["CCDM_D0"]=float(np.mean([P.ccdm(Y[i],ref) for i in i0])) if len(i0) else np.nan
                rows.append(rec)
        print(f"  T_ref={T_REF} ok | {(time.time()-t0)/60:.1f} min",flush=True)
    pd.DataFrame(rows).to_csv(CK,index=False)
d=pd.concat([prev,pd.DataFrame(rows)],ignore_index=True).drop_duplicates(["banda","T_ref","T_test","metodo"])
d.to_csv(os.path.join(OUT,"tref_full.csv"),index=False)
print(f"\n✅ tref consolidado: {d.shape} | bandas={sorted(d.banda.unique())} | {(time.time()-t0)/60:.1f} min",flush=True)

# ===== FIG: heatmap melhor T_ref (RMSD saudável) por banda × método =====
MJ=["AE","Park","RF_direct","ExtraTrees"]
fig,axes=plt.subplots(1,4,figsize=(19,4.6),dpi=170)
for ax,m in zip(axes,MJ):
    piv=d[d.metodo==m].pivot_table(index="banda",columns="T_ref",values="RMSD_D0",aggfunc="mean").reindex([b for b in ALLB if b in d.banda.unique()])
    im=ax.imshow(piv.values,cmap="viridis_r",aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([int(c) for c in piv.columns],rotation=0,fontsize=8)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("T_ref (°C)"); ax.set_title(f"{LB[m]}")
    for i in range(piv.shape[0]):
        j=int(np.nanargmin(piv.values[i])); ax.scatter(j,i,marker="*",s=160,c="white",edgecolors="k",linewidths=0.6,zorder=5)
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
axes[0].set_ylabel("Banda (kHz)")
fig.suptitle("RMSD saudável por temperatura de referência e banda (estrela branca = melhor T_ref de cada banda; mais escuro = melhor)",y=1.03,fontsize=13)
fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figT_tref_heatmap.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figT_tref_heatmap",flush=True)

# ===== FIG: degradação com a distância térmica |T-T_ref| =====
fig,ax=plt.subplots(figsize=(8.5,4.8),dpi=170)
for m in MJ+["Original"]:
    s=d[d.metodo==m].copy(); g=s.groupby(pd.cut(s.dist,[-1,5,15,25,35,50,100],labels=["≤5","5-15","15-25","25-35","35-50",">50"]),observed=True).RMSD_D0.mean()
    ax.plot(range(len(g)),g.values,marker="o",lw=2.2,color=CM[m],label=LB[m])
ax.set_xticks(range(6)); ax.set_xticklabels(["≤5","5-15","15-25","25-35","35-50",">50"])
ax.set_xlabel("Distância térmica à referência |T − T_ref| (°C)"); ax.set_ylabel("RMSD saudável")
ax.set_title("Degradação da compensação com a distância à temperatura de referência")
ax.legend(frameon=False); fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figT_degradacao_distancia.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figT_degradacao_distancia",flush=True)

# ===== resumo texto =====
print("\n=== MELHOR T_ref (RMSD saudável) por banda × método ===",flush=True)
sumrows=[]
for banda in [b for b in ALLB if b in d.banda.unique()]:
    for m in MJ:
        pv=d[(d.banda==banda)&(d.metodo==m)].groupby("T_ref").RMSD_D0.mean()
        if len(pv): sumrows.append({"banda":banda,"metodo":m,"melhor_Tref":int(pv.idxmin()),"rmsd_melhor":round(pv.min(),3),"pior_Tref":int(pv.idxmax()),"rmsd_pior":round(pv.max(),3),"amplitude":round(pv.max()-pv.min(),3)})
sr=pd.DataFrame(sumrows); sr.to_csv(os.path.join(OUT,"melhor_tref_resumo.csv"),index=False)
print(sr.to_string(index=False),flush=True)
# sensibilidade a T_ref (amplitude média) por método
print("\n=== SENSIBILIDADE a T_ref (amplitude RMSD média entre T_refs) ===",flush=True)
for m in MJ: print(f"  {LB[m]:14s}: {sr[sr.metodo==m].amplitude.mean():.3f}",flush=True)
print("\n✅ varredura completa de T_ref concluída",flush=True)

# -*- coding: utf-8 -*-
"""
ANÁLISE DE LARGURA DE BANDA E JANELAS DE 5 kHz (RMSD, CCDM, NRMSE) — Park, RF, AE.
(A) Bandas PROGRESSIVAS a partir de 30 kHz: 30-40, 30-50, ..., 30-100 (efeito da largura).
(B) Janelas de 5 kHz DESLIZANTES: 30-35, 35-40, ..., 95-100 (qual região é melhor; amplitude interfere?).
NRMSE (normalizado pela amplitude) separa "amplitude" de "dificuldade real".
Figuras: 3 métodos juntos + só AE + só RF + só Park; RMSD e CCDM lado a lado.
Checkpoint por banda. LOTO, configs tunadas.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"03_frequency_bands"); os.makedirs(OUT,exist_ok=True)
CK=os.path.join(ROOT,"checkpoints","faixas_largura.csv")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":12,"axes.labelsize":13,"axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold",
 "xtick.labelsize":10,"ytick.labelsize":11,"legend.fontsize":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
CM={"Park":"#2ca02c","RF_direct":"#d62728","AE":"#1f5fd0"}; LB={"Park":"Park","RF_direct":"Random Forest","AE":"Autoencoder"}
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json"))) if os.path.exists(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")) else {}
def cfg(banda,m,d):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
DEF={"Park":{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5},
     "RF_direct":{"n_estimators":300,"max_depth":10,"min_samples_leaf":2,"input_decim":8,"smooth_win":5},
     "AE":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"epochs":450,"patience":55}}
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
PROG=[(30,40),(30,50),(30,60),(30,70),(30,80),(30,90),(30,100)]
WIN=[(a,a+5) for a in range(30,96,5)]
ALL=[("prog",b) for b in PROG]+[("win",b) for b in WIN]

done=set(); rows=[]
if os.path.exists(CK):
    prev=pd.read_csv(CK); rows=prev.to_dict("records"); done={(r["tipo"],r["banda"]) for r in rows}
t0=time.time()
for tipo,(lo,hi) in ALL:
    banda=f"{lo}-{hi}"
    if (tipo,banda) in done: continue
    fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    if len(fc)<15: continue
    X=df[fc].to_numpy(np.float64); ref=np.median(X[np.isclose(T,30.0)&(y==0)],axis=0)
    acc={m:{"RMSD":[],"CCDM":[],"NRMSE":[]} for m in ["Park","RF_direct","AE"]}
    for T_test in FOLDS:
        te=np.isclose(T,T_test); trh=(~te)&(y==0); idx0=np.where(te&(y==0))[0]
        if trh.sum()<8 or len(idx0)==0: continue
        try:
            comps={"Park":P.comp_park(X,ref,f,cfg(banda,"Park",DEF["Park"]))[0],
                   "RF_direct":P.comp_rf(X,T,ref,trh,cfg(banda,"RF_direct",DEF["RF_direct"]),"direct")[0],
                   "AE":P.comp_ae(X,T,ref,trh,cfg(banda,"AE",DEF["AE"]),30.0,seed=42)[0]}
        except Exception as e: print("err",banda,e); continue
        for m,Y in comps.items():
            M=[P.all_metrics(Y[i],ref) for i in idx0]
            acc[m]["RMSD"].append(np.mean([x["RMSD"] for x in M]))
            acc[m]["CCDM"].append(np.mean([x["CCDM"] for x in M]))
            acc[m]["NRMSE"].append(np.mean([x["NRMSE"] for x in M]))
    for m in ["Park","RF_direct","AE"]:
        if acc[m]["RMSD"]:
            rows.append({"tipo":tipo,"banda":banda,"lo":lo,"hi":hi,"largura":hi-lo,"centro":(lo+hi)/2,"metodo":m,
                         "RMSD":float(np.mean(acc[m]["RMSD"])),"CCDM":float(np.mean(acc[m]["CCDM"])),"NRMSE":float(np.mean(acc[m]["NRMSE"]))})
    pd.DataFrame(rows).to_csv(CK,index=False)
    print(f"  {tipo} {banda} ok | {(time.time()-t0)/60:.1f} min",flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"faixas_largura.csv"),index=False)

def savef(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n,flush=True)

# ===== (A) BANDAS PROGRESSIVAS — 3 métodos, RMSD|CCDM|NRMSE =====
dp=d[d.tipo=="prog"].copy(); order=[f"{a}-{b}" for a,b in PROG]
fig,axes=plt.subplots(1,3,figsize=(18,4.8),dpi=170)
for ax,met,tit in zip(axes,["RMSD","CCDM","NRMSE"],["RMSD","CCDM","NRMSE (normalizado)"]):
    for m in ["Park","RF_direct","AE"]:
        s=dp[dp.metodo==m].set_index("banda").reindex(order)
        ax.plot(range(len(order)),s[met],marker="o",lw=2.2,color=CM[m],label=LB[m])
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=30,ha="right")
    ax.set_xlabel("Banda (a partir de 30 kHz)"); ax.set_ylabel(met); ax.set_title(f"({'abc'[list(['RMSD','CCDM','NRMSE']).index(met)]}) {tit}")
axes[2].legend(frameon=False)
fig.suptitle("Efeito da LARGURA da banda (progressiva a partir de 30 kHz) na compensação saudável",y=1.02,fontsize=13)
fig.tight_layout(); savef(fig,"figF_largura_progressiva")

# ===== (B) JANELAS 5 kHz DESLIZANTES — 3 métodos =====
dw=d[d.tipo=="win"].copy(); dw=dw.sort_values("centro")
fig,axes=plt.subplots(1,3,figsize=(18,4.8),dpi=170)
for ax,met,tit in zip(axes,["RMSD","CCDM","NRMSE"],["RMSD","CCDM","NRMSE (normalizado)"]):
    for m in ["Park","RF_direct","AE"]:
        s=dw[dw.metodo==m].sort_values("centro")
        ax.plot(s.centro,s[met],marker="o",lw=2.2,color=CM[m],label=LB[m])
    ax.set_xlabel("Centro da janela de 5 kHz (kHz)"); ax.set_ylabel(met); ax.set_title(f"({'abc'[list(['RMSD','CCDM','NRMSE']).index(met)]}) {tit}")
axes[2].legend(frameon=False)
fig.suptitle("Janelas de 5 kHz deslizantes: qual região é mais fácil de compensar?",y=1.02,fontsize=13)
fig.tight_layout(); savef(fig,"figF_janelas_5khz")

# ===== POR MÉTODO (só AE, só RF, só Park): RMSD|CCDM por banda (progressiva + janelas) =====
for m,fname in [("AE","figF_so_AE"),("RF_direct","figF_so_RF"),("Park","figF_so_Park")]:
    fig,axes=plt.subplots(1,2,figsize=(15,4.8),dpi=170)
    sp=dp[dp.metodo==m].set_index("banda").reindex(order); sw=dw[dw.metodo==m].sort_values("centro")
    for ax,met in zip(axes,["RMSD","CCDM"]):
        ax.plot(range(len(order)),sp[met],marker="o",lw=2.2,color=CM[m],label="bandas progressivas (30–X)")
        ax2=ax.twiny(); ax2.plot(sw.centro,sw[met],marker="s",lw=2,color="#888",ls="--",label="janelas 5 kHz")
        ax.set_ylabel(met); ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=30,ha="right",fontsize=8)
        ax.set_xlabel("Banda progressiva"); ax2.set_xlabel("Centro da janela 5 kHz (kHz)",fontsize=9)
        ax.set_title(f"{met}")
        if met=="RMSD":
            h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,frameon=False,fontsize=8)
    fig.suptitle(f"{LB[m]}: RMSD e CCDM por faixa de frequência",y=1.03,fontsize=13)
    fig.tight_layout(); savef(fig,fname)

# ===== ranking: melhor janela 5 kHz por método/metrica =====
print("\n=== MELHOR JANELA DE 5 kHz (menor RMSD saudável) ===",flush=True)
for m in ["Park","RF_direct","AE"]:
    s=dw[dw.metodo==m]
    br=s.loc[s.RMSD.idxmin()]; bc=s.loc[s.CCDM.idxmin()]
    print(f"  {LB[m]:14s}: melhor RMSD em {br.banda} ({br.RMSD:.2f}); melhor CCDM em {bc.banda} ({bc.CCDM:.4f})",flush=True)
print("\n=== EFEITO DA LARGURA (30-40 -> 30-100) ===",flush=True)
for m in ["Park","RF_direct","AE"]:
    s=dp[dp.metodo==m].set_index("banda")
    print(f"  {LB[m]:14s}: RMSD 30-40={s.loc['30-40','RMSD']:.2f} -> 30-100={s.loc['30-100','RMSD']:.2f} (var {100*(s.loc['30-100','RMSD']-s.loc['30-40','RMSD'])/s.loc['30-40','RMSD']:+.0f}%)",flush=True)
print(f"\n✅ análise de largura/janelas concluída em {(time.time()-t0)/60:.1f} min",flush=True)

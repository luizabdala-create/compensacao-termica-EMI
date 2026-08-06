# -*- coding: utf-8 -*-
"""
FIGURAS DO ARTIGO — geradas a partir da FASE 2 (2340 registros) e FASE 3.
PNG 600 dpi + PDF vetorial. Cores consistentes em todo o artigo.
"""
import os,sys,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); os.makedirs(FIG,exist_ok=True)

plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10,
 "pdf.fonttype":42,"ps.fonttype":42,"axes.linewidth":0.9})

# paleta consistente do artigo
COR={"Original":"#7f7f7f","Park":"#2ca02c","RF_direct":"#d62728","RF_temponly":"#9467bd",
     "AE":"#1f5fd0","TAU_T":"#7b2fbe"}
LBL={"Original":"Original","Park":"Park","RF_direct":"RF (direto)","RF_temponly":"RF (só temp.)",
     "AE":"Autoencoder","TAU_T":r"$\tau(T)$ low-rank (aux.)"}
ORD=["Original","Park","RF_direct","RF_temponly","AE","TAU_T"]
BANDORD=["30-40","30-50","30-60","30-70","30-80","30-90","30-100",
         "40-50","50-60","60-70","70-80","80-90","90-100"]

def save(fig,name):
    for ext,dpi in [("png",600),("pdf",None)]:
        fig.savefig(os.path.join(FIG,f"{name}.{ext}"),dpi=dpi,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",name)
def ax_clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(alpha=.25,linewidth=.6)

d=pd.read_csv(os.path.join(ROOT,"checkpoints","fase2_master.csv"))
d=d[~d.T_test_eh_T_ref].copy()
d["dT"]=np.abs(d.T_test-d.T_ref)
MET=[m for m in ORD if m in d.metodo.unique()]

# ---------- FIG 3/4: RMSD_D0 e CCDM_D0 por temperatura ----------
for metrica,fname,lab in [("RMSD_D0","fig03_RMSD_D0_por_temperatura","RMSD (curvas saudáveis)"),
                          ("CCDM_D0","fig04_CCDM_D0_por_temperatura","CCDM (curvas saudáveis)")]:
    fig,ax=plt.subplots(figsize=(7.2,4.6),dpi=170)
    for m in MET:
        g=d[d.metodo==m].groupby("T_test")[metrica].agg(["mean","std"])
        ax.plot(g.index,g["mean"],marker="o",ms=5,lw=2,color=COR[m],label=LBL[m])
        ax.fill_between(g.index,g["mean"]-g["std"],g["mean"]+g["std"],color=COR[m],alpha=.10)
    ax.set_xlabel("Temperatura de teste (°C)"); ax.set_ylabel({"RMSD (curvas saudáveis)":"RMSD (curvas saudáveis)","CCDM (curvas saudáveis)":"CCDM (curvas saudáveis)"}.get(lab,lab))
    ax.set_yscale("log" if metrica=="RMSD_D0" else "linear")
    ax.legend(ncol=2,frameon=False); ax_clean(ax)
    save(fig,fname)

# ---------- FIG 5/6: RMSD e CCDM por classe, painel por método ----------
for metrica,fname in [("RMSD","fig05_RMSD_por_classe"),("CCDM","fig06_CCDM_por_classe")]:
    fig,axes=plt.subplots(2,3,figsize=(13,7),dpi=170,sharex=True,sharey=True)
    CD={0:"#1f77b4",1:"#ff7f0e",2:"#d62728"}; NM={0:"D0 (saudável)",1:"D1",2:"D2"}
    for ax,m in zip(axes.ravel(),MET):
        for c in [0,1,2]:
            g=d[d.metodo==m].groupby("T_test")[f"{metrica}_D{c}"].mean()
            ax.plot(g.index,g.values,marker="o",ms=4,lw=1.8,color=CD[c],label=NM[c])
        ax.set_title(LBL[m]); ax_clean(ax)
        if metrica=="RMSD": ax.set_yscale("log")
    for ax in axes[1]: ax.set_xlabel("Temperatura de teste (°C)")
    for ax in axes[:,0]: ax.set_ylabel(metrica)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle(f"{metrica} por classe de dano — LOTO, todas as bandas e temperaturas de referência",y=.98)
    fig.tight_layout(); save(fig,fname)

# ---------- FIG 7: separação de dano ----------
fig,axes=plt.subplots(1,2,figsize=(11,4.4),dpi=170,sharey=True)
for ax,col,tit in zip(axes,["sep_RMSD_D1","sep_RMSD_D2"],["D1 − D0","D2 − D0"]):
    vals=[d[d.metodo==m][col].dropna().values for m in MET]
    bp=ax.boxplot(vals,patch_artist=True,widths=.6,showfliers=False,medianprops=dict(color="k",lw=1.4))
    for p,m in zip(bp["boxes"],MET): p.set_facecolor(COR[m]); p.set_alpha(.75)
    ax.axhline(0,color="k",lw=1,ls="--")
    ax.set_xticks(range(1,len(MET)+1)); ax.set_xticklabels([LBL[m] for m in MET],rotation=25,ha="right")
    ax.set_title(f"Separação do índice de dano: {tit}"); ax_clean(ax)
axes[0].set_ylabel("Diferença de RMSD")
fig.tight_layout(); save(fig,"fig07_separacao_dano")

# ---------- FIG 8: healthy_sep (critério primário) ----------
fig,ax=plt.subplots(figsize=(7.6,4.4),dpi=170)
g=d.groupby("metodo")["healthy_sep"].mean().reindex(MET)
g2=d.groupby("metodo")["full_order"].mean().reindex(MET)
x=np.arange(len(MET)); w=.38
ax.bar(x-w/2,g.values,w,color=[COR[m] for m in MET],edgecolor="k",lw=.7,label="D0 < min(D1,D2)  [primary]")
ax.bar(x+w/2,g2.values,w,color=[COR[m] for m in MET],edgecolor="k",lw=.7,alpha=.45,hatch="//",
       label="D0 < D1 < D2  [descriptive only]")
for xi,v in zip(x-w/2,g.values): ax.text(xi,v+.02,f"{v:.2f}",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([LBL[m] for m in MET],rotation=20,ha="right")
ax.set_ylabel("Fração de folds LOTO"); ax.set_ylim(0,1.12)
ax.legend(frameon=False,loc="upper left"); ax_clean(ax)
ax.set_title("Ordenação do índice de dano entre folds")
fig.tight_layout(); save(fig,"fig08_monotonicidade")

# ---------- FIG 9/10: heatmaps banda x T_ref ----------
for metrica,fname,cmap,tit in [("RMSD_D0","fig09_heatmap_banda_Tref_RMSD","viridis_r","RMSD (saudável)"),
                               ("healthy_sep","fig10_heatmap_banda_Tref_sep","viridis","D0 < min(D1,D2)")]:
    fig,axes=plt.subplots(1,len(MET),figsize=(3.0*len(MET),4.6),dpi=170,sharey=True)
    bands=[b for b in BANDORD if b in d.banda.unique()]
    vv=d[metrica].astype(float).values
    vmin=float(np.nanmin(vv)); vmax=float(np.nanpercentile(vv,97))
    for ax,m in zip(axes,MET):
        pv=d[d.metodo==m].pivot_table(index="banda",columns="T_ref",values=metrica,aggfunc="mean").reindex(bands)
        im=ax.imshow(pv.values,aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels([f"{c:.0f}" for c in pv.columns])
        ax.set_title(LBL[m],fontsize=10.5); ax.set_xlabel("$T_{ref}$ (°C)")
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v=pv.values[i,j]
                if v==v: ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7,
                                 color="w" if (v-vmin)/(vmax-vmin+1e-9)>.5 else "k")
    axes[0].set_yticks(range(len(bands))); axes[0].set_yticklabels(bands)
    axes[0].set_ylabel("Banda de frequência (kHz)")
    fig.colorbar(im,ax=axes,fraction=.02,pad=.01,label=tit)
    fig.suptitle(f"{tit} — banda × temperatura de referência",y=1.0)
    save(fig,fname)

# ---------- FIG 16: pareado por fold ----------
key=["banda","T_ref","T_test"]
w=d.pivot_table(index=key,columns="metodo",values="RMSD_D0").dropna()
sel=[m for m in ["Park","RF_direct","AE"] if m in w.columns]
fig,ax=plt.subplots(figsize=(6.4,5),dpi=170)
sub=w.sample(min(120,len(w)),random_state=0)
for _,r in sub.iterrows():
    ax.plot(range(len(sel)),[r[m] for m in sel],"-",color="gray",alpha=.16,lw=.8)
ax.plot(range(len(sel)),[w[m].mean() for m in sel],"-o",color="crimson",lw=3,ms=10,label="média")
ax.set_xticks(range(len(sel))); ax.set_xticklabels([LBL[m] for m in sel])
ax.set_ylabel("RMSD (saudável)"); ax.set_yscale("log")
ax.set_title(f"Comparação pareada por fold (n={len(w)} folds)")
ax.legend(frameon=False); ax_clean(ax)
fig.tight_layout(); save(fig,"fig16_pareado_por_fold")

# ---------- FIG 17: Pareto compensação x preservação ----------
fig,ax=plt.subplots(figsize=(6.6,5),dpi=170)
for m in MET:
    s=d[d.metodo==m]
    ax.scatter(s.RMSD_D0.mean(),s.healthy_sep.mean(),s=210,color=COR[m],edgecolor="k",lw=1,zorder=3)
    ax.annotate(LBL[m],(s.RMSD_D0.mean(),s.healthy_sep.mean()),
                textcoords="offset points",xytext=(10,6),fontsize=10)
ax.set_xscale("log"); ax.set_xlabel("RMSD nas curvas saudáveis  (← melhor)")
ax.set_ylabel("Fração de folds com D0 < min(D1,D2)  (melhor ↑)")
ax.set_title("Pareto: compensação térmica vs. separabilidade de dano"); ax_clean(ax)
fig.tight_layout(); save(fig,"fig17_pareto")

# ---------- FIG 18: desempenho x distância térmica ----------
fig,axes=plt.subplots(1,2,figsize=(11,4.4),dpi=170)
d["bin_dT"]=pd.cut(d.dT,[-1,10,30,200],labels=["≤10","10–30",">30"])
for ax,col,lab in zip(axes,["RMSD_D0","healthy_sep"],["RMSD (saudável)","D0 < min(D1,D2)"]):
    for m in MET:
        g=d[d.metodo==m].groupby("bin_dT",observed=True)[col].mean()
        ax.plot(range(len(g)),g.values,marker="o",ms=7,lw=2,color=COR[m],label=LBL[m])
    ax.set_xticks(range(3)); ax.set_xticklabels(["≤10","10–30",">30"])
    ax.set_xlabel(r"$|T_{test}-T_{ref}|$  (°C)"); ax.set_ylabel({"RMSD (curvas saudáveis)":"RMSD (curvas saudáveis)","CCDM (curvas saudáveis)":"CCDM (curvas saudáveis)"}.get(lab,lab)); ax_clean(ax)
    if col=="RMSD_D0": ax.set_yscale("log")
axes[1].legend(ncol=2,frameon=False,fontsize=9)
fig.suptitle("Robustez à distância térmica da referência",y=1.0)
fig.tight_layout(); save(fig,"fig18_distancia_termica")

# ---------- FIG 20: distribuição por fold ----------
fig,axes=plt.subplots(1,2,figsize=(11,4.4),dpi=170)
for ax,col,lab in zip(axes,["RMSD_D0","CCDM_D0"],["RMSD (saudável)","CCDM (healthy)"]):
    vals=[d[d.metodo==m][col].dropna().values for m in MET]
    vp=ax.violinplot(vals,showmedians=True,widths=.8)
    for b,m in zip(vp["bodies"],MET): b.set_facecolor(COR[m]); b.set_alpha(.7)
    ax.set_xticks(range(1,len(MET)+1)); ax.set_xticklabels([LBL[m] for m in MET],rotation=25,ha="right")
    ax.set_ylabel({"RMSD (curvas saudáveis)":"RMSD (curvas saudáveis)","CCDM (curvas saudáveis)":"CCDM (curvas saudáveis)"}.get(lab,lab)); ax_clean(ax)
    if col=="RMSD_D0": ax.set_yscale("log")
fig.suptitle("Distribuição sobre todos os folds LOTO, bandas e temperaturas de referência",y=1.0)
fig.tight_layout(); save(fig,"fig20_distribuicoes")

print(f"\n✅ figuras em {FIG}")

# -*- coding: utf-8 -*-
"""Regenera figuras de comparação a partir da bateria AMPLIADA (Fase 8, 7 bandas)."""
import os,sys,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; FIG=os.path.join(ROOT,"10_figuras_artigo")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,"axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold",
 "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9.5,"pdf.fonttype":42})
COR={"Original":"#7f7f7f","Park":"#2ca02c","RF_direct":"#d62728","ExtraTrees":"#ff7f0e","RF_temponly":"#9467bd","AE":"#1f5fd0","TAU_T":"#7b2fbe"}
LBL={"Park":"Park","RF_direct":"RF (direto)","ExtraTrees":"RF otimizado","RF_temponly":"RF (temp-only)","AE":"Autoencoder","TAU_T":r"$\tau(T)$ (aux.)"}
def save(fig,n):
    for e,d in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=d,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n)
def clean(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=.25,lw=.6)

p=os.path.join(ROOT,"02_compensacao","fase8_tuning_ampliado.csv")
if not os.path.exists(p): p=os.path.join(ROOT,"02_compensacao","comparacao_justa_todos_tunados.csv")
d=pd.read_csv(p); d=d[d.metodo!="TAU_T"]
MJ=[m for m in ["AE","Park","RF_direct","ExtraTrees"] if m in d.metodo.unique()]
bands=[b for b in ["30-40","40-50","50-60","60-70","70-80","80-90","90-100"] if b in d.banda.unique()]

# fig03b — RMSD_D0 por banda (tunado)
piv=d.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean").reindex(bands)
fig,ax=plt.subplots(figsize=(10,4.8),dpi=170); x=np.arange(len(bands)); w=.8/len(MJ)
for i,m in enumerate(MJ):
    ax.bar(x+(i-(len(MJ)-1)/2)*w,piv[m],w,color=COR[m],edgecolor="k",lw=.6,label=LBL[m])
ax.set_xticks(x); ax.set_xticklabels([b+"\nkHz" for b in bands]); ax.set_ylabel("RMSD nas curvas saudáveis")
ax.set_ylim(0,np.nanmax(piv.values)*1.18)
ax.set_title("Comparação justa (todos ajustados por CV interna) — RMSD por banda",pad=8)
ax.legend(frameon=False,ncol=len(MJ),loc="upper center",bbox_to_anchor=(0.5,-0.16)); clean(ax); fig.tight_layout(); save(fig,"fig03b_comparacao_justa_RMSD_banda")

# fig_winner — mapa de vencedor por banda
fig,ax=plt.subplots(figsize=(9,3.2),dpi=170)
main=[m for m in ["AE","Park","RF_direct"] if m in piv.columns]
wins=piv[main].idxmin(axis=1)
for i,b in enumerate(bands):
    m=wins[b]; ax.barh(0,1,left=i,color=COR[m],edgecolor="w")
    ax.text(i+.5,0,f"{b}\n{LBL[m]}\n{piv.loc[b,m]:.2f}",ha="center",va="center",fontsize=8.5,color="w",weight="bold")
ax.set_xlim(0,len(bands)); ax.set_ylim(-.5,.5); ax.axis("off")
ax.set_title("Melhor compensador por banda (menor RMSD saudável)")
fig.tight_layout(); save(fig,"fig_winner_por_banda")

# fig healthy_sep por banda
if "healthy_sep" in d.columns:
    hs=d.pivot_table(index="banda",columns="metodo",values="healthy_sep",aggfunc="mean").reindex(bands)
    fig,ax=plt.subplots(figsize=(10,4.4),dpi=170)
    for i,m in enumerate(MJ):
        ax.bar(x+(i-(len(MJ)-1)/2)*w,hs[m],w,color=COR[m],edgecolor="k",lw=.6,label=LBL[m])
    ax.set_xticks(x); ax.set_xticklabels([b for b in bands]); ax.set_ylabel("D0 < min(D1,D2)"); ax.set_ylim(0,1.15)
    ax.set_title("healthy_sep por banda (fração de folds)",pad=8); ax.legend(frameon=False,ncol=len(MJ),loc="upper center",bbox_to_anchor=(0.5,-0.14)); clean(ax)
    fig.tight_layout(); save(fig,"fig08b_healthy_sep_banda")

# pareto tunado
fig,ax=plt.subplots(figsize=(6.6,5),dpi=170)
for m in MJ:
    s=d[d.metodo==m]
    ax.scatter(s.RMSD_D0.mean(),s.healthy_sep.mean() if "healthy_sep" in s else np.nan,
               s=210,color=COR[m],edgecolor="k",lw=1,zorder=3)
    ax.annotate(LBL[m],(s.RMSD_D0.mean(),s.healthy_sep.mean() if "healthy_sep" in s else np.nan),
                textcoords="offset points",xytext=(9,5),fontsize=10)
ax.set_xlabel("RMSD saudável (← melhor)"); ax.set_ylabel("healthy_sep (melhor ↑)")
ax.set_title("Pareto (tunado): compensação vs. separabilidade"); clean(ax)
fig.tight_layout(); save(fig,"fig17b_pareto_tunado")
print("✅ figuras finais (Fase 8) geradas")

# -*- coding: utf-8 -*-
"""Figura de estabilidade entre seeds (suplementar)."""
import os,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; FIG=os.path.join(ROOT,"10_figuras_artigo")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"pdf.fonttype":42})
d=pd.read_csv(os.path.join(ROOT,"checkpoints","fase7_seeds.csv"))
per=d.groupby(["banda","metodo","seed"])["RMSD_D0"].mean().reset_index()
ag=per.groupby(["banda","metodo"])["RMSD_D0"].agg(["mean","std"]).reset_index()
bands=["30-40","30-60","60-70","70-80"]
COR={"AE":"#1f5fd0","RF_direct":"#d62728"}; LBL={"AE":"Autoencoder","RF_direct":"RF (direto)"}
fig,ax=plt.subplots(figsize=(8,4.6),dpi=170); x=np.arange(len(bands)); w=.36
for i,m in enumerate(["AE","RF_direct"]):
    s=ag[ag.metodo==m].set_index("banda").reindex(bands)
    ax.bar(x+(i-.5)*w,s["mean"],w,yerr=s["std"],capsize=4,color=COR[m],edgecolor="k",lw=.7,label=LBL[m])
    for xi,mn,sd in zip(x+(i-.5)*w,s["mean"],s["std"]):
        ax.text(xi,mn+sd+.08,f"{mn:.2f}\n±{sd:.3f}",ha="center",fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([b+" kHz" for b in bands])
ax.set_ylabel("RMSD nas curvas saudáveis"); ax.set_title("Estabilidade entre sementes (3 cada): AE 42/123/2026, RF 0/1/2")
ax.legend(frameon=False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=.25)
fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figS1_seeds.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
print("figS1 seeds ok")

# -*- coding: utf-8 -*-
"""FIGURA 11 — mapa das janelas móveis: frequência central × largura, RMSD_D0."""
import os,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; FIG=os.path.join(ROOT,"10_figuras_artigo")
plt.rcParams.update({"font.family":"serif","font.size":11,"pdf.fonttype":42})
d=pd.read_csv(os.path.join(ROOT,"07_janelas","janelas_moveis_coarse.csv"))
METS=["Park","RF_direct","TAU_T"]; LBL={"Park":"Park","RF_direct":"RF (direct)","TAU_T":r"$\tau(T)$ (aux.)"}
fig,axes=plt.subplots(1,3,figsize=(15,4.6),dpi=170,sharey=True)
vmax=np.nanpercentile(d.RMSD_D0,90)
for ax,m in zip(axes,METS):
    s=d[d.metodo==m]
    sc=ax.scatter(s.centro,s.largura,c=s.RMSD_D0.clip(upper=vmax),cmap="viridis_r",
                  s=90,edgecolor="k",lw=.5,vmin=0.5,vmax=vmax)
    # marca as 3 melhores janelas
    top=s.nsmallest(3,"RMSD_D0")
    ax.scatter(top.centro,top.largura,s=260,facecolor="none",edgecolor="red",lw=2)
    for _,r in top.iterrows():
        ax.annotate(f"{r.RMSD_D0:.2f}",(r.centro,r.largura),textcoords="offset points",
                    xytext=(6,6),fontsize=8,color="red")
    ax.set_title(LBL[m]); ax.set_xlabel("Window centre (kHz)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=.2)
axes[0].set_ylabel("Window width (kHz)")
cb=fig.colorbar(sc,ax=axes,fraction=.02,pad=.01); cb.set_label("RMSD on healthy curves")
fig.suptitle("Moving-window scan (30-100 kHz): healthy RMSD vs window centre and width",y=1.0)
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"fig11_janelas_moveis.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
print("fig11 ok | melhor global:",d.loc[d.RMSD_D0.idxmin(),["metodo","lo","hi","RMSD_D0"]].to_dict())

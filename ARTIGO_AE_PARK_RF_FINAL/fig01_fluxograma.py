# -*- coding: utf-8 -*-
"""FIGURA 1 — fluxograma metodológico."""
import os,matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; FIG=os.path.join(ROOT,"10_figuras_artigo")
plt.rcParams.update({"font.family":"serif","pdf.fonttype":42})
fig,ax=plt.subplots(figsize=(11,7),dpi=170); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
def box(x,y,w,h,txt,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",fc=c,ec="black",lw=1.3))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=10.5)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,lw=1.4,color="#333"))
box(3.2,9.0,3.6,0.8,"EMI dataset (164 curves, 1 Hz)\n10 temps with D0/D1/D2","#e8eef7")
box(3.2,7.7,3.6,0.8,"Leave-One-Temperature-Out\n(outer loop, 10 folds)","#dbe9d5")
arrow(5,9.0,5,8.5)
box(0.3,6.2,3.0,0.9,"Frozen healthy reference\n(median @ T_ref)","#f7f0da")
box(6.7,6.2,3.0,0.9,"Inner CV on train temps\n(hyperparameters, band)","#f7f0da")
arrow(4.2,7.7,2.4,7.1); arrow(5.8,7.7,7.8,7.1)
box(2.0,4.7,6.0,0.9,"Compensation (train on healthy only):\nOriginal · Park · RF(direct/temp-only) · Autoencoder · [tau(T) aux.]","#e7dff2")
arrow(1.8,6.2,3.5,5.6); arrow(8.2,6.2,6.5,5.6)
box(0.4,3.1,4.2,0.9,"PART A — Compensation\nRMSD/CCDM, sep_D1/D2, healthy_sep","#dfeaf2")
box(5.4,3.1,4.2,0.9,"PART B — Classification\nbinary + D0/D1/D2, false-healthy","#f2dfe0")
arrow(4,4.7,2.5,4.0); arrow(6,4.7,7.5,4.0)
box(1.5,1.6,7.0,0.9,"Paired statistics (Friedman + Wilcoxon/Holm) · Negative control (shuffled labels)","#eeeeee")
arrow(2.5,3.1,4,2.5); arrow(7.5,3.1,6,2.5)
box(3.0,0.3,4.0,0.8,"Band-resolved comparison\n(no method dominates)","#dbe9d5")
arrow(5,1.6,5,1.1)
fig.tight_layout()
for e,d in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"fig01_fluxograma.{e}"),dpi=d,bbox_inches="tight",facecolor="white")
print("fig01 ok")

# -*- coding: utf-8 -*-
"""Pareto (A) remoção térmica vs (B) classificação de dano — out-of-sample."""
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=r"C:\Users\luize\IC_EMI\etapa3_v7_split"; os.makedirs(OUT,exist_ok=True)

# (A) RMSD saudável (menor melhor), (B) balanced accuracy (maior melhor)
data={
 "B_ref30":{"Original":(10.923,0.333),"Park":(1.852,0.875),"Físico (núcleo V7)":(5.550,0.333),"AE V7 completo":(1.437,0.667)},
 "A_ref20":{"Original":(13.352,0.458),"Park":(2.239,0.792),"Físico (núcleo V7)":(15.529,0.417),"AE V7 completo":(1.954,0.667)},
}
cores={"Original":"tab:red","Park":"tab:green","Físico (núcleo V7)":"tab:gray","AE V7 completo":"tab:blue"}
mark={"B_ref30":"o","A_ref20":"s"}

fig,ax=plt.subplots(figsize=(10,7),dpi=200)
for split,pts in data.items():
    for m,(a,b) in pts.items():
        ax.scatter(a,b,s=170,color=cores[m],marker=mark[split],edgecolor="black",linewidth=1.2,zorder=3)
        ax.annotate(f"{m}",(a,b),textcoords="offset points",xytext=(9,5),fontsize=10)
ax.axhline(1/3,color="gray",ls=":",lw=1); ax.text(13.5,0.345,"acaso (3 classes)",fontsize=9,color="gray")
ax.set_xlabel("(A) RMSD saudável — remoção térmica  (← melhor)",fontsize=13)
ax.set_ylabel("(B) Balanced accuracy — detecção de dano  (melhor ↑)",fontsize=13)
ax.set_title("Pareto out-of-sample — o que importa é o eixo Y (detecção)",fontsize=14)
ax.invert_xaxis()
# legenda de forma
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker="o",color="w",markerfacecolor="k",markersize=11,label="Split B (ref30)"),
     Line2D([0],[0],marker="s",color="w",markerfacecolor="k",markersize=11,label="Split A' (ref20)")]
ax.legend(handles=leg,loc="lower left",fontsize=10)
ax.grid(alpha=0.25)
# seta destacando o trade-off do V7
ax.annotate("V7 ganha em (A) mas\nPERDE em (B): apaga dano",
            xy=(1.44,0.667),xytext=(4.5,0.55),fontsize=10,color="tab:blue",
            arrowprops=dict(arrowstyle="->",color="tab:blue"))
fig.tight_layout()
p=os.path.join(OUT,"pareto_A_vs_B.png"); fig.savefig(p,dpi=200,bbox_inches="tight",facecolor="white")
print("salvo",p)

# -*- coding: utf-8 -*-
"""
RADAR COMPARATIVO — perfil multicritério de cada método, normalizado 0..1 (1 = melhor).
Usa apenas dados já computados (fase8, extrapolação, T_ref). Critérios: compensação, preservação
de dano, robustez à banda larga, robustez à T_ref, robustez à extrapolação, custo baixo / sem-treino.
"""
import os,sys,json,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; FIG=os.path.join(ROOT,"10_figuras_artigo")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.titlesize":14,
 "axes.titleweight":"bold","pdf.fonttype":42})
CM={"Park":"#2ca02c","RF_direct":"#d62728","ExtraTrees":"#ff7f0e","AE":"#1f5fd0"}
LB={"Park":"Park","RF_direct":"Random Forest","ExtraTrees":"RF otimizado","AE":"Autoencoder"}
MET=["Park","RF_direct","ExtraTrees","AE"]
f8=pd.read_csv(os.path.join(ROOT,"02_compensacao","fase8_tuning_ampliado.csv")); f8=f8[f8.metodo!="TAU_T"]
wide=["30-70","30-100"];
def norm_lowbest(d):
    v=np.array([d[m] for m in MET]); lo,hi=v.min(),v.max(); return {m:(hi-d[m])/(hi-lo+1e-9) for m in MET}
def norm_highbest(d):
    v=np.array([d[m] for m in MET]); lo,hi=v.min(),v.max(); return {m:(d[m]-lo)/(hi-lo+1e-9) for m in MET}
comp={m:f8[f8.metodo==m].RMSD_D0.mean() for m in MET}
pres={m:f8[f8.metodo==m].healthy_sep.mean() for m in MET}
wideb={m:f8[(f8.metodo==m)&(f8.banda.isin(wide))].RMSD_D0.mean() for m in MET}
# T_ref amplitude
mtr=pd.read_csv(os.path.join(ROOT,"06_sensibilidade_referencia","melhor_tref_resumo.csv"))
tref={m:mtr[mtr.metodo==m].amplitude.mean() for m in MET}
# extrapolação
ext=pd.read_csv(os.path.join(ROOT,"08_analises_avancadas","extrapolacao.csv"))
extmap={"Park":"Park","RF_direct":"RF","ExtraTrees":"ExtraTrees","AE":"AE"}
extr={m:ext[extmap[m]].mean() for m in MET}
# custo (Park=0 treino; ML alto; AE médio)
custo={"Park":0.0,"RF_direct":150.0,"ExtraTrees":150.0,"AE":4.5}
crit=[("Compensação",norm_lowbest(comp)),("Preservação de dano",norm_highbest(pres)),
      ("Robustez banda larga",norm_lowbest(wideb)),("Robustez a T_ref",norm_lowbest(tref)),
      ("Robustez à extrapolação",norm_lowbest(extr)),("Baixo custo / sem treino",norm_lowbest(custo))]
labels=[c[0] for c in crit]; N=len(labels)
ang=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); ang+=ang[:1]
fig,ax=plt.subplots(figsize=(8.5,8),dpi=170,subplot_kw=dict(polar=True))
for m in MET:
    vals=[c[1][m] for c in crit]; vals+=vals[:1]
    ax.plot(ang,vals,lw=2.2,color=CM[m],label=LB[m]); ax.fill(ang,vals,color=CM[m],alpha=.08)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels,fontsize=10.5)
ax.set_yticks([.25,.5,.75,1.0]); ax.set_yticklabels(["0,25","0,50","0,75","1,0"],fontsize=8); ax.set_ylim(0,1)
ax.set_title("Perfil comparativo multicritério dos métodos\n(1 = melhor em cada eixo; normalizado)",pad=24)
ax.legend(loc="upper right",bbox_to_anchor=(1.28,1.12),frameon=False,fontsize=10)
fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figX_radar.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figX_radar")
# salva tabela normalizada
dd=pd.DataFrame({c[0]:c[1] for c in crit}).T; dd.columns=[LB[m] for m in MET]; dd.to_csv(os.path.join(ROOT,"08_analises_avancadas","radar_norm.csv"))
print(dd.round(2).to_string())
print("✅ radar comparativo gerado")

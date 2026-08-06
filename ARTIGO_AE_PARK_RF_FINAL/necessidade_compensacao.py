# -*- coding: utf-8 -*-
"""
POR QUE COMPENSAR? — demonstração de que a compensação térmica é NECESSÁRIA.
Compara o sinal ORIGINAL (sem compensação) com os métodos compensados em três eixos:
 (a) distância à referência das curvas SAUDÁVEIS (a temperatura infla o DI do saudável);
 (b) SEPARAÇÃO saudável–dano (sem compensar, o dano fica mascarado);
 (c) AUC de DETECÇÃO do dano pela distância à referência (Original ~0,5 = acaso).
Usa dados já computados (di_por_curva.csv, auc_deteccao.csv) — sem recomputar.
Também consolida CLASSIFICAÇÃO Original vs compensado (parteB) antes/depois.
"""
import os,sys,json,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9.5,"pdf.fonttype":42,
 "axes.spines.top":False,"axes.spines.right":False})
CM={"Original":"#7f7f7f","Park":"#2ca02c","RF_direct":"#d62728","AE":"#1f5fd0"}
LB={"Original":"Original\n(sem comp.)","Park":"Park","RF_direct":"Random\nForest","AE":"Auto-\nencoder"}
MET=["Original","Park","RF_direct","AE"]
def savef(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n,flush=True)

pc=pd.read_csv(os.path.join(OUT,"di_por_curva.csv"))
auc=pd.read_csv(os.path.join(OUT,"auc_deteccao.csv")).rename(columns={"Unnamed: 0":"metodo"}).set_index("metodo")

# métricas agregadas (todas as bandas)
di_saud={m:pc[(pc.metodo==m)&(pc.dano==0)].DI_rmsd.mean() for m in MET}
di_dano={m:pc[(pc.metodo==m)&(pc.dano>0)].DI_rmsd.mean() for m in MET}
sep={m:di_dano[m]-di_saud[m] for m in MET}
aucg={m:auc.loc[m,"global"] for m in MET}
res=pd.DataFrame({"DI_saudavel":di_saud,"DI_dano":di_dano,"separacao":sep,"AUC_deteccao":aucg}).T
res.to_csv(os.path.join(OUT,"necessidade_compensacao.csv"))
print("=== NECESSIDADE DA COMPENSAÇÃO (todas as bandas) ===\n",res.round(3).to_string(),flush=True)
red_saud=100*(di_saud["Original"]-min(di_saud["Park"],di_saud["RF_direct"],di_saud["AE"]))/di_saud["Original"]
print(f"\nRedução do DI saudável (Original -> melhor compensado): {red_saud:.0f}%",flush=True)
print(f"AUC de detecção: Original={aucg['Original']:.3f} (acaso) -> AE={aucg['AE']:.3f}",flush=True)

# ===== FIGURA: por que compensar (3 painéis antes vs depois) =====
fig,axes=plt.subplots(1,3,figsize=(15,4.6),dpi=170)
x=np.arange(len(MET)); cols=[CM[m] for m in MET]
# (a) DI saudável — quanto menor, mais temperatura removida
va=[di_saud[m] for m in MET]; axes[0].bar(x,va,color=cols,edgecolor="k",lw=.6)
for i,v in enumerate(va): axes[0].text(i,v+max(va)*.02,f"{v:.1f}",ha="center",fontsize=9)
axes[0].set_ylim(0,max(va)*1.16); axes[0].set_xticks(x); axes[0].set_xticklabels([LB[m] for m in MET])
axes[0].set_ylabel("DI das curvas saudáveis (RMSD à ref.)"); axes[0].set_title("(a) A temperatura infla o sinal saudável\n(menor = mais removida)")
# (b) separação saudável-dano — quanto maior, dano mais destacado
vb=[sep[m] for m in MET]; axes[1].bar(x,vb,color=cols,edgecolor="k",lw=.6)
for i,v in enumerate(vb): axes[1].text(i,v+max(vb)*.02,f"{v:.2f}",ha="center",fontsize=9)
axes[1].set_ylim(min(0,min(vb)*1.1),max(vb)*1.16); axes[1].axhline(0,color="k",lw=.8)
axes[1].set_xticks(x); axes[1].set_xticklabels([LB[m] for m in MET])
axes[1].set_ylabel("Separação (DI dano − DI saudável)"); axes[1].set_title("(b) Sem compensar, o dano fica mascarado\n(maior = mais separável)")
# (c) AUC de detecção — Original ~0.5 = acaso
vc=[aucg[m] for m in MET]; bars=axes[2].bar(x,vc,color=cols,edgecolor="k",lw=.6)
axes[2].axhline(0.5,color="crimson",ls="--",lw=1.2); axes[2].text(len(MET)-1,0.52,"acaso (0,5)",color="crimson",fontsize=8,ha="right")
for i,v in enumerate(vc): axes[2].text(i,v+.02,f"{v:.2f}",ha="center",fontsize=9)
axes[2].set_ylim(0,1.08); axes[2].set_xticks(x); axes[2].set_xticklabels([LB[m] for m in MET])
axes[2].set_ylabel("AUC de detecção de dano"); axes[2].set_title("(c) Sem compensação a detecção é\nquase aleatória (AUC≈0,5)")
fig.suptitle("Por que compensar? O sinal original mascara o dano; a compensação o revela",y=1.03,fontsize=13)
fig.tight_layout(); savef(fig,"figB_necessidade")

# ===== consolidação de CLASSIFICAÇÃO Original vs compensado (parteB) =====
pb=None
for p in ["checkpoints/parteB.csv"]:
    fp=os.path.join(ROOT,p)
    if os.path.exists(fp): pb=pd.read_csv(fp)
if pb is not None:
    pb=pb[(pb.metodo!="TAU_T")&(pb.controle=="real")]
    rowsc=[]
    for m in ["Original","Park","RF_direct","AE"]:
        s=pb[pb.metodo==m]
        rowsc.append({"metodo":m,
            "bin_bal_acc":s[s.task=="bin"].bal_acc.mean(),
            "multi_bal_acc":s[s.task=="multi"].bal_acc.mean(),
            "falso_saudavel":s[s.task=="bin"].taxa_falso_saudavel.mean() if "taxa_falso_saudavel" in s.columns else np.nan})
    cc=pd.DataFrame(rowsc).set_index("metodo"); cc.to_csv(os.path.join(OUT,"classificacao_antes_depois.csv"))
    print("\n=== CLASSIFICAÇÃO Original vs compensado ===\n",cc.round(3).to_string(),flush=True)
print("\n✅ necessidade da compensação concluída",flush=True)

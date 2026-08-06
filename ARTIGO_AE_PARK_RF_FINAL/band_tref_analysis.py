# -*- coding: utf-8 -*-
"""
ANÁLISE COMPLETA — FAIXAS DE FREQUÊNCIA e TEMPERATURA DE REFERÊNCIA (heatmaps + rankings).
Produz:
 - heatmap método×banda (RMSD_D0 e healthy_sep) — qual método vence onde
 - heatmap vencedor por (banda×T_ref) — a partir do screening (fase2, 13 bandas×3 Tref)
 - "faixa mais analisável para o dano": classificação e separabilidade por banda
 - heatmap T_ref×método (varredura completa) e degradação com distância térmica
 - relatório-texto com as melhores faixas/T_ref
"""
import os,sys,json,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"05_sensibilidade_faixas")
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
LBL={"AE":"AE","Park":"Park","RF_direct":"RF","ExtraTrees":"RF otim.","RF_temponly":"RF-tmp","TAU_T":"tau(T)","Original":"Orig"}
def Rd(p):
    fp=os.path.join(ROOT,p); return pd.read_csv(fp) if os.path.exists(fp) else None
def savef(fig,n):
    for e,d in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=d,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n)
def heat(ax,piv,cmap,vmin,vmax,fmt="{:.2f}",thr=None,cbar=None):
    im=ax.imshow(piv.values,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns,rotation=30,ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v=piv.values[i,j]
            if v==v:
                t=(v-vmin)/(vmax-vmin+1e-9); ax.text(j,i,fmt.format(v),ha="center",va="center",fontsize=7,color="w" if (t>0.55 if thr is None else v>thr) else "k")
    return im
report=[]
def log(s): print(s); report.append(str(s))

BORDER=["30-40","40-50","50-60","60-70","70-80","80-90","90-100","30-50","30-70","30-100"]

# ===== 1) fase8 tunado: método × banda =====
comp=Rd("02_compensacao/fase8_tuning_ampliado.csv")
if comp is not None:
    comp=comp[comp.metodo!="TAU_T"]
    MJ=[m for m in ["AE","Park","RF_direct","ExtraTrees"] if m in comp.metodo.unique()]
    bands=[b for b in BORDER if b in comp.banda.unique()]
    pr=comp.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean").reindex(bands)[MJ]
    pr.columns=[LBL[m] for m in MJ]
    fig,axes=plt.subplots(1,2,figsize=(14,5),dpi=170)
    im=heat(axes[0],pr,"viridis_r",float(np.nanmin(pr.values)),float(np.nanpercentile(pr.values,95)))
    axes[0].set_title("(a) RMSD saudável (menor=melhor)"); axes[0].set_ylabel("Banda (kHz)")
    fig.colorbar(im,ax=axes[0],fraction=.046,pad=.04)
    hs=comp.pivot_table(index="banda",columns="metodo",values="healthy_sep",aggfunc="mean").reindex(bands)[MJ]; hs.columns=[LBL[m] for m in MJ]
    im2=heat(axes[1],hs,"viridis",0.5,1.0); axes[1].set_title("(b) healthy_sep (maior=melhor)")
    fig.colorbar(im2,ax=axes[1],fraction=.046,pad=.04)
    fig.suptitle("Método × banda de frequência (todos ajustados por validação interna)",y=1.0,fontsize=12)
    fig.tight_layout(); savef(fig,"figH_metodo_banda")
    # vencedor por banda
    main=[m for m in ["AE","Park","RF_direct","ExtraTrees"] if m in comp.metodo.unique()]
    prm=comp.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean").reindex(bands)[main]
    log("=== VENCEDOR (RMSD saudável) por banda ===");
    for b in bands: log(f"  {b:8s}: {prm.loc[b].idxmin()} ({prm.loc[b].min():.2f})")

# ===== 2) fase2 screening: vencedor por (banda × T_ref) =====
f2=Rd("checkpoints/fase2_master.csv")
if f2 is not None:
    f2=f2[~f2.T_test_eh_T_ref]
    main=["AE","Park","RF_direct","ExtraTrees"]
    bands2=sorted(f2.banda.unique(),key=lambda s:(int(s.split("-")[0]),int(s.split("-")[1])))
    trefs=sorted(f2.T_ref.unique())
    # matriz vencedor (codigo cor): 0=AE,1=Park,2=RF
    Wc=np.full((len(bands2),len(trefs)),np.nan)
    for i,b in enumerate(bands2):
        for j,tr in enumerate(trefs):
            s=f2[(f2.banda==b)&(np.isclose(f2.T_ref,tr))]
            g={m:s[s.metodo==m].RMSD_D0.mean() for m in main if (s.metodo==m).any()}
            if g: Wc[i,j]=[m for m in main].index(min(g,key=g.get))
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap(["#1f5fd0","#2ca02c","#d62728"])
    fig,ax=plt.subplots(figsize=(6,7),dpi=170)
    im=ax.imshow(Wc,cmap=cmap,vmin=0,vmax=2,aspect="auto")
    ax.set_xticks(range(len(trefs))); ax.set_xticklabels([f"{int(t)}" for t in trefs]); ax.set_xlabel("T_ref (°C)")
    ax.set_yticks(range(len(bands2))); ax.set_yticklabels(bands2); ax.set_ylabel("Banda (kHz)")
    ax.set_title("Método vencedor por (banda × T_ref) — screening")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f5fd0",label="AE"),Patch(color="#2ca02c",label="Park"),Patch(color="#d62728",label="RF")],
              loc="upper center",bbox_to_anchor=(0.5,-0.08),ncol=3,frameon=False)
    fig.tight_layout(); savef(fig,"figH_vencedor_banda_tref")

# ===== 3) FAIXA MAIS ANALISÁVEL PARA O DANO (classificação por banda) =====
pb=Rd("checkpoints/parteB.csv")
if pb is not None:
    real=pb[pb.controle=="real"]
    bands3=[b for b in BORDER if b in real.banda.unique()]
    rows=[]
    for b in bands3:
        s=real[real.banda==b]
        rows.append({"banda":b,"bin_bal_acc":s[s.task=="bin"].bal_acc.mean(),"multi_bal_acc":s[s.task=="multi"].bal_acc.mean(),
                     "falso_saud":s[s.task=="bin"].taxa_falso_saudavel.mean()})
    dd=pd.DataFrame(rows).set_index("banda")
    fig,ax=plt.subplots(figsize=(11,4.6),dpi=170); x=np.arange(len(dd)); w=.38
    ax.bar(x-w/2,dd.bin_bal_acc,w,color="#1f5fd0",edgecolor="k",label="Binária")
    ax.bar(x+w/2,dd.multi_bal_acc,w,color="#d62728",edgecolor="k",label="Multiclasse")
    ax.set_xticks(x); ax.set_xticklabels(dd.index,rotation=20); ax.set_ylabel("Acurácia balanceada"); ax.set_ylim(0,1.12)
    ax.axhline(dd.bin_bal_acc.max(),ls=":",color="#1f5fd0",alpha=.5); ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(0.5,1.0))
    ax.set_title("Detectabilidade do dano por banda — qual faixa é mais analisável",pad=8)
    fig.tight_layout(); savef(fig,"figH_dano_por_banda")
    best_bin=dd.bin_bal_acc.idxmax(); best_multi=dd.multi_bal_acc.idxmax(); best_fs=dd.falso_saud.idxmin()
    log(f"\n=== FAIXA MAIS ANALISÁVEL PARA O DANO ===")
    log(f"  melhor detecção binária: {best_bin} ({dd.bin_bal_acc.max():.3f})")
    log(f"  melhor multiclasse: {best_multi} ({dd.multi_bal_acc.max():.3f})")
    log(f"  menor falso-saudável: {best_fs} ({dd.falso_saud.min():.3f})")
    dd.round(4).to_csv(os.path.join(OUT,"dano_por_banda.csv"))

# ===== 4) T_ref completo: heatmap T_ref × método + distância térmica =====
tf=Rd("06_sensibilidade_referencia/tref_full.csv")
if tf is not None:
    fig,axes=plt.subplots(1,2,figsize=(14,5),dpi=170)
    for ax,banda in zip(axes,sorted(tf.banda.unique())):
        s=tf[tf.banda==banda]; pv=s.pivot_table(index="T_ref",columns="metodo",values="RMSD_D0",aggfunc="mean")[["AE","Park","RF_direct"]]
        pv.columns=["AE","Park","RF"]
        im=heat(ax,pv,"viridis_r",float(np.nanmin(pv.values)),float(np.nanpercentile(pv.values,95)))
        ax.set_title(f"{banda} kHz — RMSD saudável por T_ref"); ax.set_ylabel("T_ref (°C)")
    fig.colorbar(im,ax=axes,fraction=.02,pad=.02)
    fig.suptitle("Temperatura de referência × método",y=1.0,fontsize=12); savef(fig,"figH_tref_metodo")
    # degradação com distância
    fig,ax=plt.subplots(figsize=(8,4.6),dpi=170)
    for m,c in [("AE","#1f5fd0"),("Park","#2ca02c"),("RF_direct","#d62728")]:
        s=tf[tf.metodo==m]; g=s.groupby(pd.cut(s.dist,[-1,10,20,30,50,100]),observed=True).RMSD_D0.mean()
        ax.plot(range(len(g)),g.values,marker="o",lw=2,color=c,label=LBL[m])
    ax.set_xticks(range(5)); ax.set_xticklabels(["≤10","10-20","20-30","30-50",">50"]); ax.set_xlabel("|T - T_ref| (°C)")
    ax.set_ylabel("RMSD saudável"); ax.legend(frameon=False); ax.set_title("Degradação com a distância térmica à referência")
    fig.tight_layout(); savef(fig,"figH_distancia_tref")
    log("\n=== MELHOR T_ref (RMSD saudável) ===")
    for banda in sorted(tf.banda.unique()):
        s=tf[tf.banda==banda]
        for m in ["AE","Park","RF_direct"]:
            g=s[s.metodo==m].groupby("T_ref").RMSD_D0.mean()
            log(f"  {banda} {LBL[m]:5s}: melhor T_ref={g.idxmin():.0f} ({g.min():.2f}) | pior={g.idxmax():.0f} ({g.max():.2f})")

open(os.path.join(OUT,"analise_banda_tref.txt"),"w",encoding="utf-8").write("\n".join(report))
print("\n✅ análise de banda e T_ref concluída")

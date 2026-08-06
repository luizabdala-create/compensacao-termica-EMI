# -*- coding: utf-8 -*-
"""
GRÁFICOS-CHAVE pedidos pelo revisor (usam dados já computados; sem recomputar ML):
 (1) figX_pareto           — Pareto compensação (RMSD saudável) × preservação de dano (healthy_sep).
 (2) figX_heat_banda_temp  — heatmap banda × temperatura de teste (RMSD) por método + diferenças.
 (3) figX_dist_fold        — distribuição por fold (violino + linhas pareadas por temperatura).
 (4) figX_matriz_seletor   — método escolhido pela CV interna em cada (banda × temperatura).
 (5) figX_superficie_dz    — superfície da correção térmica ΔZ(f,T) observada (estado saudável).
"""
import os,sys,json,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","legend.fontsize":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
CM={"Park":"#2ca02c","RF_direct":"#d62728","AE":"#1f5fd0","ExtraTrees":"#ff7f0e","RF_temponly":"#8c564b"}
LB={"Park":"Park","RF_direct":"Random Forest","AE":"Autoencoder","ExtraTrees":"RF otimizado","RF_temponly":"RF (só temp.)"}
def save(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n,flush=True)
f8=pd.read_csv(os.path.join(ROOT,"02_compensacao","fase8_tuning_ampliado.csv")); f8=f8[f8.metodo!="TAU_T"]
MJ=["AE","Park","RF_direct","ExtraTrees"]

# ===== (1) PARETO: RMSD saudável × healthy_sep (preservação) =====
g=f8.groupby("metodo").agg(RMSD=("RMSD_D0","mean"),hsep=("healthy_sep","mean"),sepD1=("sep_RMSD_D1","mean")).reindex(MJ+["RF_temponly"])
fig,ax=plt.subplots(figsize=(8,5.2),dpi=170)
for m in g.index:
    if m not in CM: continue
    ax.scatter(g.loc[m,"RMSD"],g.loc[m,"hsep"],s=220,color=CM[m],edgecolor="k",zorder=5)
    ax.annotate(LB[m],(g.loc[m,"RMSD"],g.loc[m,"hsep"]),xytext=(8,6),textcoords="offset points",fontsize=10)
ax.set_xlabel("RMSD saudável (← melhor compensação)"); ax.set_ylabel("healthy_sep (preservação de dano →)")
ax.annotate("ideal",(g.RMSD.min()*0.97,min(1.0,g.hsep.max()*1.005)),fontsize=11,style="italic",color="#2a7")
ax.set_title("Pareto: compensação térmica × preservação de dano\n(canto inferior-direito ideal: baixo erro e alta preservação)")
ax.grid(alpha=.25); fig.tight_layout(); save(fig,"figX_pareto")

# ===== (2) HEATMAP banda × temperatura (RMSD) por método + diferenças =====
bands=[b for b in ["30-40","40-50","50-60","60-70","70-80","80-90","90-100"] if b in f8.banda.unique()]
temps=sorted(f8.T_test.unique())
def mat(m): return f8[f8.metodo==m].pivot_table(index="banda",columns="T_test",values="RMSD_D0").reindex(bands)[temps]
fig,axes=plt.subplots(2,3,figsize=(16,7),dpi=165)
for ax,m in zip(axes[0],MJ):
    M=mat(m); im=ax.imshow(M.values,cmap="viridis_r",aspect="auto",vmin=0,vmax=6)
    ax.set_title(LB[m]); ax.set_xticks(range(len(temps))); ax.set_xticklabels([int(t) for t in temps],fontsize=7,rotation=90)
    ax.set_yticks(range(len(bands))); ax.set_yticklabels(bands,fontsize=8)
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
axes[0,0].set_ylabel("Banda (kHz)")
for ax,(a,b,tt) in zip(axes[1],[("AE","Park","AE − Park"),("RF_direct","Park","RF − Park"),("AE","RF_direct","AE − RF")]):
    D=mat(a).values-mat(b).values; vlim=np.nanmax(np.abs(D))
    im=ax.imshow(D,cmap="coolwarm",aspect="auto",vmin=-vlim,vmax=vlim)
    ax.set_title(tt+"  (azul: 1º melhor)"); ax.set_xticks(range(len(temps))); ax.set_xticklabels([int(t) for t in temps],fontsize=7,rotation=90)
    ax.set_yticks(range(len(bands))); ax.set_yticklabels(bands,fontsize=8); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
axes[1,0].set_ylabel("Banda (kHz)")
fig.suptitle("RMSD saudável por banda × temperatura de teste (linha 1) e mapas de diferença entre métodos (linha 2)",y=1.01,fontsize=13)
fig.text(0.5,-0.01,"Temperatura de teste (°C)",ha="center",fontsize=11); fig.tight_layout(); save(fig,"figX_heat_banda_temp")

# ===== (3) DISTRIBUIÇÃO POR FOLD (violino + pareado) — banda 70-80 =====
b="70-80" if "70-80" in f8.banda.unique() else bands[-1]
sub=f8[f8.banda==b]; piv=sub.pivot_table(index="T_test",columns="metodo",values="RMSD_D0")[MJ].dropna()
fig,ax=plt.subplots(figsize=(8.5,5),dpi=170)
parts=ax.violinplot([piv[m].values for m in MJ],showmeans=True,showextrema=False)
for pc,m in zip(parts['bodies'],MJ): pc.set_facecolor(CM[m]); pc.set_alpha(.35)
for _,row in piv.iterrows():
    ax.plot(range(1,len(MJ)+1),[row[m] for m in MJ],color="#999",lw=.7,alpha=.6,marker="o",ms=3)
ax.set_xticks(range(1,len(MJ)+1)); ax.set_xticklabels([LB[m] for m in MJ])
ax.set_ylabel("RMSD saudável (por temperatura)"); ax.set_title(f"Distribuição por fold em {b} kHz — linhas cinza conectam a mesma temperatura\n(médias em barras escondem a variabilidade e a dependência entre condições)")
ax.grid(alpha=.25,axis="y"); fig.tight_layout(); save(fig,"figX_dist_fold")

# ===== (4) MATRIZ DO SELETOR: método escolhido pela CV interna por (banda × temperatura) =====
sel=f8.pivot_table(index="banda",columns="T_test",values="cv_inner",aggfunc="mean")  # placeholder p/ shape
choice=np.full((len(bands),len(temps)),-1)
cmapc={"AE":0,"Park":1,"RF_direct":2}
for i,bb in enumerate(bands):
    for j,tt in enumerate(temps):
        cv={m:f8[(f8.banda==bb)&(np.isclose(f8.T_test,tt))&(f8.metodo==m)].cv_inner.mean() for m in ["AE","Park","RF_direct"]}
        cv={m:v for m,v in cv.items() if v==v}
        if cv: choice[i,j]=cmapc[min(cv,key=cv.get)]
from matplotlib.colors import ListedColormap
cols=ListedColormap(["#1f5fd0","#2ca02c","#d62728"]); Cm=np.ma.masked_where(choice<0,choice)
fig,ax=plt.subplots(figsize=(11,4),dpi=170)
ax.imshow(Cm,cmap=cols,aspect="auto",vmin=0,vmax=2)
ax.set_xticks(range(len(temps))); ax.set_xticklabels([int(t) for t in temps],fontsize=7,rotation=90)
ax.set_yticks(range(len(bands))); ax.set_yticklabels(bands,fontsize=8)
ax.set_xlabel("Temperatura de teste (°C)"); ax.set_ylabel("Banda (kHz)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#1f5fd0",label="Autoencoder"),Patch(color="#2ca02c",label="Park"),Patch(color="#d62728",label="Random Forest")],
          loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=3,frameon=False)
ax.set_title("Método escolhido pela CV interna em cada (banda × temperatura) — estabilidade do seletor",pad=8)
fig.tight_layout(); save(fig,"figX_matriz_seletor")

# ===== (5) SUPERFÍCIE ΔZ(f,T) observada (estado saudável) =====
lo,hi=70,80; fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//600); fc,f=P.band(lo,hi,decim=dec)
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); yv=df["falha"].to_numpy(int)
X=df[fc].to_numpy(np.float64); ref=np.median(X[np.isclose(T,30.0)&(yv==0)],axis=0)
htemps=sorted([t for t in np.unique(T[yv==0])])
DZ=np.array([ref-X[np.isclose(T,t)&(yv==0)].mean(0) for t in htemps])  # (n_temp, n_freq)
fig,ax=plt.subplots(figsize=(10,4.8),dpi=170)
im=ax.pcolormesh(f/1e3,htemps,DZ,cmap="RdBu_r",shading="auto",vmin=-np.nanpercentile(np.abs(DZ),98),vmax=np.nanpercentile(np.abs(DZ),98))
ax.set_xlabel("Frequência (kHz)"); ax.set_ylabel("Temperatura (°C)")
ax.set_title("Superfície da correção térmica observada ΔZ(f,T) = z_ref − z_saudável(T)  —  70–80 kHz\n(suave e estruturada: por isso a interpolação é competitiva)")
fig.colorbar(im,ax=ax,label="ΔZ (correção térmica)"); fig.tight_layout(); save(fig,"figX_superficie_dz")
print("✅ gráficos do revisor concluídos",flush=True)

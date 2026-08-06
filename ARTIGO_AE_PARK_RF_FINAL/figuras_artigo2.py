# -*- coding: utf-8 -*-
"""
FIGURAS DO ARTIGO — parte 2: curvas, comparação justa, classificação, confusão, ablation.
Usa hiperparâmetros TUNADOS. PNG 600 dpi + PDF.
"""
import os,sys,json,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,"axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold",
 "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9.5,"pdf.fonttype":42,"ps.fonttype":42})
COR={"Original":"#7f7f7f","Park":"#2ca02c","RF_direct":"#d62728","RF_temponly":"#9467bd",
     "AE":"#1f5fd0","TAU_T":"#7b2fbe"}
LBL={"Original":"Original","Park":"Park","RF_direct":"RF (direto)","RF_temponly":"RF (só temp.)",
     "AE":"Autoencoder","TAU_T":r"$\tau(T)$ (aux.)"}
def save(fig,n):
    for e,d in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=d,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n)
def clean(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=.25,lw=.6)

# configs TUNADAS (após re-tuning expandido; seleção por CV interna, sem tocar no teste)
TUNED=json.loads('{"30-40":{"park":{"max_shift_frac":0.05,"nsteps":121,"smooth_win":9},"rf":{"n_estimators":600,"max_depth":16,"min_samples_leaf":1,"min_samples_split":4,"max_features":"sqrt","smooth_win":5,"input_decim":4},"ae":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":0.001,"dropout":0.1,"noise":0.01,"epochs":550,"patience":75,"lambda_d1":0.3}},"70-80":{"park":{"max_shift_frac":0.15,"nsteps":121,"smooth_win":9},"rf":{"n_estimators":300,"max_depth":10,"min_samples_leaf":1,"min_samples_split":4,"max_features":"sqrt","smooth_win":5,"input_decim":4},"ae":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":384,"lr":0.001,"dropout":0.1,"noise":0.01,"epochs":550,"patience":75,"lambda_d1":0.3}}}')
TAU={"rank":8,"max_shift_frac":0.14,"nsteps":81}
df,_,_=P.load_base()
T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int); T_REF=30.0

# ===== FIG 2: curvas Original/ref/Park/RF/AE por classe (LOTO: temp fora do treino) =====
def fig_curvas(banda,T_show,fname):
    lo,hi=map(int,banda.split("-"))
    fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF); tp=TUNED[banda]
    te=np.isclose(T,T_show); trh=(~te)&(y==0)
    comps={"Park":P.comp_park(X,ref,f,tp["park"])[0],
           "RF_direct":P.comp_rf(X,T,ref,trh,tp["rf"],"direct")[0],
           "AE":P.comp_ae(X,T,ref,trh,tp["ae"],T_REF,seed=42)[0]}
    fk=f/1e3; NM={0:"D0 (healthy)",1:"D1",2:"D2"}
    fig,axes=plt.subplots(2,3,figsize=(15,8),dpi=170,sharex=True)
    lo_y=min(min(c.min() for c in comps.values()),ref.min())-3
    hi_y=max(max(c.max() for c in comps.values()),ref.max())+3
    for j,c in enumerate([0,1,2]):
        ii=np.where(te&(y==c))[0]; ax,ax2=axes[0,j],axes[1,j]
        if len(ii)==0: ax.set_title(f"{NM[c]} — n/a"); continue
        k=ii[0]
        ax.plot(fk,ref,"--",color="k",lw=1.5,label="Referência (saudável 30°C)",zorder=6)
        ax.plot(fk,X[k],color=COR["Original"],lw=1,alpha=.6,label=f"Original {T_show:.0f}°C")
        for m in ["Park","RF_direct","AE"]:
            ax.plot(fk,comps[m][k],lw=1.5,color=COR[m],alpha=.9,label=LBL[m])
        ax.set_title(NM[c]); ax.set_ylim(lo_y,hi_y); clean(ax)
        ax2.axhline(0,color="k",ls="--",lw=1)
        for m in ["Park","RF_direct","AE"]:
            ax2.plot(fk,comps[m][k]-ref,lw=1.2,color=COR[m],alpha=.85)
        ax2.set_xlabel("Frequência (kHz)"); ax2.set_ylim(-13,13); clean(ax2)
    axes[0,0].set_ylabel("Impedância compensada"); axes[1,0].set_ylabel("Resíduo (comp − ref)")
    axes[0,0].legend(frameon=False,fontsize=8.5,loc="upper right")
    fig.suptitle(f"Curvas compensadas — {banda} kHz — T = {T_show:.0f}°C (fora da amostra)",y=.99)
    fig.tight_layout(); save(fig,fname)
fig_curvas("70-80",-10.0,"fig02a_curvas_70-80_Tm10")
fig_curvas("70-80",60.0,"fig02b_curvas_70-80_T60")
fig_curvas("30-40",60.0,"fig02c_curvas_30-40_T60")

# ===== FIG fair: RMSD_D0 por banda (todos tunados) — fonte única: fase8 (params re-tunados) =====
_fp=os.path.join(ROOT,"02_compensacao","fase8_tuning_ampliado.csv")
fj=pd.read_csv(_fp if os.path.exists(_fp) else os.path.join(ROOT,"02_compensacao","comparacao_justa_todos_tunados.csv"))
fj=fj[fj.metodo!="TAU_T"]
piv=fj.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean")
piv=piv.reindex([b for b in ["30-40","40-50","60-70","70-80"] if b in piv.index])
MJ=["AE","Park","RF_direct"]
fig,ax=plt.subplots(figsize=(8,4.6),dpi=170); x=np.arange(len(piv)); w=.2
for i,m in enumerate(MJ):
    ax.bar(x+(i-1.5)*w,piv[m],w,color=COR[m],edgecolor="k",lw=.7,label=LBL[m])
    for xi,v in zip(x+(i-1.5)*w,piv[m]): ax.text(xi,v+.05,f"{v:.2f}",ha="center",fontsize=8,rotation=90)
ax.set_xticks(x); ax.set_xticklabels(piv.index); ax.set_ylabel("RMSD nas curvas saudáveis")
ax.set_ylim(0,np.nanmax(piv.values)*1.32)  # folga para os rótulos girados
ax.set_title("Comparação justa (todos ajustados por CV interna) — RMSD por banda",pad=8)
ax.legend(frameon=False,ncol=4,fontsize=9,loc="upper center",bbox_to_anchor=(0.5,-0.12)); clean(ax)
fig.tight_layout(); save(fig,"fig03b_comparacao_justa_RMSD_banda")

# ===== Parte B =====
pb=pd.read_csv(os.path.join(ROOT,"checkpoints","parteB.csv"))
pb=pb[pb.metodo!="TAU_T"]; real=pb[pb.controle=="real"]; MB=["Original","Park","RF_direct","RF_temponly","AE"]

# FIG 12: binário
b=real[real.task=="bin"].groupby("metodo")[["bal_acc","recall_dano","taxa_falso_saudavel"]].mean().reindex(MB)
fig,axes=plt.subplots(1,3,figsize=(14,4.2),dpi=170)
for ax,col,tit,best in zip(axes,["bal_acc","recall_dano","taxa_falso_saudavel"],
    ["Acurácia balanceada ↑","Recall de dano ↑","Taxa de falso-saudável ↓"],["max","max","min"]):
    vals=b[col].values; bars=ax.bar(range(len(MB)),vals,color=[COR[m] for m in MB],edgecolor="k",lw=.7)
    idx=int(np.argmax(vals)) if best=="max" else int(np.argmin(vals))
    bars[idx].set_edgecolor("gold"); bars[idx].set_linewidth(2.5)
    for i,v in enumerate(vals): ax.text(i,v+max(vals)*.02,f"{v:.3f}",ha="center",fontsize=8)
    ax.set_xticks(range(len(MB))); ax.set_xticklabels([LBL[m] for m in MB],rotation=30,ha="right")
    ax.set_ylim(0,max(vals)*1.16); ax.set_title(tit); clean(ax)
fig.suptitle("Detecção binária de dano (saudável vs. com dano) — fora da amostra",y=1.0)
fig.tight_layout(); save(fig,"fig12_classificacao_binaria")

# FIG 14: multiclasse
m=real[real.task=="multi"].groupby("metodo")[["bal_acc","macro_f1","f1_D0","f1_D1","f1_D2"]].mean().reindex(MB)
fig,axes=plt.subplots(1,2,figsize=(13,4.6),dpi=170)
axes[0].bar(range(len(MB)),m["macro_f1"],color=[COR[x] for x in MB],edgecolor="k",lw=.7)
for i,v in enumerate(m["macro_f1"]): axes[0].text(i,v+.008,f"{v:.3f}",ha="center",fontsize=8)
axes[0].set_xticks(range(len(MB))); axes[0].set_xticklabels([LBL[x] for x in MB],rotation=30,ha="right")
axes[0].set_ylim(0,float(m["macro_f1"].max())*1.16); axes[0].set_title("Macro-F1 (D0/D1/D2) ↑"); clean(axes[0])
CD={"f1_D0":"#1f77b4","f1_D1":"#ff7f0e","f1_D2":"#d62728"}; w=.25
for i,c in enumerate(["f1_D0","f1_D1","f1_D2"]):
    axes[1].bar(np.arange(len(MB))+(i-1)*w,m[c],w,color=CD[c],edgecolor="k",lw=.6,label=c.replace("f1_",""))
axes[1].set_xticks(range(len(MB))); axes[1].set_xticklabels([LBL[x] for x in MB],rotation=30,ha="right")
axes[1].set_ylim(0,float(m[["f1_D0","f1_D1","f1_D2"]].values.max())*1.18)
axes[1].set_title("F1 por classe"); axes[1].legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.28)); clean(axes[1])
fig.suptitle("Reconhecimento multiclasse de dano — fora da amostra",y=1.0)
fig.tight_layout(); save(fig,"fig14_classificacao_multiclasse")

# FIG 15: matrizes de confusão multiclasse (soma dos folds), FS1+logreg p/ comparabilidade
def sum_conf(sub):
    M=np.zeros((3,3),int)
    for s in sub["conf"].dropna(): M+=np.array(json.loads(s))
    return M
sel=real[(real.task=="multi")&(real.feature_set=="FS1")&(real.clf=="logreg")]
fig,axes=plt.subplots(2,3,figsize=(13,8.2),dpi=170)
for ax,mth in zip(axes.ravel(),MB):
    M=sum_conf(sel[sel.metodo==mth]); Mn=M/M.sum(1,keepdims=True).clip(min=1)
    im=ax.imshow(Mn,cmap="Blues",vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{M[i,j]}\n{Mn[i,j]:.2f}",ha="center",va="center",
                    fontsize=10,color="white" if Mn[i,j]>.5 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(["D0","D1","D2"]); ax.set_yticks(range(3)); ax.set_yticklabels(["D0","D1","D2"])
    ax.set_title(LBL[mth]); ax.set_xlabel("Predito"); ax.set_ylabel("Real")
fig.suptitle("Matrizes de confusão multiclasse (FS1 + regressão logística, somadas sobre folds)",y=1.0)
fig.tight_layout(); save(fig,"fig15_confusao_multiclasse")

# FIG 13: confusão binária
def sum_conf_bin(sub):
    tn=fp=fn=tp=0
    for _,r in sub.iterrows():
        fn+=r["n_falso_saudavel"]; tp+=r["n_dano_real"]-r["n_falso_saudavel"]
    return fn,tp
selb=real[(real.task=="bin")&(real.feature_set=="FS1")&(real.clf=="logreg")]
fig,axes=plt.subplots(2,3,figsize=(13,8.2),dpi=170)
for ax,mth in zip(axes.ravel(),MB):
    s=selb[selb.metodo==mth]
    fn=int(s["n_falso_saudavel"].sum()); tp=int((s["n_dano_real"]-s["n_falso_saudavel"]).sum())
    ndr=int(s["n_dano_real"].sum())
    ax.text(.5,.62,f"Dano detectado corretamente (VP): {tp}",ha="center",fontsize=11,transform=ax.transAxes)
    ax.text(.5,.42,f"Falso-saudável (FN): {fn}",ha="center",fontsize=11,color="crimson",transform=ax.transAxes)
    ax.text(.5,.22,f"Recall = {tp/max(ndr,1):.3f}",ha="center",fontsize=11,transform=ax.transAxes)
    ax.set_title(LBL[mth]); ax.axis("off")
fig.suptitle("Detecção binária: contagem de falso-saudável (erro crítico em SHM) — somada sobre folds",y=1.0)
fig.tight_layout(); save(fig,"fig13_confusao_binaria")

# FIG 19: ablation (a) RF direct vs temponly sep_D1  (b) efeito do shift
fig,axes=plt.subplots(1,2,figsize=(12,4.6),dpi=170)
sub=fj[fj.metodo.isin(["RF_direct","RF_temponly"])]
g=sub.groupby("metodo")[["sep_RMSD_D1","RMSD_D0","healthy_sep"]].mean()
x=np.arange(2); w=.35
axes[0].bar(x-w/2,g["sep_RMSD_D1"],w,color="#8888ff",edgecolor="k",label="Separação D1−D0")
axes[0].bar(x+w/2,g["RMSD_D0"],w,color="#ff8888",edgecolor="k",label="RMSD saudável")
axes[0].set_xticks(x); axes[0].set_xticklabels(["RF direto\n(vê a curva)","RF só-temp\n(vê só T)"])
axes[0].set_ylim(0,float(g[["sep_RMSD_D1","RMSD_D0"]].values.max())*1.28)
axes[0].set_title("Ablação: o compensador reage ao dano?"); axes[0].legend(frameon=False,loc="upper right",ncol=1); clean(axes[0])
# (b) healthy_sep dos métodos principais (tunado)
sh=fj[fj.metodo.isin(["Original","Park","RF_direct","AE"])].groupby("metodo")["healthy_sep"].mean().reindex(["Original","Park","RF_direct","AE"])
labs=["Original","Park","RF","AE"]; cols=["#9e9e9e","#2ca02c","#d62728","#1f5fd0"]
axes[1].bar(range(len(sh)),sh.values,color=cols,edgecolor="k")
for i,v in enumerate(sh.values): axes[1].text(i,v+.02,f"{v:.2f}",ha="center",fontsize=9)
axes[1].set_xticks(range(len(sh))); axes[1].set_xticklabels(labs); axes[1].set_ylim(0,1.15)
axes[1].set_ylabel("D0 < min(D1,D2)"); axes[1].set_title("healthy_sep por método"); clean(axes[1])
fig.tight_layout(); save(fig,"fig19_ablation")
print("\n✅ figuras parte 2 concluídas")

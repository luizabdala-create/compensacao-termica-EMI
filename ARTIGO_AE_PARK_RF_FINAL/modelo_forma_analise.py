# -*- coding: utf-8 -*-
"""
ANALISE do MODELO FORMA — figuras + veredito honesto (dois eixos), bilingue.
Le 13_modelo_forma/metricas/*.csv e curvas/*.npz (funciona em resultado parcial).
ART_LANG=pt (default) -> graficos/ ; ART_LANG=en -> graficos_en/.

Produz: fig_A_shape_por_banda, fig_B_class_{multi,bin}_por_banda, fig_tradeoff,
        fig_C_guarda_dano, fig_curvas_* , modelo_forma_veredito.csv + resumo impresso.
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; OUT=os.path.join(ROOT,"13_modelo_forma")
LANG=os.environ.get("ART_LANG","pt").lower()
FIG=os.path.join(OUT,"graficos" if LANG=="pt" else "graficos_en"); MET=os.path.join(OUT,"metricas"); CUR=os.path.join(OUT,"curvas")
os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,
 "axes.titlesize":13,"axes.titleweight":"bold","axes.labelsize":12,"legend.fontsize":9,
 "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})

ORDER=["Original","Park","RF_amp","RF_forma","AE_amp","AE_forma"]
COL={"Original":"#7f7f7f","Park":"#2ca02c","RF_amp":"#d62728","RF_forma":"#ff7f0e",
     "AE_amp":"#1f5fd0","AE_forma":"#9467bd"}
# rotulos por idioma
if LANG=="en":
    LBL={"Original":"Original","Park":"Park","RF_amp":"Optimized RF (amplitude)","RF_forma":"Optimized RF (shape)",
         "AE_amp":"Autoencoder (amplitude)","AE_forma":"Autoencoder (shape)"}
    TX=dict(
      A_ccdm="CCDM on healthy (lower = better)", A_peak="Peak error |Hz| (lower = better)",
      A_a="(a) Shape of compensated healthy curve vs. reference", A_b="(b) Main-resonance alignment",
      A_sup="(A) THERMAL REMOVAL — shape on the healthy test curves (LOTO)",
      B_y="Balanced accuracy (higher = better)", chance="chance",
      B_title=lambda t:f"(B) DAMAGE PRESERVATION — {'multiclass 0/1/2' if t=='multi' else 'binary'} classification after compensation (Original = before)",
      TR_x="CCDM on healthy (shape; LOWER = better, to the left)", TR_y="Balanced damage accuracy (HIGHER = better)",
      TR_t="Shape vs. damage-preservation trade-off\n(ideal = upper-left corner; arrow = gain from the shape objective)",
      C_y="CCDM of the DAMAGED curve vs. healthy reference",
      C_t="(C) DAMAGE GUARD — damaged curves must not collapse onto the reference (CCDM->0 = damage erased)",
      dano=lambda d:f"Damage {d}", ref="Reference (healthy @Tref)", refs="Healthy ref.",
      healthy=lambda tt,b:f"Healthy @ T={tt:.0f} C — {b} kHz",
      dmg=lambda lab,tt:f"{lab} @ T={tt:.0f} C — the damage peak must REMAIN",
      band="Band (kHz)", freq="Frequency (kHz)", z="|Z| (compensated)",
      sup=lambda b,tt:f"Example curves — compensated shape ({b} kHz, T={tt:.0f} C)")
    danolbl={1:"Damage 1",2:"Damage 2"}
else:
    LBL={"Original":"Original","Park":"Park","RF_amp":"RF otimizado (amplitude)","RF_forma":"RF otimizado (forma)",
         "AE_amp":"Autoencoder (amplitude)","AE_forma":"Autoencoder (forma)"}
    TX=dict(
      A_ccdm="CCDM saudavel (menor = melhor)", A_peak="Erro de pico |Hz| (menor = melhor)",
      A_a="(a) Forma da curva saudavel compensada vs. referencia", A_b="(b) Alinhamento da ressonancia principal",
      A_sup="(A) REMOCAO TERMICA — formato no SAUDAVEL de teste (LOTO)",
      B_y="Acuracia balanceada (maior = melhor)", chance="acaso",
      B_title=lambda t:f"(B) PRESERVACAO DO DANO — classificacao {'multiclasse 0/1/2' if t=='multi' else 'binaria'} apos compensacao (Original = antes)",
      TR_x="CCDM no saudavel (forma; MENOR = melhor, a esquerda)", TR_y="Acuracia balanceada do dano (MAIOR = melhor)",
      TR_t="Trade-off FORMA x PRESERVACAO DO DANO\n(ideal = canto superior esquerdo; seta = ganho do objetivo de forma)",
      C_y="CCDM da curva de DANO vs. referencia saudavel",
      C_t="(C) GUARDA DE DANO — curvas de dano NAO podem colapsar para a referencia (CCDM->0 = dano apagado)",
      dano=lambda d:f"Dano {d}", ref="Referencia (saudavel @Tref)", refs="Ref. saudavel",
      healthy=lambda tt,b:f"Saudavel @ T={tt:.0f}C — {b} kHz",
      dmg=lambda lab,tt:f"{lab} @ T={tt:.0f}C — o pico do dano deve PERMANECER",
      band="Banda (kHz)", freq="Frequencia (kHz)", z="|Z| (compensada)",
      sup=lambda b,tt:f"Curvas de exemplo — formato compensado ({b} kHz, T={tt:.0f}C)")
    danolbl={1:"Dano 1",2:"Dano 2"}

def order(ms): return [m for m in ORDER if m in ms]
def rd(p): return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
def savef(fig,n):
    for e,d in [("png",600),("pdf",None)]:
        fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=d,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n)

A=rd(os.path.join(MET,"shapeA_saudavel.csv"))
B=rd(os.path.join(MET,"classB_dano.csv"))
C=rd(os.path.join(MET,"guardaC_dano.csv"))
if not len(A): print("sem dados ainda em shapeA_saudavel.csv"); sys.exit(0)
BANDS=[b for b in ["30-40","40-50","50-60","60-70","70-80","30-70"] if b in A.banda.unique()]

# =================== (A) forma no saudavel por banda ===================
gA=A.groupby(["banda","metodo"])[["CCDM","CORR","peak_hz","RMSD","SAM_deg"]].mean().reset_index()
ms=order(A.metodo.unique())
fig,axes=plt.subplots(1,2,figsize=(14,4.8),dpi=170)
x=np.arange(len(BANDS)); w=0.8/max(1,len(ms))
for i,m in enumerate(ms):
    v=[gA[(gA.banda==b)&(gA.metodo==m)].CCDM.mean() for b in BANDS]
    axes[0].bar(x+(i-len(ms)/2+.5)*w,v,w,color=COL[m],edgecolor="k",lw=.4,label=LBL[m])
axes[0].set_xticks(x); axes[0].set_xticklabels(BANDS); axes[0].set_ylabel(TX["A_ccdm"]); axes[0].set_title(TX["A_a"])
for i,m in enumerate(ms):
    v=[gA[(gA.banda==b)&(gA.metodo==m)].peak_hz.mean() for b in BANDS]
    axes[1].bar(x+(i-len(ms)/2+.5)*w,v,w,color=COL[m],edgecolor="k",lw=.4,label=LBL[m])
axes[1].set_xticks(x); axes[1].set_xticklabels(BANDS); axes[1].set_ylabel(TX["A_peak"]); axes[1].set_title(TX["A_b"]); axes[1].set_yscale("log")
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",bbox_to_anchor=(0.5,1.10),ncol=3,frameon=False)
fig.suptitle(TX["A_sup"],y=1.16,fontsize=13.5)
fig.tight_layout(); savef(fig,"fig_A_shape_por_banda")

# =================== (B) classificacao por banda ===================
if len(B):
    real=B[B.controle=="real"]
    for task in [t for t in ["multi","bin"] if t in real.task.unique()]:
        s=real[real.task==task]
        gB=s.groupby(["banda","metodo"])["bal_acc"].mean().reset_index()
        fig,ax=plt.subplots(figsize=(11.5,4.8),dpi=170)
        for i,m in enumerate(order(s.metodo.unique())):
            v=[gB[(gB.banda==b)&(gB.metodo==m)].bal_acc.mean() for b in BANDS]
            ax.bar(x+(i-len(ms)/2+.5)*w,v,w,color=COL[m],edgecolor="k",lw=.4,label=LBL[m])
        ax.set_xticks(x); ax.set_xticklabels(BANDS); ax.set_ylabel(TX["B_y"])
        ax.axhline(1/3 if task=="multi" else .5,ls=":",color="k",alpha=.5,label=TX["chance"]); ax.set_ylim(0,1.05)
        ax.set_title(TX["B_title"](task))
        ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.12))
        fig.tight_layout(); savef(fig,f"fig_B_class_{task}_por_banda")

# =================== FIGURA-CHAVE: tradeoff ===================
if len(B):
    real=B[(B.controle=="real")&(B.task=="multi")] if "multi" in B.task.unique() else B[B.controle=="real"]
    ca={m:A[A.metodo==m].CCDM.mean() for m in ms}
    ba={m:real[real.metodo==m].bal_acc.mean() for m in order(real.metodo.unique())}
    fig,ax=plt.subplots(figsize=(8.6,6.6),dpi=170)
    plotted=[]
    for m in ms:
        if m in ba:
            ax.scatter(ca[m],ba[m],s=190,color=COL[m],edgecolor="k",zorder=3,label=LBL[m])
            coincide=any(abs(ca[m]-px)<1e-3 and abs(ba[m]-py)<1e-3 for px,py in plotted)
            if not coincide:
                lab=LBL[m]
                if m=="RF_amp" and "RF_forma" in ba and abs(ca["RF_amp"]-ca.get("RF_forma",1))<1e-3:
                    lab=("Optimized RF (ampl.=shape)" if LANG=="en" else "RF otimizado (ampl.=forma)")
                ax.annotate(lab.replace(" (","\n("),(ca[m],ba[m]),textcoords="offset points",xytext=(9,6),fontsize=8.5)
            plotted.append((ca[m],ba[m]))
    for a_m,f_m in [("AE_amp","AE_forma"),("RF_amp","RF_forma")]:
        if a_m in ba and f_m in ba and (abs(ca[a_m]-ca[f_m])>1e-3 or abs(ba[a_m]-ba[f_m])>1e-3):
            ax.annotate("",xy=(ca[f_m],ba[f_m]),xytext=(ca[a_m],ba[a_m]),arrowprops=dict(arrowstyle="->",color=COL[f_m],lw=2,alpha=.85))
    ax.set_xlabel(TX["TR_x"]); ax.set_ylabel(TX["TR_y"]); ax.set_title(TX["TR_t"])
    fig.tight_layout(); savef(fig,"fig_tradeoff")

# =================== (C) guarda de dano ===================
if len(C):
    gC=C.groupby(["metodo","dano"])[["CCDM","RMSD"]].mean().reset_index()
    fig,ax=plt.subplots(figsize=(10,4.6),dpi=170)
    danos=sorted(C.dano.unique()); x2=np.arange(len(danos)); w2=0.8/max(1,len(ms))
    for i,m in enumerate(order(C.metodo.unique())):
        v=[gC[(gC.metodo==m)&(gC.dano==d)].CCDM.mean() for d in danos]
        ax.bar(x2+(i-len(ms)/2+.5)*w2,v,w2,color=COL[m],edgecolor="k",lw=.4,label=LBL[m])
    ax.set_xticks(x2); ax.set_xticklabels([TX["dano"](int(d)) for d in danos]); ax.set_ylabel(TX["C_y"]); ax.set_title(TX["C_t"])
    ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.12))
    fig.tight_layout(); savef(fig,"fig_C_guarda_dano")

# =================== curvas de exemplo ===================
for fn in sorted(os.listdir(CUR)) if os.path.isdir(CUR) else []:
    if not fn.endswith(".npz"): continue
    z=np.load(os.path.join(CUR,fn),allow_pickle=True)
    f=z["f"]/1000.0; ref=z["ref"]; banda=str(z["banda"]); Tt=float(z["T_test"])
    show=[m for m in ["Original","Park","AE_amp","AE_forma","RF_forma"] if f"Y_{m}_healthy" in z.files]
    npan=1+sum(1 for k in ["d1","d2"] if f"Y_AE_forma_{k}" in z.files)
    fig,axes=plt.subplots(1,npan,figsize=(6.2*npan,4.6),dpi=170,squeeze=False); axes=axes[0]
    ax=axes[0]; ax.plot(f,ref,color="k",lw=2.2,label=TX["ref"],zorder=5)
    for m in show: ax.plot(f,z[f"Y_{m}_healthy"],color=COL[m],lw=1.1,alpha=.9,label=LBL[m])
    ax.set_title(TX["healthy"](Tt,banda)); ax.set_xlabel(TX["freq"]); ax.set_ylabel(TX["z"]); ax.legend(frameon=False,fontsize=8)
    pi=1
    for k,dnum in [("d1",1),("d2",2)]:
        if f"Y_AE_forma_{k}" not in z.files: continue
        ax=axes[pi]; pi+=1
        ax.plot(f,ref,color="k",lw=2.0,ls="--",label=TX["refs"],zorder=5)
        for m in [mm for mm in ["Original","AE_amp","AE_forma"] if f"Y_{mm}_{k}" in z.files]:
            ax.plot(f,z[f"Y_{m}_{k}"],color=COL[m],lw=1.3,alpha=.9,label=LBL[m])
        ax.set_title(TX["dmg"](danolbl[dnum],Tt)); ax.set_xlabel(TX["freq"]); ax.legend(frameon=False,fontsize=8)
    fig.suptitle(TX["sup"](banda,Tt),y=1.03,fontsize=12.5)
    fig.tight_layout(); savef(fig,f"fig_curvas_{banda}_T{int(round(Tt))}")

# =================== VEREDITO (CSV + print) — so no PT run ===================
if LANG=="pt":
    rows=[]
    for m in ms:
        a=A[A.metodo==m]
        r={"metodo":LBL[m],"CCDM_saud":round(a.CCDM.mean(),4),"CORR_saud":round(a.CORR.mean(),4),
           "peak_hz":round(a.peak_hz.mean(),1),"RMSD_saud":round(a.RMSD.mean(),3)}
        if len(B):
            rb=B[(B.controle=="real")&(B.metodo==m)]
            for tk in ["bin","multi"]:
                s=rb[rb.task==tk]
                if len(s): r[f"balacc_{tk}"]=round(s.bal_acc.mean(),4)
        if len(C):
            for d in sorted(C.dano.unique()):
                r[f"danoCCDM_D{int(d)}"]=round(C[(C.metodo==m)&(C.dano==d)].CCDM.mean(),4)
        rows.append(r)
    V=pd.DataFrame(rows); V.to_csv(os.path.join(MET,"modelo_forma_veredito.csv"),index=False)
    print("\n"+"="*70+"\nVEREDITO — MODELO FORMA (refino)\n"+"="*70); print(V.to_string(index=False))
    def cmp(a_m,f_m,tag):
        if a_m not in A.metodo.unique() or f_m not in A.metodo.unique(): return
        dccdm=A[A.metodo==f_m].CCDM.mean()-A[A.metodo==a_m].CCDM.mean()
        line=f"\n{tag}: forma vs amplitude -> dCCDM={dccdm:+.4f}"
        if len(B):
            for tk in ["multi","bin"]:
                sa=B[(B.controle=='real')&(B.task==tk)&(B.metodo==a_m)].bal_acc.mean()
                sf=B[(B.controle=='real')&(B.task==tk)&(B.metodo==f_m)].bal_acc.mean()
                line+=f" | dbalacc_{tk}={sf-sa:+.4f}"
        print(line)
    cmp("AE_amp","AE_forma","AE"); cmp("RF_amp","RF_forma","ExtraTrees")
print(f"\nfiguras em {FIG}")

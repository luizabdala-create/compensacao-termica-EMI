# -*- coding: utf-8 -*-
"""
ETAPA 7 — RESULTADOS FINAIS + GRÁFICOS (foco: MONOTONICIDADE do índice de dano)
==============================================================================
Avaliação LEAVE-ONE-TEMPERATURE-OUT: para cada temperatura, o modelo térmico é
treinado em TODAS as outras temperaturas e a temperatura avaliada fica fora.
=> cada ponto dos gráficos é out-of-sample.

Hiperparâmetros: r=24, smooth=7, trava=offset_gain — valores selecionados pela CV
(só no treino) na bateria de 10 splits (etapa6). Referência: mediana saudável 30°C.

Entrega:
 1) RMSD e CCDM por temperatura, por classe de dano (V8.1 e Park)
 2) MONOTONICIDADE: em quantas temperaturas vale D0<D1 e D0<D2 (saudável embaixo)
 3) barras-resumo por método x dano
 4) curvas de exemplo
 5) CSVs
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42)
FMIN,FMAX=30.0,40.0; REF_TEMP=30.0
R_STAR,SMOOTH_STAR=24,7
OUT=os.path.join(PROJ,"etapa7_final"); os.makedirs(OUT,exist_ok=True)
CACHE=os.path.join(PROJ,"etapa5_final","aligned_cache.npz")

plt.rcParams.update({"font.family":"serif","font.size":15,"axes.labelsize":17,
    "axes.titlesize":18,"xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,
    "pdf.fonttype":42})
CD={0:"#1f77b4",1:"#ff7f0e",2:"#d62728"}
NOME={0:"Sem dano (D0)",1:"Dano 1",2:"Dano 2"}

df=pd.read_pickle(BASE).reset_index(drop=True)
df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX)
X=df[fcols].to_numpy(np.float64); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
y_ref=np.median(X[np.isclose(T,REF_TEMP)&(y==0)],axis=0)
A=np.load(CACHE)["A"]; R=A-y_ref[None,:]
d2=np.abs(np.gradient(np.gradient(y_ref))); base_m=(d2<=np.percentile(d2,70))

def pack(mask):
    hm=mask&(y==0); temps=np.array(sorted(np.unique(T[hm])))
    Mm=np.vstack([np.median(R[hm&np.isclose(T,tk)],axis=0) for tk in temps])
    _,_,Vt=np.linalg.svd(Mm,full_matrices=False); return {"temps":temps,"Mm":Mm,"Vt":Vt}
def smc(Cf,w):
    if w<=1: return Cf
    k=np.ones(w)/w; pad=w//2
    return np.column_stack([np.convolve(np.pad(Cf[:,j],(pad,pad),mode="edge"),k,mode="valid")[:len(Cf)] for j in range(Cf.shape[1])])
def mh_fn(pk,r,smo):
    temps=pk["temps"]; re=min(r,len(temps),pk["Vt"].shape[0])
    Vr=pk["Vt"][:re]; Cf=smc(pk["Mm"]@Vr.T,smo)
    return lambda t:(np.array([np.interp(t,temps,Cf[:,j]) for j in range(re)])@Vr)
def vlock(yc):
    yc=yc+np.median(y_ref-yc)
    a=float(np.clip(np.polyfit(yc[base_m],y_ref[base_m],1)[0],0.97,1.03))
    return a*yc+np.median(y_ref-a*yc)

# ---------- magnitude REAL do dano na referência (sem compensação envolvida) ----------
print("="*80); print("MAGNITUDE DO DANO MEDIDA EM 30°C (referência — sem compensação)"); print("="*80)
for d in [1,2]:
    ii=np.where(np.isclose(T,REF_TEMP)&(y==d))[0]
    print(f"  Dano {d}: RMSD vs referência = {np.mean([v7.rmsd(X[i],y_ref) for i in ii]):.3f} | "
          f"CCDM = {np.mean([v7.ccdm(X[i],y_ref) for i in ii]):.4f}  (n={len(ii)})")

# ---------- LOTO por temperatura ----------
temps_all=np.array(sorted(np.unique(T)))
temps_eval=[t for t in temps_all if not np.isclose(t,REF_TEMP)]
rows=[]; Yv8_store={}
print(f"\nAvaliando {len(temps_eval)} temperaturas em leave-one-temperature-out...")
for t in temps_eval:
    ho=np.isclose(T,t); keep=~ho
    if (keep&(y==0)).sum()<5: continue
    pk=pack(keep); f=mh_fn(pk,R_STAR,SMOOTH_STAR)
    idx=np.where(ho)[0]
    Yv=np.vstack([vlock(A[i]-f(T[i])) for i in idx])
    Yp=np.vstack([v7.park_single(X[i],y_ref,fHz) for i in idx])
    Yv8_store[float(t)]=(idx,Yv,Yp)
    for nome,Y in [("Park",Yp),("V8.1",Yv)]:
        for c in [0,1,2]:
            sel=np.where(y[idx]==c)[0]
            if len(sel)==0: continue
            rows.append({"temperatura_c":float(t),"metodo":nome,"falha":c,"n":len(sel),
                "RMSD":float(np.mean([v7.rmsd(Y[j],y_ref) for j in sel])),
                "CCDM":float(np.mean([v7.ccdm(Y[j],y_ref) for j in sel]))})
    # original
    for c in [0,1,2]:
        sel=np.where(y[idx]==c)[0]
        if len(sel)==0: continue
        rows.append({"temperatura_c":float(t),"metodo":"Original","falha":c,"n":len(sel),
            "RMSD":float(np.mean([v7.rmsd(X[idx[j]],y_ref) for j in sel])),
            "CCDM":float(np.mean([v7.ccdm(X[idx[j]],y_ref) for j in sel]))})
res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,"metricas_por_temperatura_LOTO.csv"),index=False)

# ---------- MONOTONICIDADE ----------
print("\n"+"="*80); print("MONOTONICIDADE — saudável (D0) tem que ficar ABAIXO"); print("="*80)
mono=[]
for metodo in ["Original","Park","V8.1"]:
    sub=res[res.metodo==metodo]
    ok01=ok02=ok12=tot=0
    for t,g in sub.groupby("temperatura_c"):
        gd=g.set_index("falha")
        if not {0,1,2}.issubset(gd.index): continue
        tot+=1
        v0,v1,v2=gd.loc[0,"RMSD"],gd.loc[1,"RMSD"],gd.loc[2,"RMSD"]
        ok01+= v0<v1; ok02+= v0<v2; ok12+= v1<v2
    mono.append({"metodo":metodo,"n_temps":tot,"D0<D1":ok01,"D0<D2":ok02,"D1<D2":ok12,
                 "D0_abaixo_dos_dois":min(ok01,ok02)})
    print(f"  {metodo:9s} (n={tot} temps c/ 3 classes): D0<D1 em {ok01}/{tot} | D0<D2 em {ok02}/{tot} "
          f"| D0 abaixo dos DOIS em {min(ok01,ok02)}/{tot} | D1<D2 em {ok12}/{tot}")
pd.DataFrame(mono).to_csv(os.path.join(OUT,"monotonicidade.csv"),index=False)

med=res.groupby(["metodo","falha"])[["RMSD","CCDM"]].mean().round(4)
print("\nMédia geral (todas as temperaturas, out-of-sample):"); print(med.to_string())

# ================= GRÁFICOS =================
def eixo(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=.25)

# FIG 1 — RMSD e CCDM por temperatura: Park vs V8.1
for metrica in ["RMSD","CCDM"]:
    fig,axes=plt.subplots(1,2,figsize=(19,7),dpi=170,sharey=True)
    for ax,metodo in zip(axes,["Park","V8.1"]):
        sub=res[res.metodo==metodo]
        for c in [0,1,2]:
            s=sub[sub.falha==c].sort_values("temperatura_c")
            ax.plot(s["temperatura_c"],s[metrica],marker="o",ms=7,lw=2.4,
                    color=CD[c],label=NOME[c])
        ax.axvline(REF_TEMP,color="k",ls="--",lw=1.2,alpha=.6)
        ax.text(REF_TEMP+1,ax.get_ylim()[1]*0.96,"referência 30°C",fontsize=11,color="k")
        ax.set_xlabel("Temperatura (°C)"); ax.set_title(metodo); eixo(ax)
    axes[0].set_ylabel(f"{metrica} vs referência saudável")
    axes[1].legend(frameon=True,loc="upper right")
    fig.suptitle(f"{metrica} por temperatura e classe de dano — avaliação leave-one-temperature-out (out-of-sample)",
                 fontsize=18,y=1.01)
    fig.tight_layout()
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(OUT,f"fig1_{metrica}_por_temperatura.{ext}"),dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("figura: fig1_"+metrica)

# FIG 2 — barras resumo por método x dano
fig,axes=plt.subplots(1,2,figsize=(17,6.5),dpi=170)
metodos=["Original","Park","V8.1"]; w=0.26; xs=np.arange(len(metodos))
for ax,metrica in zip(axes,["RMSD","CCDM"]):
    for i,c in enumerate([0,1,2]):
        vals=[med.loc[(m,c),metrica] for m in metodos]
        ax.bar(xs+(i-1)*w,vals,width=w,color=CD[c],edgecolor="black",lw=.8,label=NOME[c])
        for xx,vv in zip(xs+(i-1)*w,vals):
            ax.text(xx,vv,f"{vv:.2f}" if metrica=="RMSD" else f"{vv:.3f}",
                    ha="center",va="bottom",fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(metodos); ax.set_ylabel(f"{metrica} médio"); eixo(ax)
    ax.set_title(metrica)
axes[1].legend(frameon=True)
fig.suptitle("Resumo — média sobre todas as temperaturas (out-of-sample)",fontsize=18,y=1.02)
fig.tight_layout()
for ext in ["png","pdf"]:
    fig.savefig(os.path.join(OUT,f"fig2_barras_resumo.{ext}"),dpi=170,bbox_inches="tight",facecolor="white")
plt.close(fig); print("figura: fig2")

# FIG 3 — curvas de exemplo em 3 temperaturas
for Tt in [-10.0,20.0,60.0]:
    if float(Tt) not in Yv8_store: continue
    idx,Yv,Yp=Yv8_store[float(Tt)]
    fig,axes=plt.subplots(2,3,figsize=(20,10),dpi=170,sharex=True)
    lo=min(Yv.min(),y_ref.min())-4; hi=max(Yv.max(),y_ref.max())+4
    for j,c in enumerate([0,1,2]):
        sel=np.where(y[idx]==c)[0]
        ax,ax2=axes[0,j],axes[1,j]
        if len(sel)==0: ax.set_title(f"{NOME[c]} — sem amostra"); continue
        k=sel[0]
        ax.plot(fHz/1e3,y_ref,"--",color="k",lw=1.6,label="Referência saudável 30°C",zorder=5)
        ax.plot(fHz/1e3,X[idx[k]],color="#e8736a",lw=1.0,alpha=.45,label=f"Original {Tt:.0f}°C")
        ax.plot(fHz/1e3,Yp[k],color="#2ca02c",lw=1.4,alpha=.85,label="Park")
        ax.plot(fHz/1e3,Yv[k],color="#1f5fd0",lw=1.8,label="V8.1")
        ax.set_title(NOME[c]); ax.set_ylim(lo,hi); eixo(ax)
        ax2.axhline(0,color="k",ls="--",lw=1)
        ax2.plot(fHz/1e3,Yp[k]-y_ref,color="#2ca02c",lw=1.2,alpha=.8)
        ax2.plot(fHz/1e3,Yv[k]-y_ref,color="#1f5fd0",lw=1.4)
        ax2.set_ylim(-12,12); ax2.set_xlabel("Frequência (kHz)"); eixo(ax2)
    axes[0,0].set_ylabel("Impedância compensada")
    axes[1,0].set_ylabel("Assinatura de dano\n(compensada − referência)")
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=4,frameon=True,bbox_to_anchor=(0.5,-0.02))
    fig.suptitle(f"Curvas compensadas — T = {Tt:.0f}°C (out-of-sample) — 30–40 kHz",fontsize=19,y=1.0)
    fig.tight_layout(rect=[0,0.04,1,0.97])
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(OUT,f"fig3_curvas_T{int(Tt)}C.{ext}"),dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"figura: fig3_T{int(Tt)}")
print("\n✅",OUT)

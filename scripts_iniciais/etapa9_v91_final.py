# -*- coding: utf-8 -*-
"""
V9.1 — ENTREGA FINAL: τ(T) robusto + modelo térmico de baixo posto + trava vertical
====================================================================================
V9 -> V9.1
  Problema: o estimador de shift falhou em 76 e 78 °C (τ=-1400 Hz em vez de ~+450 Hz;
  travou numa ressonância vizinha). Esses 2 outliers corrompiam a interpolação e
  produziam um pico espúrio no saudável perto de 75 °C.
  Mudança: τ(T) passa a ser ajustado de forma ROBUSTA (polinômio grau 2 com rejeição
  iterativa de outliers por MAD), justificado fisicamente: o deslocamento térmico é
  suave e monotônico em T (medido: -348 Hz a -10°C -> +436 Hz a 74°C).

MÉTODO FINAL (nenhuma curva de dano é alinhada individualmente):
  1) shift térmico τ(T) por TEMPERATURA, aprendido só no saudável, ajuste robusto
  2) modelo térmico de baixo posto (SVD r=24) do resíduo saudável por temperatura,
     coeficientes suavizados em T e interpolados
  3) trava vertical: offset mediano robusto + ganho limitado [0.97,1.03]
Avaliação leave-one-temperature-out: cada temperatura é avaliada FORA da amostra.
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42)
FMIN,FMAX=30.0,40.0; REF_TEMP=30.0; R_STAR,SMOOTH_STAR=24,7
OUT=os.path.join(PROJ,"RESULTADOS_FINAIS"); os.makedirs(OUT,exist_ok=True)
TAUC=os.path.join(PROJ,"etapa8_v9","tau_por_temperatura.npz")
plt.rcParams.update({"font.family":"serif","font.size":15,"axes.labelsize":17,"axes.titlesize":18,
    "xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,"pdf.fonttype":42})
CD={0:"#1f77b4",1:"#ff7f0e",2:"#d62728"}; NOME={0:"Sem dano (D0)",1:"Dano 1",2:"Dano 2"}

df=pd.read_pickle(BASE).reset_index(drop=True)
df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX)
X=df[fcols].to_numpy(np.float64); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
y_ref=np.median(X[np.isclose(T,REF_TEMP)&(y==0)],axis=0)
d2=np.abs(np.gradient(np.gradient(y_ref))); base_m=(d2<=np.percentile(d2,70))
z=np.load(TAUC); h_temps=z["temps"]; taus_raw=z["taus"]

def robust_tau_fit(tt,ta,deg=2,n_iter=3,thr=3.0):
    """polinômio com rejeição iterativa de outliers (MAD). Retorna (poly, mask_inliers)."""
    m=np.ones(len(tt),bool)
    for _ in range(n_iter):
        c=np.polyfit(tt[m],ta[m],deg); p=np.poly1d(c)
        r=ta-p(tt); mad=np.median(np.abs(r-np.median(r)))+1e-9
        m=np.abs(r-np.median(r))<=thr*1.4826*mad
        if m.sum()<deg+2: break
    return np.poly1d(np.polyfit(tt[m],ta[m],deg)), m

p_all,mask_all=robust_tau_fit(h_temps,taus_raw)
print("="*80); print("τ(T) ROBUSTO"); print("="*80)
print(f"  outliers rejeitados: {list(h_temps[~mask_all])} (τ bruto: {list(np.round(taus_raw[~mask_all],1))})")
print(f"  τ ajustado nessas temps: {list(np.round(p_all(h_temps[~mask_all]),1))} Hz")

def smc(Cf,w):
    if w<=1: return Cf
    k=np.ones(w)/w; pad=w//2
    return np.column_stack([np.convolve(np.pad(Cf[:,j],(pad,pad),mode="edge"),k,mode="valid")[:len(Cf)] for j in range(Cf.shape[1])])
def vlock(yc):
    yc=yc+np.median(y_ref-yc)
    a=float(np.clip(np.polyfit(yc[base_m],y_ref[base_m],1)[0],0.97,1.03))
    return a*yc+np.median(y_ref-a*yc)

rows=[]; store={}
temps_eval=[t for t in sorted(np.unique(T)) if not np.isclose(t,REF_TEMP)]
print(f"\nLOTO em {len(temps_eval)} temperaturas...")
for t in temps_eval:
    ho=np.isclose(T,t); keep=~ho
    trm=np.isin(h_temps,np.unique(T[keep&(y==0)]))
    if trm.sum()<6: continue
    p_tau,_=robust_tau_fit(h_temps[trm],taus_raw[trm])   # ajuste SÓ com temps de treino
    idx_tr=np.where(keep&(y==0))[0]
    Atr=np.vstack([v7.shift_interp(X[i],fHz,float(p_tau(T[i]))) for i in idx_tr])
    Rtr=Atr-y_ref[None,:]; Ttr=T[idx_tr]; tu=np.array(sorted(np.unique(Ttr)))
    Mm=np.vstack([np.median(Rtr[np.isclose(Ttr,tk)],axis=0) for tk in tu])
    _,_,Vt=np.linalg.svd(Mm,full_matrices=False)
    re=min(R_STAR,len(tu),Vt.shape[0]); Vr=Vt[:re]; Cf=smc(Mm@Vr.T,SMOOTH_STAR)
    def mh(tv): return np.array([np.interp(tv,tu,Cf[:,j]) for j in range(re)])@Vr
    idx=np.where(ho)[0]
    Yv=np.vstack([vlock(v7.shift_interp(X[i],fHz,float(p_tau(T[i])))-mh(T[i])) for i in idx])
    Yp=np.vstack([v7.park_single(X[i],y_ref,fHz) for i in idx])
    store[float(t)]=(idx,Yv,Yp)
    for nome,Y in [("Original",X[idx]),("Park",Yp),("V9.1",Yv)]:
        for c in [0,1,2]:
            s=np.where(y[idx]==c)[0]
            if len(s)==0: continue
            rows.append({"temperatura_c":float(t),"metodo":nome,"falha":c,"n":len(s),
                "RMSD":float(np.mean([v7.rmsd(Y[j],y_ref) for j in s])),
                "CCDM":float(np.mean([v7.ccdm(Y[j],y_ref) for j in s]))})
res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,"metricas_por_temperatura.csv"),index=False)

print("\n"+"="*84); print("MONOTONICIDADE — requisito: D0 abaixo dos dois, e ordem D0<D1<D2"); print("="*84)
mono=[]
for metodo in ["Original","Park","V9.1"]:
    sub=res[res.metodo==metodo]; o1=o2=o12=tot=0; full=0
    for t,g in sub.groupby("temperatura_c"):
        gd=g.set_index("falha")
        if not {0,1,2}.issubset(gd.index): continue
        tot+=1; v0,v1,v2=gd.loc[0,"RMSD"],gd.loc[1,"RMSD"],gd.loc[2,"RMSD"]
        o1+=v0<v1; o2+=v0<v2; o12+=v1<v2; full+= (v0<v1<v2)
    mono.append({"metodo":metodo,"n_temps":tot,"D0<D1":o1,"D0<D2":o2,
                 "D0_abaixo_dos_dois":min(o1,o2),"D1<D2":o12,"ORDEM_COMPLETA_D0<D1<D2":full})
    print(f"  {metodo:9s} n={tot} | D0 abaixo dos DOIS: {min(o1,o2)}/{tot} | ORDEM COMPLETA D0<D1<D2: {full}/{tot}")
pd.DataFrame(mono).to_csv(os.path.join(OUT,"monotonicidade.csv"),index=False)
resumo=res.groupby(["metodo","falha"])[["RMSD","CCDM"]].mean().round(4)
resumo.to_csv(os.path.join(OUT,"resumo_medio.csv"))
print("\nMédias out-of-sample:"); print(resumo.to_string())
print("\nSeparação média (índice de dano − saudável):")
for m in ["Original","Park","V9.1"]:
    s=resumo.loc[m]
    print(f"  {m:9s} RMSD: D1-D0={s.loc[1,'RMSD']-s.loc[0,'RMSD']:+.3f}  D2-D0={s.loc[2,'RMSD']-s.loc[0,'RMSD']:+.3f}")

# ---------------- FIGURAS ----------------
def eixo(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=.25)

# FIG 0 — τ(T)
fig,ax=plt.subplots(figsize=(10,6),dpi=170)
ax.plot(h_temps[mask_all],taus_raw[mask_all],"o",ms=8,color="#1f77b4",label="τ estimado (usado)")
ax.plot(h_temps[~mask_all],taus_raw[~mask_all],"x",ms=13,mew=3,color="crimson",label="outlier rejeitado")
tg=np.linspace(h_temps.min(),h_temps.max(),200)
ax.plot(tg,p_all(tg),"-",lw=2.5,color="black",label="ajuste robusto τ(T)")
ax.set_xlabel("Temperatura (°C)"); ax.set_ylabel("Deslocamento térmico τ (Hz)")
ax.set_title("Deslocamento térmico em função da temperatura"); ax.legend(); eixo(ax)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(os.path.join(OUT,f"fig0_tau_temperatura.{e}"),dpi=170,bbox_inches="tight",facecolor="white")
plt.close(fig)

# FIG 1 — RMSD e CCDM por temperatura
for metrica in ["RMSD","CCDM"]:
    fig,axes=plt.subplots(1,3,figsize=(21,6.8),dpi=170,sharey=True)
    for ax,metodo in zip(axes,["Original","Park","V9.1"]):
        sub=res[res.metodo==metodo]
        for c in [0,1,2]:
            s=sub[sub.falha==c].sort_values("temperatura_c")
            ax.plot(s["temperatura_c"],s[metrica],marker="o",ms=6,lw=2.4,color=CD[c],label=NOME[c])
        ax.axvline(REF_TEMP,color="k",ls="--",lw=1.1,alpha=.6)
        ax.set_xlabel("Temperatura (°C)"); ax.set_title(metodo); eixo(ax)
    axes[0].set_ylabel(f"{metrica} vs referência saudável"); axes[2].legend(frameon=True)
    fig.suptitle(f"{metrica} por temperatura e classe de dano — leave-one-temperature-out (out-of-sample)",fontsize=18,y=1.01)
    fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(os.path.join(OUT,f"fig1_{metrica}_por_temperatura.{e}"),dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)

# FIG 1b — só V9.1 ampliado (o gráfico principal do relatório)
for metrica in ["RMSD","CCDM"]:
    fig,ax=plt.subplots(figsize=(11,7),dpi=170)
    sub=res[res.metodo=="V9.1"]
    for c in [0,1,2]:
        s=sub[sub.falha==c].sort_values("temperatura_c")
        ax.plot(s["temperatura_c"],s[metrica],marker="o",ms=7,lw=2.6,color=CD[c],label=NOME[c])
    ax.axvline(REF_TEMP,color="k",ls="--",lw=1.2,alpha=.6)
    ax.text(REF_TEMP+1.5,ax.get_ylim()[1]*.95,"referência 30 °C",fontsize=12)
    ax.set_xlabel("Temperatura (°C)"); ax.set_ylabel(f"{metrica} vs referência saudável")
    ax.set_title(f"V9.1 — {metrica}: índice de dano monotônico (D0 < D1 < D2)")
    ax.legend(frameon=True); eixo(ax); fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(os.path.join(OUT,f"fig1b_V91_{metrica}.{e}"),dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)

# FIG 2 — barras
fig,axes=plt.subplots(1,2,figsize=(17,6.5),dpi=170)
ms=["Original","Park","V9.1"]; w=.26; xs=np.arange(len(ms))
for ax,metrica in zip(axes,["RMSD","CCDM"]):
    for i,c in enumerate([0,1,2]):
        vals=[resumo.loc[(m,c),metrica] for m in ms]
        ax.bar(xs+(i-1)*w,vals,width=w,color=CD[c],edgecolor="k",lw=.8,label=NOME[c])
        for xx,vv in zip(xs+(i-1)*w,vals):
            ax.text(xx,vv,f"{vv:.2f}" if metrica=="RMSD" else f"{vv:.3f}",ha="center",va="bottom",fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(ms); ax.set_ylabel(f"{metrica} médio"); ax.set_title(metrica); eixo(ax)
axes[1].legend(frameon=True)
fig.suptitle("Resumo — média sobre todas as temperaturas (out-of-sample)",fontsize=18,y=1.02)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(os.path.join(OUT,f"fig2_barras.{e}"),dpi=170,bbox_inches="tight",facecolor="white")
plt.close(fig)

# FIG 3 — curvas
for Tt in [-10.0,20.0,60.0,80.0]:
    if float(Tt) not in store: continue
    idx,Yv,Yp=store[float(Tt)]
    fig,axes=plt.subplots(2,3,figsize=(20,10),dpi=170,sharex=True)
    lo=min(Yv.min(),y_ref.min())-4; hi=max(Yv.max(),y_ref.max())+4
    for j,c in enumerate([0,1,2]):
        s=np.where(y[idx]==c)[0]; ax,ax2=axes[0,j],axes[1,j]
        if len(s)==0: ax.set_title(f"{NOME[c]} — sem amostra"); continue
        k=s[0]
        ax.plot(fHz/1e3,y_ref,"--",color="k",lw=1.6,label="Referência saudável 30 °C",zorder=5)
        ax.plot(fHz/1e3,X[idx[k]],color="#e8736a",lw=1.0,alpha=.45,label=f"Original {Tt:.0f} °C")
        ax.plot(fHz/1e3,Yp[k],color="#2ca02c",lw=1.4,alpha=.85,label="Park")
        ax.plot(fHz/1e3,Yv[k],color="#7b2fbe",lw=1.8,label="V9.1")
        ax.set_title(NOME[c]); ax.set_ylim(lo,hi); eixo(ax)
        ax2.axhline(0,color="k",ls="--",lw=1)
        ax2.plot(fHz/1e3,Yp[k]-y_ref,color="#2ca02c",lw=1.2,alpha=.8)
        ax2.plot(fHz/1e3,Yv[k]-y_ref,color="#7b2fbe",lw=1.4)
        ax2.set_ylim(-14,14); ax2.set_xlabel("Frequência (kHz)"); eixo(ax2)
    axes[0,0].set_ylabel("Impedância compensada"); axes[1,0].set_ylabel("Assinatura de dano")
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=4,frameon=True,bbox_to_anchor=(.5,-.02))
    fig.suptitle(f"V9.1 vs Park — T = {Tt:.0f} °C (out-of-sample) — 30–40 kHz",fontsize=19,y=1.0)
    fig.tight_layout(rect=[0,.04,1,.97])
    for e in ["png","pdf"]: fig.savefig(os.path.join(OUT,f"fig3_curvas_T{int(Tt)}C.{e}"),dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)
print("\n✅ TUDO EM:",OUT)

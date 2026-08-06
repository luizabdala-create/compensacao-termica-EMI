# -*- coding: utf-8 -*-
"""
V9 — shift térmico POR TEMPERATURA (τ(T)) em vez de alinhamento por curva
=========================================================================
V8.1 -> V9
  Problema: o alinhamento POR CURVA absorve o deslocamento próprio do dano 2
  (dano 2 tem componente horizontal), invertendo a ordem: no cru D2>D1 (RMSD 6,22
  vs 5,29 em 30°C) mas depois de compensar D1>D2.
  Mudança: τ é estimado SÓ no saudável, por temperatura, e interpolado em T; a MESMA
  correção é aplicada a todas as curvas daquela temperatura. Nenhuma curva de dano
  é alinhada individualmente -> o deslocamento do dano é PRESERVADO.
  Isso só é viável porque há 46 temperaturas saudáveis (espaçamento ~2°C).
Avaliação leave-one-temperature-out (cada temperatura fora da amostra).
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42)
FMIN,FMAX=30.0,40.0; REF_TEMP=30.0; R_STAR,SMOOTH_STAR=24,7
OUT=os.path.join(PROJ,"etapa8_v9"); os.makedirs(OUT,exist_ok=True)
CACHEV8=os.path.join(PROJ,"etapa5_final","aligned_cache.npz")
TAUC=os.path.join(OUT,"tau_por_temperatura.npz")
plt.rcParams.update({"font.family":"serif","font.size":15,"axes.labelsize":17,
    "axes.titlesize":18,"xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,"pdf.fonttype":42})
CD={0:"#1f77b4",1:"#ff7f0e",2:"#d62728"}; NOME={0:"Sem dano (D0)",1:"Dano 1",2:"Dano 2"}

df=pd.read_pickle(BASE).reset_index(drop=True)
df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX)
X=df[fcols].to_numpy(np.float64); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
y_ref=np.median(X[np.isclose(T,REF_TEMP)&(y==0)],axis=0)
d2=np.abs(np.gradient(np.gradient(y_ref))); base_m=(d2<=np.percentile(d2,70))
A_v8=np.load(CACHEV8)["A"]   # alinhamento por curva (V8.1), p/ comparação

# ---- τ(Tk) do saudável, uma vez por temperatura (independente do fold) ----
h_temps=np.array(sorted(np.unique(T[y==0])))
if os.path.exists(TAUC):
    z=np.load(TAUC); h_temps=z["temps"]; taus=z["taus"]; print("τ(T): cache")
else:
    print(f"estimando τ para {len(h_temps)} temperaturas saudáveis...")
    taus=np.zeros(len(h_temps))
    for i,tk in enumerate(h_temps):
        med=np.median(X[np.isclose(T,tk)&(y==0)],axis=0)
        taus[i]=v7.estimate_shift_multiscale(med,y_ref,fHz,max_frac=0.14,n_coarse=101,n_fine=81,
                    prior_tau=0.0,prior_penalty=0.015,min_improvement=0.0,return_info=False)
    np.savez(TAUC,temps=h_temps,taus=taus)
print("τ(T): min=%.1f Hz max=%.1f Hz"%(taus.min(),taus.max()))

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
print(f"LOTO em {len(temps_eval)} temperaturas...")
for t in temps_eval:
    ho=np.isclose(T,t); keep=~ho
    tr_mask=np.isin(h_temps,np.unique(T[keep&(y==0)]))
    if tr_mask.sum()<5: continue
    tt_tr=h_temps[tr_mask]; ta_tr=taus[tr_mask]
    tau_ho=float(np.interp(t,tt_tr,ta_tr))            # τ da temp de teste: INTERPOLADO

    # curvas de treino deslocadas pelo τ da sua própria temperatura (dado de treino)
    idx_tr=np.where(keep&(y==0))[0]
    Atr=np.vstack([v7.shift_interp(X[i],fHz,float(np.interp(T[i],tt_tr,ta_tr))) for i in idx_tr])
    Rtr=Atr-y_ref[None,:]
    Ttr=T[idx_tr]
    temps_u=np.array(sorted(np.unique(Ttr)))
    Mm=np.vstack([np.median(Rtr[np.isclose(Ttr,tk)],axis=0) for tk in temps_u])
    _,_,Vt=np.linalg.svd(Mm,full_matrices=False)
    re=min(R_STAR,len(temps_u),Vt.shape[0]); Vr=Vt[:re]; Cf=smc(Mm@Vr.T,SMOOTH_STAR)
    def mh(tv): return np.array([np.interp(tv,temps_u,Cf[:,j]) for j in range(re)])@Vr

    idx=np.where(ho)[0]
    # V9: shift por TEMPERATURA (igual p/ todas as classes) + térmico + trava
    Yv9=np.vstack([vlock(v7.shift_interp(X[i],fHz,tau_ho)-mh(T[i])) for i in idx])
    Yp=np.vstack([v7.park_single(X[i],y_ref,fHz) for i in idx])
    store[float(t)]=(idx,Yv9,Yp)
    for nome,Y in [("Park",Yp),("V9",Yv9)]:
        for c in [0,1,2]:
            s=np.where(y[idx]==c)[0]
            if len(s)==0: continue
            rows.append({"temperatura_c":float(t),"metodo":nome,"falha":c,"n":len(s),
                "RMSD":float(np.mean([v7.rmsd(Y[j],y_ref) for j in s])),
                "CCDM":float(np.mean([v7.ccdm(Y[j],y_ref) for j in s]))})
res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,"metricas_v9_LOTO.csv"),index=False)

# junta com V8.1 (etapa7) p/ comparar
p7=os.path.join(PROJ,"etapa7_final","metricas_por_temperatura_LOTO.csv")
if os.path.exists(p7):
    old=pd.read_csv(p7); res=pd.concat([old[old.metodo.isin(["Original","V8.1"])],res],ignore_index=True)

print("\n"+"="*84); print("MONOTONICIDADE (RMSD) — D0 abaixo + ordem D1<D2"); print("="*84)
mono=[]
for metodo in ["Original","Park","V8.1","V9"]:
    sub=res[res.metodo==metodo]
    if len(sub)==0: continue
    o1=o2=o12=tot=0
    for t,g in sub.groupby("temperatura_c"):
        gd=g.set_index("falha")
        if not {0,1,2}.issubset(gd.index): continue
        tot+=1; v0,v1,v2=gd.loc[0,"RMSD"],gd.loc[1,"RMSD"],gd.loc[2,"RMSD"]
        o1+=v0<v1; o2+=v0<v2; o12+=v1<v2
    mono.append({"metodo":metodo,"n":tot,"D0<D1":o1,"D0<D2":o2,"D0_abaixo":min(o1,o2),"D1<D2":o12})
    print(f"  {metodo:9s} n={tot} | D0<D1 {o1}/{tot} | D0<D2 {o2}/{tot} | D0 abaixo dos dois {min(o1,o2)}/{tot} | D1<D2 {o12}/{tot}")
pd.DataFrame(mono).to_csv(os.path.join(OUT,"monotonicidade_v9.csv"),index=False)
print("\nMédias (out-of-sample):")
print(res.groupby(["metodo","falha"])[["RMSD","CCDM"]].mean().round(4).to_string())

for metrica in ["RMSD","CCDM"]:
    ms=[m for m in ["Park","V8.1","V9"] if m in res.metodo.unique()]
    fig,axes=plt.subplots(1,len(ms),figsize=(7.0*len(ms),6.8),dpi=170,sharey=True)
    for ax,metodo in zip(np.atleast_1d(axes),ms):
        sub=res[res.metodo==metodo]
        for c in [0,1,2]:
            s=sub[sub.falha==c].sort_values("temperatura_c")
            ax.plot(s["temperatura_c"],s[metrica],marker="o",ms=6,lw=2.3,color=CD[c],label=NOME[c])
        ax.axvline(REF_TEMP,color="k",ls="--",lw=1.1,alpha=.6)
        ax.set_xlabel("Temperatura (°C)"); ax.set_title(metodo)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=.25)
    np.atleast_1d(axes)[0].set_ylabel(f"{metrica} vs referência saudável")
    np.atleast_1d(axes)[-1].legend(frameon=True)
    fig.suptitle(f"{metrica} por temperatura — leave-one-temperature-out (out-of-sample)",fontsize=18,y=1.01)
    fig.tight_layout()
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(OUT,f"fig_v9_{metrica}.{ext}"),dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("figura fig_v9_"+metrica)

for Tt in [-10.0,20.0,60.0]:
    if float(Tt) not in store: continue
    idx,Yv,Yp=store[float(Tt)]
    fig,axes=plt.subplots(2,3,figsize=(20,10),dpi=170,sharex=True)
    lo=min(Yv.min(),y_ref.min())-4; hi=max(Yv.max(),y_ref.max())+4
    for j,c in enumerate([0,1,2]):
        s=np.where(y[idx]==c)[0]; ax,ax2=axes[0,j],axes[1,j]
        if len(s)==0: ax.set_title(f"{NOME[c]} — sem amostra"); continue
        k=s[0]
        ax.plot(fHz/1e3,y_ref,"--",color="k",lw=1.6,label="Referência saudável 30°C",zorder=5)
        ax.plot(fHz/1e3,X[idx[k]],color="#e8736a",lw=1.0,alpha=.45,label=f"Original {Tt:.0f}°C")
        ax.plot(fHz/1e3,Yp[k],color="#2ca02c",lw=1.4,alpha=.85,label="Park")
        ax.plot(fHz/1e3,Yv[k],color="#7b2fbe",lw=1.8,label="V9")
        ax.set_title(NOME[c]); ax.set_ylim(lo,hi)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=.2)
        ax2.axhline(0,color="k",ls="--",lw=1)
        ax2.plot(fHz/1e3,Yp[k]-y_ref,color="#2ca02c",lw=1.2,alpha=.8)
        ax2.plot(fHz/1e3,Yv[k]-y_ref,color="#7b2fbe",lw=1.4)
        ax2.set_ylim(-12,12); ax2.set_xlabel("Frequência (kHz)")
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False); ax2.grid(alpha=.2)
    axes[0,0].set_ylabel("Impedância compensada")
    axes[1,0].set_ylabel("Assinatura de dano")
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=4,frameon=True,bbox_to_anchor=(0.5,-0.02))
    fig.suptitle(f"V9 vs Park — T = {Tt:.0f}°C (out-of-sample)",fontsize=19,y=1.0)
    fig.tight_layout(rect=[0,0.04,1,0.97])
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(OUT,f"fig_v9_curvas_T{int(Tt)}C.{ext}"),dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)
print("\n✅",OUT)

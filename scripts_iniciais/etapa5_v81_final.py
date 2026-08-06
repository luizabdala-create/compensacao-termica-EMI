# -*- coding: utf-8 -*-
"""
V8.1 — V8 + TRAVA VERTICAL (offset constante robusto)
=====================================================
V8 -> V8.1
  Problema: a forma/picos ficaram corretos (CCDM melhor que Park), mas o NÍVEL
  vertical ficava deslocado (~6 unidades) -> RMSD pior que o Park.
  Mudança: após subtrair a térmica m̃(T), aplica um OFFSET CONSTANTE robusto
  (mediana de y_ref - compensada). Offset constante NÃO altera forma nem
  amplitude de pico -> a assinatura de dano é preservada exatamente.
  Variante opcional: offset + ganho robusto (limitado), testada por CV.
Seleção de (r, modo vertical) por leave-one-temperature-out SÓ no treino.
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42)
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa5_v81"); os.makedirs(OUT,exist_ok=True)
CACHE=os.path.join(PROJ,"etapa5_final","aligned_cache.npz")
TEST_TEMPS=[-10.0,20.0,40.0,60.0]; REF_TEMP=30.0
RS=[2,3,4,6,8]; MODES=["none","offset","offset_gain"]
plt.rcParams.update({"font.family":"serif","font.size":12,"axes.labelsize":13,
                     "axes.titlesize":14,"legend.fontsize":11})

df=pd.read_pickle(BASE).reset_index(drop=True)
df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX)
X=df[fcols].to_numpy(np.float64); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
is_test=np.zeros(len(df),bool)
for t in TEST_TEMPS: is_test|=np.isclose(T,t)
is_train=~is_test
y_ref=np.median(X[np.isclose(T,REF_TEMP)&(y==0)],axis=0)
A=np.load(CACHE)["A"]; R=A-y_ref[None,:]

# pesos de baseline (baixa curvatura da referência) p/ ganho robusto
d2=np.abs(np.gradient(np.gradient(y_ref)))
w_base=(d2<=np.percentile(d2,70)).astype(float)

def build_thermal(train_mask,r):
    hm=train_mask&(y==0); temps=np.array(sorted(np.unique(T[hm])))
    Mmat=np.vstack([np.median(R[hm&np.isclose(T,tk)],axis=0) for tk in temps])
    r_eff=min(r,len(temps),Mmat.shape[1])
    _,_,Vt=np.linalg.svd(Mmat,full_matrices=False); Vr=Vt[:r_eff]; Cf=Mmat@Vr.T
    def mh(t):
        c=np.array([np.interp(t,temps,Cf[:,j]) for j in range(r_eff)]); return c@Vr
    return mh

def vertical_lock(yc,mode):
    """offset constante robusto (mediana) — não altera forma. ganho opcional limitado."""
    if mode=="none": return yc
    off=np.median(y_ref-yc); yc=yc+off
    if mode=="offset_gain":
        m=w_base>0
        a=np.polyfit(yc[m],y_ref[m],1)[0]
        a=float(np.clip(a,0.97,1.03))
        yc=a*yc+np.median(y_ref-a*yc)
    return yc

def compensate(train_mask,r,mode,idx):
    mh=build_thermal(train_mask,r)
    return np.vstack([vertical_lock(A[i]-mh(T[i]),mode) for i in idx])

# ---- CV (leave-one-temperature-out) no TREINO, saudáveis ----
train_h_temps=np.array(sorted(np.unique(T[is_train&(y==0)])))
cv={}
for r in RS:
    for mode in MODES:
        sc=[]
        for tk in train_h_temps:
            ho=is_train&np.isclose(T,tk)&(y==0); keep=is_train&~np.isclose(T,tk)
            if ho.sum()==0: continue
            idx=np.where(ho)[0]; Yc=compensate(keep,r,mode,idx)
            sc.append(np.mean([v7.rmsd(Yc[j],y_ref) for j in range(len(idx))]))
        cv[(r,mode)]=float(np.mean(sc))
best=min(cv,key=cv.get); rstar,mstar=best
print("CV(treino) RMSD saudável:")
for r in RS:
    print("   r=%d: "%r + " | ".join(f"{m}={cv[(r,m)]:.3f}" for m in MODES))
print(f"   >>> escolhido: r={rstar}, modo={mstar} (CV={cv[best]:.4f})")

allidx=np.arange(len(X))
Y_v81=compensate(is_train,rstar,mstar,allidx)
Y_v8=compensate(is_train,rstar,"none",allidx)
Y_park=np.vstack([v7.park_single(x,y_ref,fHz) for x in X])
METHODS={"Original":X,"Park":Y_park,"V8 (sem trava)":Y_v8,f"V8.1 ({mstar})":Y_v81}

rows=[]
for mn,Y in METHODS.items():
    for d in [0,1,2]:
        idx=np.where(is_test&(y==d))[0]
        rows.append({"Metodo":mn,"falha":d,
            "RMSD":float(np.mean([v7.rmsd(Y[i],y_ref) for i in idx])),
            "CCDM":float(np.mean([v7.ccdm(Y[i],y_ref) for i in idx]))})
met=pd.DataFrame(rows); met.to_csv(os.path.join(OUT,"metricas_v81.csv"),index=False)
print("\n"+"="*74); print("RMSD / CCDM — TESTE (out-of-sample)"); print("="*74)
print(met.pivot(index="Metodo",columns="falha",values=["RMSD","CCDM"]).round(4).to_string())
print("\nSeparação dano-saudável:")
for mn in METHODS:
    s=met[met["Metodo"]==mn].set_index("falha")
    print(f"  {mn:18s} RMSD D1-D0={s.loc[1,'RMSD']-s.loc[0,'RMSD']:+.3f} D2-D0={s.loc[2,'RMSD']-s.loc[0,'RMSD']:+.3f}"
          f" | CCDM D1-D0={s.loc[1,'CCDM']-s.loc[0,'CCDM']:+.4f} D2-D0={s.loc[2,'CCDM']-s.loc[0,'CCDM']:+.4f}")

def sm(Rr,w=51): return np.vstack([v7.moving_average(r,w) for r in Rr])
def nc(Ftr,ytr,Fte):
    cs=sorted(np.unique(ytr)); C=np.vstack([Ftr[ytr==c].mean(0) for c in cs])
    return np.array([cs[int(np.argmin(((C-r)**2).sum(1)))] for r in Fte])
def bal(a,b): return float(np.mean([(b[a==c]==c).mean() for c in np.unique(a)]))
print("\nClassificação de dano out-of-sample:")
clf={}
for mn,Y in METHODS.items():
    F=sm(Y-y_ref[None,:]); p=nc(F[is_train],y[is_train],F[is_test]); ba=bal(y[is_test],p)
    M3=np.zeros((3,3),int)
    for a,b in zip(y[is_test],p): M3[a,b]+=1
    clf[mn]=ba; print(f"  {mn:18s} bal_acc={ba:.3f} conf={M3.tolist()}")
json.dump({"rstar":int(rstar),"modo":mstar,"cv":{f"{k[0]}_{k[1]}":v for k,v in cv.items()},
           "metricas":rows,"bal_acc":clf},open(os.path.join(OUT,"resumo_v81.json"),"w"),indent=2,ensure_ascii=False)

fkhz=fHz/1e3
cor={"Original":"#e8736a","Park":"#2ca02c",f"V8.1 ({mstar})":"#1f5fd0"}
for Tt in TEST_TEMPS:
    fig,axes=plt.subplots(2,3,figsize=(19,9),dpi=150,sharex=True)
    sel=np.isclose(T,Tt)
    lo=min(Y_v81[sel].min(),y_ref.min())-4; hi=max(Y_v81[sel].max(),y_ref.max())+4
    for j,d in enumerate([0,1,2]):
        idx=np.where(sel&(y==d))[0]; ax=axes[0,j]; ax2=axes[1,j]
        if len(idx)==0: ax.set_title(f"Dano {d} — sem amostra"); continue
        i=idx[0]
        ax.plot(fkhz,y_ref,"--",color="black",lw=1.5,label=f"Referência saudável {REF_TEMP:.0f}°C",zorder=5)
        ax.plot(fkhz,X[i],color=cor["Original"],lw=1.0,alpha=0.45,label="Original")
        ax.plot(fkhz,Y_park[i],color=cor["Park"],lw=1.4,alpha=0.85,label="Park")
        ax.plot(fkhz,Y_v81[i],color=cor[f"V8.1 ({mstar})"],lw=1.7,label=f"V8.1")
        ax.set_title(f"Dano {d}"); ax.set_ylim(lo,hi)
        ax2.axhline(0,color="black",lw=1,ls="--")
        ax2.plot(fkhz,Y_park[i]-y_ref,color=cor["Park"],lw=1.2,alpha=0.8)
        ax2.plot(fkhz,Y_v81[i]-y_ref,color=cor[f"V8.1 ({mstar})"],lw=1.4)
        ax2.set_xlabel("Frequência (kHz)"); ax2.set_ylim(-12,12)
        for a in (ax,ax2):
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    axes[0,0].set_ylabel("Impedância (compensada)")
    axes[1,0].set_ylabel("Assinatura de dano\n(compensada − referência)")
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=4,frameon=True,bbox_to_anchor=(0.5,-0.03))
    fig.suptitle(f"V8.1 vs Park — T = {Tt:.0f}°C (TESTE, out-of-sample) — 30–40 kHz",fontsize=16,y=1.0)
    fig.tight_layout(rect=[0,0.04,1,0.97])
    p=os.path.join(OUT,f"v81_T{int(Tt)}C.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("figura:",p)
print("\n✅",OUT)

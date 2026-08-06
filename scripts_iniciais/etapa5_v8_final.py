# -*- coding: utf-8 -*-
"""
V8 — Compensação térmica por VARIEDADE TÉRMICA SAUDÁVEL DE BAIXO POSTO
=======================================================================
Corrige os artefatos da versão anterior (picos invertidos / AE não-linear falhando).

PROBLEMA ANTERIOR:
  curva_compensada = y_ref + [R - proj_k(R)]  -> projetava o resíduo DA PRÓPRIA CURVA
  no subespaço saudável. Isso (a) comia parte do dano e (b) subtraía a média saudável
  'mu' de curvas danificadas, criando picos invertidos.

MUDANÇA (V8):
  A correção térmica é uma FUNÇÃO DA TEMPERATURA, aprendida só no saudável, e é
  SUBTRAÍDA de qualquer curva:
      m(T) = mediana( saudável_alinhada(T) ) - y_ref          [por temperatura de treino]
      m(T) é denoised por SVD de baixo posto (r comps) e interpolado em T
      curva_compensada = curva_alinhada - m̃(T)
  Como m̃(T) não depende do resíduo da curva, o DANO NUNCA É PROJETADO FORA.
  Funciona agora porque há 46 temperaturas saudáveis (vizinhas a ~2°C), então a
  interpolação em T é precisa (antes, com vãos de 20°C, falhava).

HIPÓTESE TESTADA: remove térmica melhor que o Park (que só faz shift+offset rígido)
  preservando integralmente a assinatura de dano.
MÉTRICAS: RMSD/CCDM (A) + classificação de dano out-of-sample (B).
Seleção de r: leave-one-temperature-out APENAS no treino.
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42)
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa5_v8"); os.makedirs(OUT,exist_ok=True)
CACHE=os.path.join(PROJ,"etapa5_final","aligned_cache.npz")
TEST_TEMPS=[-10.0,20.0,40.0,60.0]; REF_TEMP=30.0
RS=[1,2,3,4,5,6,8,10]

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

# alinhamento (cache do run anterior)
if os.path.exists(CACHE):
    A=np.load(CACHE)["A"]; print("alinhamento: cache")
else:
    A=np.zeros_like(X)
    for i in range(len(X)):
        tau=v7.estimate_shift_multiscale(X[i],y_ref,fHz,max_frac=0.14,n_coarse=101,n_fine=81,
              prior_tau=0.0,prior_penalty=0.015,min_improvement=0.0,return_info=False)
        A[i]=v7.shift_interp(X[i],fHz,tau)
    np.savez_compressed(CACHE,A=A)
R=A-y_ref[None,:]

def build_thermal_model(train_mask, r):
    """m(T) por temperatura saudável de treino + SVD baixo posto; devolve interpolador."""
    hm=train_mask&(y==0)
    temps=np.array(sorted(np.unique(T[hm])))
    Mmat=np.vstack([np.median(R[hm&np.isclose(T,tk)],axis=0) for tk in temps])
    r_eff=min(r,len(temps),Mmat.shape[1])
    U,S,Vt=np.linalg.svd(Mmat,full_matrices=False)
    Vr=Vt[:r_eff]; Cf=Mmat@Vr.T                      # coefs por temperatura (n_temps x r)
    def m_hat(t):
        c=np.array([np.interp(t,temps,Cf[:,j]) for j in range(r_eff)])
        return c@Vr
    return m_hat

def compensate(train_mask,r,idx):
    mh=build_thermal_model(train_mask,r)
    return np.vstack([A[i]-mh(T[i]) for i in idx])

# ---------- seleção de r por LOTO-CV no TREINO (só saudáveis) ----------
train_h_temps=np.array(sorted(np.unique(T[is_train&(y==0)])))
cv={r:[] for r in RS}
for tk in train_h_temps:
    ho=is_train&np.isclose(T,tk)&(y==0)
    keep=is_train&~np.isclose(T,tk)
    if ho.sum()==0: continue
    idx=np.where(ho)[0]
    for r in RS:
        Yc=compensate(keep,r,idx)
        cv[r].append(np.mean([v7.rmsd(Yc[j],y_ref) for j in range(len(idx))]))
cvm={r:float(np.mean(v)) for r,v in cv.items() if len(v)}
rstar=min(cvm,key=cvm.get)
print("CV(treino) RMSD saudável por r:")
for r in RS:
    if r in cvm: print(f"   r={r:2d}: {cvm[r]:.4f}"+("   <= r* escolhido" if r==rstar else ""))

# ---------- aplica ----------
allidx=np.arange(len(X))
Y_v8=compensate(is_train,rstar,allidx)
Y_park=np.vstack([v7.park_single(x,y_ref,fHz) for x in X])
METHODS={"Original":X,"Park":Y_park,f"V8 (r={rstar})":Y_v8}

# ---------- métricas ----------
rows=[]
for mn,Y in METHODS.items():
    for d in [0,1,2]:
        idx=np.where(is_test&(y==d))[0]
        rows.append({"Metodo":mn,"falha":d,"n":len(idx),
            "RMSD":float(np.mean([v7.rmsd(Y[i],y_ref) for i in idx])),
            "CCDM":float(np.mean([v7.ccdm(Y[i],y_ref) for i in idx]))})
met=pd.DataFrame(rows); met.to_csv(os.path.join(OUT,"metricas_v8.csv"),index=False)
print("\n"+"="*72); print("RMSD / CCDM — TEMPERATURAS DE TESTE (out-of-sample)"); print("="*72)
print(met.pivot(index="Metodo",columns="falha",values=["RMSD","CCDM"]).round(4).to_string())
print("\nSeparação dano-saudável (maior = dano melhor preservado):")
for mn in METHODS:
    s=met[met["Metodo"]==mn].set_index("falha")
    print(f"  {mn:14s} RMSD D1-D0={s.loc[1,'RMSD']-s.loc[0,'RMSD']:+.3f} D2-D0={s.loc[2,'RMSD']-s.loc[0,'RMSD']:+.3f}"
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
    clf[mn]=ba; print(f"  {mn:14s} bal_acc={ba:.3f} conf={M3.tolist()}")
json.dump({"rstar":int(rstar),"cv":cvm,"metricas":rows,"bal_acc":clf},
          open(os.path.join(OUT,"resumo_v8.json"),"w"),indent=2,ensure_ascii=False)

# ---------- gráficos: curva compensada (cima) + assinatura de dano (baixo) ----------
fkhz=fHz/1e3
cor={"Original":"#e8736a","Park":"#2ca02c",f"V8 (r={rstar})":"#1f5fd0"}
for Tt in TEST_TEMPS:
    fig,axes=plt.subplots(2,3,figsize=(19,9),dpi=150,sharex=True)
    lo=min(np.min(Y_v8[np.isclose(T,Tt)]),np.min(y_ref))-4
    hi=max(np.max(Y_v8[np.isclose(T,Tt)]),np.max(y_ref))+4
    for j,d in enumerate([0,1,2]):
        idx=np.where(np.isclose(T,Tt)&(y==d))[0]
        ax=axes[0,j]; ax2=axes[1,j]
        if len(idx)==0: ax.set_title(f"Dano {d} — sem amostra"); continue
        i=idx[0]
        ax.plot(fkhz,y_ref,"--",color="black",lw=1.5,label=f"Referência saudável {REF_TEMP:.0f}°C",zorder=5)
        ax.plot(fkhz,X[i],color=cor["Original"],lw=1.0,alpha=0.5,label="Original")
        ax.plot(fkhz,Y_park[i],color=cor["Park"],lw=1.5,alpha=0.85,label="Park")
        ax.plot(fkhz,Y_v8[i],color=cor[f"V8 (r={rstar})"],lw=1.7,label=f"V8 (r={rstar})")
        ax.set_title(f"Dano {d}"); ax.set_ylim(lo,hi)
        ax2.axhline(0,color="black",lw=1,ls="--")
        ax2.plot(fkhz,Y_park[i]-y_ref,color=cor["Park"],lw=1.2,alpha=0.8,label="Park")
        ax2.plot(fkhz,Y_v8[i]-y_ref,color=cor[f"V8 (r={rstar})"],lw=1.4,label="V8")
        ax2.set_xlabel("Frequência (kHz)"); ax2.set_ylim(-12,12)
        for a in (ax,ax2):
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    axes[0,0].set_ylabel("Impedância (compensada)"); axes[1,0].set_ylabel("Assinatura de dano\n(compensada − referência)")
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=4,frameon=True,bbox_to_anchor=(0.5,-0.03))
    fig.suptitle(f"V8 vs Park — T = {Tt:.0f}°C (TESTE, out-of-sample) — 30–40 kHz",fontsize=16,y=1.0)
    fig.tight_layout(rect=[0,0.04,1,0.97])
    p=os.path.join(OUT,f"v8_T{int(Tt)}C.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("figura:",p)
print("\n✅",OUT)

# -*- coding: utf-8 -*-
"""
ETAPA 6 — BATERIA MULTI-SPLIT: busca do ótimo do V8.1 com estatística robusta
=============================================================================
Objetivo: (1) dar média ± desvio das métricas em vários conjuntos de teste
          (resolve a fragilidade do teste de 24 curvas);
          (2) otimizar os hiperparâmetros de forma HONESTA (CV só no treino).

Método V8.x:
  1) registro por shift multiescala à referência (y_ref = mediana saudável @30°C)
  2) modelo térmico: mediana do resíduo saudável por temperatura -> SVD baixo posto (r)
     -> coeficientes suavizados ao longo de T (novo) -> interpolados em T
  3) compensada = alinhada - m̃(T)      [não depende do resíduo da própria curva]
  4) trava vertical: offset mediano robusto (+ ganho limitado, opcional)

Hiperparâmetros varridos por LOTO-CV no TREINO: r x modo_vertical x suavização_coef.
REF_TEMP=30 fixo em todos os splits (nunca é temp de teste) -> alinhamento em cache.
"""
import os, sys, json, itertools, numpy as np, pandas as pd
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42)
FMIN,FMAX=30.0,40.0; REF_TEMP=30.0
OUT=os.path.join(PROJ,"etapa6_bateria"); os.makedirs(OUT,exist_ok=True)
CACHE=os.path.join(PROJ,"etapa5_final","aligned_cache.npz")

SPLITS=[[-10,20,40,60],[0,25,50,75],[5,35,55,80],[-5,15,45,70],
        [10,45,65,80],[-10,0,50,70],[15,25,55,65],[5,35,60,75]]
RS=[2,3,4,6,8]; VMODES=["offset","offset_gain"]; SMOOTHS=[0,3,5]
CV_STRIDE=3          # usa 1 a cada 3 temperaturas de treino na CV (custo)
CLF_WIN=51

df=pd.read_pickle(BASE).reset_index(drop=True)
df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX)
X=df[fcols].to_numpy(np.float64); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
y_ref=np.median(X[np.isclose(T,REF_TEMP)&(y==0)],axis=0)
A=np.load(CACHE)["A"]; R=A-y_ref[None,:]
d2=np.abs(np.gradient(np.gradient(y_ref))); base_m=(d2<=np.percentile(d2,70))

def pack_thermal(mask):
    """SVD uma vez; todos os r são truncamentos do mesmo SVD."""
    hm=mask&(y==0); temps=np.array(sorted(np.unique(T[hm])))
    if len(temps)<3: return None
    Mm=np.vstack([np.median(R[hm&np.isclose(T,tk)],axis=0) for tk in temps])
    U,S,Vt=np.linalg.svd(Mm,full_matrices=False)
    return {"temps":temps,"Mm":Mm,"Vt":Vt}

def smooth_cols(Cf,w):
    if w<=1: return Cf
    k=np.ones(w)/w; pad=w//2
    return np.column_stack([np.convolve(np.pad(Cf[:,j],(pad,pad),mode="edge"),k,mode="valid")[:len(Cf)]
                            for j in range(Cf.shape[1])])

def make_mh(pk,r,smooth):
    temps=pk["temps"]; re=min(r,len(temps),pk["Vt"].shape[0])
    Vr=pk["Vt"][:re]; Cf=smooth_cols(pk["Mm"]@Vr.T,smooth)
    def mh(t): return np.array([np.interp(t,temps,Cf[:,j]) for j in range(re)])@Vr
    return mh

def vlock(yc,mode):
    off=np.median(y_ref-yc); yc=yc+off
    if mode=="offset_gain":
        a=float(np.clip(np.polyfit(yc[base_m],y_ref[base_m],1)[0],0.97,1.03))
        yc=a*yc+np.median(y_ref-a*yc)
    return yc

def comp(pk,r,smooth,mode,idx):
    mh=make_mh(pk,r,smooth)
    return np.vstack([vlock(A[i]-mh(T[i]),mode) for i in idx])

def sm(Rr,w=CLF_WIN): return np.vstack([v7.moving_average(r,w) for r in Rr])
def nc(Ftr,ytr,Fte):
    cs=sorted(np.unique(ytr)); C=np.vstack([Ftr[ytr==c].mean(0) for c in cs])
    return np.array([cs[int(np.argmin(((C-r)**2).sum(1)))] for r in Fte])
def bal(a,b):
    u=np.unique(a); return float(np.mean([(b[a==c]==c).mean() for c in u]))

rows=[]
for si,test_temps in enumerate(SPLITS,1):
    is_test=np.zeros(len(df),bool)
    for t in test_temps: is_test|=np.isclose(T,float(t))
    is_train=~is_test
    assert not any(np.isclose(REF_TEMP,test_temps)), "ref não pode estar no teste"

    # ---- CV honesta no TREINO p/ escolher (r, modo, smooth) ----
    th=np.array(sorted(np.unique(T[is_train&(y==0)])))[::CV_STRIDE]
    scores={}
    for tk in th:
        keep=is_train&~np.isclose(T,tk); ho=is_train&np.isclose(T,tk)&(y==0)
        if ho.sum()==0: continue
        pk=pack_thermal(keep)
        if pk is None: continue
        idx=np.where(ho)[0]
        for r,mode,smo in itertools.product(RS,VMODES,SMOOTHS):
            Yc=comp(pk,r,smo,mode,idx)
            scores.setdefault((r,mode,smo),[]).append(
                np.mean([v7.rmsd(Yc[j],y_ref) for j in range(len(idx))]))
    cvm={k:float(np.mean(v)) for k,v in scores.items()}
    rstar,mstar,sstar=min(cvm,key=cvm.get)

    # ---- aplica e avalia no TESTE ----
    pk=pack_thermal(is_train)
    Yv8=comp(pk,rstar,sstar,mstar,np.arange(len(X)))
    Ypk=np.vstack([v7.park_single(x,y_ref,fHz) for x in X])
    for nome,Y in [("Park",Ypk),("V8.1",Yv8)]:
        d={"split":si,"test":str(test_temps),"metodo":nome,
           "r":rstar if nome=="V8.1" else None,"modo":mstar if nome=="V8.1" else None,
           "smooth":sstar if nome=="V8.1" else None}
        for c in [0,1,2]:
            ii=np.where(is_test&(y==c))[0]
            d[f"RMSD_D{c}"]=float(np.mean([v7.rmsd(Y[i],y_ref) for i in ii]))
            d[f"CCDM_D{c}"]=float(np.mean([v7.ccdm(Y[i],y_ref) for i in ii]))
        d["sep_RMSD_D1"]=d["RMSD_D1"]-d["RMSD_D0"]; d["sep_RMSD_D2"]=d["RMSD_D2"]-d["RMSD_D0"]
        F=sm(Y-y_ref[None,:]); p=nc(F[is_train],y[is_train],F[is_test])
        d["bal_acc"]=bal(y[is_test],p)
        rows.append(d)
    print(f"split {si} {test_temps}: r*={rstar} modo={mstar} smooth={sstar} | "
          f"V8.1 RMSD_D0={rows[-1]['RMSD_D0']:.3f} bal={rows[-1]['bal_acc']:.3f} | "
          f"Park RMSD_D0={rows[-2]['RMSD_D0']:.3f} bal={rows[-2]['bal_acc']:.3f}")

res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,"bateria_multisplit.csv"),index=False)
print("\n"+"="*84); print(f"RESUMO — média ± desvio sobre {len(SPLITS)} splits (out-of-sample)"); print("="*84)
cols=["RMSD_D0","CCDM_D0","RMSD_D1","RMSD_D2","sep_RMSD_D1","sep_RMSD_D2","bal_acc"]
agg=res.groupby("metodo")[cols].agg(["mean","std"]).round(4)
print(agg.to_string())
print("\nVitórias do V8.1 sobre o Park, split a split:")
for si in sorted(res["split"].unique()):
    a=res[(res.split==si)&(res.metodo=="V8.1")].iloc[0]; b=res[(res.split==si)&(res.metodo=="Park")].iloc[0]
    print(f"  split {si}: RMSD_D0 {a.RMSD_D0:.3f} vs {b.RMSD_D0:.3f} "
          f"{'V8.1' if a.RMSD_D0<b.RMSD_D0 else 'Park':>5} | bal_acc {a.bal_acc:.3f} vs {b.bal_acc:.3f} "
          f"{'V8.1' if a.bal_acc>b.bal_acc else ('empate' if a.bal_acc==b.bal_acc else 'Park')}")
w_rmsd=int((res[res.metodo=='V8.1'].set_index('split').RMSD_D0 < res[res.metodo=='Park'].set_index('split').RMSD_D0).sum())
w_bal=int((res[res.metodo=='V8.1'].set_index('split').bal_acc > res[res.metodo=='Park'].set_index('split').bal_acc).sum())
t_bal=int((res[res.metodo=='V8.1'].set_index('split').bal_acc == res[res.metodo=='Park'].set_index('split').bal_acc).sum())
print(f"\nPlacar: RMSD_D0 V8.1 vence {w_rmsd}/{len(SPLITS)} | bal_acc V8.1 vence {w_bal}, empata {t_bal}, perde {len(SPLITS)-w_bal-t_bal}")
print(f"\n✅ {os.path.join(OUT,'bateria_multisplit.csv')}")

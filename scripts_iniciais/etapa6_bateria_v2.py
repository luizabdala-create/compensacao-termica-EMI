# -*- coding: utf-8 -*-
"""
ETAPA 6 v2 — BATERIA MULTI-SPLIT CORRIGIDA + grade estendida
============================================================
Correções sobre a v1:
 (a) splits de teste só com temperaturas que têm SAUDÁVEL **e** DANO
     ({-10,0,10,20,40,50,60,70,80}; 30 é a referência, sempre no treino).
     As temps múltiplas ímpares de 5 (-5,5,15,...) só têm dano -> RMSD_D0 era NaN.
     Elas continuam no TREINO (contribuem centróides de dano).
 (b) grade estendida: r ate 16 e suavização ate 9 (o ótimo batia na borda: r*=8, smooth=5).
Hiperparâmetros por LOTO-CV só no TREINO, por split. Estatística sobre os splits.
"""
import os, sys, json, itertools, numpy as np, pandas as pd
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42)
FMIN,FMAX=30.0,40.0; REF_TEMP=30.0
OUT=os.path.join(PROJ,"etapa6_bateria"); os.makedirs(OUT,exist_ok=True)
CACHE=os.path.join(PROJ,"etapa5_final","aligned_cache.npz")

# só temps com saudável E dano (30 = referência, fica sempre no treino)
SPLITS=[[-10,20,40,60],[0,10,50,70],[-10,40,80],[0,20,60,80],
        [10,40,70],[-10,50,80],[0,40,60],[10,20,70,80],[-10,0,60],[20,50,80]]
RS=[4,6,8,10,12,16]; VMODES=["offset","offset_gain"]; SMOOTHS=[0,3,5,7,9]
CV_STRIDE=3; CLF_WIN=51

df=pd.read_pickle(BASE).reset_index(drop=True)
df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX)
X=df[fcols].to_numpy(np.float64); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
y_ref=np.median(X[np.isclose(T,REF_TEMP)&(y==0)],axis=0)
A=np.load(CACHE)["A"]; R=A-y_ref[None,:]
d2=np.abs(np.gradient(np.gradient(y_ref))); base_m=(d2<=np.percentile(d2,70))

def pack_thermal(mask):
    hm=mask&(y==0); temps=np.array(sorted(np.unique(T[hm])))
    if len(temps)<3: return None
    Mm=np.vstack([np.median(R[hm&np.isclose(T,tk)],axis=0) for tk in temps])
    _,_,Vt=np.linalg.svd(Mm,full_matrices=False)
    return {"temps":temps,"Mm":Mm,"Vt":Vt}
def smooth_cols(Cf,w):
    if w<=1: return Cf
    k=np.ones(w)/w; pad=w//2
    return np.column_stack([np.convolve(np.pad(Cf[:,j],(pad,pad),mode="edge"),k,mode="valid")[:len(Cf)]
                            for j in range(Cf.shape[1])])
def make_mh(pk,r,smo):
    temps=pk["temps"]; re=min(r,len(temps),pk["Vt"].shape[0])
    Vr=pk["Vt"][:re]; Cf=smooth_cols(pk["Mm"]@Vr.T,smo)
    return lambda t:(np.array([np.interp(t,temps,Cf[:,j]) for j in range(re)])@Vr)
def vlock(yc,mode):
    yc=yc+np.median(y_ref-yc)
    if mode=="offset_gain":
        a=float(np.clip(np.polyfit(yc[base_m],y_ref[base_m],1)[0],0.97,1.03))
        yc=a*yc+np.median(y_ref-a*yc)
    return yc
def comp(pk,r,smo,mode,idx):
    mh=make_mh(pk,r,smo); return np.vstack([vlock(A[i]-mh(T[i]),mode) for i in idx])
def sm(Rr,w=CLF_WIN): return np.vstack([v7.moving_average(r,w) for r in Rr])
def nc(Ftr,ytr,Fte):
    cs=sorted(np.unique(ytr)); C=np.vstack([Ftr[ytr==c].mean(0) for c in cs])
    return np.array([cs[int(np.argmin(((C-r)**2).sum(1)))] for r in Fte])
def bal(a,b): return float(np.mean([(b[a==c]==c).mean() for c in np.unique(a)]))

rows=[]; hp=[]
for si,tt in enumerate(SPLITS,1):
    is_test=np.zeros(len(df),bool)
    for t in tt: is_test|=np.isclose(T,float(t))
    is_train=~is_test
    th=np.array(sorted(np.unique(T[is_train&(y==0)])))[::CV_STRIDE]
    sc={}
    for tk in th:
        keep=is_train&~np.isclose(T,tk); ho=is_train&np.isclose(T,tk)&(y==0)
        if ho.sum()==0: continue
        pk=pack_thermal(keep)
        if pk is None: continue
        idx=np.where(ho)[0]
        for r,mode,smo in itertools.product(RS,VMODES,SMOOTHS):
            Yc=comp(pk,r,smo,mode,idx)
            sc.setdefault((r,mode,smo),[]).append(np.mean([v7.rmsd(Yc[j],y_ref) for j in range(len(idx))]))
    cvm={k:float(np.mean(v)) for k,v in sc.items()}
    rstar,mstar,sstar=min(cvm,key=cvm.get)
    hp.append({"split":si,"r":rstar,"modo":mstar,"smooth":sstar,"cv":cvm[(rstar,mstar,sstar)]})

    pk=pack_thermal(is_train)
    Yv=comp(pk,rstar,sstar,mstar,np.arange(len(X)))
    Yp=np.vstack([v7.park_single(x,y_ref,fHz) for x in X])
    for nome,Y in [("Park",Yp),("V8.1",Yv)]:
        d={"split":si,"test":str(tt),"metodo":nome}
        for c in [0,1,2]:
            ii=np.where(is_test&(y==c))[0]
            d[f"RMSD_D{c}"]=float(np.mean([v7.rmsd(Y[i],y_ref) for i in ii])) if len(ii) else np.nan
            d[f"CCDM_D{c}"]=float(np.mean([v7.ccdm(Y[i],y_ref) for i in ii])) if len(ii) else np.nan
        d["sep_RMSD_D1"]=d["RMSD_D1"]-d["RMSD_D0"]; d["sep_RMSD_D2"]=d["RMSD_D2"]-d["RMSD_D0"]
        F=sm(Y-y_ref[None,:]); p=nc(F[is_train],y[is_train],F[is_test])
        d["bal_acc"]=bal(y[is_test],p); d["n_test"]=int(is_test.sum())
        rows.append(d)
    print(f"split {si} {tt}: r*={rstar} modo={mstar} smooth={sstar} | "
          f"V8.1 RMSD_D0={rows[-1]['RMSD_D0']:.3f} bal={rows[-1]['bal_acc']:.3f} | "
          f"Park RMSD_D0={rows[-2]['RMSD_D0']:.3f} bal={rows[-2]['bal_acc']:.3f}")

res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,"bateria_v2.csv"),index=False)
pd.DataFrame(hp).to_csv(os.path.join(OUT,"hiperparametros_v2.csv"),index=False)
cols=["RMSD_D0","CCDM_D0","RMSD_D1","RMSD_D2","sep_RMSD_D1","sep_RMSD_D2","bal_acc"]
print("\n"+"="*86); print(f"RESUMO — média ± desvio sobre {len(SPLITS)} splits (out-of-sample)"); print("="*86)
print(res.groupby("metodo")[cols].agg(["mean","std"]).round(4).to_string())
a=res[res.metodo=="V8.1"].set_index("split"); b=res[res.metodo=="Park"].set_index("split")
print("\nsplit a split:")
for si in a.index:
    print(f"  {si:2d} {a.loc[si,'test']:<20} RMSD_D0 {a.loc[si,'RMSD_D0']:.3f} vs {b.loc[si,'RMSD_D0']:.3f}"
          f" {'V8.1' if a.loc[si,'RMSD_D0']<b.loc[si,'RMSD_D0'] else 'Park':>5}"
          f" | bal {a.loc[si,'bal_acc']:.3f} vs {b.loc[si,'bal_acc']:.3f}"
          f" {'V8.1' if a.loc[si,'bal_acc']>b.loc[si,'bal_acc'] else ('empate' if a.loc[si,'bal_acc']==b.loc[si,'bal_acc'] else 'Park')}")
print(f"\nPlacar RMSD_D0: V8.1 {int((a.RMSD_D0<b.RMSD_D0).sum())}/{len(a)}")
print(f"Placar bal_acc: V8.1 vence {int((a.bal_acc>b.bal_acc).sum())}, empata {int((a.bal_acc==b.bal_acc).sum())}, perde {int((a.bal_acc<b.bal_acc).sum())}")
print("\nHiperparâmetros escolhidos (por CV no treino):")
print(pd.DataFrame(hp).to_string(index=False))
print(f"\n✅ {OUT}")

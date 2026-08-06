# -*- coding: utf-8 -*-
"""
ETAPA 6 final — confirma o ÓTIMO (grade estendida) + estatística pareada
========================================================================
 (a) estende r ate 32 e suavização ate 11 (r* batia na borda 16) -> confirma platô
 (b) teste t pareado V8.1 vs Park em cada métrica, sobre os 10 splits
 (c) figura final: comparação pareada
"""
import os, sys, itertools, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42)
FMIN,FMAX=30.0,40.0; REF_TEMP=30.0
OUT=os.path.join(PROJ,"etapa6_bateria"); os.makedirs(OUT,exist_ok=True)
CACHE=os.path.join(PROJ,"etapa5_final","aligned_cache.npz")
SPLITS=[[-10,20,40,60],[0,10,50,70],[-10,40,80],[0,20,60,80],
        [10,40,70],[-10,50,80],[0,40,60],[10,20,70,80],[-10,0,60],[20,50,80]]
RS=[8,12,16,24,32]; VMODES=["offset_gain"]; SMOOTHS=[5,7,9,11]
CV_STRIDE=3; CLF_WIN=51
plt.rcParams.update({"font.family":"serif","font.size":12})

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
    _,_,Vt=np.linalg.svd(Mm,full_matrices=False)
    return {"temps":temps,"Mm":Mm,"Vt":Vt}
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
def comp(pk,r,smo,idx):
    f=mh_fn(pk,r,smo); return np.vstack([vlock(A[i]-f(T[i])) for i in idx])
def sm(Rr,w=CLF_WIN): return np.vstack([v7.moving_average(r,w) for r in Rr])
def nc(Ftr,ytr,Fte):
    cs=sorted(np.unique(ytr)); C=np.vstack([Ftr[ytr==c].mean(0) for c in cs])
    return np.array([cs[int(np.argmin(((C-r)**2).sum(1)))] for r in Fte])
def bal(a,b): return float(np.mean([(b[a==c]==c).mean() for c in np.unique(a)]))

rows=[]; hps=[]
for si,tt in enumerate(SPLITS,1):
    is_test=np.zeros(len(df),bool)
    for t in tt: is_test|=np.isclose(T,float(t))
    is_train=~is_test
    th=np.array(sorted(np.unique(T[is_train&(y==0)])))[::CV_STRIDE]
    sc={}
    for tk in th:
        keep=is_train&~np.isclose(T,tk); ho=is_train&np.isclose(T,tk)&(y==0)
        if ho.sum()==0: continue
        pk=pack(keep); idx=np.where(ho)[0]
        for r,smo in itertools.product(RS,SMOOTHS):
            Yc=comp(pk,r,smo,idx)
            sc.setdefault((r,smo),[]).append(np.mean([v7.rmsd(Yc[j],y_ref) for j in range(len(idx))]))
    cvm={k:float(np.mean(v)) for k,v in sc.items()}
    rstar,sstar=min(cvm,key=cvm.get); hps.append({"split":si,"r":rstar,"smooth":sstar,"cv":cvm[(rstar,sstar)]})
    pk=pack(is_train); Yv=comp(pk,rstar,sstar,np.arange(len(X)))
    Yp=np.vstack([v7.park_single(x,y_ref,fHz) for x in X])
    for nome,Y in [("Park",Yp),("V8.1",Yv)]:
        d={"split":si,"metodo":nome}
        for c in [0,1,2]:
            ii=np.where(is_test&(y==c))[0]
            d[f"RMSD_D{c}"]=float(np.mean([v7.rmsd(Y[i],y_ref) for i in ii]))
            d[f"CCDM_D{c}"]=float(np.mean([v7.ccdm(Y[i],y_ref) for i in ii]))
        d["sep_RMSD_D1"]=d["RMSD_D1"]-d["RMSD_D0"]; d["sep_RMSD_D2"]=d["RMSD_D2"]-d["RMSD_D0"]
        F=sm(Y-y_ref[None,:]); d["bal_acc"]=bal(y[is_test],nc(F[is_train],y[is_train],F[is_test]))
        rows.append(d)
    print(f"split {si} {tt}: r*={rstar} smooth={sstar} | V8.1 RMSD={rows[-1]['RMSD_D0']:.3f} bal={rows[-1]['bal_acc']:.3f}"
          f" | Park RMSD={rows[-2]['RMSD_D0']:.3f} bal={rows[-2]['bal_acc']:.3f}")

res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,"final_otimo.csv"),index=False)
hp=pd.DataFrame(hps); print("\nHiperparâmetros por CV:"); print(hp.to_string(index=False))
print(f"  r escolhido: {sorted(hp.r.unique())}  (grade ia ate {max(RS)}) -> {'PLATÔ confirmado' if hp.r.max()<max(RS) else 'ainda na borda'}")

def ttest_pair(a,b):
    d=np.asarray(a)-np.asarray(b); n=len(d); m=d.mean(); s=d.std(ddof=1)
    t=m/(s/np.sqrt(n))
    try:
        from scipy import stats; p=float(2*(1-stats.t.cdf(abs(t),n-1)))
    except Exception:
        p=float("nan")
    return m,t,p

A_=res[res.metodo=="V8.1"].set_index("split"); B_=res[res.metodo=="Park"].set_index("split")
print("\n"+"="*88); print("TESTE t PAREADO (V8.1 - Park) sobre 10 splits"); print("="*88)
print(f"{'métrica':16s} {'V8.1':>16s} {'Park':>16s} {'dif média':>11s} {'t':>7s} {'p':>8s}  concl.")
for met,melhor in [("RMSD_D0","menor"),("CCDM_D0","menor"),("sep_RMSD_D1","maior"),
                   ("sep_RMSD_D2","maior"),("bal_acc","maior")]:
    m,t,p=ttest_pair(A_[met],B_[met])
    sig = "SIGNIFICATIVO" if (not np.isnan(p) and p<0.05) else "n.s."
    print(f"{met:16s} {A_[met].mean():7.4f}±{A_[met].std():.4f} {B_[met].mean():7.4f}±{B_[met].std():.4f} "
          f"{m:+11.4f} {t:7.2f} {p:8.4f}  {sig}")

fig,axes=plt.subplots(1,3,figsize=(16,5),dpi=150)
for ax,met,tit in zip(axes,["RMSD_D0","CCDM_D0","bal_acc"],
                      ["(A) RMSD saudável ↓","(A) CCDM saudável ↓","(B) Balanced accuracy ↑"]):
    for si in A_.index:
        ax.plot([0,1],[B_.loc[si,met],A_.loc[si,met]],"-o",color="gray",alpha=.55,ms=5)
    ax.plot([0,1],[B_[met].mean(),A_[met].mean()],"-o",color="crimson",lw=3,ms=10,label="média")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Park","V8.1"]); ax.set_title(tit)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(alpha=.25)
axes[0].legend()
fig.suptitle("V8.1 vs Park — 10 splits de teste independentes (out-of-sample)",fontsize=15)
fig.tight_layout()
p=os.path.join(OUT,"comparacao_pareada.png"); fig.savefig(p,dpi=150,bbox_inches="tight",facecolor="white")
print("\nfigura:",p); print("✅",OUT)

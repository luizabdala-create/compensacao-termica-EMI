# -*- coding: utf-8 -*-
"""
V8.1 — melhora do CLASSIFICADOR (compensação inalterada)
========================================================
O dano 2 tem assinatura fina/localizada nas ressonâncias; a suavização de 51 pontos
pode estar apagando-a. Varre (janela de suavização × métrica de distância) e escolhe
por leave-one-temperature-out SÓ no treino. Compensação V8.1 (r=4, offset_gain) fixa.
"""
import os, sys, json, numpy as np, pandas as pd
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa5_v81"); os.makedirs(OUT,exist_ok=True)
CACHE=os.path.join(PROJ,"etapa5_final","aligned_cache.npz")
TEST_TEMPS=[-10.0,20.0,40.0,60.0]; REF_TEMP=30.0; R_STAR=4
WINS=[5,11,21,51,101,201]; METRICS=["euclid","corr","abs"]

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
d2=np.abs(np.gradient(np.gradient(y_ref))); w_base=(d2<=np.percentile(d2,70)).astype(float)

def build_thermal(mask,r):
    hm=mask&(y==0); temps=np.array(sorted(np.unique(T[hm])))
    Mm=np.vstack([np.median(R[hm&np.isclose(T,tk)],axis=0) for tk in temps])
    re=min(r,len(temps),Mm.shape[1]); _,_,Vt=np.linalg.svd(Mm,full_matrices=False)
    Vr=Vt[:re]; Cf=Mm@Vr.T
    return lambda t:(np.array([np.interp(t,temps,Cf[:,j]) for j in range(re)])@Vr)
def vlock(yc):
    off=np.median(y_ref-yc); yc=yc+off
    m=w_base>0; a=float(np.clip(np.polyfit(yc[m],y_ref[m],1)[0],0.97,1.03))
    return a*yc+np.median(y_ref-a*yc)
def compensate(mask,idx):
    mh=build_thermal(mask,R_STAR); return np.vstack([vlock(A[i]-mh(T[i])) for i in idx])

def feat(Y,win):
    Rr=Y-y_ref[None,:]
    return np.vstack([v7.moving_average(r,win) for r in Rr]) if win>1 else Rr
def classify(Ftr,ytr,Fte,metric):
    cs=sorted(np.unique(ytr))
    if metric=="abs":
        C=np.vstack([np.abs(Ftr[ytr==c]).mean(0) for c in cs]); Fte=np.abs(Fte)
    else:
        C=np.vstack([Ftr[ytr==c].mean(0) for c in cs])
    out=[]
    for r in Fte:
        if metric=="corr":
            def n(v): v=v-v.mean(); return v/(np.linalg.norm(v)+1e-12)
            out.append(cs[int(np.argmax(np.array([n(c)@n(r) for c in C])))])
        else:
            out.append(cs[int(np.argmin(((C-r)**2).sum(1)))])
    return np.array(out)
def bal(a,b): return float(np.mean([(b[a==c]==c).mean() for c in np.unique(a)]))

# CV no treino
train_temps=np.array(sorted(np.unique(T[is_train])))
cv={}
for tk in train_temps:
    ho=is_train&np.isclose(T,tk); keep=is_train&~np.isclose(T,tk)
    if ho.sum()==0 or len(np.unique(y[ho]))<2: continue
    Yk=compensate(keep,np.where(keep)[0]); Yh=compensate(keep,np.where(ho)[0])
    for w in WINS:
        Ftr=feat(Yk,w); Fte=feat(Yh,w)
        for mt in METRICS:
            cv.setdefault((w,mt),[]).append(bal(y[ho],classify(Ftr,y[keep],Fte,mt)))
cvm={k:float(np.mean(v)) for k,v in cv.items()}
best=max(cvm,key=cvm.get)
print("CV(treino) balanced-acc por (janela, métrica):")
for w in WINS:
    print(f"   win={w:3d}: "+" | ".join(f"{m}={cvm.get((w,m),float('nan')):.3f}" for m in METRICS))
print(f"   >>> escolhido: janela={best[0]}, métrica={best[1]} (CV={cvm[best]:.3f})")

# teste
Yall=compensate(is_train,np.arange(len(X)))
Ypark=np.vstack([v7.park_single(x,y_ref,fHz) for x in X])
w,mt=best
for nome,Y in [("Park",Ypark),("V8.1",Yall)]:
    F=feat(Y,w); p=classify(F[is_train],y[is_train],F[is_test],mt); ba=bal(y[is_test],p)
    M=np.zeros((3,3),int)
    for a,b in zip(y[is_test],p): M[a,b]+=1
    print(f"\n[{nome}] classificador escolhido (win={w},{mt}): bal_acc={ba:.3f}")
    for r in M: print("     ",r)
json.dump({"best":[int(best[0]),best[1]],"cv":{f"{k[0]}_{k[1]}":v for k,v in cvm.items()}},
          open(os.path.join(OUT,"classif_sweep.json"),"w"),indent=2,ensure_ascii=False)

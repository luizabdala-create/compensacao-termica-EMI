# -*- coding: utf-8 -*-
"""
ETAPA 5 — EXP 7: AE NÃO-LINEAR (rede) que bate o Park — versão honesta
======================================================================
Contribuição ML: autoencoder não-linear que modela a variedade (manifold) TÉRMICA
das curvas SAUDÁVEIS; o resíduo fora-da-variedade é a assinatura de dano.

Pipeline (sem Park, sem usar dano na parte térmica):
  1) registro por shift multiescala a y_ref
  2) R = alinhado - y_ref ; reduz para M coefs na base PCA do SAUDÁVEL de treino
  3) AE não-linear (M->bottleneck->M) treinado SÓ no saudável de treino
  4) feature de dano = c - AE(c)  (resíduo fora da variedade saudável)
  5) classifica (nearest-centroid), centróides do treino
Hiperparâmetros (M, bottleneck) escolhidos por LOTO-CV no TREINO. Teste só no fim.
Compara com Park e com o AE-linear (PCA-removal).
"""
import os, sys, json, numpy as np, pandas as pd
import torch, torch.nn as nn
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42); torch.manual_seed(42)
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa5_exp7"); os.makedirs(OUT,exist_ok=True)
SPLITS={
 "B_ref30":{"train":[0,10,30,50,70],"test":[-10,20,40,60],"ref":30.0},
 "A_ref20":{"train":[0,10,20,40,60],"test":[-10,30,50,70],"ref":20.0},
}
GRID=[(20,4),(20,6),(30,4),(30,6),(30,8),(40,6)]  # (M coefs, bottleneck)

def load():
    df=pd.read_pickle(BASE).reset_index(drop=True)
    df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
    df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
    fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX); return df[["temperatura_c","falha"]+fcols].copy(),fcols,fHz
def subset(df,temps):
    m=np.zeros(len(df),bool)
    for T in temps: m|=np.isclose(df["temperatura_c"],T)
    return df[m].copy()
def align_to_ref(X,y_ref,fHz):
    Y=np.zeros_like(X)
    for i in range(len(X)):
        tau=v7.estimate_shift_multiscale(X[i],y_ref,fHz,max_frac=0.14,n_coarse=101,n_fine=81,
             prior_tau=0.0,prior_penalty=0.015,min_improvement=0.0,return_info=False)
        Y[i]=v7.shift_interp(X[i],fHz,tau)
    return Y
def nc(Ftr,ytr,Fte):
    cs=sorted(np.unique(ytr)); C=np.vstack([Ftr[ytr==c].mean(0) for c in cs])
    return np.array([cs[int(np.argmin(((C-r)**2).sum(1)))] for r in Fte])
def bal(y,p): return float(np.mean([(p[y==c]==c).mean() for c in np.unique(y)]))
def conf(y,p):
    M=np.zeros((3,3),int)
    for a,b in zip(y,p): M[int(a),int(b)]+=1
    return M

class AE(nn.Module):
    def __init__(self,M,bott):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(M,32),nn.GELU(),nn.Linear(32,bott))
        self.dec=nn.Sequential(nn.Linear(bott,32),nn.GELU(),nn.Linear(32,M))
    def forward(self,x): return self.dec(self.enc(x))

def train_ae(Ch,M,bott,epochs=400):
    """treina só nos coefs saudáveis Ch (n x M)"""
    torch.manual_seed(42)
    model=AE(M,bott); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-3)
    X=torch.tensor(Ch,dtype=torch.float32)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out=model(X); loss=((out-X)**2).mean()
        loss.backward(); opt.step()
    model.eval(); return model

def pca_coefs(Rtr,ytr,Rte,M):
    Rh=Rtr[ytr==0]; mu=Rh.mean(0)
    _,_,Vt=np.linalg.svd(Rh-mu,full_matrices=False); Vm=Vt[:M]
    Ctr=(Rtr-mu)@Vm.T; Cte=(Rte-mu)@Vm.T
    # normaliza por desvio dos coefs saudáveis (estabiliza AE)
    sd=Ctr[ytr==0].std(0)+1e-8
    return Ctr/sd, Cte/sd

def ae_feature(Ctr,ytr,Cte,M,bott):
    model=train_ae(Ctr[ytr==0],M,bott)
    with torch.no_grad():
        Ftr=Ctr-model(torch.tensor(Ctr,dtype=torch.float32)).numpy()
        Fte=Cte-model(torch.tensor(Cte,dtype=torch.float32)).numpy()
    return Ftr,Fte

df,fcols,fHz=load(); rep=[]
for name,cfg in SPLITS.items():
    print("="*92); print(f"SPLIT {name}"); print("="*92)
    df_tr=subset(df,cfg["train"]).reset_index(drop=True); df_te=subset(df,cfg["test"]).reset_index(drop=True)
    v7._CALIBRATION_CACHE.clear(); v7.REF_TEMP=cfg["ref"]
    hbt,tt,y_ref,_=v7.get_healthy_references_by_temperature(df_tr,fcols,cfg["ref"])
    Xtr=df_tr[fcols].to_numpy(np.float64); ytr=df_tr["falha"].to_numpy(int); Ttr=df_tr["temperatura_c"].to_numpy(float)
    Xte=df_te[fcols].to_numpy(np.float64); yte=df_te["falha"].to_numpy(int)
    Rtr=align_to_ref(Xtr,y_ref,fHz)-y_ref[None,:]; Rte=align_to_ref(Xte,y_ref,fHz)-y_ref[None,:]

    # CV LOTO no treino p/ escolher (M,bott)
    train_temps=sorted(np.unique(Ttr)); cvsc={g:[] for g in GRID}
    for th in train_temps:
        ho=np.isclose(Ttr,th); ke=~ho
        for (M,bott) in GRID:
            Ctr,Cho=pca_coefs(Rtr[ke],ytr[ke],Rtr[ho],M)
            Ftr,Fho=ae_feature(Ctr,ytr[ke],Cho,M,bott)
            cvsc[(M,bott)].append(bal(ytr[ho],nc(Ftr,ytr[ke],Fho)))
    cvm={g:float(np.mean(v)) for g,v in cvsc.items() if len(v)}
    gstar=max(cvm,key=cvm.get)
    print("CV(treino) por (M,bottleneck):")
    for g in GRID: print(f"   M={g[0]:2d} bott={g[1]}: CV={cvm[g]:.3f}"+("  <= escolhido" if g==gstar else ""))

    # Park baseline
    Pk_tr=np.vstack([v7.park_single(x,y_ref,fHz) for x in Xtr]); Pk_te=np.vstack([v7.park_single(x,y_ref,fHz) for x in Xte])
    def sm(R): return np.vstack([v7.moving_average(r,51) for r in R])
    park_ba=bal(yte,nc(sm(Pk_tr-y_ref[None,:]),ytr,sm(Pk_te-y_ref[None,:])))

    # teste no g*
    M,bott=gstar
    Ctr,Cte=pca_coefs(Rtr,ytr,Rte,M); Ftr,Fte=ae_feature(Ctr,ytr,Cte,M,bott)
    p=nc(Ftr,ytr,Fte); ae_ba=bal(yte,p)
    print(f"\n>>> TESTE: AE não-linear(M={M},bott={bott}) = {ae_ba:.3f} | Park = {park_ba:.3f} | Δ={ae_ba-park_ba:+.3f}")
    for r in conf(yte,p): print("      ",r)
    rep.append({"name":name,"gstar":list(gstar),"ae_test":ae_ba,"park_test":park_ba,"conf":conf(yte,p).tolist(),"cv":{f"{a}_{b}":cvm[(a,b)] for (a,b) in cvm}})

with open(os.path.join(OUT,"exp7.json"),"w",encoding="utf-8") as f: json.dump(rep,f,indent=2,ensure_ascii=False)
print(f"\n✅ salvo em {os.path.join(OUT,'exp7.json')}")

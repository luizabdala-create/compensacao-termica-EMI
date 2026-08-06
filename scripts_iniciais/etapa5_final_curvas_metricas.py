# -*- coding: utf-8 -*-
"""
ETAPA 5 — ENTREGÁVEL: curvas + RMSD/CCDM + classificação, out-of-sample
=======================================================================
Métodos comparados (todos avaliados SÓ nas temperaturas de TESTE):
  Original | Park | AE-linear (remoção do subespaço térmico saudável) | AE não-linear

Treino do modelo térmico: TODAS as temperaturas SAUDÁVEIS que não são de teste
(~84 curvas). Isso NÃO é leakage — temps de teste nunca entram — e é a vantagem
legítima do ML sobre o Park (que só usa a referência).

Curva compensada de um método AE = y_ref + (resíduo fora da variedade térmica saudável)
  -> saudável tende a y_ref (A cai); dano preserva sua assinatura (B se mantém).

Saídas: CSV de métricas, PNGs de curvas por temperatura de teste.
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch, torch.nn as nn
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
np.random.seed(42); torch.manual_seed(42)
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa5_final"); os.makedirs(OUT,exist_ok=True)
CACHE=os.path.join(OUT,"aligned_cache.npz")

TEST_TEMPS=[-10.0,20.0,40.0,60.0]   # todas têm dano 0/1/2
REF_TEMP=30.0
K_LINEAR=5      # k* escolhido por CV no treino (exp6)
M_COEF, BOTT = 30, 6

plt.rcParams.update({"font.family":"serif","font.size":13,"axes.labelsize":14,
                     "axes.titlesize":15,"legend.fontsize":11})

def load():
    df=pd.read_pickle(BASE).reset_index(drop=True)
    df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
    df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
    fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX)
    return df[["temperatura_c","falha"]+fcols].copy(),fcols,fHz

df,fcols,fHz=load()
X=df[fcols].to_numpy(np.float64); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
is_test=np.zeros(len(df),bool)
for t in TEST_TEMPS: is_test|=np.isclose(T,t)
is_train=~is_test

# referência: mediana saudável em REF_TEMP (REF_TEMP é de TREINO)
assert not any(np.isclose(REF_TEMP,TEST_TEMPS)), "REF_TEMP não pode ser temp de teste"
y_ref=np.median(X[np.isclose(T,REF_TEMP)&(y==0)],axis=0)
print(f"Referência: mediana saudável em {REF_TEMP}°C (n={int((np.isclose(T,REF_TEMP)&(y==0)).sum())})")
print(f"Treino: {int(is_train.sum())} curvas ({int((is_train&(y==0)).sum())} saudáveis) | Teste: {int(is_test.sum())} curvas")

# ---------- registro (alinhamento) com cache ----------
if os.path.exists(CACHE):
    A=np.load(CACHE)["A"]; print("alinhamento carregado do cache")
else:
    print("alinhando curvas (shift multiescala)...")
    A=np.zeros_like(X)
    for i in range(len(X)):
        tau=v7.estimate_shift_multiscale(X[i],y_ref,fHz,max_frac=0.14,n_coarse=101,n_fine=81,
              prior_tau=0.0,prior_penalty=0.015,min_improvement=0.0,return_info=False)
        A[i]=v7.shift_interp(X[i],fHz,tau)
        if (i+1)%40==0: print(f"  {i+1}/{len(X)}")
    np.savez_compressed(CACHE,A=A)

R=A-y_ref[None,:]                      # resíduo pós-registro
Rh_tr=R[is_train&(y==0)]               # saudáveis de TREINO
mu=Rh_tr.mean(0)
_,_,Vt=np.linalg.svd(Rh_tr-mu,full_matrices=False)

# ---------- AE-linear: remove k comps térmicas ----------
Vk=Vt[:K_LINEAR]
Rc=R-mu
R_lin=Rc-(Rc@Vk.T)@Vk                  # resíduo de dano
Y_lin=y_ref[None,:]+R_lin              # curva compensada

# ---------- AE não-linear em espaço reduzido ----------
Vm=Vt[:M_COEF]
Cc=(Rc@Vm.T); sd=Cc[is_train&(y==0)].std(0)+1e-8; Cn=Cc/sd
R_perp=Rc-Cc@Vm                        # parte fora do top-M (preservada)
class AE(nn.Module):
    def __init__(s,M,b):
        super().__init__(); s.enc=nn.Sequential(nn.Linear(M,32),nn.GELU(),nn.Linear(32,b))
        s.dec=nn.Sequential(nn.Linear(b,32),nn.GELU(),nn.Linear(32,M))
    def forward(s,x): return s.dec(s.enc(x))
torch.manual_seed(42)
ae=AE(M_COEF,BOTT); opt=torch.optim.AdamW(ae.parameters(),lr=3e-3,weight_decay=1e-3)
Xtr_t=torch.tensor(Cn[is_train&(y==0)],dtype=torch.float32)
for ep in range(1500):
    ae.train(); opt.zero_grad(); loss=((ae(Xtr_t)-Xtr_t)**2).mean(); loss.backward(); opt.step()
ae.eval()
with torch.no_grad(): Chat=ae(torch.tensor(Cn,dtype=torch.float32)).numpy()
D=(Cn-Chat)*sd                          # coefs de dano (fora da variedade saudável)
R_nl=R_perp+D@Vm
Y_nl=y_ref[None,:]+R_nl
print(f"AE não-linear treinado (M={M_COEF}, bottleneck={BOTT}, loss final={float(loss):.5f})")

# ---------- Park ----------
Y_park=np.vstack([v7.park_single(x,y_ref,fHz) for x in X])

METHODS={"Original":X,"Park":Y_park,f"AE-linear(k={K_LINEAR})":Y_lin,f"AE nao-linear":Y_nl}

# ---------- métricas RMSD/CCDM out-of-sample ----------
rows=[]
for mname,Y in METHODS.items():
    for d in [0,1,2]:
        idx=np.where(is_test&(y==d))[0]
        rows.append({"Metodo":mname,"falha":d,"n":len(idx),
                     "RMSD":float(np.mean([v7.rmsd(Y[i],y_ref) for i in idx])),
                     "CCDM":float(np.mean([v7.ccdm(Y[i],y_ref) for i in idx]))})
met=pd.DataFrame(rows)
met.to_csv(os.path.join(OUT,"metricas_RMSD_CCDM_teste.csv"),index=False)
print("\n"+"="*78); print("RMSD e CCDM — SÓ TEMPERATURAS DE TESTE (out-of-sample)"); print("="*78)
piv=met.pivot(index="Metodo",columns="falha",values=["RMSD","CCDM"]).round(4)
print(piv.to_string())

# separação (B): dano deve ficar LONGE do saudável
print("\nSeparação (dano - saudável) — maior = dano mais preservado:")
for mname in METHODS:
    s=met[met["Metodo"]==mname].set_index("falha")
    print(f"  {mname:20s} RMSD: D1-D0={s.loc[1,'RMSD']-s.loc[0,'RMSD']:+.3f} | D2-D0={s.loc[2,'RMSD']-s.loc[0,'RMSD']:+.3f}"
          f" || CCDM: D1-D0={s.loc[1,'CCDM']-s.loc[0,'CCDM']:+.4f} | D2-D0={s.loc[2,'CCDM']-s.loc[0,'CCDM']:+.4f}")

# ---------- classificação out-of-sample ----------
def sm(Rr,w=51): return np.vstack([v7.moving_average(r,w) for r in Rr])
def nc(Ftr,ytr,Fte):
    cs=sorted(np.unique(ytr)); C=np.vstack([Ftr[ytr==c].mean(0) for c in cs])
    return np.array([cs[int(np.argmin(((C-r)**2).sum(1)))] for r in Fte])
def bal(a,b): return float(np.mean([(b[a==c]==c).mean() for c in np.unique(a)]))
print("\nClassificação de dano out-of-sample (balanced accuracy):")
clf={}
for mname,Y in METHODS.items():
    F=sm(Y-y_ref[None,:])
    p=nc(F[is_train],y[is_train],F[is_test]); ba=bal(y[is_test],p); clf[mname]=ba
    M3=np.zeros((3,3),int)
    for a,b in zip(y[is_test],p): M3[a,b]+=1
    print(f"  {mname:20s} bal_acc={ba:.3f}  conf={M3.tolist()}")
json.dump({"rmsd_ccdm":rows,"bal_acc":clf},open(os.path.join(OUT,"resumo.json"),"w"),indent=2,ensure_ascii=False)

# ---------- gráficos ----------
fkhz=fHz/1e3
cores={"Original":"tab:red","Park":"tab:green",f"AE-linear(k={K_LINEAR})":"tab:blue","AE nao-linear":"tab:purple"}
for Tt in TEST_TEMPS:
    fig,axes=plt.subplots(1,3,figsize=(21,6),dpi=140,sharey=True)
    for ax,d in zip(axes,[0,1,2]):
        idx=np.where(np.isclose(T,Tt)&(y==d))[0]
        if len(idx)==0: ax.set_title(f"Dano {d} — sem amostra"); continue
        i=idx[0]
        ax.plot(fkhz,y_ref,"--",color="black",lw=1.6,label=f"Referência saudável {REF_TEMP:.0f}°C")
        for mname,Y in METHODS.items():
            ax.plot(fkhz,Y[i],lw=1.9 if mname!="Original" else 1.2,
                    alpha=0.55 if mname=="Original" else 0.95,
                    color=cores[mname],label=mname)
        ax.set_title(f"Dano {d}"); ax.set_xlabel("Frequência (kHz)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Impedância")
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=5,frameon=True,bbox_to_anchor=(0.5,-0.04))
    fig.suptitle(f"Curvas compensadas — T = {Tt:.0f}°C (TESTE, out-of-sample) — 30–40 kHz",fontsize=17,y=1.02)
    fig.tight_layout(rect=[0,0.05,1,0.97])
    p=os.path.join(OUT,f"curvas_teste_T{int(Tt)}C.png"); fig.savefig(p,dpi=140,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("figura:",p)
print("\n✅ tudo salvo em",OUT)

# -*- coding: utf-8 -*-
"""
Verificação empírica: a ordem D1 < D2 existe no SINAL CRU?
==========================================================
Mede, sem compensação alguma, a distância de cada classe de dano à referência
saudável NA PRÓPRIA TEMPERATURA (assim não há efeito térmico envolvido).
Feito banda a banda e temperatura a temperatura.

Objetivo: mostrar que D0<D1<D2 NÃO é uma propriedade garantida dos dados,
e portanto não pode ser usada como critério de sucesso de um compensador.
"""
import os, sys, numpy as np, pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"00_auditoria")

df,_,_=P.load_base()
T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
BANDS=[(30,40),(30,50),(30,60),(30,70),(30,80),(30,90),(30,100),
       (40,50),(50,60),(60,70),(70,80),(80,90),(90,100)]
# temperaturas com as 3 classes -> comparação intra-temperatura (sem efeito térmico)
TEMPS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
print(f"temperaturas com as 3 classes: {TEMPS}\n")

rows=[]
for lo,hi in BANDS:
    fc,f=P.band(lo,hi,decim=1)
    dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64)
    for t in TEMPS:
        m0=np.isclose(T,t)&(y==0)
        ref_t=np.median(X[m0],axis=0)          # referência saudável DA PRÓPRIA temperatura
        r={}
        for c in [1,2]:
            ii=np.where(np.isclose(T,t)&(y==c))[0]
            r[c]={"RMSD":float(np.mean([P.rmsd(X[i],ref_t) for i in ii])),
                  "CCDM":float(np.mean([P.ccdm(X[i],ref_t) for i in ii]))}
        rows.append({"banda":f"{lo}-{hi}","T":t,
                     "RMSD_D1":r[1]["RMSD"],"RMSD_D2":r[2]["RMSD"],
                     "CCDM_D1":r[1]["CCDM"],"CCDM_D2":r[2]["CCDM"],
                     "D1_menor_D2_RMSD":bool(r[1]["RMSD"]<r[2]["RMSD"]),
                     "D1_menor_D2_CCDM":bool(r[1]["CCDM"]<r[2]["CCDM"])})
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"ordem_fisica_dano.csv"),index=False)

print("="*86)
print("ORDEM FÍSICA D1 vs D2 NO SINAL CRU (referência = saudável da MESMA temperatura)")
print("="*86)
print(f"{'banda':>9} | {'RMSD_D1':>8} {'RMSD_D2':>8} | {'D1<D2 (RMSD)':>14} | {'D1<D2 (CCDM)':>14}")
for b,g in d.groupby("banda",sort=False):
    n=len(g); k=int(g.D1_menor_D2_RMSD.sum()); kc=int(g.D1_menor_D2_CCDM.sum())
    print(f"{b:>9} | {g.RMSD_D1.mean():8.3f} {g.RMSD_D2.mean():8.3f} | {k:>6}/{n:<7} | {kc:>6}/{n:<7}")

tot=len(d); kt=int(d.D1_menor_D2_RMSD.sum()); ktc=int(d.D1_menor_D2_CCDM.sum())
print("\n" + "-"*86)
print(f"GLOBAL: D1<D2 em RMSD -> {kt}/{tot} casos ({100*kt/tot:.0f}%)")
print(f"GLOBAL: D1<D2 em CCDM -> {ktc}/{tot} casos ({100*ktc/tot:.0f}%)")
print("-"*86)
if kt<tot*0.9:
    print(">>> CONCLUSÃO: a ordem D1<D2 NÃO é uma propriedade consistente dos dados.")
    print(">>> Portanto D0<D1<D2 NÃO pode ser usado como critério de sucesso do compensador.")
    print(">>> O critério válido é healthy_sep: D0 < min(D1,D2).")
else:
    print(">>> A ordem D1<D2 é majoritariamente consistente no sinal cru.")

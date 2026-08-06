# -*- coding: utf-8 -*-
"""Auditoria por banda: quais faixas são numericamente utilizáveis."""
import os,re,numpy as np,pandas as pd
BASE=r"C:\Users\luize\base-completo--.pkl"
OUT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL\00_auditoria"
def fz(c):
    m=re.match(r"^f_(\d+(?:\.\d+)?)Hz$",str(c)); return float(m.group(1)) if m else None
df=pd.read_pickle(BASE).reset_index(drop=True)
fcols=[c for c in df.columns if fz(c) is not None]
fr=np.array([fz(c) for c in fcols]); o=np.argsort(fr); fcols=[fcols[i] for i in o]; fr=fr[o]
rows=[]
print(f"{'banda kHz':>12} | {'min':>13} {'mediana':>10} {'max':>13} | {'|v|>1e3':>8} {'|v|>1e5':>8} | usavel")
for lo in range(0,125,5):
    hi=lo+5
    sel=[c for c,f in zip(fcols,fr) if lo*1e3<=f<hi*1e3]
    if not sel: continue
    X=df[sel].to_numpy(np.float64)
    n=X.size
    big3=int((np.abs(X)>1e3).sum()); big5=int((np.abs(X)>1e5).sum())
    usavel = (big3/n)<1e-4
    rows.append({"banda":f"{lo}-{hi}","lo_khz":lo,"hi_khz":hi,"n_pontos":len(sel),
                 "min":float(X.min()),"mediana":float(np.median(X)),"max":float(X.max()),
                 "frac_abs_gt_1e3":big3/n,"frac_abs_gt_1e5":big5/n,"usavel":bool(usavel)})
    print(f"{lo:5d}-{hi:<6d} | {X.min():13.4g} {np.median(X):10.3f} {X.max():13.4g} | {big3:8d} {big5:8d} | {'SIM' if usavel else 'NAO'}")
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"bandas_usaveis.csv"),index=False)
ok=d[d.usavel]
print(f"\nBandas de 5 kHz utilizáveis: {len(ok)}/{len(d)}")
if len(ok):
    print(f"  faixa contínua utilizável: {ok.lo_khz.min()}–{ok.hi_khz.max()} kHz")
bad=d[~d.usavel]
if len(bad): print(f"  bandas PROBLEMÁTICAS: {bad.banda.tolist()}")

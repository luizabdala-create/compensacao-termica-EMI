# -*- coding: utf-8 -*-
"""FASE 1 — SMOKE TEST: 30-40 kHz, T_ref=30, 1 fold LOTO, todos os métodos."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL")
import pipeline as P

df, _, _ = P.load_base()
T = df["temperatura_c"].to_numpy(float); y = df["falha"].to_numpy(int)
fcols, f = P.band(30, 40, decim=1)
X = df[fcols].to_numpy(np.float64)
print(f"banda 30-40 kHz: {len(fcols)} pontos | {len(X)} curvas")

T_REF = 30.0
ref, n_ref = P.build_reference(df, fcols, T_REF)     # PROTOCOLO A (congelada)
print(f"referência: T_ref={T_REF}, n={n_ref}")

T_TEST = 60.0                                        # fold de teste
te = np.isclose(T, T_TEST); tr = ~te
trh = tr & (y == 0)
print(f"fold teste T={T_TEST}: {te.sum()} curvas | treino saudável: {trh.sum()}")

res = {}
t0=time.time(); res["Original"] = (X.copy(), {}); print(f"Original      ok  {time.time()-t0:.1f}s")
t0=time.time(); res["Park"] = P.comp_park(X, ref, f, {"max_shift_frac":0.10,"nsteps":111,"smooth_win":5}); print(f"Park          ok  {time.time()-t0:.1f}s")
t0=time.time(); res["RF_direct"] = P.comp_rf(X, T, ref, trh, {"n_estimators":250,"max_depth":10,"smooth_win":5}, mode="direct"); print(f"RF_direct     ok  {time.time()-t0:.1f}s")
t0=time.time(); res["RF_temponly"] = P.comp_rf(X, T, ref, trh, {"n_estimators":250,"max_depth":10,"smooth_win":5}, mode="temponly"); print(f"RF_temponly   ok  {time.time()-t0:.1f}s")
t0=time.time(); res["AE"] = P.comp_ae(X, T, ref, trh, {"n_input":1000,"n_anchors":64,"epochs":400,"patience":60}, T_REF, seed=42); print(f"AE            ok  {time.time()-t0:.1f}s  info={res['AE'][1]}")
t0=time.time(); res["TAU_T"] = P.comp_tauT(X, T, ref, trh, f, {"rank":8}); print(f"TAU_T(aux)    ok  {time.time()-t0:.1f}s")

print(f"\n{'metodo':14s} | {'RMSD D0':>8} {'RMSD D1':>8} {'RMSD D2':>8} | {'CCDM D0':>8} | ordem")
rows=[]
for name,(Y,info) in res.items():
    m={}
    for c in [0,1,2]:
        ii=np.where(te&(y==c))[0]
        m[c]={"RMSD":float(np.mean([P.rmsd(Y[i],ref) for i in ii])),
              "CCDM":float(np.mean([P.ccdm(Y[i],ref) for i in ii]))}
    hs,fo=P.monotonicity({c:m[c]["RMSD"] for c in [0,1,2]})
    print(f"{name:14s} | {m[0]['RMSD']:8.3f} {m[1]['RMSD']:8.3f} {m[2]['RMSD']:8.3f} | {m[0]['CCDM']:8.4f} | "
          f"D0<min:{'S' if hs else 'N'} D0<D1<D2:{'S' if fo else 'N'}")
    rows.append({"metodo":name,**{f"RMSD_D{c}":m[c]["RMSD"] for c in [0,1,2]},
                 **{f"CCDM_D{c}":m[c]["CCDM"] for c in [0,1,2]},"healthy_sep":hs,"full_order":fo})
pd.DataFrame(rows).to_csv(r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL\00_auditoria\smoke_test.csv",index=False)
print("\n✅ SMOKE TEST OK")

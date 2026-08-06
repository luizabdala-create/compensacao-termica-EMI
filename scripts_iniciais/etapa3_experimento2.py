# -*- coding: utf-8 -*-
"""
ETAPA 3 — EXPERIMENTO 2 (hipótese-chave da V8, sem rede)
========================================================
Hipótese: o Físico do V7 falha out-of-sample porque interpola a CURVA saudável
entre temps de treino distantes (picos viram borrão). Conserto: modelar os
PARÂMETROS térmicos τ(T), a(T), b(T), c(T) como funções SUAVES de T, calibradas
no saudável por temperatura de treino, e aplicá-los por temperatura a TODAS as
curvas (damage-safe: sem re-alinhar cada curva).

Compara out-of-sample (só temps de teste):
  Original | Park | Fisico_curva (V7) | Fisico_suave(τ(T)) [deg1 e deg2]
Mede (A) remoção térmica no saudável e (B) preservação de dano (D-D0, sem micro-shift).
"""
import os, sys, json, numpy as np, pandas as pd
PROJ = r"C:\Users\luize\IC_EMI"; BASE_ABS = r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0, PROJ); import ae_hibrido_v7 as v7
FMIN, FMAX = 30.0, 40.0
OUT = os.path.join(PROJ, "etapa3_exp2"); os.makedirs(OUT, exist_ok=True)

SPLITS = {
    "B_ref30": {"train":[0,10,30,50,70], "test":[-10,20,40,60], "ref":30.0},
    "A_ref20": {"train":[0,10,20,40,60], "test":[-10,30,50,70], "ref":20.0},
}

def load():
    df = pd.read_pickle(BASE_ABS).reset_index(drop=True)
    df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
    df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
    fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX)
    return df[["temperatura_c","falha"]+fcols].copy(),fcols,fHz

def subset(df,temps):
    m=np.zeros(len(df),bool)
    for T in temps: m|=np.isclose(df["temperatura_c"],T)
    return df[m].copy()

def fit_scalar_T(Ts, vals, deg):
    """Ajuste polinomial vals~T com extrapolação; deg limitado ao nº de pontos."""
    Ts=np.asarray(Ts,float); vals=np.asarray(vals,float)
    deg=min(deg, len(Ts)-1)
    coef=np.polyfit(Ts, vals, deg)
    return np.poly1d(coef)

def build_smooth_calib(df_tr, fcols, fHz, ref_temp, deg_tau=1, deg_aff=1):
    """Calibra por temperatura de treino (saudável) e ajusta funções suaves de T."""
    healthy_by_temp, train_temps, y_ref, ref_used = v7.get_healthy_references_by_temperature(df_tr, fcols, ref_temp)
    Ts=[]; taus=[]; aa=[]; bb=[]; cc=[]
    for T in train_temps:
        h=healthy_by_temp[float(T)]
        tau=v7.estimate_healthy_shift(h, y_ref, fHz)
        h_sh=v7.shift_interp(h, fHz, tau)
        a,b,c=v7.fit_affine_tilt_healthy(h_sh, y_ref)
        Ts.append(float(T)); taus.append(tau); aa.append(a); bb.append(b); cc.append(c)
    fn={"tau":fit_scalar_T(Ts,taus,deg_tau),"a":fit_scalar_T(Ts,aa,deg_aff),
        "b":fit_scalar_T(Ts,bb,deg_aff),"c":fit_scalar_T(Ts,cc,deg_aff)}
    return fn, y_ref, (Ts,taus,aa,bb,cc), ref_used

def apply_smooth(x, T, fn, fHz):
    z=np.linspace(-1,1,len(x))
    tau=float(fn["tau"](T)); a=float(fn["a"](T)); b=float(fn["b"](T)); c=float(fn["c"](T))
    xs=v7.shift_interp(np.asarray(x,float), fHz, tau)
    return a*xs+b+c*z

def metrics_by_class(Y,y_ref,falha):
    rows={}
    for d in [0,1,2]:
        idx=np.where(falha==d)[0]
        if len(idx)==0: continue
        rows[d]={"RMSD":float(np.mean([v7.rmsd(Y[i],y_ref) for i in idx])),
                 "CCDM":float(np.mean([v7.ccdm(Y[i],y_ref) for i in idx])),"n":len(idx)}
    return rows

def compensa_fisico_curva(Xte,Tte,df_tr,fcols,fHz,ref_temp):
    v7._CALIBRATION_CACHE.clear()
    hbt,tt,y_ref,_=v7.get_healthy_references_by_temperature(df_tr,fcols,ref_temp)
    Y=np.zeros_like(Xte)
    for i,(x,T) in enumerate(zip(Xte,Tte)):
        p=v7.get_calibration_params_for_T(float(T),hbt,tt,y_ref,fHz)
        Y[i]=v7.apply_calibration_to_curve(x,p,fHz,alpha=1.0)
    return Y,y_ref

def tabela(results):
    print(f"{'metodo':18s} | {'D0 RMSD':>8} {'D0 CCDM':>8} | {'D1 RMSD':>8} {'D2 RMSD':>8} | {'D1-D0':>7} {'D2-D0':>7}")
    for m,rr in results.items():
        d0=rr.get(0,{});d1=rr.get(1,{});d2=rr.get(2,{})
        r0=d0.get("RMSD",np.nan);r1=d1.get("RMSD",np.nan);r2=d2.get("RMSD",np.nan)
        print(f"{m:18s} | {r0:8.3f} {d0.get('CCDM',np.nan):8.3f} | {r1:8.3f} {r2:8.3f} | {r1-r0:7.3f} {r2-r0:7.3f}")

def run(name,cfg,df,fcols,fHz):
    print("\n"+"="*94); print(f"SPLIT {name} | treino={cfg['train']} teste={cfg['test']} ref={cfg['ref']}"); print("="*94)
    df_tr=subset(df,cfg["train"]); df_te=subset(df,cfg["test"])
    Xte=df_te[fcols].to_numpy(np.float64); Tte=df_te["temperatura_c"].to_numpy(float); fal=df_te["falha"].to_numpy(int)
    res={}
    res["Original"]=metrics_by_class(Xte,None if False else None,fal) if False else None
    # ref para Original/Park:
    _,_,y_ref0,_=(None,None,None,None)
    # Park + ref
    v7._CALIBRATION_CACHE.clear(); v7.REF_TEMP=cfg["ref"]
    hbt,tt,y_ref,ref_used=v7.get_healthy_references_by_temperature(df_tr,fcols,cfg["ref"])
    res={}
    res["Original"]=metrics_by_class(Xte,y_ref,fal)
    dfp=v7.compensar_park(df_te,fcols,fHz,y_ref); res["Park"]=metrics_by_class(dfp[fcols].to_numpy(np.float64),y_ref,fal)
    Yfc,_=compensa_fisico_curva(Xte,Tte,df_tr,fcols,fHz,cfg["ref"]); res["Fisico_curva(V7)"]=metrics_by_class(Yfc,y_ref,fal)
    for deg in [1,2]:
        v7._CALIBRATION_CACHE.clear()
        fn,yr,_,_=build_smooth_calib(df_tr,fcols,fHz,cfg["ref"],deg_tau=deg,deg_aff=deg)
        Ys=np.vstack([apply_smooth(Xte[i],Tte[i],fn,fHz) for i in range(len(Xte))])
        res[f"Fisico_suave(deg{deg})"]=metrics_by_class(Ys,yr,fal)
    tabela(res)
    # por temperatura no saudável (A) — para ver extrapolação -10
    print("\n(A) RMSD saudável por temperatura de teste:")
    print(f"{'T':>5} | {'Park':>7} {'Fis_curva':>9} {'Fis_suave1':>10} {'Fis_suave2':>10}")
    for T in cfg["test"]:
        idx=np.where(np.isclose(Tte,T)&(fal==0))[0]
        if len(idx)==0: continue
        def rm(Y): return np.mean([v7.rmsd(Y[i],y_ref) for i in idx])
        v7._CALIBRATION_CACHE.clear(); fn1,_,_,_=build_smooth_calib(df_tr,fcols,fHz,cfg["ref"],1,1)
        v7._CALIBRATION_CACHE.clear(); fn2,_,_,_=build_smooth_calib(df_tr,fcols,fHz,cfg["ref"],2,2)
        Ys1=np.vstack([apply_smooth(Xte[i],Tte[i],fn1,fHz) for i in idx])
        Ys2=np.vstack([apply_smooth(Xte[i],Tte[i],fn2,fHz) for i in idx])
        rk=np.mean([v7.rmsd(dfp[fcols].to_numpy(np.float64)[i],y_ref) for i in idx])
        rc=np.mean([v7.rmsd(Yfc[i],y_ref) for i in idx])
        print(f"{T:>5} | {rk:7.3f} {rc:9.3f} {np.mean([v7.rmsd(Ys1[k],y_ref) for k in range(len(idx))]):10.3f} {np.mean([v7.rmsd(Ys2[k],y_ref) for k in range(len(idx))]):10.3f}")
    return {"name":name,"results":res}

if __name__=="__main__":
    df,fcols,fHz=load(); rep=[]
    for name,cfg in SPLITS.items(): rep.append(run(name,cfg,df,fcols,fHz))
    with open(os.path.join(OUT,"exp2_resumo.json"),"w",encoding="utf-8") as f: json.dump(rep,f,indent=2,ensure_ascii=False)
    print(f"\n✅ salvo em {os.path.join(OUT,'exp2_resumo.json')}")

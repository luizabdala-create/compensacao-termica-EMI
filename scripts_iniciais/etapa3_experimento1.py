# -*- coding: utf-8 -*-
"""
ETAPA 3 + ablação parcial — EXPERIMENTO 1 (rápido, sem rede neural)
===================================================================
Objetivo: com SPLIT correto por temperatura (sem leakage), comparar out-of-sample:
    Original | Park | Físico(1 shift + afim) | Físico + micro-shift por amostra
Responder a pergunta central da V8:
    o micro-shift por amostra melhora (A) remoção térmica no SAUDÁVEL?
    quanto ele desloca (apaga) as curvas de DANO (B)?

Regras anti-leakage:
- referência = mediana saudável em REF_TEMP, REF_TEMP OBRIGATORIAMENTE no treino;
- healthy_by_temp montado só com temps de TREINO;
- params de calibração para uma temp de teste = interpolação sobre temps de TREINO
  (nunca usa curvas da própria temp de teste);
- avaliação só em temps de TESTE.
Reusa as funções-folha comprovadas do ae_hibrido_v7 (sem reescrever o método).
"""
import os, sys, json, numpy as np, pandas as pd

PROJ = r"C:\Users\luize\IC_EMI"
BASE_ABS = r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0, PROJ)
import ae_hibrido_v7 as v7

FMIN, FMAX = 30.0, 40.0
OUT = os.path.join(PROJ, "etapa3_exp1")
os.makedirs(OUT, exist_ok=True)

# micro-shift (mesmos parâmetros do V7)
MICRO_MAX_FRAC = 0.018
MICRO_NC, MICRO_NF = 61, 61
MICRO_PRIOR_PEN = 0.18
MICRO_MIN_IMPROV = 0.010

SPLITS = {
    "B_ref30":  {"train": [0,10,30,50,70], "test": [-10,20,40,60],  "ref": 30.0},
    "A_ref20":  {"train": [0,10,20,40,60], "test": [-10,30,50,70],  "ref": 20.0},
}

def load():
    df = pd.read_pickle(BASE_ABS).reset_index(drop=True)
    df["temperatura_c"] = pd.to_numeric(df["temperatura_c"], errors="coerce")
    df["falha"] = pd.to_numeric(df["falha"], errors="coerce").astype(int)
    fcols, fHz = v7.get_freq_columns(df, FMIN, FMAX)
    return df[["temperatura_c","falha"]+fcols].copy(), fcols, fHz

def subset(df, temps):
    m = np.zeros(len(df), bool)
    for T in temps: m |= np.isclose(df["temperatura_c"], T)
    return df[m].copy()

def compensa_fisico(X, T_all, healthy_by_temp, train_temps, y_ref, fHz):
    Y = np.zeros_like(X)
    taus = np.zeros(len(X))
    for i,(x,T) in enumerate(zip(X,T_all)):
        p = v7.get_calibration_params_for_T(float(T), healthy_by_temp, train_temps, y_ref, fHz)
        Y[i] = v7.apply_calibration_to_curve(x, p, fHz, alpha=1.0)
        taus[i] = p["tau"]
    return Y, taus

def micro_shift(Y, y_ref, fHz):
    Y2 = np.zeros_like(Y); micro = np.zeros(len(Y))
    for i,y in enumerate(Y):
        tau = v7.estimate_shift_multiscale(y, y_ref, fHz, max_frac=MICRO_MAX_FRAC,
            n_coarse=MICRO_NC, n_fine=MICRO_NF, prior_tau=0.0,
            prior_penalty=MICRO_PRIOR_PEN, min_improvement=MICRO_MIN_IMPROV, return_info=False)
        Y2[i] = v7.shift_interp(y, fHz, tau); micro[i] = tau
    return Y2, micro

def metrics_by_class(Y, y_ref, falha):
    rows={}
    for d in [0,1,2]:
        m = (falha==d)
        if m.sum()==0: continue
        r=[v7.rmsd(Y[i],y_ref) for i in np.where(m)[0]]
        c=[v7.ccdm(Y[i],y_ref) for i in np.where(m)[0]]
        rows[d]={"RMSD":float(np.mean(r)),"CCDM":float(np.mean(c)),"n":int(m.sum())}
    return rows

def run_split(name, cfg, df, fcols, fHz):
    print("\n"+"="*90); print(f"SPLIT {name} | treino={cfg['train']} teste={cfg['test']} ref={cfg['ref']}"); print("="*90)
    df_tr = subset(df, cfg["train"]); df_te = subset(df, cfg["test"])
    v7._CALIBRATION_CACHE.clear()
    v7.REF_TEMP = cfg["ref"]  # usado internamente em algumas funções
    healthy_by_temp, train_temps, y_ref, ref_used = v7.get_healthy_references_by_temperature(df_tr, fcols, cfg["ref"])
    assert any(np.isclose(ref_used, cfg["train"])), "REF_TEMP tem que estar no treino!"

    Xte = df_te[fcols].to_numpy(np.float64); Tte = df_te["temperatura_c"].to_numpy(float)
    fal = df_te["falha"].to_numpy(int)

    results={}
    # Original
    results["Original"]=metrics_by_class(Xte, y_ref, fal)
    # Park (ref treino)
    dfp = v7.compensar_park(df_te, fcols, fHz, y_ref)
    results["Park"]=metrics_by_class(dfp[fcols].to_numpy(np.float64), y_ref, fal)
    # Físico
    Yf, tauf = compensa_fisico(Xte, Tte, healthy_by_temp, train_temps, y_ref, fHz)
    results["Fisico"]=metrics_by_class(Yf, y_ref, fal)
    # Físico + micro-shift
    Ym, micro = micro_shift(Yf, y_ref, fHz)
    results["Fisico+micro"]=metrics_by_class(Ym, y_ref, fal)

    # tabela (A) e (B)
    print(f"\n(A) SAUDÁVEL (dano0) e (B) DANO — RMSD | CCDM out-of-sample:")
    print(f"{'metodo':16s} | {'D0 RMSD':>8} {'D0 CCDM':>8} | {'D1 RMSD':>8} | {'D2 RMSD':>8} | {'D1-D0':>7} {'D2-D0':>7}")
    for m,rr in results.items():
        d0=rr.get(0,{}); d1=rr.get(1,{}); d2=rr.get(2,{})
        r0=d0.get("RMSD",np.nan); r1=d1.get("RMSD",np.nan); r2=d2.get("RMSD",np.nan)
        print(f"{m:16s} | {r0:8.3f} {d0.get('CCDM',np.nan):8.3f} | {r1:8.3f} | {r2:8.3f} | {r1-r0:7.3f} {r2-r0:7.3f}")

    # micro-shift aplicado por classe (o item 3)
    print(f"\nMicro-shift aplicado (|Hz|) por classe — deve ser ~0 se não apaga dano:")
    for d in [0,1,2]:
        s=np.abs(micro[fal==d])
        if len(s): print(f"   dano {d}: média={s.mean():6.1f} | máx={s.max():6.1f} | n={len(s)}")

    # variação de (A) ao adicionar micro-shift (no saudável)
    a_fis = results["Fisico"].get(0,{}).get("RMSD",np.nan)
    a_mic = results["Fisico+micro"].get(0,{}).get("RMSD",np.nan)
    print(f"\n>>> (A) saudável: Físico={a_fis:.3f} -> +micro={a_mic:.3f}  (Δ={100*(a_mic-a_fis)/a_fis:+.1f}%)")
    print(f">>> Park saudável={results['Park'].get(0,{}).get('RMSD',np.nan):.3f} | Original={results['Original'].get(0,{}).get('RMSD',np.nan):.3f}")

    return {"name":name,"cfg":cfg,"ref_used":float(ref_used),
            "results":results,
            "micro_by_class":{int(d):float(np.abs(micro[fal==d]).mean()) for d in [0,1,2] if (fal==d).sum()}}

if __name__=="__main__":
    df, fcols, fHz = load()
    allrep=[]
    for name,cfg in SPLITS.items():
        allrep.append(run_split(name,cfg,df,fcols,fHz))
    with open(os.path.join(OUT,"exp1_resumo.json"),"w",encoding="utf-8") as f:
        json.dump(allrep,f,indent=2,ensure_ascii=False)
    print(f"\n✅ salvo em {os.path.join(OUT,'exp1_resumo.json')}")

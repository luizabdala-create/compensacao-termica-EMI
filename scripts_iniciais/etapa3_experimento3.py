# -*- coding: utf-8 -*-
"""
ETAPA 3 — EXPERIMENTO 3 (a pergunta central do projeto)
=======================================================
1) O Park também apaga o dano 2? -> medir o shift que o Park aplica por classe.
2) O dano é detectável APÓS compensação, e o quanto é "shift-invariante"?
   Para cada temperatura de teste (que tem D0,D1,D2), após Park:
     - separação bruta:  RMSD(medD1,medD0), RMSD(medD2,medD0)
     - separação shift-invariante: alinha medD1/medD2 a medD0 pelo MELHOR shift
       e mede o RESÍDUO (parte que NÃO some com deslocamento = assinatura de forma)
   Se o resíduo shift-invariante do D2 ~ 0, o dano 2 é só um shift -> some com qualquer
   compensação por curva. Se for grande, há esperança via features de forma.
"""
import os, sys, json, numpy as np, pandas as pd
PROJ=r"C:\Users\luize\IC_EMI"; BASE=r"C:\Users\luize\base-completo--.pkl"
sys.path.insert(0,PROJ); import ae_hibrido_v7 as v7
FMIN,FMAX=30.0,40.0
OUT=os.path.join(PROJ,"etapa3_exp3"); os.makedirs(OUT,exist_ok=True)
SPLITS={"B_ref30":{"train":[0,10,30,50,70],"test":[-10,20,40,60],"ref":30.0}}

def load():
    df=pd.read_pickle(BASE).reset_index(drop=True)
    df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
    df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)
    fcols,fHz=v7.get_freq_columns(df,FMIN,FMAX)
    return df[["temperatura_c","falha"]+fcols].copy(),fcols,fHz
def subset(df,temps):
    m=np.zeros(len(df),bool)
    for T in temps: m|=np.isclose(df["temperatura_c"],T)
    return df[m].copy()

def park_tau(x,ref,fHz):
    band=fHz[-1]-fHz[0]; tmax=v7.PARK_MAX_SHIFT_FRAC*band
    best=(np.inf,0.0,0.0)
    for tau in np.linspace(-tmax,tmax,v7.PARK_NSTEPS):
        xs=v7.shift_interp(x,fHz,tau); dS=np.mean(ref-xs); err=np.mean((ref-(xs+dS))**2)
        if err<best[0]: best=(err,tau,dS)
    _,tau,dS=best
    return v7.shift_interp(x,fHz,tau)+dS, tau

def best_align_resid(y, target, fHz, max_frac=0.03, nsteps=301):
    """Alinha y a target pelo melhor shift+offset; retorna RMSD residual (shape-only)."""
    band=fHz[-1]-fHz[0]; tmax=max_frac*band
    best=np.inf
    for tau in np.linspace(-tmax,tmax,nsteps):
        ys=v7.shift_interp(y,fHz,tau); ys=ys-np.mean(ys)+np.mean(target)
        e=np.sqrt(np.mean((ys-target)**2))
        if e<best: best=e
    return best

df,fcols,fHz=load()
report=[]
for name,cfg in SPLITS.items():
    print("="*90); print(f"SPLIT {name} treino={cfg['train']} teste={cfg['test']} ref={cfg['ref']}"); print("="*90)
    df_tr=subset(df,cfg["train"]); df_te=subset(df,cfg["test"])
    v7._CALIBRATION_CACHE.clear(); v7.REF_TEMP=cfg["ref"]
    hbt,tt,y_ref,_=v7.get_healthy_references_by_temperature(df_tr,fcols,cfg["ref"])
    Xte=df_te[fcols].to_numpy(np.float64); Tte=df_te["temperatura_c"].to_numpy(float); fal=df_te["falha"].to_numpy(int)

    # 1) shift do Park por classe
    Ypark=np.zeros_like(Xte); taus=np.zeros(len(Xte))
    for i in range(len(Xte)): Ypark[i],taus[i]=park_tau(Xte[i],y_ref,fHz)
    print("\n[1] Shift aplicado pelo PARK por classe (|Hz|):")
    for d in [0,1,2]:
        s=np.abs(taus[fal==d])
        print(f"   dano {d}: média={s.mean():6.1f} | máx={s.max():6.1f}")

    # 2) separabilidade por temperatura (medianas de classe após Park)
    print("\n[2] Após PARK — separação de dano por temperatura de teste:")
    print(f"{'T':>5} | {'RMSD raw D1-D0':>14} {'RMSD raw D2-D0':>14} | {'RESID inv D1':>12} {'RESID inv D2':>12}")
    rows=[]
    for T in cfg["test"]:
        def med(d):
            idx=np.where(np.isclose(Tte,T)&(fal==d))[0]
            return np.median(Ypark[idx],axis=0) if len(idx) else None
        m0,m1,m2=med(0),med(1),med(2)
        if m0 is None: continue
        raw1=v7.rmsd(m1,m0) if m1 is not None else np.nan
        raw2=v7.rmsd(m2,m0) if m2 is not None else np.nan
        inv1=best_align_resid(m1,m0,fHz) if m1 is not None else np.nan
        inv2=best_align_resid(m2,m0,fHz) if m2 is not None else np.nan
        print(f"{T:>5} | {raw1:14.3f} {raw2:14.3f} | {inv1:12.3f} {inv2:12.3f}")
        rows.append({"T":float(T),"raw_D1":float(raw1),"raw_D2":float(raw2),
                     "inv_D1":float(inv1),"inv_D2":float(inv2)})
    # fração do sinal de dano que é shift-invariante (forma) vs que some com shift
    a2=np.nanmean([r["raw_D2"] for r in rows]); i2=np.nanmean([r["inv_D2"] for r in rows])
    a1=np.nanmean([r["raw_D1"] for r in rows]); i1=np.nanmean([r["inv_D1"] for r in rows])
    print(f"\n   Dano 1: sinal bruto={a1:.3f} | shape-only(shift-invariante)={i1:.3f}  ({100*i1/a1:.0f}% sobrevive ao shift)")
    print(f"   Dano 2: sinal bruto={a2:.3f} | shape-only(shift-invariante)={i2:.3f}  ({100*i2/a2:.0f}% sobrevive ao shift)")
    report.append({"name":name,"park_tau_by_class":{int(d):float(np.abs(taus[fal==d]).mean()) for d in [0,1,2]},
                   "rows":rows,"inv_frac_D1":float(i1/a1),"inv_frac_D2":float(i2/a2)})

with open(os.path.join(OUT,"exp3_resumo.json"),"w",encoding="utf-8") as f: json.dump(report,f,indent=2,ensure_ascii=False)
print(f"\n✅ salvo em {os.path.join(OUT,'exp3_resumo.json')}")

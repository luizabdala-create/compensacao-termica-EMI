# -*- coding: utf-8 -*-
"""
RE-AVALIAÇÃO CONGELADA de fase8_tuning_ampliado.csv com os hiperparâmetros do RE-TUNING.
Só recalcula AE e RF_direct nas bandas que o retune alterou (30-40,40-50,60-70,70-80,30-70),
usando a config MODAL de checkpoints/fase8_bestcfg.json (atualizado pelo retune). Park,
RF_temponly, TAU_T e bandas não-retunadas são mantidos VERBATIM do CSV antigo (config idêntica
=> números idênticos, determinístico). Reusa EXATAMENTE o bloco de métricas da Fase 8 (mesmas
fórmulas: all_metrics por classe, sep_*, healthy_sep, full_order, alteracao_RMSD).
Sem grid search, sem tocar no teste para selecionar. cv_inner vem do retune (cv_novo).
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
OUT=os.path.join(ROOT,"02_compensacao"); CSV=os.path.join(OUT,"fase8_tuning_ampliado.csv")
T_REF=30.0
RETUNED_BANDS={"30-40","40-50","60-70","70-80","30-70"}
RECOMP_METHODS={"AE","RF_direct"}
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
retune=pd.read_csv(os.path.join(OUT,"retune_expandido.csv")) if os.path.exists(os.path.join(OUT,"retune_expandido.csv")) else None
def cvnovo(banda,m):
    if retune is None: return np.nan
    s=retune[(retune.banda==banda)&(retune.metodo==m)]
    return float(s.cv_novo.iloc[0]) if len(s) else np.nan
def cfg(banda,m):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else None

old=pd.read_csv(CSV); print("CSV antigo:",old.shape,"| bandas:",sorted(old.banda.unique()),flush=True)
# linhas a recomputar vs manter
mask_recomp=old.metodo.isin(RECOMP_METHODS)&old.banda.isin(RETUNED_BANDS)
carry=old[~mask_recomp].copy()
alvo=old[mask_recomp][["banda","T_test","metodo"]].drop_duplicates()
print(f"recomputar {len(alvo)} linhas (AE/RF_direct × bandas retunadas); manter {len(carry)}",flush=True)

df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
cols=list(old.columns); newrows=[]; t0=time.time()
for banda in sorted(alvo.banda.unique()):
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
    sub=alvo[alvo.banda==banda]
    print(f"\n=== {banda} ({len(fc)} pts) — {sub.metodo.nunique()} métodos × {sub.T_test.nunique()} folds ===",flush=True)
    for T_test in sorted(sub.T_test.unique()):
        te=np.isclose(T,T_test); tr=~te; trh=tr&(y==0)
        for m in sorted(sub[sub.T_test==T_test].metodo.unique()):
            g=cfg(banda,m)
            try:
                Y=P.comp_ae(X,T,ref,trh,g,T_REF,seed=42)[0] if m=="AE" else P.comp_rf(X,T,ref,trh,g,"direct")[0]
            except Exception as e:
                print("  ERRO",banda,T_test,m,e,flush=True); continue
            rec={"banda":banda,"T_test":T_test,"metodo":m,"cv_inner":cvnovo(banda,m),"cfg":json.dumps(g)}
            pc={}
            for c in [0,1,2]:
                ii=np.where(te&(y==c))[0]
                if len(ii)==0: continue
                M=[P.all_metrics(Y[i],ref) for i in ii]
                for kk in M[0]: rec[f"{kk}_D{c}"]=float(np.mean([mm[kk] for mm in M]))
                pc[c]=rec[f"RMSD_D{c}"]
            rec["alteracao_RMSD"]=float(np.mean([P.rmsd(Y[i],X[i]) for i in np.where(te)[0]]))
            if {0,1,2}<=set(pc):
                rec["sep_RMSD_D1"]=pc[1]-pc[0]; rec["sep_RMSD_D2"]=pc[2]-pc[0]
                hs,fo=P.monotonicity(pc); rec["healthy_sep"]=hs; rec["full_order"]=fo
            newrows.append(rec)
        print(f"  T={T_test} ok | {(time.time()-t0)/60:.1f} min",flush=True)

newdf=pd.DataFrame(newrows)
# garante todas as colunas na mesma ordem
for c in cols:
    if c not in newdf.columns: newdf[c]=np.nan
newdf=newdf[cols]
out=pd.concat([carry,newdf],ignore_index=True).sort_values(["banda","T_test","metodo"]).reset_index(drop=True)
out.to_csv(CSV,index=False); out.to_csv(os.path.join(ROOT,"checkpoints","fase8_tuning.csv"),index=False)
print(f"\n✅ RE-AVALIAÇÃO: {out.shape} salvo | {(time.time()-t0)/60:.1f} min",flush=True)
# resumo do efeito
piv=out[out.metodo.isin(["AE","Park","RF_direct"])].pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean").round(3)
print("\nRMSD_D0 por banda (após retune):"); print(piv.to_string(),flush=True)

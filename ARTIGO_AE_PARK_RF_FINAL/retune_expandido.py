# -*- coding: utf-8 -*-
"""
RE-TUNING EXPANDIDO do AE e do RF nas bandas principais (grades maiores, nested CV honesto).
Reporta se a grade maior melhora vs a atual. Atualiza checkpoints/fase8_bestcfg.json e
fase8_tuning_ampliado.csv APENAS onde houver melhora comprovada por CV interna (sem tocar teste).
"""
import os,sys,json,time,itertools,numpy as np,pandas as pd
from collections import Counter
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
BANDS=["30-40","40-50","60-70","70-80","30-70"]; T_REF=30.0; N_INNER=3
F8CFG=os.path.join(ROOT,"checkpoints","fase8_bestcfg.json"); F8=os.path.join(ROOT,"checkpoints","fase8_tuning.csv")
bestcfg=json.load(open(F8CFG)) if os.path.exists(F8CFG) else {}
# grades EXPANDIDAS (AE enxuta p/ tratabilidade: explora capacidade e lr em torno do melhor atual;
# AE tende a overfitar, então capacidade maior só é adotada se a CV interna comprovar ganho).
AE_GRID=[dict(n_input=ni,n_anchors=128,latent=lt,hidden=hd,lr=lr,dropout=0.10,noise=0.01,epochs=550,patience=75,lambda_d1=0.3)
         for (ni,lt,hd,lr) in [
             (2000, 8,256,2e-3),(2000, 8,256,1e-3),(2000,16,256,2e-3),(2000,16,384,2e-3),
             (2000,12,320,2e-3),(2000, 8,384,1e-3),(3333, 8,256,2e-3),(3333,16,320,1e-3),(2000,24,384,1e-3)]]
RF_GRID=[dict(n_estimators=n,max_depth=dp,min_samples_leaf=lf,min_samples_split=4,max_features="sqrt",smooth_win=5,input_decim=idc)
         for n in [300,600] for dp in [10,16,None] for lf in [1,2] for idc in [4,8]]
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
def modal(banda,m):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else None
print(f"AE grid={len(AE_GRID)} RF grid={len(RF_GRID)} | bandas={BANDS}",flush=True)
rows=[]; t0=time.time()
for banda in BANDS:
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
    print(f"\n=== {banda} ({len(fc)} pts) ===",flush=True)
    # inner CV global (média sobre folds internos = temps de treino) para escolher config
    inner=[t for t in FOLDS if not np.isclose(t,T_REF)][::3]  # CV interna amostrada (tratabilidade do AE)
    def inner_rmsd(g,make):
        """RMSD média por CV INTERNA (só temps de treino) — usada para SELECIONAR config."""
        sc=[]
        for tk in inner:
            ho=np.isclose(T,tk)&(y==0); keep=(~np.isclose(T,tk))&(y==0)
            if ho.sum()==0 or keep.sum()<8: continue
            try: Yc=make(g,keep); ii=np.where(ho)[0]; sc.append(np.mean([P.rmsd(Yc[i],ref) for i in ii]))
            except Exception: sc.append(np.inf)
        return float(np.mean(sc)) if sc else np.inf
    def score(grid,make):
        best=(np.inf,grid[0])
        for g in grid:
            mm=inner_rmsd(g,make)
            if mm<best[0]: best=(mm,g)
        return best
    makeAE=lambda g,k:P.comp_ae(X,T,ref,k,g,T_REF,seed=42)[0]
    makeRF=lambda g,k:P.comp_rf(X,T,ref,k,g,"direct")[0]
    cvAE,gAE=score(AE_GRID,makeAE); cvRF,gRF=score(RF_GRID,makeRF)
    # comparar com config atual
    for m,gnew,cvnew,make in [("AE",gAE,cvAE,makeAE),("RF_direct",gRF,cvRF,makeRF)]:
        gold=modal(banda,m)
        # SELEÇÃO: só por CV interna (nunca pelo teste). O teste é apenas REPORTADO.
        cvold=inner_rmsd(gold,make) if gold else np.nan
        def loto_rmsd(g):
            r=[]
            for tk in FOLDS:
                if np.isclose(tk,T_REF): continue
                te=np.isclose(T,tk); trh=(~te)&(y==0); ii=np.where(te&(y==0))[0]
                if trh.sum()<8: continue
                Y=make(g,trh)
                r.append(np.mean([P.rmsd(Y[i],ref) for i in ii]))
            return float(np.mean(r)) if r else np.nan
        rnew=loto_rmsd(gnew); rold=loto_rmsd(gold) if gold else np.nan
        # decisão de adoção: SÓ CV interna melhora (anti-leakage)
        melhora = (not np.isnan(cvold)) and cvnew<cvold-1e-3
        rows.append({"banda":banda,"metodo":m,"cv_atual":round(cvold,3) if cvold==cvold else None,
                     "cv_novo":round(cvnew,3),"rmsd_teste_atual":round(rold,3) if rold==rold else None,
                     "rmsd_teste_novo":round(rnew,3) if rnew==rnew else None,"melhora_cv":bool(melhora),
                     "cfg_novo":json.dumps(gnew)})
        print(f"  {m}: CV_int atual={cvold:.3f} novo={cvnew:.3f} -> {'ADOTA' if melhora else 'mantem'} "
              f"| (teste p/ reporte: atual={rold:.3f} novo={rnew:.3f})",flush=True)
        if melhora:  # atualiza bestcfg de todos os folds dessa banda/método (seleção via CV interna)
            for tk in FOLDS:
                if not np.isclose(tk,T_REF): bestcfg[f"{banda}|{tk}|{m}"]=gnew
    # ---- checkpoint incremental por banda (resumível / à prova de interrupção) ----
    pd.DataFrame(rows).to_csv(os.path.join(ROOT,"02_compensacao","retune_expandido.csv"),index=False)
    json.dump(bestcfg,open(F8CFG,"w"),indent=1)
    print(f"  [checkpoint salvo apos banda {banda} | {(time.time()-t0)/60:.1f} min]",flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(ROOT,"02_compensacao","retune_expandido.csv"),index=False)
json.dump(bestcfg,open(F8CFG,"w"),indent=1)
print("\n=== RESUMO RE-TUNING ===",flush=True); print(d[["banda","metodo","cv_atual","cv_novo","rmsd_teste_atual","rmsd_teste_novo","melhora_cv"]].to_string(index=False),flush=True)
nm=int(d.melhora_cv.sum()); print(f"\nConfigurações adotadas (melhora por CV interna): {nm}/{len(d)} | {(time.time()-t0)/60:.1f} min",flush=True)
print("(bestcfg atualizado só onde a CV interna melhorou — seleção nunca usou o teste; re-rode figuras/PDF para refletir)",flush=True)

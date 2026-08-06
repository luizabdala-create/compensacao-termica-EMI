# -*- coding: utf-8 -*-
"""Estatística pareada da COMPARAÇÃO JUSTA (todos tunados) — Fase 3 + 3b."""
import os,sys,itertools,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
OUT=os.path.join(ROOT,"09_estatistica"); os.makedirs(OUT,exist_ok=True)
ae=pd.read_csv(os.path.join(ROOT,"checkpoints","fase3_AE.csv")); ae["metodo"]="AE"
pr=pd.read_csv(os.path.join(ROOT,"checkpoints","fase3b_ParkRF.csv"))
pr["metodo"]=pr["metodo"].str.replace("_tuned","",regex=False)
d=pd.concat([ae,pr],ignore_index=True)
d.to_csv(os.path.join(ROOT,"02_compensacao","comparacao_justa_todos_tunados.csv"),index=False)
MET=["AE","Park","RF_direct","RF_temponly"]
try:
    from scipy import stats; HAVE=True
except Exception: HAVE=False

print("="*94); print("COMPARAÇÃO JUSTA — todos com tuning por inner CV"); print("="*94)
g=d.groupby("metodo").agg(RMSD_D0=("RMSD_D0","mean"),RMSD_sd=("RMSD_D0","std"),
    CCDM_D0=("CCDM_D0","mean"),healthy_sep=("healthy_sep","mean"),
    sep_D1=("sep_RMSD_D1","mean"),sep_D2=("sep_RMSD_D2","mean"),n=("RMSD_D0","size")).round(4)
print(g.reindex(MET).to_string())

print("\n--- por banda (RMSD_D0) ---")
pv=d.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean").round(3)[MET]
print(pv.to_string())
print("\nvencedor por banda:");
for b in pv.index: print(f"   {b:8s}: {pv.loc[b].idxmin()}  ({pv.loc[b].min():.3f})")

print("\n" + "="*94); print("TESTES PAREADOS (mesmos folds banda x T_test)"); print("="*94)
key=["banda","T_test"]
rows=[]
for metrica in ["RMSD_D0","CCDM_D0","sep_RMSD_D1","sep_RMSD_D2"]:
    w=d.pivot_table(index=key,columns="metodo",values=metrica).dropna()
    if len(w)<6: continue
    if HAVE:
        fr=stats.friedmanchisquare(*[w[m] for m in MET if m in w.columns])
        print(f"\n[{metrica}] Friedman chi2={fr.statistic:.2f} p={fr.pvalue:.3e} (n={len(w)})")
    ps=[]
    for a,b in itertools.combinations([m for m in MET if m in w.columns],2):
        dm=float((w[a]-w[b]).mean())
        p=float(stats.wilcoxon(w[a],w[b]).pvalue) if HAVE else np.nan
        ps.append((a,b,dm,p))
    if HAVE:
        idx=np.argsort([p for *_,p in ps]); mt=len(ps); adj=[None]*mt; prev=0
        for rk,i in enumerate(idx):
            v=min(1.0,max(prev,(mt-rk)*ps[i][3])); adj[i]=v; prev=v
    else: adj=[np.nan]*len(ps)
    for (a,b,dm,p),pa in zip(ps,adj):
        sig="SIM" if (pa==pa and pa<0.05) else "nao"
        rows.append({"metrica":metrica,"A":a,"B":b,"dif_A_menos_B":round(dm,4),"p_holm":pa,"signif":sig})
        if metrica=="RMSD_D0":
            print(f"   {a:12s} vs {b:12s}: dif={dm:+7.3f}  p_holm={pa:.4f}  signif={sig}")
st=pd.DataFrame(rows); st.to_csv(os.path.join(OUT,"stats_comparacao_justa.csv"),index=False)

print("\n" + "="*94); print("POR BANDA: AE vs cada um (n=9 folds por banda)"); print("="*94)
for b in sorted(d.banda.unique()):
    s=d[d.banda==b]; w=s.pivot_table(index="T_test",columns="metodo",values="RMSD_D0").dropna()
    if "AE" not in w.columns or len(w)<5: continue
    out=[]
    for m in ["Park","RF_direct","RF_temponly"]:
        if m not in w.columns: continue
        dm=float((w["AE"]-w[m]).mean())
        p=float(stats.wilcoxon(w["AE"],w[m]).pvalue) if HAVE else np.nan
        out.append(f"AE-{m}={dm:+.3f}(p={p:.3f})")
    print(f"  {b:8s} n={len(w)}: "+"  ".join(out))
print("\n(n=9 por banda -> poder estatístico baixo; p>=0.05 nao prova equivalencia)")

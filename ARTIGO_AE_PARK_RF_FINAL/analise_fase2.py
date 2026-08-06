# -*- coding: utf-8 -*-
"""Análise da FASE 2: por banda, por T_ref, estatística pareada."""
import os,sys,json,itertools,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
d=pd.read_csv(os.path.join(ROOT,"checkpoints","fase2_master.csv"))
d=d[~d.T_test_eh_T_ref]
OUT5=os.path.join(ROOT,"05_sensibilidade_faixas"); OUT6=os.path.join(ROOT,"06_sensibilidade_referencia")
OUT9=os.path.join(ROOT,"09_estatistica")
for o in (OUT5,OUT6,OUT9): os.makedirs(o,exist_ok=True)
MET=["Original","Park","RF_direct","RF_temponly","AE","TAU_T"]

print("="*100); print("Q9 — QUAL BANDA É MELHOR? (RMSD_D0 médio por método x banda)"); print("="*100)
pv=d.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean").round(3)
pv=pv[[m for m in MET if m in pv.columns]]
order=["30-40","30-50","30-60","30-70","30-80","30-90","30-100","40-50","50-60","60-70","70-80","80-90","90-100"]
pv=pv.reindex([b for b in order if b in pv.index])
print(pv.to_string()); pv.to_csv(os.path.join(OUT5,"RMSD_D0_por_banda.csv"))
print("\nmelhor banda por método (menor RMSD_D0):")
for m in pv.columns: print(f"   {m:12s}: {pv[m].idxmin()}  ({pv[m].min():.3f})")

print("\n" + "="*100); print("healthy_sep (D0 < min(D1,D2)) por banda — CRITÉRIO PRIMÁRIO"); print("="*100)
pv2=d.pivot_table(index="banda",columns="metodo",values="healthy_sep",aggfunc="mean").round(3)
pv2=pv2[[m for m in MET if m in pv2.columns]].reindex([b for b in order if b in pv2.index])
print(pv2.to_string()); pv2.to_csv(os.path.join(OUT5,"healthy_sep_por_banda.csv"))

print("\n" + "="*100); print("Q11 — QUAL T_ref É MELHOR? (RMSD_D0 por método x T_ref)"); print("="*100)
pv3=d.pivot_table(index="T_ref",columns="metodo",values="RMSD_D0",aggfunc="mean").round(3)
pv3=pv3[[m for m in MET if m in pv3.columns]]
print(pv3.to_string()); pv3.to_csv(os.path.join(OUT6,"RMSD_D0_por_Tref.csv"))
pv4=d.pivot_table(index="T_ref",columns="metodo",values="healthy_sep",aggfunc="mean").round(3)
pv4=pv4[[m for m in MET if m in pv4.columns]]
print("\nhealthy_sep por T_ref:"); print(pv4.to_string()); pv4.to_csv(os.path.join(OUT6,"healthy_sep_por_Tref.csv"))

print("\n" + "="*100); print("C7 — ALTERAÇÃO DA CURVA (controle: não pode 'vencer' virando a referência)"); print("="*100)
alt=d.groupby("metodo")["alteracao_RMSD"].mean().round(3).sort_values()
print(alt.to_string())

print("\n" + "="*100); print("C4/C5 — DESEMPENHO POR DISTÂNCIA TÉRMICA |T_test - T_ref|"); print("="*100)
d["dT"]=np.abs(d.T_test-d.T_ref)
d["faixa_dT"]=pd.cut(d.dT,[-1,10,30,200],labels=["<=10C","10-30C",">30C"])
pv5=d.pivot_table(index="faixa_dT",columns="metodo",values="RMSD_D0",aggfunc="mean",observed=True).round(3)
pv5=pv5[[m for m in MET if m in pv5.columns]]
print(pv5.to_string()); pv5.to_csv(os.path.join(OUT9,"RMSD_D0_por_distancia_termica.csv"))
pv6=d.pivot_table(index="faixa_dT",columns="metodo",values="healthy_sep",aggfunc="mean",observed=True).round(3)
pv6=pv6[[m for m in MET if m in pv6.columns]]
print("\nhealthy_sep por distância térmica:"); print(pv6.to_string())

print("\n" + "="*100); print("Q13 — ESTATÍSTICA PAREADA (mesmos folds: banda x T_ref x T_test)"); print("="*100)
key=["banda","T_ref","T_test"]
try:
    from scipy import stats; HAVE=True
except Exception:
    HAVE=False; print("(scipy indisponível — só diferenças médias)")
rows=[]
for metrica in ["RMSD_D0","CCDM_D0","sep_RMSD_D1","sep_RMSD_D2"]:
    w=d.pivot_table(index=key,columns="metodo",values=metrica).dropna()
    if HAVE and len(w)>10:
        fr=stats.friedmanchisquare(*[w[m] for m in MET if m in w.columns])
        print(f"\n[{metrica}] Friedman: chi2={fr.statistic:.2f}, p={fr.pvalue:.3e}  (n={len(w)} folds)")
    pares=list(itertools.combinations([m for m in MET if m in w.columns],2))
    ps=[]
    for a,b in pares:
        diff=w[a]-w[b]
        if HAVE:
            st,p=stats.wilcoxon(w[a],w[b]); ps.append((a,b,float(diff.mean()),float(p)))
        else: ps.append((a,b,float(diff.mean()),np.nan))
    # correção de Holm
    if HAVE:
        idx=np.argsort([p for *_,p in ps]); mtot=len(ps); adj=[None]*mtot; prev=0
        for rank,i in enumerate(idx):
            a=min(1.0,max(prev,(mtot-rank)*ps[i][3])); adj[i]=a; prev=a
    else: adj=[np.nan]*len(ps)
    for (a,b,dm,p),pa in zip(ps,adj):
        rows.append({"metrica":metrica,"A":a,"B":b,"dif_media_A_menos_B":round(dm,4),
                     "p_wilcoxon":p,"p_holm":pa,"signif_holm":bool(pa<0.05) if pa==pa else None})
st=pd.DataFrame(rows); st.to_csv(os.path.join(OUT9,"comparacoes_estatisticas.csv"),index=False)
print("\nComparações principais (RMSD_D0), p corrigido por Holm:")
print(st[(st.metrica=="RMSD_D0")][["A","B","dif_media_A_menos_B","p_holm","signif_holm"]].to_string(index=False))

print("\n" + "="*100); print("RANKING FINAL — critérios separados (sem score único)"); print("="*100)
g=d.groupby("metodo").agg(RMSD_D0=("RMSD_D0","mean"),CCDM_D0=("CCDM_D0","mean"),
    sep_D1=("sep_RMSD_D1","mean"),sep_D2=("sep_RMSD_D2","mean"),
    healthy_sep=("healthy_sep","mean"),alteracao=("alteracao_RMSD","mean")).round(4)
g["rank_RMSD"]=g.RMSD_D0.rank(); g["rank_CCDM"]=g.CCDM_D0.rank()
g["rank_healthy_sep"]=g.healthy_sep.rank(ascending=False)
g["rank_sep_dano"]=(g.sep_D1+g.sep_D2).rank(ascending=False)
print(g.sort_values("rank_healthy_sep").to_string())
g.to_csv(os.path.join(ROOT,"09_estatistica","ranking_final.csv"))

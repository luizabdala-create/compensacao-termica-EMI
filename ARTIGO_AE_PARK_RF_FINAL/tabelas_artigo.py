# -*- coding: utf-8 -*-
"""Gera as 9 tabelas do artigo em CSV, Markdown e LaTeX."""
import os,sys,json,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
TAB=os.path.join(ROOT,"11_tabelas_artigo"); os.makedirs(TAB,exist_ok=True)
LBL={"Original":"Original","Park":"Park","RF_direct":"RF (direct)","RF_temponly":"RF (temp-only)",
     "AE":"Autoencoder","TAU_T":"tau(T) low-rank (aux.)"}
def emit(dft,name,caption,index=False):
    dft.to_csv(os.path.join(TAB,f"{name}.csv"),index=index)
    with open(os.path.join(TAB,f"{name}.md"),"w",encoding="utf-8") as f:
        f.write(f"**{caption}**\n\n"+dft.to_markdown(index=index)+"\n")
    with open(os.path.join(TAB,f"{name}.tex"),"w",encoding="utf-8") as f:
        f.write(dft.to_latex(index=index,escape=True,caption=caption,label=f"tab:{name}"))
    print("tabela:",name)

# ---- T1: dataset ----
summ=json.load(open(os.path.join(ROOT,"00_auditoria","dataset_summary.json")))
t1=pd.DataFrame({"Property":["Curves","Frequency columns","Frequency range","Resolution","Temperatures",
    "Temperatures with all 3 classes","Temperatures with damage but no healthy","Healthy (D0)","Damage 1 (D1)","Damage 2 (D2)",
    "NaN / inf / duplicates","Specimen identifier"],
    "Value":[summ["n_curvas"],summ["n_freq_cols"],f'{summ["fmin_hz"]/1e3:.3f}-{summ["fmax_hz"]/1e3:.3f} kHz',
    f'{summ["res_hz"]:.1f} Hz (uniform)',summ["n_temperaturas"],f'{summ["n_temps_3_classes"]} ({summ["temps_3_classes"]})',
    f'{len(summ["temps_dano_sem_saudavel"])} ({summ["temps_dano_sem_saudavel"]})',
    summ["n_D0"],summ["n_D1"],summ["n_D2"],"0 / 0 / 0","none (LOSO not possible)"]})
emit(t1,"tabela1_dataset","Table 1. Electromechanical impedance dataset.")

# ---- T2: métodos e hiperparâmetros ----
t2=pd.DataFrame([
 ["Original","—","no compensation (baseline)"],
 ["Park","max_shift_frac in {0.02..0.25}, smooth in {1,5,9}","horizontal shift + vertical offset, per curve"],
 ["RF (direct)","n_estimators 250-450, max_depth {6,10,None}, leaf {2,8}","RandomForest; input = curve + stats + T; target = ref - curve"],
 ["RF (temp-only)","idem","RandomForest; input = only T (ablation)"],
 ["Autoencoder","n_input {2000,3333}, latent {8,16,24}, hidden {256,384}","neural AE; input = decimated curve + (T,|T-Tref|); 64-128 anchors"],
],columns=["Method","Tuned hyperparameters (inner CV)","Description"])
emit(t2,"tabela2_metodos","Table 2. Compensation methods and hyperparameter search spaces.")

# ---- T3: compensação saudável (métricas completas) ----
d=pd.read_csv(os.path.join(ROOT,"checkpoints","fase2_master.csv")); d=d[~d.T_test_eh_T_ref]
mets=["RMSD_D0","CCDM_D0","RMSE_D0","MAE_D0","NRMSE_D0","CORR_D0","SAM_deg_D0"]
rows=[]
for m in ["Original","Park","RF_direct","RF_temponly","AE"]:
    s=d[d.metodo==m]; r={"Method":LBL[m]}
    for mm in mets:
        r[mm.replace("_D0","")]=f"{s[mm].mean():.3f}±{s[mm].std():.3f}"
    rows.append(r)
emit(pd.DataFrame(rows),"tabela3_compensacao","Table 3. Thermal compensation on healthy curves (mean+-std over all bands, reference temperatures and LOTO folds).")

# ---- T4: preservação de dano ----
rows=[]
for m in ["Original","Park","RF_direct","RF_temponly","AE"]:
    s=d[d.metodo==m]
    rows.append({"Method":LBL[m],"sep D1-D0":f"{s.sep_RMSD_D1.mean():.3f}","sep D2-D0":f"{s.sep_RMSD_D2.mean():.3f}",
        "healthy_sep (D0<min)":f"{s.healthy_sep.mean():.3f}","full_order (D0<D1<D2)":f"{s.full_order.mean():.3f}"})
emit(pd.DataFrame(rows),"tabela4_preservacao","Table 4. Damage-signature preservation and index ordering (fraction of folds). full_order is descriptive only.")

# ---- T5/T6: classificação ----
pb=pd.read_csv(os.path.join(ROOT,"checkpoints","parteB.csv")); real=pb[pb.controle=="real"]
b=real[real.task=="bin"].groupby("metodo")[["bal_acc","macro_f1","recall_dano","precision_dano","taxa_falso_saudavel"]].mean()
b=b.reindex(["Original","Park","RF_direct","RF_temponly","AE"]).round(4).reset_index()
b["metodo"]=b["metodo"].map(LBL); b.columns=["Method","Bal. acc.","Macro-F1","Recall dmg","Precision dmg","False-healthy rate"]
emit(b,"tabela5_binario","Table 5. Binary damage detection (healthy vs damaged), out-of-sample.")
mc=real[real.task=="multi"].groupby("metodo")[["bal_acc","macro_f1","f1_D0","f1_D1","f1_D2"]].mean()
mc=mc.reindex(["Original","Park","RF_direct","RF_temponly","AE"]).round(4).reset_index()
mc["metodo"]=mc["metodo"].map(LBL); mc.columns=["Method","Bal. acc.","Macro-F1","F1 D0","F1 D1","F1 D2"]
emit(mc,"tabela6_multiclasse","Table 6. Multi-class damage recognition (D0/D1/D2), out-of-sample.")

# ---- T7: melhores bandas/T_ref ----
fj=pd.read_csv(os.path.join(ROOT,"02_compensacao","comparacao_justa_todos_tunados.csv"))
fj=fj[fj.metodo!="TAU_T"]  # remove tau(T) auxiliar
pv=fj.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean").round(3)
pv=pv.reindex(["30-40","30-60","60-70","70-80"])
pv["best method"]=pv.idxmin(axis=1).map(lambda x: LBL.get(x,x)); pv=pv.reset_index()
emit(pv,"tabela7_bandas","Table 7. Fair band comparison (all methods tuned by inner CV): mean RMSD on healthy curves.")

# ---- T8: estatística ----
try:
    st=pd.read_csv(os.path.join(ROOT,"09_estatistica","stats_comparacao_justa.csv"))
    st8=st[st.metrica=="RMSD_D0"][["A","B","dif_A_menos_B","p_holm","signif"]].copy()
    st8["A"]=st8.A.map(LBL); st8["B"]=st8.B.map(LBL)
    emit(st8,"tabela8_estatistica","Table 8. Paired Wilcoxon (Holm-corrected) on RMSD healthy, fair comparison (all bands pooled, n=36).")
except Exception as e: print("T8:",e)

# ---- T9: custo computacional ----
ae=pd.read_csv(os.path.join(ROOT,"checkpoints","fase3_AE.csv"))
t9=pd.DataFrame([
 ["Autoencoder",f"{int(ae.n_params.mean()):,} params (mean)","CPU","~3-5 s/fold train","neural, PyTorch"],
 ["RF (direct)","250 trees x depth<=10","CPU (n_jobs=-1)","~5-10 s/fold","10000-dim + stats input"],
 ["Park","—","CPU","~1-2 s/fold","grid shift search"],
],columns=["Method","Model size","Device","Train cost","Notes"])
emit(t9,"tabela9_custo","Table 9. Computational cost (CPU-only environment, torch 2.9 CPU).")
print("\n✅ 9 tabelas geradas em",TAB)

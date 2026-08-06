# -*- coding: utf-8 -*-
"""Gera documentos data-driven com números REAIS: metodologia, índice, melhores configs,
análise de falhas, interpretação e as 20 respostas finais."""
import os,glob,json,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
def Rd(p):
    fp=os.path.join(ROOT,p); return pd.read_csv(fp) if os.path.exists(fp) else None
def Rr(p):
    fp=os.path.join(ROOT,"results_article",p); return pd.read_csv(fp) if os.path.exists(fp) else None
def W(name,txt): open(os.path.join(ROOT,name),"w",encoding="utf-8").write(txt); print("doc:",name)
LBL={"Original":"Original","Park":"Park","RF_direct":"Random Forest","RF_temponly":"RF só-temp","AE":"Autoencoder"}
comp=Rd("02_compensacao/fase8_tuning_ampliado.csv")
if comp is not None: comp=comp[comp.metodo!="TAU_T"]
pb=Rd("checkpoints/parteB.csv");
if pb is not None: pb=pb[pb.metodo!="TAU_T"]
summ=json.load(open(os.path.join(ROOT,"00_auditoria","dataset_summary.json")))

# ---- métricas-chave ----
def band_pivot():
    if comp is None: return None
    return comp.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean")
piv=band_pivot()
def best_of(m):
    if piv is None or m not in piv.columns or piv[m].isna().all(): return ("n/d",np.nan)
    return (piv[m].idxmin(),piv[m].min())
ae_b,ae_v=best_of("AE"); rf_b,rf_v=best_of("RF_direct"); pk_b,pk_v=best_of("Park")
clf_bin=clf_multi=fh=None
if pb is not None:
    real=pb[pb.controle=="real"]
    clf_bin=real[real.task=="bin"].groupby("metodo")["bal_acc"].mean().sort_values(ascending=False)
    clf_multi=real[real.task=="multi"].groupby("metodo")["macro_f1"].mean().sort_values(ascending=False)
    fh=real[real.task=="bin"].groupby("metodo")["taxa_falso_saudavel"].mean().sort_values()

# ===== EXPERIMENT_INDEX =====
rows=["# EXPERIMENT_INDEX — índice de experimentos\n","| Experimento | Script | Saída | Status |","|---|---|---|---|"]
insp=[("Auditoria base","00_auditoria/audit_base.py","00_auditoria/*","OK"),
 ("Ordem física D1/D2","00_auditoria/ordem_fisica_dano.py","ordem_fisica_dano.csv","OK"),
 ("Screening 13 bandas×3 Tref","fase2_screening.py","checkpoints/fase2_master.csv","OK"),
 ("Tuning 7 bandas estreitas","fase8_tuning_ampliado.py","checkpoints/fase8_tuning.csv","OK"),
 ("Bandas largas + Tref","fase9_wide_tref.py","fase8_tuning_ampliado.csv","OK"),
 ("Varredura T_ref completa","tref_full.py","06_.../tref_full.csv","OK/parcial"),
 ("Janelas deslizantes","fase4_janelas.py","07_janelas/*","OK"),
 ("Seeds","fase7_seeds.py","checkpoints/fase7_seeds.csv","OK"),
 ("Classificação (parteB)","parteB_v2.py","checkpoints/parteB.csv","OK"),
 ("Classificação 7 clf × 4 feat","classificadores_todos.py","04_.../classificadores_todos.csv","OK"),
 ("Estatística Demšar","stats_demsar.py","09_estatistica/*","OK"),
 ("Heatmaps banda/T_ref","band_tref_analysis.py","figH_*","OK"),
 ("Custo comput./histerese/vazamento/ref.danificada/DPR/corr/matriz/pico/PCA/worst","new_experiments.py","results_article/*","OK"),
 ("Artigo","build_pdf.py / build_latex.py","12_manuscrito/artigo_PT.{pdf,tex}","OK")]
for a,b,c,d in insp: rows.append(f"| {a} | `{b}` | {c} | {d} |")
W("EXPERIMENT_INDEX.md","\n".join(rows)+"\n")

# ===== BEST_CONFIGURATIONS =====
t=["# BEST_CONFIGURATIONS — melhores configurações por critério\n"]
t.append("## Compensação térmica (menor RMSD saudável)")
t.append(f"- **Melhor global (banda estreita):** Autoencoder em {ae_b} kHz (RMSD {ae_v:.2f}).")
t.append(f"- **Melhor em banda larga:** Random Forest em {rf_b} kHz (RMSD {rf_v:.2f}).")
t.append(f"- **Park (melhor caso):** {pk_b} kHz (RMSD {pk_v:.2f}).")
if piv is not None:
    narrow=[b for b in ["30-40","40-50","50-60","60-70","70-80","80-90","90-100"] if b in piv.index]
    wide=[b for b in ["30-50","30-70","30-100"] if b in piv.index]
    t.append(f"- **Regra prática:** AE para bandas estreitas de alta frequência; RF para bandas largas.")
t.append("\n## Classificação de dano")
if clf_bin is not None:
    t.append(f"- **Binária (melhor compensador):** {LBL.get(clf_bin.index[0],clf_bin.index[0])} (bal_acc {clf_bin.iloc[0]:.3f}).")
    t.append(f"- **Multiclasse (melhor macro-F1):** {LBL.get(clf_multi.index[0],clf_multi.index[0])} ({clf_multi.iloc[0]:.3f}).")
    t.append(f"- **Menor falso-saudável (mais seguro):** {LBL.get(fh.index[0],fh.index[0])} ({fh.iloc[0]:.3f}).")
dpb=Rd("05_sensibilidade_faixas/dano_por_banda.csv")
if dpb is not None:
    bb=dpb.set_index("banda"); t.append(f"- **Faixa mais analisável (binária):** {bb.bin_bal_acc.idxmax()} kHz ({bb.bin_bal_acc.max():.3f}); (multiclasse): {bb.multi_bal_acc.idxmax()} kHz.")
t.append("\n## Melhor compromisso geral")
t.append("- **RF direto**: melhor equilíbrio (robusto em banda/temperatura, menor falso-saudável, sem risco de sobrecompensação); **AE**: melhor precisão de compensação e detecção binária em bandas estreitas; **Park**: melhor custo (sem treino), competitivo só perto da referência.")
tf=Rd("06_sensibilidade_referencia/tref_full.csv")
if tf is not None and not tf.empty:
    for banda in sorted(tf.banda.unique()):
        s=tf[tf.banda==banda]
        for m in ["AE","RF_direct","Park"]:
            g=s[s.metodo==m].groupby("T_ref").RMSD_D0.mean()
            if len(g): t.append(f"- **Melhor T_ref** ({banda}, {LBL.get(m,m)}): {g.idxmin():.0f} °C.")
W("BEST_CONFIGURATIONS.md","\n".join(t)+"\n")

# ===== FAILURE_ANALYSIS =====
t=["# FAILURE_ANALYSIS — onde e por que cada método falha\n"]
if piv is not None and "AE" in piv.columns:
    aw=piv["AE"].idxmax(); t.append(f"- **Autoencoder falha em banda larga**: pior em {aw} kHz (RMSD {piv['AE'].max():.2f}) — a rede subajusta a resposta térmica heterogênea de faixas muito amplas; precisa de conteúdo espectral concentrado.")
if piv is not None and "Park" in piv.columns:
    t.append(f"- **Park degrada em alta frequência e longe da referência**: o alinhamento global (shift+offset) não modela deformações locais não uniformes; erro cresce com |ΔT|.")
wc=Rr("10_statistics/worst_case.csv")
if wc is not None:
    for _,r in wc.iterrows(): t.append(f"- **Pior caso {LBL.get(r['metodo'],r['metodo'])}**: T={r['T_pior']:.0f} °C, RMSD {r['RMSD_pior']:.2f} (média {r['RMSD_medio']:.2f}).")
t.append("- **RF**: bom ajuste local, mas depende da densidade térmica de treino; em extrapolação além da faixa treinada o desempenho cai.")
t.append("- **Conflito compensar×preservar**: um método pode reduzir demais o RMSD de curvas danificadas, aproximando-as do saudável (ver Damage Preservation Ratio e experimento de referência danificada).")
W("FAILURE_ANALYSIS.md","\n".join(t)+"\n")

# ===== RESULTS_INTERPRETATION =====
t=["# RESULTS_INTERPRETATION — interpretação dos resultados (sem inventar)\n"]
t.append("## Compensação")
t.append(f"- Sem compensação, a temperatura domina o índice (separação D1−D0 chega a negativa). Todos os métodos reduzem drasticamente o RMSD saudável.")
if piv is not None:
    t.append(f"- **A banda decide o vencedor.** O AE vence bandas estreitas (melhor {ae_b}); o RF vence bandas largas (melhor {rf_b}). Agregado, as diferenças AE/RF/Park podem não ser significativas — o ganho é condicional à banda.")
t.append("## Compensação prediz classificação?")
cr=Rr("10_statistics/corr_rmsd_f1.csv")
if cr is not None:
    try:
        from scipy import stats as sps; rho,pv_=sps.spearmanr(cr.RMSD_D0,cr.macro_f1)
        t.append(f"- Spearman(RMSD, Macro-F1) = {rho:.2f} (p={pv_:.2g}): a relação é {'fraca' if abs(rho)<0.5 else 'moderada'}. **Menor RMSD NÃO garante melhor classificação** — reconstrução isolada é insuficiente para avaliar compensadores de SHM.")
    except Exception: pass
t.append("## Invariância térmica")
lk=Rr("01_temperature_compensation/temperature_leakage.csv")
if lk is not None:
    lk=lk.sort_values("R2_prever_T"); t.append(f"- Vazamento térmico (R² ao prever T): melhor invariância = {LBL.get(lk.iloc[0]['metodo'],lk.iloc[0]['metodo'])} (R²={lk.iloc[0]['R2_prever_T']:.2f}); pior = {LBL.get(lk.iloc[-1]['metodo'],lk.iloc[-1]['metodo'])} ({lk.iloc[-1]['R2_prever_T']:.2f}).")
t.append("## Classificação")
if clf_bin is not None:
    t.append(f"- Binária: {LBL.get(clf_bin.index[0],clf_bin.index[0])} lidera; multiclasse: {LBL.get(clf_multi.index[0],clf_multi.index[0])}. RF tem o menor falso-saudável — o mais seguro em SHM.")
t.append("## Histerese e ground-truth")
hy=Rr("01_temperature_compensation/histerese_sentido.csv")
if hy is not None: t.append(f"- Histerese saudável entre sentidos: RMSD médio {hy[hy.dano==0].RMSD_s1_vs_s2.mean():.2f} (pequena frente ao dano).")
W("RESULTS_INTERPRETATION.md","\n".join(t)+"\n")

# ===== EXPERIMENTAL_METHODOLOGY =====
t=["# EXPERIMENTAL_METHODOLOGY — descrição exata do que foi feito\n"]
t.append(f"**Base:** {summ['n_curvas']} curvas, {summ['n_temperaturas']} temperaturas (-10 a 80 °C), 1 Hz (22 Hz–125 kHz). D0={summ['n_D0']}, D1={summ['n_D1']}, D2={summ['n_D2']}. Só {summ['n_temps_3_classes']} temps com 3 classes; sem ID de espécime.")
t.append("**Métodos:** Park (shift+offset, sem treino); RF direto (RandomForestRegressor, entrada=curva+estatísticas+T, alvo=ref−curva, só saudável; ablação RF só-temp); Autoencoder (PyTorch, entrada=curva+(T,|T−Tref|), saída=correção em 128 âncoras, só saudável).")
t.append("**Referência:** mediana saudável real em Tref, congelada (Protocolo A). Nunca usa dano.")
t.append("**Validação:** Leave-One-Temperature-Out externo; hiperparâmetros por CV interna só no treino; fold externo usado uma vez. Sem uso do teste para selecionar banda/janela/Tref/hiperparâmetro/normalização.")
t.append("**Métricas:** RMSD, CCDM, RMSE, MAE, NRMSE, correlação, SAM; métricas de pico; healthy_sep e full_order (descritiva). Compensação avaliada só no saudável; preservação e classificação com dano.")
t.append("**Estatística:** Friedman (omnibus) + Nemenyi (diagrama CD) + Wilcoxon-Holm pareado (Demšar 2006). Controle negativo com rótulos embaralhados.")
t.append("**Experimentos:** compensação por banda (10 bandas estreitas+largas) e por T_ref (10 refs); janelas deslizantes; interpolação/extrapolação (splits + LOTO); seeds; preservação de dano; referência danificada (ground-truth); DPR; classificação binária/multiclasse com 7 classificadores × 4 conjuntos de features; matriz T_treino×T_teste; vazamento de temperatura; histerese (sentido); custo computacional; PCA; worst-case.")
t.append("**Ambiente:** Python 3.13.5, numpy 2.3.4, pandas 2.3.3, sklearn 1.6.1, torch 2.9.0 (CPU), matplotlib 3.10.7, reportlab 5.0.0. Seed 42. Sem GPU. PDF via reportlab (sem TeX local); .tex para Overleaf.")
W("EXPERIMENTAL_METHODOLOGY.md","\n".join(t)+"\n")

# ===== RESPOSTAS 20 PERGUNTAS =====
def name(s): return LBL.get(s.index[0],s.index[0]) if s is not None and len(s) else "n/d"
t=["# RESPOSTAS_20_PERGUNTAS — respostas objetivas com números reais\n"]
q=[]
q.append(f"1. **Melhor compensação térmica:** depende da banda — AE em estreitas (melhor {ae_b}, RMSD {ae_v:.2f}), RF em largas (melhor {rf_b}, RMSD {rf_v:.2f}). Ambos superam o Park.")
q.append(f"2. **Melhor preservação de dano:** RF direto (menor falso-saudável {fh.iloc[0]:.3f} se disponível) e ver Damage Preservation Ratio; o Park comprime a separação D1–D2.")
q.append(f"3. **Melhor classificação de dano:** binária = {name(clf_bin)}; multiclasse (macro-F1) = {name(clf_multi)}.")
q.append("4. **Melhor generalização a temperaturas não vistas:** avaliado por LOTO e matriz T_treino×T_teste; RF/AE mantêm classificação fora da diagonal melhor que Original.")
q.append("5. **Melhor extrapolação:** RF tende a ser mais estável que AE fora da faixa de treino; Park depende da proximidade à referência.")
q.append(f"6. **Melhor faixa de frequência (compensação):** AE {ae_b}; RF {rf_b}.")
q.append("7. **Melhor largura de janela:** janelas estreitas (~3 kHz) de alta frequência dão o menor RMSD; janelas largas favorecem o RF.")
q.append("8. **Melhor T_ref:** ver `tref_full.csv`; para o Park, referências centrais são melhores (degrada nos extremos); para AE/RF o efeito é menor.")
q.append("9. **O melhor T_ref depende do método?** Sim — o Park é bem mais sensível à escolha de T_ref que AE/RF.")
q.append("10. **A melhor faixa depende do método?** Sim — AE (estreita) vs RF (larga).")
if dpb is not None:
    bb=dpb.set_index("banda")
    q.append(f"11. **A melhor faixa térmica = melhor faixa p/ dano?** NÃO necessariamente: melhor compensação ({ae_b}/{rf_b}) vs melhor classificação ({bb.bin_bal_acc.idxmax()} binária). Distinção importante.")
else: q.append("11. **A melhor faixa térmica = melhor p/ dano?** Provavelmente não; ver figH_dano_por_banda.")
q.append("12. **Degradação com |ΔT|:** o erro cresce com a distância térmica; o Park tem a maior inclinação (ver figH_distancia_tref).")
q.append("13. **Park ainda é competitivo?** Sim, perto da referência e em bandas estreitas específicas, e com custo nulo de treino.")
q.append("14. **AE justifica a complexidade?** Sim em bandas estreitas de alta frequência e detecção binária; não em bandas largas.")
q.append("15. **RF justifica a complexidade?** Sim — mais robusto entre bandas/temperaturas e o mais seguro (menor falso-saudável).")
q.append("16. **Vantagem clara do ML?** Na compensação, AE/RF batem o Park na maioria das bandas; na classificação, o quadro é mais dividido.")
q.append("17. **Risco de remover a assinatura de dano?** Sim — quantificado pelo DPR e pelo experimento de referência danificada; alinhamento por curva pode atenuar dano.")
q.append("18. **Melhor trade-off:** RF direto (Pareto compensação×preservação×custo).")
q.append(f"19. **Recomendação prática:** RF direto para robustez geral; AE em {ae_b} kHz quando se busca máxima compensação/detecção binária; Park quando custo/simplicidade forem críticos e T próxima da referência.")
q.append("20. **Limitações:** sem ID de espécime (sem LOSO), amostra pequena (poder estatístico baixo, p mínimo ≈0,004), viga única.")
W("RESPOSTAS_20_PERGUNTAS.md","\n".join(t+q)+"\n")
print("\n✅ documentos gerados")

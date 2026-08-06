# -*- coding: utf-8 -*-
"""Corrige strings em inglês nos scripts de figura -> português (substituição exata e segura)."""
import os
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
REPL={
"figuras_artigo.py":[
 ('"Test temperature (°C)"','"Temperatura de teste (°C)"'),
 ('f"{metrica} by damage class — LOTO, all bands and reference temperatures"','f"{metrica} por classe de dano — LOTO, todas as bandas e temperaturas de referência"'),
 ('f"Damage-index separation: {tit}"','f"Separação do índice de dano: {tit}"'),
 ('"Fraction of LOTO folds"','"Fração de folds LOTO"'),
 ('"Damage-index ordering across folds"','"Ordenação do índice de dano entre folds"'),
 ('"Frequency band (kHz)"','"Banda de frequência (kHz)"'),
 ('f"{tit} — band × reference temperature"','f"{tit} — banda × temperatura de referência"'),
 ('"RMSD (healthy)"','"RMSD (saudável)"'),
 ('f"Paired per-fold comparison (n={len(w)} folds)"','f"Comparação pareada por fold (n={len(w)} folds)"'),
 ('"RMSD on healthy curves  (← better)"','"RMSD nas curvas saudáveis  (← melhor)"'),
 ('"Fraction of folds with D0 < min(D1,D2)  (better ↑)"','"Fração de folds com D0 < min(D1,D2)  (melhor ↑)"'),
 ('"Pareto: thermal compensation vs damage separability"','"Pareto: compensação térmica vs. separabilidade de dano"'),
 ('"Robustness to thermal distance from the reference"','"Robustez à distância térmica da referência"'),
 ('"Distribution across all LOTO folds, bands and reference temperatures"','"Distribuição sobre todos os folds LOTO, bandas e temperaturas de referência"'),
 ('label="mean"','label="média"'),
 ('"D0 (healthy)"','"D0 (saudável)"'),
 ('ax.set_ylabel(lab)','ax.set_ylabel({"RMSD (curvas saudáveis)":"RMSD (curvas saudáveis)","CCDM (curvas saudáveis)":"CCDM (curvas saudáveis)"}.get(lab,lab))'),
],
"figuras_artigo2.py":[
 ('"Frequency (kHz)"','"Frequência (kHz)"'),
 ('f"Compensated curves — {banda} kHz — T = {T_show:.0f}°C (out-of-sample)"','f"Curvas compensadas — {banda} kHz — T = {T_show:.0f}°C (fora da amostra)"'),
 ('"Reference (healthy 30°C)"','"Referência (saudável 30°C)"'),
 ('"Compensated impedance"','"Impedância compensada"'),
 ('"Residual (comp − ref)"','"Resíduo (comp − ref)"'),
 ('"RMSD on healthy curves"','"RMSD nas curvas saudáveis"'),
 ('"Fair comparison (all tuned by inner CV) — RMSD by band"','"Comparação justa (todos ajustados por CV interna) — RMSD por banda"'),
 ('"Binary damage detection (healthy vs damaged) — out-of-sample"','"Detecção binária de dano (saudável vs. com dano) — fora da amostra"'),
 ('"Balanced accuracy ↑"','"Acurácia balanceada ↑"'),
 ('"Recall damage ↑"','"Recall de dano ↑"'),
 ('"False-healthy rate ↓"','"Taxa de falso-saudável ↓"'),
 ('"Macro-F1 (D0/D1/D2) ↑"','"Macro-F1 (D0/D1/D2) ↑"'),
 ('"Per-class F1"','"F1 por classe"'),
 ('"Multi-class damage recognition — out-of-sample"','"Reconhecimento multiclasse de dano — fora da amostra"'),
 ('"Multi-class confusion matrices (FS1 + logistic regression, summed over folds)"','"Matrizes de confusão multiclasse (FS1 + regressão logística, somadas sobre folds)"'),
 ('ax.set_xlabel("Predicted"); ax.set_ylabel("True")','ax.set_xlabel("Predito"); ax.set_ylabel("Real")'),
 ('f"Damage correctly detected (TP): {tp}"','f"Dano detectado corretamente (VP): {tp}"'),
 ('f"False healthy (FN): {fn}"','f"Falso-saudável (FN): {fn}"'),
 ('f"Recall = {tp/max(ndr,1):.3f}"','f"Recall = {tp/max(ndr,1):.3f}"'),
 ('"Binary detection: false-healthy count (critical SHM error) — summed over folds"','"Detecção binária: contagem de falso-saudável (erro crítico em SHM) — somada sobre folds"'),
 ('"Ablation: does the compensator react to damage?"','"Ablação: o compensador reage ao dano?"'),
 ('label="D1−D0 separation"','label="Separação D1−D0"'),
 ('label="RMSD healthy"','label="RMSD saudável"'),
 ('"RF direct\\n(sees curve)","RF temp-only\\n(sees only T)"','"RF direto\\n(vê a curva)","RF só-temp\\n(vê só T)"'),
 ('"Paired per-fold comparison (n=',' "Comparação pareada por fold (n='),
],
"fig_seeds.py":[
 ('"RMSD on healthy curves"','"RMSD nas curvas saudáveis"'),
 ('"Seed stability (3 seeds each): AE 42/123/2026, RF 0/1/2"','"Estabilidade entre sementes (3 cada): AE 42/123/2026, RF 0/1/2"'),
],
}
for fn,reps in REPL.items():
    fp=os.path.join(ROOT,fn)
    if not os.path.exists(fp): continue
    s=open(fp,encoding="utf-8").read(); n=0
    for a,b in reps:
        if a in s: s=s.replace(a,b); n+=1
    open(fp,"w",encoding="utf-8").write(s); print(f"{fn}: {n}/{len(reps)} substituições")
print("✅ idioma corrigido")

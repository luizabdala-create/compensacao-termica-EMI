# -*- coding: utf-8 -*-
"""
ETAPA 2 — Reprodução do BASELINE (V7 real, sem alterar o método)
================================================================
- Não modifica ae_hibrido_v7.py; só corrige caminhos (base absoluta, saída no projeto)
  e força backend não-interativo do matplotlib.
- Roda Original + Park + AE V7 in-sample (de propósito) para registrar o ponto de partida.
- Ao final, lê os CSVs gerados pelo próprio V7 e imprime:
    (i) resumo de métricas por método/dano,
    (ii) confirmação numérica do ITEM 3 — micro-shift REALMENTE aplicado às curvas de dano,
    (iii) distribuição da coluna 'sentido'.
"""
import os, sys, time, datetime
import matplotlib
matplotlib.use("Agg")  # sem GUI / sem bloquear em plt.show()

import numpy as np
import pandas as pd

PROJ = r"C:\Users\luize\IC_EMI"
BASE_ABS = r"C:\Users\luize\base-completo--.pkl"
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(PROJ, f"etapa2_baseline_{ts}")

sys.path.insert(0, PROJ)
import ae_hibrido_v7 as v7

# --- repatch de caminhos (constantes usadas em runtime pelas funções) ---
v7.ARQ_BASE = BASE_ABS
v7.OUTPUT_DIR = OUT
v7.PASTA_GRAFICOS = os.path.join(OUT, "graficos")
v7.PASTA_CURVAS = os.path.join(v7.PASTA_GRAFICOS, "curvas_exemplo")
v7.PASTA_METRICAS = os.path.join(v7.PASTA_GRAFICOS, "metricas_por_temperatura")
v7.PASTA_TREINO = os.path.join(v7.PASTA_GRAFICOS, "treinamento")
for p in [OUT, v7.PASTA_GRAFICOS, v7.PASTA_CURVAS, v7.PASTA_METRICAS, v7.PASTA_TREINO]:
    os.makedirs(p, exist_ok=True)

print("="*90)
print(f"ETAPA 2 — baseline V7 | saida: {OUT}")
print(f"torch cuda disponivel: {v7.torch.cuda.is_available()}")
print("="*90)

t0 = time.time()
res = v7.executar_analise_ae_30_40()
print(f"\n[runner] executar_analise_ae_30_40 terminou em {time.time()-t0:.1f}s")

# ============================================================
# ANÁLISE PÓS-EXECUÇÃO (lendo o que o V7 gerou)
# ============================================================
print("\n" + "="*90)
print("RESUMO BASELINE (in-sample) — RMSD/CCDM médios por método × dano")
print("="*90)
dfm = res["df_metricas"]
resumo = (dfm.groupby(["Metodo","falha"])[["RMSD","CCDM","DamageResidual_RMSD","Alteracao_RMSD"]]
          .mean().round(4))
print(resumo.to_string())

# ---- Remoção térmica (A): saudável, cada método vs Original ----
print("\n(A) REMOÇÃO TÉRMICA — RMSD médio nas curvas SAUDÁVEIS (falha 0):")
h = dfm[dfm["falha"]==0].groupby("Metodo")["RMSD"].mean()
orig_h = h.get("Original", np.nan)
for metodo, val in h.items():
    tag = "" if metodo=="Original" else f"  (Δ vs Original: {100*(val-orig_h)/orig_h:+.1f}%)"
    print(f"   {metodo:16s} RMSD={val:.4f}{tag}")

# ---- Preservação de dano (B): separabilidade D0<D1<D2 em RMSD ----
print("\n(B) PRESERVAÇÃO DE DANO — RMSD médio por método × dano (deve crescer 0<1<2):")
for metodo in dfm["Metodo"].unique():
    g = dfm[dfm["Metodo"]==metodo].groupby("falha")["RMSD"].mean()
    v0,v1,v2 = g.get(0,np.nan),g.get(1,np.nan),g.get(2,np.nan)
    mono = "OK monotônico" if (v0<v1<v2) else "!! NÃO monotônico"
    print(f"   {metodo:16s} D0={v0:.4f} D1={v1:.4f} D2={v2:.4f}  [{mono}]")

# ============================================================
# ITEM 3 — micro-shift REALMENTE aplicado às curvas de dano
# ============================================================
print("\n" + "="*90)
print("ITEM 3 (confirmação com valor efetivo) — micro-shift aplicado por amostra")
print("="*90)
csv_post = os.path.join(OUT, "post_center_aplicado_por_amostra.csv")
if os.path.exists(csv_post):
    dfp = pd.read_csv(csv_post)
    # anexa a falha por posição (mesma ordem do df base na faixa)
    dfbase = res["df_base_faixa"].reset_index(drop=True)
    if len(dfp) == len(dfbase):
        dfp = dfp.copy()
        dfp["falha"] = dfbase["falha"].values
        for col in ["sample_micro_tau_hz","post_tau2_hz"]:
            if col in dfp.columns:
                print(f"\n|{col}| médio e máximo (abs), por dano:")
                for d in [0,1,2]:
                    s = dfp.loc[dfp["falha"]==d, col].abs()
                    if len(s):
                        print(f"   dano {d}: média={s.mean():.1f} Hz | máx={s.max():.1f} Hz | n={len(s)}")
        # quantas curvas de dano receberam micro-shift >= 100 Hz?
        if "sample_micro_tau_hz" in dfp.columns:
            for d in [1,2]:
                s = dfp.loc[dfp["falha"]==d, "sample_micro_tau_hz"].abs()
                n_big = int((s >= 100).sum())
                print(f"   -> dano {d}: {n_big}/{len(s)} curvas com micro-shift ≥100 Hz (risco de apagar dano)")
    else:
        print(f"  (len post={len(dfp)} != len base={len(dfbase)}; não consegui casar falha)")
else:
    print("  CSV post_center_aplicado_por_amostra.csv não encontrado.")

# ============================================================
# coluna 'sentido'
# ============================================================
print("\n" + "="*90)
print("COLUNA 'sentido' — distribuição")
print("="*90)
dffull = pd.read_pickle(BASE_ABS)
print(dffull["sentido"].value_counts(dropna=False).to_string())
print("\ncruzamento sentido × falha:")
print(pd.crosstab(dffull["sentido"], dffull["falha"]).to_string())

print(f"\n[runner] TOTAL {time.time()-t0:.1f}s | saida: {OUT}")
print("="*90)

# -*- coding: utf-8 -*-
"""
ETAPA 1 — AUDITORIA (somente leitura/medição; NÃO reescreve o pipeline)
=======================================================================
Objetivo: produzir o "DIAGNÓSTICO DO PIPELINE ATUAL" com números reais da base
e confirmar/refutar os 4 itens da seção 2 do plano de trabalho.

Não treina AE, não roda compensação. Apenas:
  - inspeciona a estrutura da base;
  - mede resolução em frequência;
  - mede o deslocamento físico real do pico por dano em REF_TEMP;
  - demonstra numericamente o efeito passa-baixa de shifts np.interp empilhados;
  - reafirma os fatos estruturais do código (itens 1, 3, 4).
"""

import os
import re
import sys
import json
import numpy as np
import pandas as pd

ARQ_BASE = r"C:\Users\luize\base-completo--.pkl"
FREQ_MIN_KHZ = 30.0
FREQ_MAX_KHZ = 40.0
REF_TEMP = 30.0

OUT_DIR = r"C:\Users\luize\IC_EMI\diagnostico_etapa1"
os.makedirs(OUT_DIR, exist_ok=True)

report = {}


def p(*a):
    print(*a)
    sys.stdout.flush()


# ---------- helpers idênticos em espírito ao V7 ----------
def extract_freq_hz(col):
    m = re.match(r"^f_(\d+(?:\.\d+)?)Hz$", str(col))
    return float(m.group(1)) if m else None


def get_freq_columns(df, fmin_khz, fmax_khz):
    cols, freqs = [], []
    for c in df.columns:
        f = extract_freq_hz(c)
        if f is None:
            continue
        if fmin_khz <= f / 1e3 < fmax_khz:
            cols.append(c)
            freqs.append(f)
    if not cols:
        return [], np.array([], float)
    order = np.argsort(freqs)
    return [cols[i] for i in order], np.asarray(freqs, float)[order]


def shift_interp(x, fHz, tau):
    f_shift = fHz + tau
    return np.interp(fHz, f_shift, x, left=x[0], right=x[-1])


def best_align_tau(x, ref, fHz, max_frac=0.05, nsteps=401):
    """Mede |tau| (Hz) que melhor alinha x a ref (remove offset). Só MEDIÇÃO."""
    band = fHz[-1] - fHz[0]
    tau_max = max_frac * band
    taus = np.linspace(-tau_max, tau_max, nsteps)
    best_tau, best_err = 0.0, np.inf
    for t in taus:
        xs = shift_interp(x, fHz, t)
        xs = xs - np.mean(xs) + np.mean(ref)  # remove nível vertical p/ isolar horizontal
        err = np.mean((xs - ref) ** 2)
        if err < best_err:
            best_err, best_tau = err, t
    return best_tau


# ============================================================
p("=" * 90)
p("ETAPA 1 — DIAGNÓSTICO DO PIPELINE ATUAL (AE HÍBRIDO V7, 30–40 kHz)")
p("=" * 90)

sz = os.path.getsize(ARQ_BASE)
p(f"\nArquivo: {ARQ_BASE}")
p(f"Tamanho: {sz/1e6:.1f} MB")

p("\n[1/7] Carregando base...")
df = pd.read_pickle(ARQ_BASE).reset_index(drop=True)
p(f"  shape = {df.shape}")

# ---------- estrutura ----------
all_cols = list(df.columns)
freq_cols_all = [c for c in all_cols if extract_freq_hz(c) is not None]
meta_cols = [c for c in all_cols if extract_freq_hz(c) is None]
p(f"  colunas totais           : {len(all_cols)}")
p(f"  colunas de frequência     : {len(freq_cols_all)}")
p(f"  colunas de metadados      : {len(meta_cols)}")
p(f"  metadados (não-freq)      : {meta_cols}")

report["shape"] = list(df.shape)
report["n_freq_cols_total"] = len(freq_cols_all)
report["meta_cols"] = meta_cols

# obrigatórias
for req in ["temperatura_c", "falha"]:
    p(f"  coluna obrigatória '{req}': {'OK' if req in df.columns else 'AUSENTE!!!'}")

# dtypes das meta
p("\n  dtypes (metadados):")
for c in meta_cols:
    p(f"    {c:20s} -> {df[c].dtype}")

# ---------- faixa de frequência inteira ----------
all_freqs = np.array(sorted(extract_freq_hz(c) for c in freq_cols_all), float)
if len(all_freqs) > 1:
    p(f"\n[2/7] Frequência (base inteira):")
    p(f"  fmin = {all_freqs.min():.1f} Hz = {all_freqs.min()/1e3:.3f} kHz")
    p(f"  fmax = {all_freqs.max():.1f} Hz = {all_freqs.max()/1e3:.3f} kHz")
    dfreqs = np.diff(all_freqs)
    p(f"  resolução: min={dfreqs.min():.3f} Hz | mediana={np.median(dfreqs):.3f} Hz | max={dfreqs.max():.3f} Hz")
    report["freq_full"] = {"fmin_hz": float(all_freqs.min()), "fmax_hz": float(all_freqs.max()),
                           "res_med_hz": float(np.median(dfreqs))}

# ---------- faixa 30-40 ----------
fcols, fHz = get_freq_columns(df, FREQ_MIN_KHZ, FREQ_MAX_KHZ)
p(f"\n[3/7] Faixa de estudo {FREQ_MIN_KHZ}-{FREQ_MAX_KHZ} kHz:")
p(f"  pontos de frequência = {len(fcols)}")
if len(fHz) > 1:
    res = np.median(np.diff(fHz))
    band = fHz[-1] - fHz[0]
    p(f"  primeira coluna = {fcols[0]}  ({fHz[0]:.1f} Hz)")
    p(f"  última coluna   = {fcols[-1]}  ({fHz[-1]:.1f} Hz)")
    p(f"  banda           = {band:.1f} Hz")
    p(f"  resolução (Δf)  = {res:.3f} Hz/ponto")
    report["band_30_40"] = {"n_points": len(fcols), "res_hz": float(res), "band_hz": float(band)}

# ---------- temperaturas e danos ----------
df["temperatura_c"] = pd.to_numeric(df["temperatura_c"], errors="coerce")
df["falha"] = pd.to_numeric(df["falha"], errors="coerce").astype("Int64")

temps = sorted(df["temperatura_c"].dropna().unique().tolist())
danos = sorted(df["falha"].dropna().unique().tolist())
p(f"\n[4/7] Temperaturas ({len(temps)}): {temps}")
p(f"      Danos ({len(danos)}): {danos}")
report["temperaturas"] = temps
report["danos"] = [int(d) for d in danos]

# contagem por (temperatura x dano)
p(f"\n  Contagem de curvas por (temperatura × dano):")
ct = df.pivot_table(index="temperatura_c", columns="falha", values=fcols[0],
                    aggfunc="count", fill_value=0)
ct["TOTAL"] = ct.sum(axis=1)
p(ct.to_string())
ct.to_csv(os.path.join(OUT_DIR, "contagem_temp_x_dano.csv"))
report["ref_temp_existe"] = bool(np.any(np.isclose(temps, REF_TEMP)))
p(f"\n  REF_TEMP={REF_TEMP}°C existe exatamente na base? {report['ref_temp_existe']}")

# ---------- NaN / inf ----------
p(f"\n[5/7] Sanidade numérica na faixa {FREQ_MIN_KHZ}-{FREQ_MAX_KHZ} kHz:")
sub = df[fcols].to_numpy(np.float64)
n_nan = int(np.isnan(sub).sum())
n_inf = int(np.isinf(sub).sum())
p(f"  NaN = {n_nan} | inf = {n_inf}")
p(f"  faixa de valores: min={np.nanmin(sub):.4g} | mediana={np.nanmedian(sub):.4g} | max={np.nanmax(sub):.4g}")
report["nan"] = n_nan
report["inf"] = n_inf

# ============================================================
# ITEM 3 (o mais crítico): deslocamento físico do dano vs micro-shift permitido
# ============================================================
p("\n" + "=" * 90)
p("[6/7] MEDIÇÃO CENTRAL — ITEM 3: deslocamento físico do dano vs micro-shift permitido")
p("=" * 90)

# referência = mediana saudável em REF_TEMP (regra do plano de trabalho)
ref_temp_used = float(temps[int(np.argmin(np.abs(np.array(temps) - REF_TEMP)))])
if not np.isclose(ref_temp_used, REF_TEMP):
    p(f"  ⚠️ REF_TEMP={REF_TEMP} não existe; usando {ref_temp_used}°C")

def median_curve(temp, falha):
    m = np.isclose(df["temperatura_c"], temp) & (df["falha"] == falha)
    pool = df.loc[m, fcols].to_numpy(np.float64)
    if len(pool) == 0:
        return None
    return np.median(pool, axis=0)

y_ref = median_curve(ref_temp_used, 0)
res = np.median(np.diff(fHz))
band = fHz[-1] - fHz[0]

p(f"\n  Referência saudável: mediana de falha==0 em T={ref_temp_used}°C")
p(f"  Resolução Δf = {res:.3f} Hz/ponto | banda = {band:.1f} Hz")

# micro-shift máximo permitido pelo V7
SAMPLE_MICRO_SHIFT_MAX_FRAC = 0.018
micro_max_hz = SAMPLE_MICRO_SHIFT_MAX_FRAC * band
p(f"\n  V7: SAMPLE_MICRO_SHIFT_MAX_FRAC = {SAMPLE_MICRO_SHIFT_MAX_FRAC}")
p(f"      => micro-shift MÁXIMO por amostra = {SAMPLE_MICRO_SHIFT_MAX_FRAC} × {band:.0f} Hz = {micro_max_hz:.1f} Hz")
p(f"      => em pontos: {micro_max_hz/res:.1f} pontos")

# deslocamento físico real do dano em REF_TEMP
p(f"\n  Deslocamento físico do PICO por dano (medido em T={ref_temp_used}°C):")
p(f"  {'dano':>6} | {'argmax_ref->dano (Hz)':>22} | {'xcorr tau (Hz)':>16} | {'micro-shift cobre?':>18}")
item3_rows = []
for d in [1, 2]:
    yd = median_curve(ref_temp_used, d)
    if yd is None or y_ref is None:
        p(f"  {d:>6} | (sem amostra)")
        continue
    # 1) deslocamento do máximo global
    shift_argmax_hz = fHz[np.argmax(yd)] - fHz[np.argmax(y_ref)]
    # 2) deslocamento por melhor alinhamento (xcorr fino) do dano ao saudável
    tau_align = best_align_tau(yd, y_ref, fHz, max_frac=0.06, nsteps=601)
    cobre = abs(tau_align) <= micro_max_hz
    p(f"  {d:>6} | {shift_argmax_hz:>22.1f} | {tau_align:>16.1f} | {'SIM (risco!)' if cobre else 'não':>18}")
    item3_rows.append({"dano": d, "shift_argmax_hz": float(shift_argmax_hz),
                       "xcorr_tau_hz": float(tau_align), "micro_cobre": bool(cobre)})

report["item3"] = {"micro_max_hz": float(micro_max_hz), "res_hz": float(res),
                   "band_hz": float(band), "deslocamentos": item3_rows}

p(f"\n  >>> Se |xcorr tau| do dano <= {micro_max_hz:.1f} Hz, o micro-shift por amostra")
p(f"      PODE empurrar o pico do dano de volta ao pico saudável (apaga dano). É o item 3.")

# ============================================================
# ITEM 2: efeito passa-baixa de shifts np.interp empilhados
# ============================================================
p("\n" + "=" * 90)
p("[7/7] ITEM 2 — efeito passa-baixa de shifts np.interp empilhados")
p("=" * 90)

# usa uma curva de dano real como sonda e mede atenuação do pico após N shifts
probe = median_curve(ref_temp_used, 2)
if probe is None:
    probe = y_ref.copy()
amp0 = np.ptp(probe)
peak0 = probe.max()

def apply_n_shifts(x, fHz, taus):
    y = x.copy()
    for t in taus:
        y = shift_interp(y, fHz, t)
    return y

# V7 (AE, dano) aplica ~3 reamostragens: fisico(tau) + post(tau2) + micro(tau_micro)
# Park aplica 1. Simulamos taus pequenos que "voltam" para ~mesma posição (ida e volta),
# isolando o efeito de reamostragem repetida (não o deslocamento líquido).
taus_1 = [+0.30 * res * 5]                 # Park: 1 shift
taus_3 = [+0.30*res*5, -0.18*res*5, +0.10*res*5]  # V7: 3 shifts encadeados
y1 = apply_n_shifts(probe, fHz, taus_1)
y3 = apply_n_shifts(probe, fHz, taus_3)
p(f"\n  Curva-sonda: mediana dano 2 em {ref_temp_used}°C | pico original = {peak0:.4g} | amplitude = {amp0:.4g}")
p(f"  Após 1 shift  (estilo Park): pico = {y1.max():.4g}  (atenuação {100*(1-y1.max()/peak0):+.2f}%) | amp {np.ptp(y1):.4g}")
p(f"  Após 3 shifts (estilo V7)  : pico = {y3.max():.4g}  (atenuação {100*(1-y3.max()/peak0):+.2f}%) | amp {np.ptp(y3):.4g}")
report["item2"] = {"peak0": float(peak0), "peak_1shift": float(y1.max()),
                   "peak_3shift": float(y3.max()),
                   "aten_1_pct": float(100*(1-y1.max()/peak0)),
                   "aten_3_pct": float(100*(1-y3.max()/peak0))}

# salva relatório JSON
with open(os.path.join(OUT_DIR, "diagnostico_etapa1.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
p(f"\n✅ Relatório salvo em: {os.path.join(OUT_DIR, 'diagnostico_etapa1.json')}")
p("=" * 90)

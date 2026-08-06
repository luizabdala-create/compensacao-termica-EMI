# -*- coding: utf-8 -*-
"""
AE HÍBRIDO CALIBRADO V7 — 30–40 kHz — alinhamento forte e nível vertical corrigido
==========================================================

Ideia:
------
Corrigir duas falhas observadas na V6:
1) picos ainda deslocados horizontalmente em relação à referência;
2) curva compensada com nível vertical sistematicamente abaixo da referência.

Mudanças principais do V7:
--------------------------
1) O shift térmico usa correlação multiescala da curva, primeira e segunda derivadas.
2) O shift horizontal é aplicado por inteiro antes do ganho/offset; não há mistura
   entre picos antigos e deslocados.
3) Há microalinhamento pequeno por amostra, regularizado e limitado.
4) O offset vertical final é calculado no baseline robusto e aplicado uma única vez.
5) O AE residual e o denoise ficaram mais fracos para não mover/achatar picos.

Importante:
-----------
O modelo NÃO recebe o dano como entrada da compensação.
O campo "falha" é usado somente para:
- montar referência saudável;
- treinar partes saudáveis;
- avaliar métricas e plotar.

Autor: adaptado para Luiz Eduardo Abdala José
"""

# ============================================================
# 1) IMPORTS
# ============================================================

import os
import re
import time
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.errors import PerformanceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2) PARÂMETROS PRINCIPAIS
# ============================================================

ARQ_BASE = "base-completo--.pkl"

FREQ_MIN_KHZ = 30.0
FREQ_MAX_KHZ = 40.0
MIN_POINTS_PER_BAND = 20

REF_TEMP = 30.0

OUTPUT_DIR = "resultados_AE_30_40_hibrido_V7_alinhado_vertical_corrigido_vs_park"
PASTA_GRAFICOS = os.path.join(OUTPUT_DIR, "graficos")
PASTA_CURVAS = os.path.join(PASTA_GRAFICOS, "curvas_exemplo")
PASTA_METRICAS = os.path.join(PASTA_GRAFICOS, "metricas_por_temperatura")
PASTA_TREINO = os.path.join(PASTA_GRAFICOS, "treinamento")

for p in [OUTPUT_DIR, PASTA_GRAFICOS, PASTA_CURVAS, PASTA_METRICAS, PASTA_TREINO]:
    os.makedirs(p, exist_ok=True)

RUN_ORIGINAL = True
RUN_PARK = True
RUN_AE_HIBRIDO = True


# ============================================================
# 3) CALIBRAÇÃO SAUDÁVEL FORTE — V7
# ============================================================

USE_HEALTHY_SHIFT = True

# O shift térmico é estimado por correlação multiescala das derivadas.
# A busca maior é segura porque usa SOMENTE a mediana saudável por temperatura.
SHIFT_MAX_FRAC = 0.140
SHIFT_NSTEPS = 101            # busca grossa; há refinamento automático
SHIFT_SMOOTH_WIN_FRAC = 0.018
SHIFT_FINE_STEPS = 81
SHIFT_PRIOR_PENALTY = 0.015

USE_GAIN = True
USE_OFFSET = True
USE_TILT = True

# Limites mais físicos. A V6 permitia ganho/offset excessivos e podia afundar a curva.
GAIN_MIN = 0.78
GAIN_MAX = 1.28
OFFSET_FRAC_LIMIT = 0.90
TILT_FRAC_LIMIT = 0.45

# Nunca extrapolar acima de 1.0. A V6 usava 1.03 e ultrapassava a calibração calculada.
PHYS_ALPHA = 1.00

ROBUST_WEIGHT_POWER = 1.35
ROBUST_MIN_WEIGHT = 0.12
FIT_SMOOTH_WIN_FRAC = 0.055


# ============================================================
# 4) AE RESIDUAL
# ============================================================

EPOCHS = 150
PATIENCE = 28
BATCH_SIZE = 16
LR = 6e-4
LATENT_DIM = 10
DROPOUT = 0.08
INPUT_NOISE_STD = 0.010
NUM_WORKERS = 0

N_ANCHORS_RESIDUAL = 35
AE_INPUT_SMOOTH_WIN_FRAC = 0.030
RESIDUAL_SMOOTH_WIN_FRAC = 0.040

# O AE fica residual de verdade: ele não deve decidir posição de pico nem nível global.
AE_RESIDUAL_BLEND = 0.12
AE_RESIDUAL_CLAMP_GAIN = 0.90
MIN_RESIDUAL_CLAMP_FRAC_STD = 0.08

LAMBDA_RECON = 0.18
LAMBDA_RESIDUAL = 1.40
LAMBDA_RESIDUAL_D1 = 0.34
LAMBDA_RESIDUAL_D2 = 0.24
LAMBDA_TEMP = 0.14

DAMAGE_PROTECT_GAIN = 2.90
DAMAGE_PROTECT_MIN_WEIGHT = 0.22

PEAK_PROTECT_GAIN = 3.60
PEAK_PROTECT_MIN_WEIGHT = 0.46
PEAK_PROTECT_SMOOTH_WIN_FRAC = 0.009

# Denoise bem mais fraco para não deslocar/achatar ressonâncias estreitas.
APPLY_FINAL_DENOISE = True
FINAL_DENOISE_WIN_FRAC = 0.010
FINAL_DENOISE_STRENGTH = 0.14
FINAL_DENOISE_PEAK_KEEP_GAIN = 6.50
FINAL_DENOISE_PEAK_KEEP_MIN = 0.62


# ============================================================
# 5) PÓS-ALINHAMENTO E TRAVA VERTICAL — V7
# ============================================================

USE_POST_CENTERING = True

# O deslocamento horizontal térmico deve ser aplicado a TODOS os danos.
# Não se reduz o shift por causa do dano, pois isso era uma fonte de desalinhamento na V6.
POST_SHIFT_BLEND_NEAR = 0.90
POST_SHIFT_BLEND_FAR = 1.00
POST_SHIFT_TEMP_POWER = 0.85

# Correção vertical aprendida apenas nas curvas saudáveis.
POST_VERTICAL_BLEND_NEAR = 0.82
POST_VERTICAL_BLEND_FAR = 1.00
POST_VERTICAL_TEMP_POWER = 0.90

POST_GAIN_MIN = 0.94
POST_GAIN_MAX = 1.06
POST_OFFSET_FRAC_LIMIT = 0.50
POST_CENTER_SMOOTH_WIN_FRAC = 0.120
POST_CENTER_CLIP_SIGMA = 3.50

# Shift residual por temperatura, calculado na mediana saudável após AE.
USE_POST_SMALL_SHIFT = True
POST_SHIFT_MAX_FRAC = 0.040
POST_SHIFT_NSTEPS = 81
POST_SHIFT_SMOOTH_WIN_FRAC = 0.018
POST_SHIFT_PRIOR_PENALTY = 0.040

# Microalinhamento por amostra: corrige pequenas variações de aquisição.
# É pequeno e regularizado para não transformar dano em referência.
USE_SAMPLE_MICRO_SHIFT = True
SAMPLE_MICRO_SHIFT_MAX_FRAC = 0.018
SAMPLE_MICRO_SHIFT_NSTEPS = 61
SAMPLE_MICRO_SHIFT_FINE_STEPS = 61
SAMPLE_MICRO_SHIFT_PRIOR_PENALTY = 0.18
SAMPLE_MICRO_SHIFT_MIN_IMPROVEMENT = 0.010

# Pesos antigos ainda são usados por funções auxiliares/ajustes robustos.
ALIGN_PEAK_WEIGHT_GAIN = 2.40
ALIGN_PEAK_WEIGHT_MIN = 0.22
ALIGN_PEAK_WEIGHT_MAX = 4.20
ALIGN_BASE_WEIGHT = 0.30

# Trava final do nível vertical. O offset constante preserva exatamente a forma do dano.
USE_FINAL_VERTICAL_LOCK = True
FINAL_VERTICAL_LOCK_WIN_FRAC = 0.145
FINAL_VERTICAL_OFFSET_BLEND = 0.96
FINAL_VERTICAL_TILT_BLEND = 0.34
FINAL_VERTICAL_OFFSET_CLIP_FRAC = 0.60
FINAL_VERTICAL_TILT_CLIP_FRAC = 0.22
FINAL_VERTICAL_TRIM_MAD = 3.0

# A centralização de baseline antiga aplicava outra correção distribuída e podia duplicar offset.
USE_BASELINE_RECENTER = False
BASELINE_RECENTER_WIN_FRAC = 0.135
BASELINE_RECENTER_BLEND_NEAR = 0.0
BASELINE_RECENTER_BLEND_FAR = 0.0
BASELINE_RECENTER_CLIP_FRAC = 0.0
BASELINE_RECENTER_PEAK_PROTECT_GAIN = 4.80
BASELINE_RECENTER_PEAK_PROTECT_MIN = 0.28

# Proteção somente para correções de forma/ganho; nunca reduz o shift térmico nem o offset global.
STRUCTURAL_PROTECT_LOW = 0.70
STRUCTURAL_PROTECT_HIGH = 3.20
STRUCTURAL_PROTECT_MAX_REDUCTION = 0.30


# ============================================================
# 6) PARK
# ============================================================

PARK_MAX_SHIFT_FRAC = 0.10
PARK_NSTEPS = 111
PARK_SMOOTH_WIN = 1


# ============================================================
# 7) GRÁFICOS
# ============================================================

SALVAR_PDF = True
TEMPS_EXEMPLO = [-10, 30, 80]
DANOS_PLOTAR = [0, 1, 2]
OCORRENCIA_CURVA = 0

METODOS_PLOT_METRICAS = ["Park", "AE híbrido V7"]
PLOTAR_SEPARACAO = True
PLOTAR_ORIGINAL_NA_COMPARACAO_GERAL = True

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.labelsize": 23,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CORES_DANO = {0: "tab:blue", 1: "tab:orange", 2: "tab:red"}
CORES_METODO = {
    "Original": "tab:red",
    "Park": "tab:green",
    "AE híbrido V7": "tab:blue",
}


# ============================================================
# 8) FUNÇÕES BÁSICAS
# ============================================================

def extract_freq_hz(col):
    m = re.match(r"^f_(\d+(?:\.\d+)?)Hz$", str(col))
    return float(m.group(1)) if m else None


def get_freq_columns(df, fmin_khz, fmax_khz):
    cols, freqs = [], []
    for c in df.columns:
        f = extract_freq_hz(c)
        if f is None:
            continue
        f_khz = f / 1e3
        if fmin_khz <= f_khz < fmax_khz:
            cols.append(c)
            freqs.append(f)
    if len(cols) == 0:
        return [], np.array([], dtype=float)
    order = np.argsort(freqs)
    cols = [cols[i] for i in order]
    freqs = np.asarray(freqs, dtype=float)[order]
    return cols, freqs


def odd_window_from_frac(n, frac, minimum=3):
    win = int(round(n * frac))
    win = max(int(minimum), win)
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = max(int(minimum), n // 5)
        if win % 2 == 0:
            win += 1
    return max(1, win)


def make_odd_window(win, n, minimum=3):
    win = int(round(win))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = max(minimum, n // 5)
        if win % 2 == 0:
            win += 1
    return max(1, win)


def moving_average(arr, win):
    arr = np.asarray(arr, dtype=float)
    if win <= 1 or len(arr) < 3:
        return arr.copy()
    win = make_odd_window(win, len(arr))
    if win <= 1:
        return arr.copy()
    pad = win // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    y = np.convolve(arr_pad, kernel, mode="valid")
    if len(y) > len(arr):
        y = y[:len(arr)]
    elif len(y) < len(arr):
        y = np.pad(y, (0, len(arr) - len(y)), mode="edge")
    return y


def smooth_matrix(X, win):
    X = np.asarray(X, dtype=float)
    return np.vstack([moving_average(x, win) for x in X])


def shift_interp(x, fHz, tau):
    f_shift = fHz + tau
    return np.interp(fHz, f_shift, x, left=x[0], right=x[-1])


def rmsd(y, ref):
    y = np.asarray(y, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return float(np.sqrt(np.mean((y - ref) ** 2)))


def pearson_corr(y, ref):
    y = np.asarray(y, dtype=float)
    ref = np.asarray(ref, dtype=float)
    y0 = y - np.mean(y)
    r0 = ref - np.mean(ref)
    den = np.sqrt(np.sum(y0 ** 2) * np.sum(r0 ** 2)) + 1e-18
    return float(np.sum(y0 * r0) / den)


def ccdm(y, ref):
    return float(1.0 - pearson_corr(y, ref))


def format_temp(T):
    T = float(T)
    if T.is_integer():
        return str(int(T))
    return f"{T:.1f}"


def faixa_label():
    return f"{int(FREQ_MIN_KHZ)}–{int(FREQ_MAX_KHZ)} kHz"


def salvar_fig(fig, pasta, nome_base):
    os.makedirs(pasta, exist_ok=True)
    png = os.path.join(pasta, nome_base + ".png")
    fig.savefig(png, dpi=500, bbox_inches="tight", facecolor="white")
    if SALVAR_PDF:
        pdf = os.path.join(pasta, nome_base + ".pdf")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        print(f"✅ Figura salva:\n{png}\n{pdf}")
    else:
        print(f"✅ Figura salva:\n{png}")


def estilo_eixos(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def temp_dist_weight(T, ref_temp, temps_available):
    temps_available = np.asarray(sorted(np.unique(temps_available)), dtype=float)
    if len(temps_available) == 0:
        return 0.0
    dmax = max(abs(temps_available.min() - ref_temp), abs(temps_available.max() - ref_temp), 1e-12)
    d = abs(float(T) - ref_temp)
    return float(np.clip(d / dmax, 0.0, 1.0))


# ============================================================
# 9) REFERÊNCIA SAUDÁVEL
# ============================================================

def get_healthy_references_by_temperature(df, fcols, ref_temp):
    df_h = df[df["falha"] == 0].copy()
    if len(df_h) == 0:
        raise ValueError("Não há curvas saudáveis com falha = 0.")

    temps_h = np.array(sorted(df_h["temperatura_c"].dropna().unique()), dtype=float)
    healthy_by_temp = {}

    for T in temps_h:
        pool = df_h.loc[np.isclose(df_h["temperatura_c"], T), fcols].to_numpy(np.float32)
        healthy_by_temp[float(T)] = np.median(pool, axis=0).astype(np.float32)

    ref_temp_used = float(temps_h[np.argmin(np.abs(temps_h - ref_temp))])
    y_ref = healthy_by_temp[ref_temp_used].astype(np.float32)

    if not np.isclose(ref_temp_used, ref_temp):
        print(f"⚠️ REF_TEMP={ref_temp}°C não existe exatamente. Usando {ref_temp_used}°C.")

    print("✅ Referência montada SOMENTE com falha == 0")
    print(f"✅ Temperatura de referência saudável usada: {format_temp(ref_temp_used)}°C")

    return healthy_by_temp, temps_h, y_ref, ref_temp_used


def interpolate_curve_by_temp(curves_by_temp, temps, T):
    temps = np.asarray(temps, dtype=float)
    T = float(T)

    if T <= temps[0]:
        return curves_by_temp[float(temps[0])].copy(), float(temps[0])
    if T >= temps[-1]:
        return curves_by_temp[float(temps[-1])].copy(), float(temps[-1])

    j = int(np.searchsorted(temps, T))
    T0 = float(temps[j - 1])
    T1 = float(temps[j])
    y0 = curves_by_temp[T0]
    y1 = curves_by_temp[T1]
    w = (T - T0) / (T1 - T0 + 1e-12)
    y = (1.0 - w) * y0 + w * y1
    return y.astype(np.float32), T


def nearest_temperature_with_all_damages(df, target_temp, danos=(0, 1, 2)):
    temps_validas = []
    for T in sorted(df["temperatura_c"].dropna().unique()):
        ok = True
        for d in danos:
            ok = ok and np.any(np.isclose(df["temperatura_c"], T) & (df["falha"] == d))
        if ok:
            temps_validas.append(float(T))
    if len(temps_validas) == 0:
        raise ValueError("Nenhuma temperatura possui todos os danos solicitados.")
    temps_validas = np.asarray(temps_validas, dtype=float)
    return float(temps_validas[np.argmin(np.abs(temps_validas - target_temp))])


# ============================================================
# 10) CALIBRAÇÃO FÍSICA SAUDÁVEL
# ============================================================

def robust_weights(x, ref):
    d2x = np.abs(np.gradient(np.gradient(x)))
    d2r = np.abs(np.gradient(np.gradient(ref)))
    score = d2x + d2r
    scale = np.median(score) + 1e-12
    w = 1.0 / (1.0 + (score / scale) ** ROBUST_WEIGHT_POWER)
    return np.clip(w, ROBUST_MIN_WEIGHT, 1.0)


def peak_alignment_weights(x, ref):
    """
    Peso para estimar deslocamento horizontal.

    Diferente do robust_weights(), aqui os picos/vales recebem mais peso,
    porque o problema visual que apareceu foi justamente pico fora do lugar.
    A correção final continua suave; isso só melhora tau.
    """
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)

    d2x = np.abs(np.gradient(np.gradient(x)))
    d2r = np.abs(np.gradient(np.gradient(ref)))
    score = d2x + d2r
    scale = np.percentile(score, 82) + 1e-12
    p = score / scale

    w = ALIGN_BASE_WEIGHT + ALIGN_PEAK_WEIGHT_GAIN * p / (1.0 + p)
    return np.clip(w, ALIGN_PEAK_WEIGHT_MIN, ALIGN_PEAK_WEIGHT_MAX)


def structural_score_curve(y, y_ref):
    """
    Score sem usar falha: mede quanto a curva compensada ainda tem assinatura
    estrutural em relação à referência.

    É usado só para não aplicar pós-centralização forte demais em dano real.
    """
    y = np.asarray(y, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    diff = y - y_ref
    diff = diff - np.median(diff)
    amp_ref = np.ptp(y_ref) + 1e-12
    rms_score = np.sqrt(np.mean(diff ** 2)) / amp_ref

    # reforça quando existem diferenças locais fortes em alguns picos/vales
    q_score = np.percentile(np.abs(diff), 90) / amp_ref
    return float(10.0 * max(rms_score, 0.75 * q_score))


def structural_protection_factor(y, y_ref):
    score = structural_score_curve(y, y_ref)
    u = (score - STRUCTURAL_PROTECT_LOW) / (STRUCTURAL_PROTECT_HIGH - STRUCTURAL_PROTECT_LOW + 1e-12)
    u = float(np.clip(u, 0.0, 1.0))
    factor = 1.0 - STRUCTURAL_PROTECT_MAX_REDUCTION * u
    return float(np.clip(factor, 1.0 - STRUCTURAL_PROTECT_MAX_REDUCTION, 1.0)), score


def _corr_segura(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 4 or len(b) < 4:
        return 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-18
    return float(np.clip(np.sum(a * b) / den, -1.0, 1.0))


def _robust_standardize(v):
    v = np.asarray(v, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    return (v - med) / (1.4826 * mad)


def _alignment_loss_multiscale(x_scales, ref_scales, fHz, tau):
    """
    Loss horizontal praticamente invariável a offset e ganho.
    Dá prioridade às derivadas, portanto alinha posição de picos/vales.
    """
    source_f = fHz - float(tau)
    mask = (source_f >= fHz[0]) & (source_f <= fHz[-1])
    idx = np.where(mask)[0]
    if len(idx) > 8:
        idx = idx[2:-2]
    if len(idx) < max(12, int(0.45 * len(fHz))):
        return np.inf

    losses = []
    for xs, rs in zip(x_scales, ref_scales):
        ys = shift_interp(xs, fHz, tau)[idx]
        rr = rs[idx]

        y0 = _robust_standardize(ys)
        r0 = _robust_standardize(rr)
        d1y = _robust_standardize(np.gradient(y0))
        d1r = _robust_standardize(np.gradient(r0))
        d2y = _robust_standardize(np.gradient(np.gradient(y0)))
        d2r = _robust_standardize(np.gradient(np.gradient(r0)))

        # Posição das ressonâncias pesa mais que amplitude absoluta.
        loss = (
            0.18 * (1.0 - _corr_segura(y0, r0))
            + 0.52 * (1.0 - _corr_segura(d1y, d1r))
            + 0.30 * (1.0 - _corr_segura(d2y, d2r))
        )
        losses.append(loss)

    return float(np.mean(losses))


def estimate_shift_multiscale(
    x,
    ref,
    fHz,
    max_frac,
    n_coarse=81,
    n_fine=81,
    prior_tau=0.0,
    prior_penalty=0.0,
    min_improvement=0.0,
    return_info=False,
):
    """
    Busca grossa + refinamento usando forma, 1ª e 2ª derivadas em três escalas.
    O termo prior_penalty impede que uma curva danificada escolha um shift exagerado.
    """
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)
    fHz = np.asarray(fHz, dtype=float)

    band = float(fHz[-1] - fHz[0])
    tau_max = max(float(max_frac) * band, 0.0)
    if tau_max <= 0 or len(x) < 12:
        info = {"loss_prior": 0.0, "loss_best": 0.0, "improvement": 0.0}
        return (0.0, info) if return_info else 0.0

    smooth_fracs = (0.008, 0.018, 0.040)
    x_scales, r_scales = [], []
    for frac in smooth_fracs:
        win = odd_window_from_frac(len(x), frac, minimum=5)
        x_scales.append(moving_average(x, win))
        r_scales.append(moving_average(ref, win))

    prior_tau = float(np.clip(prior_tau, -tau_max, tau_max))

    def objective(tau):
        base = _alignment_loss_multiscale(x_scales, r_scales, fHz, tau)
        reg = float(prior_penalty) * ((float(tau) - prior_tau) / (tau_max + 1e-12)) ** 2
        return base + reg

    n_coarse = max(21, int(n_coarse))
    coarse = np.linspace(-tau_max, tau_max, n_coarse)
    coarse_losses = np.array([objective(t) for t in coarse], dtype=float)
    j = int(np.nanargmin(coarse_losses))
    best_tau = float(coarse[j])
    best_loss = float(coarse_losses[j])

    coarse_step = float(coarse[1] - coarse[0]) if len(coarse) > 1 else tau_max
    lo = max(-tau_max, best_tau - 2.5 * coarse_step)
    hi = min(tau_max, best_tau + 2.5 * coarse_step)
    fine = np.linspace(lo, hi, max(21, int(n_fine)))
    fine_losses = np.array([objective(t) for t in fine], dtype=float)
    jf = int(np.nanargmin(fine_losses))
    if float(fine_losses[jf]) < best_loss:
        best_tau = float(fine[jf])
        best_loss = float(fine_losses[jf])

    loss_prior = float(objective(prior_tau))
    improvement = (loss_prior - best_loss) / (abs(loss_prior) + 1e-12)

    if improvement < float(min_improvement):
        best_tau = prior_tau
        best_loss = loss_prior

    info = {
        "loss_prior": loss_prior,
        "loss_best": best_loss,
        "improvement": float(improvement),
    }
    return (float(best_tau), info) if return_info else float(best_tau)


def estimate_healthy_shift(h_T, h_ref, fHz):
    if not USE_HEALTHY_SHIFT:
        return 0.0

    return estimate_shift_multiscale(
        h_T,
        h_ref,
        fHz,
        max_frac=SHIFT_MAX_FRAC,
        n_coarse=SHIFT_NSTEPS,
        n_fine=SHIFT_FINE_STEPS,
        prior_tau=0.0,
        prior_penalty=SHIFT_PRIOR_PENALTY,
        min_improvement=0.0,
        return_info=False,
    )


def fit_affine_tilt_healthy(h_T_shift, h_ref):
    n = len(h_ref)
    z = np.linspace(-1.0, 1.0, n)

    win = odd_window_from_frac(n, FIT_SMOOTH_WIN_FRAC, minimum=9)
    x_fit = moving_average(h_T_shift, win)
    r_fit = moving_average(h_ref, win)

    cols = []
    names = []

    if USE_GAIN:
        cols.append(x_fit)
        names.append("a")
    if USE_OFFSET:
        cols.append(np.ones_like(x_fit))
        names.append("b")
    if USE_TILT:
        cols.append(z)
        names.append("c")

    if len(cols) == 0:
        return 1.0, 0.0, 0.0

    A = np.column_stack(cols)
    w = robust_weights(x_fit, r_fit)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], r_fit * sw, rcond=None)

    a, b, c = 1.0, 0.0, 0.0
    for name, value in zip(names, coef):
        if name == "a":
            a = float(value)
        elif name == "b":
            b = float(value)
        elif name == "c":
            c = float(value)

    amp = np.ptp(h_ref) + 1e-12
    a = float(np.clip(a, GAIN_MIN, GAIN_MAX))
    b = float(np.clip(b, -OFFSET_FRAC_LIMIT * amp, OFFSET_FRAC_LIMIT * amp))
    c = float(np.clip(c, -TILT_FRAC_LIMIT * amp, TILT_FRAC_LIMIT * amp))
    return a, b, c


_CALIBRATION_CACHE = {}


def get_calibration_params_for_T(T, healthy_by_temp, healthy_temps, y_ref, fHz):
    """Calcula uma vez por temperatura e reutiliza em treino, aplicação e métricas."""
    key = (
        id(healthy_by_temp),
        id(y_ref),
        id(fHz),
        round(float(T), 8),
    )
    if key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[key]

    h_T, T_used = interpolate_curve_by_temp(healthy_by_temp, healthy_temps, T)
    tau = estimate_healthy_shift(h_T, y_ref, fHz)
    h_shift = shift_interp(h_T, fHz, tau)
    a, b, c = fit_affine_tilt_healthy(h_shift, y_ref)
    params = {
        "T": float(T),
        "T_used": float(T_used),
        "h_T": h_T,
        "tau": float(tau),
        "a": float(a),
        "b": float(b),
        "c": float(c),
    }
    _CALIBRATION_CACHE[key] = params
    return params


def apply_calibration_to_curve(x, params, fHz, alpha=PHYS_ALPHA):
    """
    V7: primeiro aplica TODO o shift horizontal; depois aplica apenas a transformação vertical.

    Na V6 era usado x + alpha*(y_full-x). Como y_full já continha x deslocado,
    isso misturava picos na posição antiga e nova. Com alpha > 1 também extrapolava
    o ganho/offset e podia empurrar a curva para baixo.
    """
    x = np.asarray(x, dtype=float)
    z = np.linspace(-1.0, 1.0, len(x))
    x_shift = shift_interp(x, fHz, params["tau"])
    y_vertical = params["a"] * x_shift + params["b"] + params["c"] * z
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (x_shift + alpha * (y_vertical - x_shift)).astype(np.float32)


# ============================================================
# 11) PROTEÇÕES E DENOISE
# ============================================================

def peak_protection_weight(x):
    x = np.asarray(x, dtype=float)
    win = odd_window_from_frac(len(x), PEAK_PROTECT_SMOOTH_WIN_FRAC, minimum=5)
    xs = moving_average(x, win)
    d1 = np.gradient(xs)
    d2 = np.gradient(d1)
    score = np.abs(d2)
    scale = np.percentile(score, 80) + 1e-12
    score_norm = score / scale
    w = 1.0 / (1.0 + PEAK_PROTECT_GAIN * score_norm)
    w = np.clip(w, PEAK_PROTECT_MIN_WEIGHT, 1.0)
    return moving_average(w, win).astype(np.float32)


def damage_protection_weight(y_phys, y_ref):
    diff = np.abs(np.asarray(y_phys, dtype=float) - np.asarray(y_ref, dtype=float))
    scale = np.percentile(diff, 75) + 1e-12
    w = 1.0 / (1.0 + DAMAGE_PROTECT_GAIN * diff / scale)
    return np.clip(w, DAMAGE_PROTECT_MIN_WEIGHT, 1.0).astype(np.float32)


def denoise_preservando_picos(y):
    if not APPLY_FINAL_DENOISE:
        return np.asarray(y, dtype=np.float32)

    y = np.asarray(y, dtype=float)
    win = odd_window_from_frac(len(y), FINAL_DENOISE_WIN_FRAC, minimum=7)
    ys = moving_average(y, win)

    d1 = np.gradient(y)
    d2 = np.gradient(d1)
    score = np.abs(d2)
    scale = np.percentile(score, 80) + 1e-12
    peak_keep = (score / scale) / (1.0 + score / scale)
    peak_keep = np.clip(FINAL_DENOISE_PEAK_KEEP_GAIN * peak_keep, FINAL_DENOISE_PEAK_KEEP_MIN, 1.0)
    peak_keep = moving_average(peak_keep, win)
    peak_keep = np.clip(peak_keep, FINAL_DENOISE_PEAK_KEEP_MIN, 1.0)

    y_flat = (1.0 - FINAL_DENOISE_STRENGTH) * y + FINAL_DENOISE_STRENGTH * ys
    y_final = peak_keep * y + (1.0 - peak_keep) * y_flat
    return y_final.astype(np.float32)


# ============================================================
# 12) AE RESIDUAL
# ============================================================

class ResidualAE(nn.Module):
    def __init__(self, n_points, n_anchors, latent_dim=10, dropout=0.08):
        super().__init__()
        input_dim = n_points + 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 80),
            nn.LayerNorm(80),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(80, 40),
            nn.LayerNorm(40),
            nn.GELU(),
            nn.Linear(40, latent_dim),
        )

        self.recon_head = nn.Sequential(
            nn.Linear(latent_dim, 40),
            nn.GELU(),
            nn.Linear(40, 80),
            nn.GELU(),
            nn.Linear(80, n_points),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(latent_dim, 40),
            nn.LayerNorm(40),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(40, 80),
            nn.GELU(),
            nn.Linear(80, n_anchors),
        )

        self.temp_head = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.GELU(),
            nn.Linear(24, 1),
        )

    def forward(self, x_curve_s, t_norm, dt_norm):
        x_in = torch.cat([x_curve_s, t_norm, dt_norm], dim=1)
        z = self.encoder(x_in)
        recon = self.recon_head(z)
        residual_anchor = self.residual_head(z)
        t_pred = self.temp_head(z)
        return recon, residual_anchor, z, t_pred


def deriv_loss(y_pred, y_true):
    return F.smooth_l1_loss(y_pred[:, 1:] - y_pred[:, :-1], y_true[:, 1:] - y_true[:, :-1])


def curv_loss(y_pred, y_true):
    if y_pred.shape[1] < 3:
        return torch.tensor(0.0, device=y_pred.device)
    d2p = y_pred[:, 2:] - 2 * y_pred[:, 1:-1] + y_pred[:, :-2]
    d2t = y_true[:, 2:] - 2 * y_true[:, 1:-1] + y_true[:, :-2]
    return F.smooth_l1_loss(d2p, d2t)


def preparar_targets_residual_ae(df, fcols, fHz, y_ref, healthy_by_temp, healthy_temps):
    df_h = df[df["falha"] == 0].copy()
    Xh = df_h[fcols].to_numpy(np.float32)
    T_h = df_h["temperatura_c"].to_numpy(np.float32)

    n_points = Xh.shape[1]
    input_smooth_win = odd_window_from_frac(n_points, AE_INPUT_SMOOTH_WIN_FRAC, minimum=7)
    n_anchors = min(N_ANCHORS_RESIDUAL, n_points)
    anchor_idx = np.linspace(0, n_points - 1, n_anchors).round().astype(int)

    X_phys = []
    residual_full = []
    calib_rows = []

    for x, T in zip(Xh, T_h):
        params = get_calibration_params_for_T(T, healthy_by_temp, healthy_temps, y_ref, fHz)
        y_phys = apply_calibration_to_curve(x, params, fHz, alpha=PHYS_ALPHA)
        res = y_ref - y_phys
        X_phys.append(y_phys)
        residual_full.append(res)
        calib_rows.append({
            "T": float(T),
            "T_used": params["T_used"],
            "tau": params["tau"],
            "a": params["a"],
            "b": params["b"],
            "c": params["c"],
        })

    X_phys = np.asarray(X_phys, dtype=np.float32)
    residual_full = np.asarray(residual_full, dtype=np.float32)

    X_input = smooth_matrix(X_phys, input_smooth_win).astype(np.float32)
    residual_anchor = residual_full[:, anchor_idx].astype(np.float32)

    x_scaler = StandardScaler()
    res_scaler = StandardScaler()

    X_input_s = x_scaler.fit_transform(X_input).astype(np.float32)
    residual_anchor_s = res_scaler.fit_transform(residual_anchor).astype(np.float32)

    T_norm = ((T_h - REF_TEMP) / 100.0).reshape(-1, 1).astype(np.float32)
    DT_norm = (np.abs(T_h - REF_TEMP) / 100.0).reshape(-1, 1).astype(np.float32)

    q_low = np.quantile(residual_full, 0.03, axis=0)
    q_high = np.quantile(residual_full, 0.97, axis=0)
    global_std = np.std(residual_full) + 1e-12
    min_amp = MIN_RESIDUAL_CLAMP_FRAC_STD * global_std
    lower = np.minimum(q_low * AE_RESIDUAL_CLAMP_GAIN, q_high * AE_RESIDUAL_CLAMP_GAIN)
    upper = np.maximum(q_low * AE_RESIDUAL_CLAMP_GAIN, q_high * AE_RESIDUAL_CLAMP_GAIN)
    lower = np.minimum(lower, -min_amp)
    upper = np.maximum(upper, min_amp)

    pd.DataFrame(calib_rows).to_csv(os.path.join(OUTPUT_DIR, "calibracao_saudavel_por_amostra_treino.csv"), index=False)

    return {
        "X_input_s": X_input_s,
        "T_norm": T_norm,
        "DT_norm": DT_norm,
        "residual_anchor": residual_anchor,
        "residual_anchor_s": residual_anchor_s,
        "anchor_idx": anchor_idx,
        "x_scaler": x_scaler,
        "res_scaler": res_scaler,
        "input_smooth_win": input_smooth_win,
        "residual_lower": lower.astype(np.float32),
        "residual_upper": upper.astype(np.float32),
    }


def treinar_residual_ae(df, fcols, fHz, y_ref, healthy_by_temp, healthy_temps):
    prep = preparar_targets_residual_ae(df, fcols, fHz, y_ref, healthy_by_temp, healthy_temps)

    Xs = prep["X_input_s"]
    Tn = prep["T_norm"]
    DTn = prep["DT_norm"]
    Rs = prep["residual_anchor_s"]

    n = len(Xs)
    idx = np.arange(n)
    if n >= 10:
        idx_train, idx_val = train_test_split(idx, test_size=0.20, random_state=42)
    else:
        idx_train, idx_val = idx, idx

    X_tensor = torch.tensor(Xs, dtype=torch.float32)
    T_tensor = torch.tensor(Tn, dtype=torch.float32)
    DT_tensor = torch.tensor(DTn, dtype=torch.float32)
    R_tensor = torch.tensor(Rs, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tensor[idx_train], T_tensor[idx_train], DT_tensor[idx_train], R_tensor[idx_train]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        TensorDataset(X_tensor[idx_val], T_tensor[idx_val], DT_tensor[idx_val], R_tensor[idx_val]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Treinando AE residual em: {device}")

    model = ResidualAE(
        n_points=Xs.shape[1],
        n_anchors=Rs.shape[1],
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=9)

    huber = nn.SmoothL1Loss()
    mse = nn.MSELoss()

    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    history = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, tb, dtb, rb in train_loader:
            xb = xb.to(device)
            tb = tb.to(device)
            dtb = dtb.to(device)
            rb = rb.to(device)

            xb_in = xb + INPUT_NOISE_STD * torch.randn_like(xb) if INPUT_NOISE_STD > 0 else xb

            opt.zero_grad()
            recon, res_pred, z, t_pred = model(xb_in, tb, dtb)

            loss_recon = huber(recon, xb)
            loss_res = huber(res_pred, rb)
            loss_d1 = deriv_loss(res_pred, rb)
            loss_d2 = curv_loss(res_pred, rb)
            loss_temp = mse(t_pred, tb)

            loss = (
                LAMBDA_RECON * loss_recon
                + LAMBDA_RESIDUAL * loss_res
                + LAMBDA_RESIDUAL_D1 * loss_d1
                + LAMBDA_RESIDUAL_D2 * loss_d2
                + LAMBDA_TEMP * loss_temp
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        val_res_losses = []
        val_recon_losses = []

        with torch.no_grad():
            for xb, tb, dtb, rb in val_loader:
                xb = xb.to(device)
                tb = tb.to(device)
                dtb = dtb.to(device)
                rb = rb.to(device)

                recon, res_pred, z, t_pred = model(xb, tb, dtb)
                loss_recon = huber(recon, xb)
                loss_res = huber(res_pred, rb)
                loss_d1 = deriv_loss(res_pred, rb)
                loss_d2 = curv_loss(res_pred, rb)
                loss_temp = mse(t_pred, tb)
                loss = (
                    LAMBDA_RECON * loss_recon
                    + LAMBDA_RESIDUAL * loss_res
                    + LAMBDA_RESIDUAL_D1 * loss_d1
                    + LAMBDA_RESIDUAL_D2 * loss_d2
                    + LAMBDA_TEMP * loss_temp
                )
                val_losses.append(float(loss.item()))
                val_res_losses.append(float(loss_res.item()))
                val_recon_losses.append(float(loss_recon.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_res = float(np.mean(val_res_losses))
        val_recon = float(np.mean(val_recon_losses))
        lr_now = opt.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_residual": val_res,
            "val_recon": val_recon,
            "lr": lr_now,
        })

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if ep == 1 or ep % 20 == 0:
            print(
                f"Epoch {ep:4d}/{EPOCHS} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"res={val_res:.6f} | recon={val_recon:.6f} | lr={lr_now:.2e}"
            )

        if bad_epochs >= PATIENCE:
            print(f"✅ Early stopping na epoch {ep}. Melhor val_loss = {best_val:.6f}")
            break

    model.load_state_dict(best_state)
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(OUTPUT_DIR, "historico_treinamento_ae_residual.csv"), index=False)

    return {"model": model, "device": device, "prep": prep, "history": df_hist}


def predizer_residual_ae(df, y_phys_all, ae_pack):
    model = ae_pack["model"]
    device = ae_pack["device"]
    prep = ae_pack["prep"]

    T = df["temperatura_c"].to_numpy(np.float32)
    X_input = smooth_matrix(y_phys_all, prep["input_smooth_win"]).astype(np.float32)
    Xs = prep["x_scaler"].transform(X_input).astype(np.float32)
    T_norm = ((T - REF_TEMP) / 100.0).reshape(-1, 1).astype(np.float32)
    DT_norm = (np.abs(T - REF_TEMP) / 100.0).reshape(-1, 1).astype(np.float32)

    X_tensor = torch.tensor(Xs, dtype=torch.float32)
    T_tensor = torch.tensor(T_norm, dtype=torch.float32)
    DT_tensor = torch.tensor(DT_norm, dtype=torch.float32)

    model.eval()
    res_anchor_s_list = []
    t_pred_list = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), BATCH_SIZE):
            xb = X_tensor[i:i + BATCH_SIZE].to(device)
            tb = T_tensor[i:i + BATCH_SIZE].to(device)
            dtb = DT_tensor[i:i + BATCH_SIZE].to(device)
            recon, res_anchor_s, z, t_pred = model(xb, tb, dtb)
            res_anchor_s_list.append(res_anchor_s.cpu().numpy())
            t_pred_list.append(t_pred.cpu().numpy())

    res_anchor_s = np.vstack(res_anchor_s_list).astype(np.float32)
    res_anchor = prep["res_scaler"].inverse_transform(res_anchor_s).astype(np.float32)
    t_pred = np.vstack(t_pred_list)[:, 0].astype(np.float32) * 100.0 + REF_TEMP

    n_points = y_phys_all.shape[1]
    full_idx = np.arange(n_points)
    anchor_idx = prep["anchor_idx"]

    residual_full = np.zeros_like(y_phys_all, dtype=np.float32)
    for i in range(len(y_phys_all)):
        residual_full[i] = np.interp(full_idx, anchor_idx, res_anchor[i]).astype(np.float32)

    return residual_full, t_pred


# ============================================================
# 13) PÓS-ALINHAMENTO E TRAVA VERTICAL — V7
# ============================================================

def estimate_small_shift_and_offset(x, ref, fHz):
    """
    Mantém a assinatura antiga da função, mas o offset não é mais calculado aqui.
    Shift e offset agora são estimados em etapas separadas para não contar o nível
    vertical duas vezes.
    """
    if not USE_POST_SMALL_SHIFT:
        return 0.0, 0.0

    tau, info = estimate_shift_multiscale(
        x,
        ref,
        fHz,
        max_frac=POST_SHIFT_MAX_FRAC,
        n_coarse=POST_SHIFT_NSTEPS,
        n_fine=SHIFT_FINE_STEPS,
        prior_tau=0.0,
        prior_penalty=POST_SHIFT_PRIOR_PENALTY,
        min_improvement=0.0,
        return_info=True,
    )
    return float(tau), 0.0


def fit_post_affine(med_curve, y_ref):
    """Ajuste afim pequeno, robusto e concentrado no baseline."""
    med_curve = np.asarray(med_curve, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)

    d2 = np.abs(np.gradient(np.gradient(y_ref))) + np.abs(np.gradient(np.gradient(med_curve)))
    scale = np.percentile(d2, 75) + 1e-12
    w = 1.0 / (1.0 + (d2 / scale) ** 1.35)
    w = np.clip(w, 0.16, 1.0)

    A = np.column_stack([med_curve, np.ones_like(med_curve)])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y_ref * sw, rcond=None)
    a2 = float(coef[0])
    b2 = float(coef[1])

    amp = np.ptp(y_ref) + 1e-12
    a2 = float(np.clip(a2, POST_GAIN_MIN, POST_GAIN_MAX))
    b2 = float(np.clip(b2, -POST_OFFSET_FRAC_LIMIT * amp, POST_OFFSET_FRAC_LIMIT * amp))
    return a2, b2


def build_post_center_pack(df_comp, fcols, fHz, y_ref):
    """
    Aprende por temperatura, usando somente falha == 0:
      1) shift residual saudável;
      2) ganho/offset pequeno;
      3) correção de baseline muito suave.

    Cada componente é estimado em sequência e aplicado uma única vez.
    """
    df_h = df_comp[df_comp["falha"] == 0].copy()
    temps = np.array(sorted(df_h["temperatura_c"].dropna().unique()), dtype=float)
    if len(temps) == 0:
        return None

    rows = []
    gain_by_temp = {}
    offset_by_temp = {}
    tau_by_temp = {}
    corr_by_temp = {}

    for T in temps:
        pool = df_h.loc[np.isclose(df_h["temperatura_c"], T), fcols].to_numpy(np.float32)
        med = np.median(pool, axis=0).astype(np.float32)

        tau2, _ = estimate_small_shift_and_offset(med, y_ref, fHz)
        med_shift = shift_interp(med, fHz, tau2)

        a2, b2 = fit_post_affine(med_shift, y_ref)
        med_aff = a2 * med_shift + b2

        # Só resta uma componente lenta. Nada de copiar picos da referência.
        corr = np.asarray(y_ref, dtype=float) - med_aff
        win = odd_window_from_frac(len(corr), POST_CENTER_SMOOTH_WIN_FRAC, minimum=15)
        corr_s = moving_average(corr, win)

        med_corr = np.median(corr_s)
        mad_corr = np.median(np.abs(corr_s - med_corr)) + 1e-12
        lim = POST_CENTER_CLIP_SIGMA * 1.4826 * mad_corr
        if lim > 0:
            corr_s = np.clip(corr_s, med_corr - lim, med_corr + lim)

        gain_by_temp[float(T)] = np.float32(a2)
        offset_by_temp[float(T)] = np.float32(b2)
        tau_by_temp[float(T)] = np.float32(tau2)
        corr_by_temp[float(T)] = np.asarray(corr_s, dtype=np.float32)

        rows.append({
            "temperatura_c": float(T),
            "tau2_hz": float(tau2),
            "a2": float(a2),
            "b2": float(b2),
            "corr_mean": float(np.mean(corr_s)),
            "corr_std": float(np.std(corr_s)),
        })

    df_info = pd.DataFrame(rows)
    df_info.to_csv(os.path.join(OUTPUT_DIR, "post_center_info_por_temperatura.csv"), index=False)

    return {
        "temps": temps,
        "gain_by_temp": gain_by_temp,
        "offset_by_temp": offset_by_temp,
        "tau_by_temp": tau_by_temp,
        "corr_by_temp": corr_by_temp,
        "info": df_info,
    }


def interp_scalar_by_temp(values_by_temp, temps, T):
    temps = np.asarray(temps, dtype=float)
    T = float(T)
    vals = np.array([float(values_by_temp[float(t)]) for t in temps], dtype=float)
    if T <= temps[0]:
        return float(vals[0])
    if T >= temps[-1]:
        return float(vals[-1])
    return float(np.interp(T, temps, vals))


def estimate_sample_micro_shift(y, y_ref, fHz):
    if not USE_SAMPLE_MICRO_SHIFT:
        return 0.0, {"loss_prior": np.nan, "loss_best": np.nan, "improvement": 0.0}

    return estimate_shift_multiscale(
        y,
        y_ref,
        fHz,
        max_frac=SAMPLE_MICRO_SHIFT_MAX_FRAC,
        n_coarse=SAMPLE_MICRO_SHIFT_NSTEPS,
        n_fine=SAMPLE_MICRO_SHIFT_FINE_STEPS,
        prior_tau=0.0,
        prior_penalty=SAMPLE_MICRO_SHIFT_PRIOR_PENALTY,
        min_improvement=SAMPLE_MICRO_SHIFT_MIN_IMPROVEMENT,
        return_info=True,
    )


def final_vertical_lock_curve(y, y_ref, protect_factor=1.0):
    """
    Corrige o que o usuário observa como curva 'muito para baixo'.

    O offset é constante em toda a faixa, logo NÃO altera posição nem formato dos picos.
    Uma inclinação linear pequena é opcional e recebe proteção estrutural.
    Regiões de alta curvatura e resíduos extremos têm peso reduzido.
    """
    if not USE_FINAL_VERTICAL_LOCK:
        return np.asarray(y, dtype=np.float32), 0.0, 0.0

    y = np.asarray(y, dtype=float)
    ref = np.asarray(y_ref, dtype=float)
    n = len(y)
    z = np.linspace(-1.0, 1.0, n)

    win = odd_window_from_frac(n, FINAL_VERTICAL_LOCK_WIN_FRAC, minimum=21)
    yl = moving_average(y, win)
    rl = moving_average(ref, win)
    resid = rl - yl

    # Exclui ressonâncias estreitas do cálculo do nível global.
    d2 = np.abs(np.gradient(np.gradient(y))) + np.abs(np.gradient(np.gradient(ref)))
    d2_scale = np.percentile(d2, 72) + 1e-12
    w_curv = 1.0 / (1.0 + (d2 / d2_scale) ** 1.5)
    w_curv = np.clip(w_curv, 0.05, 1.0)

    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-12
    u = np.abs(resid - med) / (FINAL_VERTICAL_TRIM_MAD * 1.4826 * mad + 1e-12)
    w_trim = 1.0 / (1.0 + u ** 4)
    w = np.clip(w_curv * w_trim, 1e-4, 1.0)

    A = np.column_stack([np.ones(n), z])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], resid * sw, rcond=None)
    offset = float(coef[0])
    tilt = float(coef[1])

    amp = np.ptp(ref) + 1e-12
    offset = float(np.clip(offset, -FINAL_VERTICAL_OFFSET_CLIP_FRAC * amp,
                           FINAL_VERTICAL_OFFSET_CLIP_FRAC * amp))
    tilt = float(np.clip(tilt, -FINAL_VERTICAL_TILT_CLIP_FRAC * amp,
                         FINAL_VERTICAL_TILT_CLIP_FRAC * amp))

    # Offset global é seguro para o dano; tilt recebe proteção.
    y_out = (
        y
        + FINAL_VERTICAL_OFFSET_BLEND * offset
        + FINAL_VERTICAL_TILT_BLEND * float(protect_factor) * tilt * z
    )
    return y_out.astype(np.float32), offset, tilt


def baseline_recenter_curve(y, y_ref, alpha):
    # Mantida apenas por compatibilidade. Na V7 fica desativada para evitar correção duplicada.
    return np.asarray(y, dtype=np.float32)


def apply_post_centering(df_comp, fcols, fHz, y_ref, post_pack):
    if (post_pack is None) or (not USE_POST_CENTERING):
        return df_comp.copy()

    temps_available = df_comp["temperatura_c"].dropna().unique()
    Y_in = df_comp[fcols].to_numpy(np.float32)
    T_all = df_comp["temperatura_c"].to_numpy(float)

    Y_out = np.zeros_like(Y_in, dtype=np.float32)
    rows = []

    for i, T in enumerate(T_all):
        dist_w = temp_dist_weight(T, REF_TEMP, temps_available)
        alpha_shift = POST_SHIFT_BLEND_NEAR + (POST_SHIFT_BLEND_FAR - POST_SHIFT_BLEND_NEAR) * (
            dist_w ** POST_SHIFT_TEMP_POWER
        )
        alpha_vertical = POST_VERTICAL_BLEND_NEAR + (POST_VERTICAL_BLEND_FAR - POST_VERTICAL_BLEND_NEAR) * (
            dist_w ** POST_VERTICAL_TEMP_POWER
        )

        # Proteção só reduz ganho/correção de forma. O shift térmico não é reduzido pelo dano.
        protect_factor, structural_score = structural_protection_factor(Y_in[i], y_ref)
        alpha_shape = alpha_vertical * protect_factor

        tau2 = interp_scalar_by_temp(post_pack["tau_by_temp"], post_pack["temps"], T)
        a2 = interp_scalar_by_temp(post_pack["gain_by_temp"], post_pack["temps"], T)
        b2 = interp_scalar_by_temp(post_pack["offset_by_temp"], post_pack["temps"], T)
        corrT, _ = interpolate_curve_by_temp(post_pack["corr_by_temp"], post_pack["temps"], T)

        y = Y_in[i].astype(np.float32)

        # 1) Shift residual térmico por temperatura — aplicado integralmente aos danos.
        y = shift_interp(y, fHz, alpha_shift * tau2).astype(np.float32)

        # 2) Microshift por amostra, pequeno e regularizado.
        tau_micro, shift_info = estimate_sample_micro_shift(y, y_ref, fHz)
        y = shift_interp(y, fHz, tau_micro).astype(np.float32)

        # 3) Ganho/offset pequeno aprendido no saudável.
        y_aff_full = a2 * y + b2
        y = y + alpha_shape * (y_aff_full - y)

        # 4) Resíduo lento saudável. Picos recebem pouca correção.
        w_peak = peak_protection_weight(y)
        y = y + alpha_shape * w_peak * corrT

        # 5) Trava final do nível vertical por amostra; não mexe nos picos.
        y, vertical_offset, vertical_tilt = final_vertical_lock_curve(
            y, y_ref, protect_factor=protect_factor
        )

        y = denoise_preservando_picos(y)
        Y_out[i] = y.astype(np.float32)

        rows.append({
            "temperatura_c": float(T),
            "post_alpha_shift": float(alpha_shift),
            "post_alpha_vertical": float(alpha_vertical),
            "post_alpha_shape": float(alpha_shape),
            "post_tau2_hz": float(tau2),
            "sample_micro_tau_hz": float(tau_micro),
            "shift_loss_before": float(shift_info.get("loss_prior", np.nan)),
            "shift_loss_after": float(shift_info.get("loss_best", np.nan)),
            "shift_improvement": float(shift_info.get("improvement", np.nan)),
            "post_a2": float(a2),
            "post_b2": float(b2),
            "final_vertical_offset": float(vertical_offset),
            "final_vertical_tilt": float(vertical_tilt),
            "structural_score": float(structural_score),
            "protect_factor": float(protect_factor),
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "post_center_aplicado_por_amostra.csv"), index=False
    )

    meta = df_comp.drop(columns=fcols).copy()
    return pd.concat([meta, pd.DataFrame(Y_out, columns=fcols, index=df_comp.index)], axis=1)


# ============================================================
# 14) COMPENSAÇÃO AE HÍBRIDO V7
# ============================================================

def compensar_ae_hibrido(df, fcols, fHz, y_ref, healthy_by_temp, healthy_temps):
    # Parte física: shift completo + correção vertical sem extrapolação.
    X = df[fcols].to_numpy(np.float32)
    T_all = df["temperatura_c"].to_numpy(float)
    Y_phys = np.zeros_like(X, dtype=np.float32)
    info_rows = []

    # Cache por temperatura: evita recalcular a busca horizontal para cada amostra.
    params_cache = {}
    for T in np.unique(T_all):
        params_cache[float(T)] = get_calibration_params_for_T(
            float(T), healthy_by_temp, healthy_temps, y_ref, fHz
        )

    for i, T in enumerate(T_all):
        params = params_cache[float(T)]
        Y_phys[i] = apply_calibration_to_curve(X[i], params, fHz, alpha=PHYS_ALPHA)
        info_rows.append({
            "temperatura_c": float(T),
            "T_healthy_usada": params["T_used"],
            "tau_hz": params["tau"],
            "gain_a": params["a"],
            "offset_b": params["b"],
            "tilt_c": params["c"],
        })

    pd.DataFrame(info_rows).to_csv(
        os.path.join(OUTPUT_DIR, "info_calibracao_fisica_todas_amostras.csv"), index=False
    )

    # AE residual: apenas pequenas diferenças suaves após a física.
    ae_pack = treinar_residual_ae(df, fcols, fHz, y_ref, healthy_by_temp, healthy_temps)
    residual_pred, t_pred = predizer_residual_ae(df, Y_phys, ae_pack)

    n_points = X.shape[1]
    smooth_res_win = odd_window_from_frac(n_points, RESIDUAL_SMOOTH_WIN_FRAC, minimum=7)
    lower = ae_pack["prep"]["residual_lower"]
    upper = ae_pack["prep"]["residual_upper"]

    Y_ae = np.zeros_like(X, dtype=np.float32)
    for i in range(len(X)):
        residual = moving_average(residual_pred[i], smooth_res_win).astype(np.float32)
        residual = np.clip(residual, lower, upper)

        w_peak = peak_protection_weight(Y_phys[i])
        w_damage = damage_protection_weight(Y_phys[i], y_ref)
        w = w_peak * w_damage

        # Não aplicar denoise aqui: o pós-alinhamento precisa enxergar a posição real dos picos.
        Y_ae[i] = (Y_phys[i] + AE_RESIDUAL_BLEND * w * residual).astype(np.float32)

    meta = df.drop(columns=fcols).copy()
    df_ae = pd.concat([meta, pd.DataFrame(Y_ae, columns=fcols, index=df.index)], axis=1)
    df_ae["temperatura_ae_pred"] = t_pred

    df_phys = pd.concat([meta.copy(), pd.DataFrame(Y_phys, columns=fcols, index=df.index)], axis=1)
    df_phys.to_pickle(os.path.join(OUTPUT_DIR, "curvas_calibracao_fisica_antes_AE.pkl"))

    if USE_POST_CENTERING:
        print("🔹 Aplicando pós-alinhamento e trava vertical V7...")
        post_pack = build_post_center_pack(df_ae, fcols, fHz, y_ref)
        df_ae = apply_post_centering(df_ae, fcols, fHz, y_ref, post_pack)
        if "temperatura_ae_pred" not in df_ae.columns:
            df_ae["temperatura_ae_pred"] = t_pred

    return df_ae, ae_pack


# ============================================================
# 15) PARK
# ============================================================

def park_single(x, ref, fHz):
    df_band = fHz[-1] - fHz[0]
    tau_max = PARK_MAX_SHIFT_FRAC * df_band
    best_err = np.inf
    best_tau = 0.0
    best_dS = 0.0

    for tau in np.linspace(-tau_max, tau_max, PARK_NSTEPS):
        x_shift = shift_interp(x, fHz, tau)
        dS = np.mean(ref - x_shift)
        y_try = x_shift + dS
        err = np.mean((ref - y_try) ** 2)
        if err < best_err:
            best_err = err
            best_tau = tau
            best_dS = dS

    y = shift_interp(x, fHz, best_tau) + best_dS
    y = moving_average(y, PARK_SMOOTH_WIN)
    return y.astype(np.float32)


def compensar_park(df, fcols, fHz, y_ref):
    X = df[fcols].to_numpy(np.float32)
    Y = np.zeros_like(X, dtype=np.float32)
    for i in range(len(X)):
        Y[i] = park_single(X[i], y_ref, fHz)
    meta = df.drop(columns=fcols).copy()
    return pd.concat([meta, pd.DataFrame(Y, columns=fcols, index=df.index)], axis=1)


# ============================================================
# 16) MÉTRICAS
# ============================================================

def calcular_metricas(df_curvas, df_original, fcols, fHz, y_ref, healthy_by_temp, healthy_temps, metodo):
    X_comp = df_curvas[fcols].to_numpy(np.float32)
    X_orig = df_original[fcols].to_numpy(np.float32)
    out = df_original[["temperatura_c", "falha"]].copy()

    rmsd_list, ccdm_list = [], []
    damage_res_rmsd, damage_res_ccdm = [], []
    alteracao_rmsd = []

    for i, (_, row) in enumerate(df_original.iterrows()):
        T = float(row["temperatura_c"])
        params = get_calibration_params_for_T(T, healthy_by_temp, healthy_temps, y_ref, fHz)
        h_T_cal = apply_calibration_to_curve(params["h_T"], params, fHz, alpha=PHYS_ALPHA)

        y_comp = X_comp[i]
        y_orig = X_orig[i]
        y_orig_cal = apply_calibration_to_curve(y_orig, params, fHz, alpha=PHYS_ALPHA)

        rmsd_list.append(rmsd(y_comp, y_ref))
        ccdm_list.append(ccdm(y_comp, y_ref))

        assinatura_esperada = y_orig_cal - h_T_cal
        assinatura_saida = y_comp - y_ref

        damage_res_rmsd.append(rmsd(assinatura_saida, assinatura_esperada))
        damage_res_ccdm.append(ccdm(assinatura_saida, assinatura_esperada))
        alteracao_rmsd.append(rmsd(y_comp, y_orig))

    out["Metodo"] = metodo
    out["Freq_min_kHz"] = FREQ_MIN_KHZ
    out["Freq_max_kHz"] = FREQ_MAX_KHZ
    out["Faixa"] = faixa_label()
    out["RMSD"] = rmsd_list
    out["CCDM"] = ccdm_list
    out["DamageResidual_RMSD"] = damage_res_rmsd
    out["DamageResidual_CCDM"] = damage_res_ccdm
    out["Alteracao_RMSD"] = alteracao_rmsd

    if "temperatura_ae_pred" in df_curvas.columns:
        out["temperatura_ae_pred"] = df_curvas["temperatura_ae_pred"].to_numpy(float)

    return out


def criar_resumo(df_metricas):
    return (
        df_metricas
        .groupby(["Metodo", "temperatura_c", "falha"], as_index=False)
        .agg(
            RMSD_medio=("RMSD", "mean"),
            RMSD_std=("RMSD", "std"),
            CCDM_medio=("CCDM", "mean"),
            CCDM_std=("CCDM", "std"),
            DamageResidual_RMSD_medio=("DamageResidual_RMSD", "mean"),
            DamageResidual_CCDM_medio=("DamageResidual_CCDM", "mean"),
            Alteracao_RMSD_medio=("Alteracao_RMSD", "mean"),
            n=("RMSD", "size"),
        )
    )


def calcular_separacao_temperatura(df_metricas):
    resumo = criar_resumo(df_metricas)
    rows = []
    for (metodo, temp), g in resumo.groupby(["Metodo", "temperatura_c"]):
        gd = g.set_index("falha")
        if not all(d in gd.index for d in [0, 1, 2]):
            continue
        for metrica in ["RMSD", "CCDM"]:
            v0 = float(gd.loc[0, f"{metrica}_medio"])
            v1 = float(gd.loc[1, f"{metrica}_medio"])
            v2 = float(gd.loc[2, f"{metrica}_medio"])
            rows.append({
                "Metodo": metodo,
                "temperatura_c": temp,
                "Metrica": metrica,
                "D0": v0,
                "D1": v1,
                "D2": v2,
                "Sep_D1_D0": v1 - v0,
                "Sep_D2_D1": v2 - v1,
                "Sep_D2_D0": v2 - v0,
                "Monotonico_D0_D1_D2": bool(v0 < v1 < v2),
            })
    return pd.DataFrame(rows)


# ============================================================
# 17) GRÁFICOS
# ============================================================

def plot_loss_ae(ae_pack):
    hist = ae_pack["history"]
    if len(hist) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(hist["epoch"], hist["train_loss"], linewidth=2.4, label="Treino")
    ax.plot(hist["epoch"], hist["val_loss"], linewidth=2.4, label="Validação")
    ax.plot(hist["epoch"], hist["val_residual"], linewidth=2.0, label="Val residual")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Treinamento do AE residual")
    ax.legend(frameon=True)
    estilo_eixos(ax)
    fig.tight_layout()
    salvar_fig(fig, PASTA_TREINO, "historico_loss_ae_residual")
    plt.show()


def plot_metricas_por_temperatura(df_metricas, metodo):
    sub = df_metricas[df_metricas["Metodo"] == metodo].copy()
    if len(sub) == 0:
        return
    resumo = criar_resumo(sub)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300, sharex=True)
    for ax, metrica in zip(axes, ["RMSD", "CCDM"]):
        for dano in DANOS_PLOTAR:
            sd = resumo[resumo["falha"] == dano].sort_values("temperatura_c")
            ax.errorbar(
                sd["temperatura_c"],
                sd[f"{metrica}_medio"],
                yerr=sd[f"{metrica}_std"].fillna(0.0),
                marker="o",
                linewidth=2.5,
                markersize=7,
                capsize=3,
                color=CORES_DANO.get(dano, None),
                label=f"Dano {dano}",
            )
        ax.axvline(REF_TEMP, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel(metrica)
        ax.set_title(f"({ 'a' if metrica == 'RMSD' else 'b' }) {metrica}")
        estilo_eixos(ax)
    axes[1].legend(frameon=True, loc="best")
    fig.suptitle(f"{metodo} — RMSD e CCDM por temperatura — {faixa_label()}", fontsize=26, y=1.03)
    fig.tight_layout()
    nome = f"metricas_por_temperatura_{metodo}".replace(" ", "_").replace("/", "_")
    salvar_fig(fig, PASTA_METRICAS, nome)
    plt.show()


def plot_separacao_por_temperatura(df_metricas, metodo):
    sep = calcular_separacao_temperatura(df_metricas)
    sep = sep[sep["Metodo"] == metodo].copy()
    if len(sep) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300, sharex=True)
    for ax, metrica in zip(axes, ["RMSD", "CCDM"]):
        sm = sep[sep["Metrica"] == metrica].sort_values("temperatura_c")
        ax.plot(sm["temperatura_c"], sm["Sep_D1_D0"], marker="o", linewidth=2.5, label="D1 − D0")
        ax.plot(sm["temperatura_c"], sm["Sep_D2_D1"], marker="s", linewidth=2.5, label="D2 − D1")
        ax.plot(sm["temperatura_c"], sm["Sep_D2_D0"], marker="^", linewidth=2.5, label="D2 − D0")
        ax.axhline(0, color="black", linewidth=1.2)
        ax.axvline(REF_TEMP, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel(f"Separação em {metrica}")
        ax.set_title(f"Separação — {metrica}")
        estilo_eixos(ax)
    axes[1].legend(frameon=True, loc="best")
    fig.suptitle(f"{metodo} — separação entre danos por temperatura — {faixa_label()}", fontsize=26, y=1.03)
    fig.tight_layout()
    nome = f"separacao_por_temperatura_{metodo}".replace(" ", "_").replace("/", "_")
    salvar_fig(fig, PASTA_METRICAS, nome)
    plt.show()


def plot_comparacao_metodos_resumo(df_metricas):
    resumo = df_metricas.groupby(["Metodo", "falha"], as_index=False).agg(RMSD=("RMSD", "mean"), CCDM=("CCDM", "mean"))
    if PLOTAR_ORIGINAL_NA_COMPARACAO_GERAL:
        metodos = [m for m in ["Original", "Park", "AE híbrido V7"] if m in resumo["Metodo"].unique()]
    else:
        metodos = [m for m in ["Park", "AE híbrido V7"] if m in resumo["Metodo"].unique()]
    danos = sorted(resumo["falha"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
    width = 0.22
    x = np.arange(len(metodos))
    for ax, metrica in zip(axes, ["RMSD", "CCDM"]):
        for i, dano in enumerate(danos):
            vals = []
            for metodo in metodos:
                v = resumo.loc[(resumo["Metodo"] == metodo) & (resumo["falha"] == dano), metrica]
                vals.append(float(v.iloc[0]) if len(v) else np.nan)
            ax.bar(x + (i - 1) * width, vals, width=width, edgecolor="black", linewidth=0.8,
                   label=f"Dano {dano}", color=CORES_DANO.get(dano, None), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metodos, rotation=15, ha="right")
        ax.set_ylabel(f"{metrica} médio")
        ax.set_title(metrica)
        estilo_eixos(ax)
    axes[1].legend(frameon=True, loc="best")
    fig.suptitle(f"Comparação geral dos métodos — {faixa_label()}", fontsize=26, y=1.03)
    fig.tight_layout()
    salvar_fig(fig, PASTA_METRICAS, "comparacao_geral_metodos_RMSD_CCDM")
    plt.show()


def plot_curvas_exemplo(df_original, curvas_por_metodo, fcols, fHz, y_ref):
    fkhz = fHz / 1e3
    for T_alvo in TEMPS_EXEMPLO:
        try:
            T = nearest_temperature_with_all_damages(df_original, T_alvo, danos=DANOS_PLOTAR)
        except Exception as e:
            print(f"⚠️ Não consegui achar temperatura exemplo {T_alvo}: {e}")
            continue

        fig, axes = plt.subplots(1, len(DANOS_PLOTAR), figsize=(7.6 * len(DANOS_PLOTAR), 6.8), dpi=300, sharey=True)
        if len(DANOS_PLOTAR) == 1:
            axes = [axes]

        for ax, dano in zip(axes, DANOS_PLOTAR):
            mask = np.isclose(df_original["temperatura_c"], T) & (df_original["falha"] == dano)
            idxs = np.where(mask.to_numpy())[0]
            if len(idxs) == 0:
                ax.set_title(f"Dano {dano}\nsem amostra")
                continue
            idx = idxs[min(OCORRENCIA_CURVA, len(idxs) - 1)]

            ax.plot(fkhz, y_ref, "--", color="black", linewidth=1.9, label=f"Referência saudável {format_temp(REF_TEMP)}°C")
            y_orig = df_original.iloc[idx][fcols].to_numpy(float)
            ax.plot(fkhz, y_orig, color="tab:red", alpha=0.50, linewidth=1.5, label=f"Original {format_temp(T)}°C")

            for metodo, df_curva in curvas_por_metodo.items():
                if metodo == "Original":
                    continue
                y = df_curva.iloc[idx][fcols].to_numpy(float)
                ax.plot(fkhz, y, linewidth=2.35, color=CORES_METODO.get(metodo, None), label=metodo)

            ax.set_title(f"Dano {dano}")
            ax.set_xlabel("Frequência (kHz)")
            estilo_eixos(ax)

        axes[0].set_ylabel("Impedância")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, fontsize=14, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f"Curvas exemplo — {faixa_label()} — T = {format_temp(T)}°C", fontsize=26, y=1.03)
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        nome = f"curvas_exemplo_T{format_temp(T).replace('-', 'm')}C"
        salvar_fig(fig, PASTA_CURVAS, nome)
        plt.show()


# ============================================================
# 18) EXECUÇÃO PRINCIPAL
# ============================================================

def executar_analise_ae_30_40():
    t0 = time.time()
    _CALIBRATION_CACHE.clear()
    print("=" * 90)
    print("AE HÍBRIDO CALIBRADO V7 — ANÁLISE 30–40 kHz")
    print("=" * 90)

    if not os.path.exists(ARQ_BASE):
        raise FileNotFoundError(f"Não encontrei o arquivo: {ARQ_BASE}")

    print(f"🔹 Carregando base: {ARQ_BASE}")
    df_base = pd.read_pickle(ARQ_BASE).reset_index(drop=True)

    required = {"temperatura_c", "falha"}
    missing = required - set(df_base.columns)
    if missing:
        raise ValueError(f"A base está sem as colunas obrigatórias: {missing}")

    df_base["temperatura_c"] = pd.to_numeric(df_base["temperatura_c"], errors="coerce")
    df_base["falha"] = pd.to_numeric(df_base["falha"], errors="coerce").astype(int)

    fcols, fHz = get_freq_columns(df_base, FREQ_MIN_KHZ, FREQ_MAX_KHZ)
    if len(fcols) < MIN_POINTS_PER_BAND:
        raise ValueError(f"Faixa {FREQ_MIN_KHZ}-{FREQ_MAX_KHZ} kHz tem só {len(fcols)} pontos.")

    df = df_base[["temperatura_c", "falha"] + fcols].copy()

    print(f"Total de amostras: {len(df)}")
    print(f"Danos encontrados: {sorted(df['falha'].unique())}")
    print(f"Temperaturas encontradas: {sorted(df['temperatura_c'].dropna().unique())}")
    print(f"Faixa analisada: {faixa_label()} | pontos de frequência: {len(fcols)}")

    healthy_by_temp, healthy_temps, y_ref, ref_temp_used = get_healthy_references_by_temperature(df=df, fcols=fcols, ref_temp=REF_TEMP)

    curvas_por_metodo = {}
    metricas_partes = []
    ae_pack = None

    if RUN_ORIGINAL:
        print("\n🔹 Métricas da curva original...")
        curvas_por_metodo["Original"] = df.copy()
        met_original = calcular_metricas(df, df, fcols, fHz, y_ref, healthy_by_temp, healthy_temps, "Original")
        metricas_partes.append(met_original)

    if RUN_PARK:
        print("\n🔹 Aplicando Park...")
        t_park = time.time()
        df_park = compensar_park(df, fcols, fHz, y_ref)
        curvas_por_metodo["Park"] = df_park
        met_park = calcular_metricas(df_park, df, fcols, fHz, y_ref, healthy_by_temp, healthy_temps, "Park")
        metricas_partes.append(met_park)
        print(f"✅ Park concluído em {time.time() - t_park:.1f} s")

    if RUN_AE_HIBRIDO:
        print("\n🔹 Aplicando AE híbrido V7...")
        t_ae = time.time()
        df_ae, ae_pack = compensar_ae_hibrido(df, fcols, fHz, y_ref, healthy_by_temp, healthy_temps)
        curvas_por_metodo["AE híbrido V7"] = df_ae
        met_ae = calcular_metricas(df_ae, df, fcols, fHz, y_ref, healthy_by_temp, healthy_temps, "AE híbrido V7")
        metricas_partes.append(met_ae)
        print(f"✅ AE híbrido V7 concluído em {time.time() - t_ae:.1f} s")

    df_metricas = pd.concat(metricas_partes, ignore_index=True)
    resumo = criar_resumo(df_metricas)
    separacao = calcular_separacao_temperatura(df_metricas)

    path_metricas = os.path.join(OUTPUT_DIR, "metricas_amostra_a_amostra.csv")
    path_resumo = os.path.join(OUTPUT_DIR, "resumo_por_metodo_temperatura_dano.csv")
    path_sep = os.path.join(OUTPUT_DIR, "separacao_danos_por_temperatura.csv")

    df_metricas.to_csv(path_metricas, index=False)
    resumo.to_csv(path_resumo, index=False)
    separacao.to_csv(path_sep, index=False)

    print(f"\n✅ Métricas salvas em:\n{path_metricas}\n{path_resumo}\n{path_sep}")

    if ae_pack is not None:
        plot_loss_ae(ae_pack)

    for metodo in METODOS_PLOT_METRICAS:
        if metodo in df_metricas["Metodo"].unique():
            plot_metricas_por_temperatura(df_metricas, metodo)
            if PLOTAR_SEPARACAO:
                plot_separacao_por_temperatura(df_metricas, metodo)

    plot_comparacao_metodos_resumo(df_metricas)
    plot_curvas_exemplo(df, curvas_por_metodo, fcols, fHz, y_ref)

    print("\nResumo médio por método/dano:")
    print(
        df_metricas
        .groupby(["Metodo", "falha"])[["RMSD", "CCDM", "DamageResidual_RMSD", "Alteracao_RMSD"]]
        .mean()
        .round(6)
    )

    print("\n" + "=" * 90)
    print("✅ ANÁLISE FINALIZADA")
    print(f"📁 Pasta de saída: {OUTPUT_DIR}")
    print(f"⏱️ Tempo total: {time.time() - t0:.1f} s")
    print("=" * 90)

    return {
        "df_base_faixa": df,
        "df_metricas": df_metricas,
        "resumo": resumo,
        "separacao": separacao,
        "curvas_por_metodo": curvas_por_metodo,
        "fcols": fcols,
        "fHz": fHz,
        "y_ref": y_ref,
        "ae_pack": ae_pack,
    }


# ============================================================
# 19) RODAR
# ============================================================

if __name__ == "__main__":
    resultados = executar_analise_ae_30_40()

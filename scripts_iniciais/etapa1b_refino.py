# -*- coding: utf-8 -*-
"""
ETAPA 1b — refinamentos honestos do diagnóstico
- Item 2 justo: 1 interp vs 3 interp para o MESMO deslocamento líquido (50/100/200 Hz)
- Efeito real do denoise V7 sobre o pico
- Tamanho do conjunto de treino do AE (contexto item 1 / overfitting)
"""
import os, re, json, numpy as np, pandas as pd

ARQ_BASE = r"C:\Users\luize\base-completo--.pkl"
FREQ_MIN_KHZ, FREQ_MAX_KHZ, REF_TEMP = 30.0, 40.0, 30.0
OUT_DIR = r"C:\Users\luize\IC_EMI\diagnostico_etapa1"

def extract_freq_hz(c):
    m = re.match(r"^f_(\d+(?:\.\d+)?)Hz$", str(c)); return float(m.group(1)) if m else None
def get_freq_columns(df, a, b):
    cols, fr = [], []
    for c in df.columns:
        f = extract_freq_hz(c)
        if f is not None and a <= f/1e3 < b: cols.append(c); fr.append(f)
    o = np.argsort(fr); return [cols[i] for i in o], np.asarray(fr, float)[o]
def shift_interp(x, fHz, tau):
    return np.interp(fHz, fHz + tau, x, left=x[0], right=x[-1])
def odd_win(n, frac, m=3):
    w = max(int(m), int(round(n*frac)))
    if w % 2 == 0: w += 1
    return w
def moving_average(a, w):
    a = np.asarray(a, float)
    if w <= 1: return a.copy()
    pad = w//2
    return np.convolve(np.pad(a,(pad,pad),mode="edge"), np.ones(w)/w, mode="valid")[:len(a)]

df = pd.read_pickle(ARQ_BASE).reset_index(drop=True)
df["temperatura_c"] = pd.to_numeric(df["temperatura_c"], errors="coerce")
df["falha"] = pd.to_numeric(df["falha"], errors="coerce").astype(int)
fcols, fHz = get_freq_columns(df, FREQ_MIN_KHZ, FREQ_MAX_KHZ)
res = np.median(np.diff(fHz))

def med(temp, fal):
    m = np.isclose(df["temperatura_c"], temp) & (df["falha"] == fal)
    pool = df.loc[m, fcols].to_numpy(np.float64)
    return np.median(pool, axis=0) if len(pool) else None

probe = med(30.0, 2)  # dano 2, pico estreito
peak0 = probe.max(); amp0 = np.ptp(probe)

print("="*80)
print("ITEM 2 (justo): 1 interp vs 3 interp para o MESMO deslocamento líquido")
print("="*80)
print(f"Sonda: mediana dano 2 @30C | pico={peak0:.4f} | amp={amp0:.4f} | Δf={res:.2f}Hz/pt\n")
print(f"{'net(Hz)':>8} | {'1 interp: aten pico%':>20} | {'3 interp: aten pico%':>20} | {'RMS(1 vs 3)':>12}")
rows2 = []
for net in [50, 100, 200, 500]:
    y1 = shift_interp(probe, fHz, net)
    # 3 interps encadeadas somando 'net'
    y3 = probe.copy()
    for t in [net*0.6, net*0.7, -net*0.3]:
        y3 = shift_interp(y3, fHz, t)
    a1 = 100*(1 - y1.max()/peak0)
    a3 = 100*(1 - y3.max()/peak0)
    rms = np.sqrt(np.mean((y1 - y3)**2))
    print(f"{net:>8} | {a1:>20.3f} | {a3:>20.3f} | {rms:>12.4f}")
    rows2.append({"net_hz": net, "aten1_pct": float(a1), "aten3_pct": float(a3), "rms_1v3": float(rms)})

print("\n" + "="*80)
print("Efeito REAL do denoise final do V7 sobre o pico (FINAL_DENOISE)")
print("="*80)
# reproduz denoise_preservando_picos do V7 com os parâmetros reais
FINAL_DENOISE_WIN_FRAC=0.010; FINAL_DENOISE_STRENGTH=0.14
FINAL_DENOISE_PEAK_KEEP_GAIN=6.50; FINAL_DENOISE_PEAK_KEEP_MIN=0.62
def denoise_v7(y):
    y=np.asarray(y,float); win=odd_win(len(y),FINAL_DENOISE_WIN_FRAC,7); ys=moving_average(y,win)
    d2=np.gradient(np.gradient(y)); s=np.abs(d2); sc=np.percentile(s,80)+1e-12
    pk=(s/sc)/(1+s/sc); pk=np.clip(FINAL_DENOISE_PEAK_KEEP_GAIN*pk,FINAL_DENOISE_PEAK_KEEP_MIN,1.0)
    pk=moving_average(pk,win); pk=np.clip(pk,FINAL_DENOISE_PEAK_KEEP_MIN,1.0)
    yf=(1-FINAL_DENOISE_STRENGTH)*y+FINAL_DENOISE_STRENGTH*ys
    return pk*y+(1-pk)*yf
yd = denoise_v7(probe)
print(f"pico antes={peak0:.4f} | depois denoise={yd.max():.4f} | atenuação={100*(1-yd.max()/peak0):+.3f}%")
print(f"RMS(antes vs depois)={np.sqrt(np.mean((probe-yd)**2)):.4f}")

print("\n" + "="*80)
print("CONTEXTO ITEM 1 — tamanho de treino do AE e superdimensionamento")
print("="*80)
n_healthy = int((df["falha"]==0).sum())
n_dano1 = int((df["falha"]==1).sum()); n_dano2 = int((df["falha"]==2).sum())
n_points = len(fcols)
print(f"curvas saudáveis (treino do AE) = {n_healthy}")
print(f"  -> após split interno 80/20 do V7: ~{int(round(n_healthy*0.8))} treino / ~{int(round(n_healthy*0.2))} val")
print(f"curvas dano 1 = {n_dano1} | dano 2 = {n_dano2} | pontos/curva = {n_points}")
# 1a camada do encoder: Linear(n_points+2, 160)
params_1a = (n_points+2)*160 + 160
print(f"parâmetros só da 1ª camada do encoder Linear({n_points+2},160) = {params_1a:,}")
print(f"razão parâmetros(1ª camada) / amostras de treino ≈ {params_1a/max(1,round(n_healthy*0.8)):,.0f} : 1")

# temperaturas com dano (restrição p/ split da ETAPA 3)
temps_dano = sorted(df.loc[df['falha'].isin([1,2]),'temperatura_c'].unique().tolist())
temps_healthy = sorted(df.loc[df['falha']==0,'temperatura_c'].unique().tolist())
print(f"\nTemperaturas COM dano (1 ou 2): {temps_dano}")
print(f"nº temps com dano = {len(temps_dano)} | nº temps com saudável = {len(temps_healthy)}")

rep = {"item2_fair": rows2,
       "denoise_aten_pct": float(100*(1-yd.max()/peak0)),
       "n_healthy": n_healthy, "n_dano1": n_dano1, "n_dano2": n_dano2,
       "n_points": n_points, "params_1a_encoder": int(params_1a),
       "temps_com_dano": temps_dano}
with open(os.path.join(OUT_DIR,"diagnostico_etapa1b.json"),"w",encoding="utf-8") as f:
    json.dump(rep, f, indent=2, ensure_ascii=False)
print(f"\n✅ salvo em {os.path.join(OUT_DIR,'diagnostico_etapa1b.json')}")

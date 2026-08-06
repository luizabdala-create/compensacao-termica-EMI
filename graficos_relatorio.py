# -*- coding: utf-8 -*-
"""
Gera os graficos de resultados usados no relatorio (compensacao termica e
classificacao de dano). Os valores ja resumidos estao embutidos abaixo, entao
o script roda sozinho e salva os PDFs na pasta 'figuras/'.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

PASTA = "figuras"
os.makedirs(PASTA, exist_ok=True)

# fonte Times New Roman (mesma dos artigos); se nao existir, usa serif padrao
for arq in [r"C:/Windows/Fonts/times.ttf"]:
    if os.path.exists(arq):
        fm.fontManager.addfont(arq)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})

# cores fixas por metodo
COR = {
    "Original": "#8c8c8c",
    "Park": "#e08214",
    "Random Forest": "#c0392b",
    "Autoencoder": "#2874a6",
    "Autoencoder (forma)": "#6a3d9a",
}


def salvar(fig, nome):
    fig.savefig(os.path.join(PASTA, nome + ".pdf"))
    plt.close(fig)
    print("gerado:", nome)


# ---------------------------------------------------------------------------
# 1) RMSD nas curvas saudaveis por temperatura
# ---------------------------------------------------------------------------
temperaturas = [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
rmsd_temp = {
    "Original":     [18.29, 15.00, 10.50, 7.78, 5.95, 6.92, 7.68, 9.67, 11.14, 11.34],
    "Park":         [4.06, 3.83, 3.01, 2.51, 1.86, 2.03, 2.34, 2.79, 3.22, 3.29],
    "Random Forest":[4.22, 2.42, 3.24, 2.61, 2.34, 1.94, 2.01, 1.41, 0.86, 1.36],
    "Autoencoder":  [5.82, 5.36, 5.33, 4.43, 4.12, 4.05, 3.75, 3.47, 3.23, 3.28],
}
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for met in ["Original", "Park", "Random Forest", "Autoencoder"]:
    ax.plot(temperaturas, rmsd_temp[met], marker="o", ms=5, lw=1.8, color=COR[met], label=met)
ax.set_xlabel("Temperatura de ensaio (\u00b0C)")
ax.set_ylabel("RMSD nas curvas saud\u00e1veis")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, columnspacing=1.3, handletextpad=0.5)
salvar(fig, "rmsd_por_temperatura")

# ---------------------------------------------------------------------------
# 2) RMSD nas curvas saudaveis por faixa de frequencia
# ---------------------------------------------------------------------------
faixas = ["30\u201340", "40\u201350", "70\u201380", "30\u201370", "30\u2013100"]
rmsd_faixa = {
    "Autoencoder":   [1.72, 1.79, 0.75, 5.46, 5.90],
    "Park":          [1.88, 2.87, 1.52, 3.45, 4.20],
    "Random Forest": [2.01, 2.37, 1.38, 2.33, 2.24],
}
x = np.arange(len(faixas))
larg = 0.26
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for i, met in enumerate(["Autoencoder", "Park", "Random Forest"]):
    ax.bar(x + (i - 1) * larg, rmsd_faixa[met], larg, color=COR[met], label=met)
ax.set_xticks(x)
ax.set_xticklabels(faixas)
ax.set_xlabel("Faixa de frequ\u00eancia (kHz)")
ax.set_ylabel("RMSD nas curvas saud\u00e1veis")
ax.set_ylim(0, 7.4)
ax.axvline(2.5, color="0.8", lw=1, ls="--")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.6)
salvar(fig, "rmsd_por_faixa")

# ---------------------------------------------------------------------------
# 3) Pareto: compensacao (RMSD) x preservacao do dano
# ---------------------------------------------------------------------------
pareto = {
    "Original":     (10.8, 0.38),
    "Park":         (2.63, 0.78),
    "Autoencoder":  (2.51, 0.90),
    "Random Forest":(2.17, 1.00),
}
desloc = {"Random Forest": (11, 3), "Autoencoder": (11, -4), "Park": (11, -11), "Original": (-12, -3)}
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for met, (rx, ry) in pareto.items():
    ax.scatter(rx, ry, s=210, color=COR[met], edgecolor="white", lw=1.5, zorder=3)
    ax.annotate(met, (rx, ry), xytext=desloc[met], textcoords="offset points",
                fontsize=12, ha=("right" if met == "Original" else "left"))
ax.set_xlabel("RMSD nas curvas saud\u00e1veis  (mais \u00e0 esquerda \u00e9 melhor)")
ax.set_ylabel("Preserva\u00e7\u00e3o do dano\n(mais acima \u00e9 melhor)")
ax.set_ylim(0.30, 1.10)
ax.set_xlim(1.2, 12.2)
salvar(fig, "pareto")

# ---------------------------------------------------------------------------
# 4) Deteccao de dano: acuracia, sensibilidade e taxa de falso-saudavel
# ---------------------------------------------------------------------------
metodos = ["Original", "Park", "Random Forest", "Autoencoder"]
rot = ["Original", "Park", "Random\nForest", "Autoencoder"]
acuracia = [0.835, 0.893, 0.846, 0.930]
sensibilidade = [0.808, 0.910, 0.965, 0.934]
falso_saudavel = [0.192, 0.090, 0.035, 0.066]
cores = [COR[m] for m in metodos]
fig, eixos = plt.subplots(1, 3, figsize=(12, 3.9))
paineis = [
    (acuracia, "Acur\u00e1cia balanceada", "(maior \u00e9 melhor)", 0.6, 1.0),
    (sensibilidade, "Sensibilidade ao dano", "(maior \u00e9 melhor)", 0.6, 1.0),
    (falso_saudavel, "Taxa de falso-saud\u00e1vel", "(menor \u00e9 melhor)", 0.0, 0.22),
]
xs = np.arange(len(metodos))
for ax, (dados, titulo, sub, ylo, yhi) in zip(eixos, paineis):
    ax.bar(xs, dados, 0.62, color=cores)
    ax.set_xticks(xs)
    ax.set_xticklabels(rot, fontsize=10)
    ax.set_ylim(ylo, yhi)
    ax.set_title(titulo + "\n" + sub, fontsize=11)
    for i, v in enumerate(dados):
        ax.text(i, v + (yhi - ylo) * 0.012, f"{v:.3f}", ha="center", fontsize=9)
fig.tight_layout()
salvar(fig, "deteccao_dano")

# ---------------------------------------------------------------------------
# 5) Identificacao do tipo de dano (multiclasse)
# ---------------------------------------------------------------------------
met_id = ["Park", "Random Forest", "Autoencoder", "Autoencoder (forma)"]
rot_id = ["Park", "Random\nForest", "Autoencoder\n(amplitude)", "Autoencoder\n(forma)"]
bal_multi = [0.824, 0.742, 0.830, 0.847]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
xs = np.arange(len(met_id))
ax.bar(xs, bal_multi, 0.62, color=[COR[m] for m in met_id])
ax.set_xticks(xs)
ax.set_xticklabels(rot_id, fontsize=10)
ax.set_ylim(0.6, 0.9)
ax.set_ylabel("Acur\u00e1cia balanceada")
for i, v in enumerate(bal_multi):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
salvar(fig, "identificacao_dano")

# ---------------------------------------------------------------------------
# 6) Custo computacional de treino por temperatura
# ---------------------------------------------------------------------------
met_custo = ["Park", "Random Forest", "Autoencoder"]
tempo = [0.0, 175.9, 8.7]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(met_custo, tempo, 0.6, color=[COR[m] for m in met_custo])
ax.set_ylabel("Tempo de treino por temperatura (s)")
ax.set_ylim(0, 190)
for i, v in enumerate(tempo):
    ax.text(i, v + 2, ("sem treino" if v == 0 else f"{v:.1f} s"), ha="center", fontsize=11)
salvar(fig, "custo_computacional")

print("\nTodos os graficos foram salvos em:", os.path.abspath(PASTA))

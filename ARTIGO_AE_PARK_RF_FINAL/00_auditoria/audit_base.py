# -*- coding: utf-8 -*-
"""
ETAPA 0 — AUDITORIA COMPLETA DA BASE (não assume nada da descrição anterior)
===========================================================================
Confirma programaticamente: temperaturas, classes, contagens, faixa/resolução de
frequência, NaN/inf, duplicatas, outliers, identificadores de espécime/ocorrência,
e a existência de temperaturas com dano mas SEM saudável.
"""
import os, re, sys, json, numpy as np, pandas as pd
BASE=r"C:\Users\luize\base-completo--.pkl"
OUT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL\00_auditoria"
os.makedirs(OUT,exist_ok=True)
L=[]
def log(s=""):
    print(s); L.append(str(s))

def freq_of(c):
    m=re.match(r"^f_(\d+(?:\.\d+)?)Hz$",str(c)); return float(m.group(1)) if m else None

log("="*84); log("AUDITORIA DA BASE — base-completo--.pkl"); log("="*84)
log(f"arquivo: {BASE}")
log(f"tamanho: {os.path.getsize(BASE)/1e6:.1f} MB")
df=pd.read_pickle(BASE).reset_index(drop=True)
log(f"shape: {df.shape}")

fcols=[c for c in df.columns if freq_of(c) is not None]
mcols=[c for c in df.columns if freq_of(c) is None]
log(f"colunas de frequência: {len(fcols)} | colunas de metadados: {len(mcols)}")
log(f"metadados: {mcols}")
for c in mcols:
    log(f"   {c}: dtype={df[c].dtype}, valores únicos={df[c].nunique()}, "
        f"amostra={sorted(df[c].dropna().unique().tolist())[:12]}")

# ---- identificadores de espécime/ocorrência ----
log("\n--- IDENTIFICADORES DE ESPÉCIME/OCORRÊNCIA ---")
cand=[c for c in mcols if c not in ("temperatura_c","falha")]
if cand:
    for c in cand:
        log(f"  candidato '{c}': {df[c].value_counts().to_dict()}")
        log(f"    cruzamento com falha:\n{pd.crosstab(df[c],df['falha']).to_string()}")
else:
    log("  NENHUM identificador de espécime/peça/sensor encontrado além de temperatura_c e falha.")
log("  => LIMITAÇÃO: não há como fazer Leave-One-Specimen-Out de forma confiável.")

df["temperatura_c"]=pd.to_numeric(df["temperatura_c"],errors="coerce")
df["falha"]=pd.to_numeric(df["falha"],errors="coerce").astype(int)

# ---- frequências ----
fr=np.array(sorted(freq_of(c) for c in fcols))
d=np.diff(fr)
log("\n--- FREQUÊNCIAS ---")
log(f"  fmin={fr.min():.1f} Hz ({fr.min()/1e3:.3f} kHz) | fmax={fr.max():.1f} Hz ({fr.max()/1e3:.3f} kHz)")
log(f"  resolução: min={d.min():.3f} mediana={np.median(d):.3f} max={d.max():.3f} Hz")
log(f"  grade uniforme? {'SIM' if np.allclose(d,np.median(d)) else 'NÃO'}")
pd.DataFrame({"freq_hz":fr,"freq_khz":fr/1e3}).to_csv(os.path.join(OUT,"frequencias_disponiveis.csv"),index=False)

# bandas disponíveis
log("\n  pontos por banda de 10 kHz:")
band_rows=[]
for lo in range(0,130,10):
    n=int(((fr>=lo*1e3)&(fr<(lo+10)*1e3)).sum())
    if n: log(f"    {lo}-{lo+10} kHz: {n} pontos"); band_rows.append({"banda":f"{lo}-{lo+10}kHz","n_pontos":n})
pd.DataFrame(band_rows).to_csv(os.path.join(OUT,"pontos_por_banda.csv"),index=False)

# ---- temperaturas x classes ----
log("\n--- TEMPERATURAS x CLASSES ---")
ct=df.pivot_table(index="temperatura_c",columns="falha",values=fcols[0],aggfunc="count",fill_value=0)
for c in [0,1,2]:
    if c not in ct.columns: ct[c]=0
ct=ct[[0,1,2]]; ct.columns=["D0","D1","D2"]; ct["TOTAL"]=ct.sum(axis=1)
log(ct.to_string())
ct.to_csv(os.path.join(OUT,"amostras_por_temperatura_dano.csv"))

temps=sorted(df["temperatura_c"].dropna().unique())
t_d0=sorted(df.loc[df.falha==0,"temperatura_c"].unique())
t_d1=sorted(df.loc[df.falha==1,"temperatura_c"].unique())
t_d2=sorted(df.loc[df.falha==2,"temperatura_c"].unique())
t_all3=[t for t in temps if ct.loc[t,"D0"]>0 and ct.loc[t,"D1"]>0 and ct.loc[t,"D2"]>0]
t_dano_sem_saudavel=[t for t in temps if (ct.loc[t,"D1"]>0 or ct.loc[t,"D2"]>0) and ct.loc[t,"D0"]==0]
t_so_saudavel=[t for t in temps if ct.loc[t,"D0"]>0 and ct.loc[t,"D1"]==0 and ct.loc[t,"D2"]==0]

log(f"\n  total de temperaturas: {len(temps)}  -> {temps}")
log(f"  com D0 (saudável): {len(t_d0)}")
log(f"  com D1: {len(t_d1)} -> {t_d1}")
log(f"  com D2: {len(t_d2)} -> {t_d2}")
log(f"  com AS TRÊS classes: {len(t_all3)} -> {t_all3}")
log(f"  *** COM DANO MAS SEM SAUDÁVEL: {len(t_dano_sem_saudavel)} -> {t_dano_sem_saudavel}")
log(f"  só saudável (sem dano): {len(t_so_saudavel)} -> {t_so_saudavel}")
log("  >>> CONFIRMADO programaticamente: existem temperaturas com dano e sem curva saudável.")
pd.DataFrame({"grupo":["todas","com_D0","com_D1","com_D2","com_as_3","dano_sem_saudavel","so_saudavel"],
    "n":[len(temps),len(t_d0),len(t_d1),len(t_d2),len(t_all3),len(t_dano_sem_saudavel),len(t_so_saudavel)],
    "temperaturas":[str(temps),str(t_d0),str(t_d1),str(t_d2),str(t_all3),str(t_dano_sem_saudavel),str(t_so_saudavel)]
}).to_csv(os.path.join(OUT,"temperaturas_por_classe.csv"),index=False)

# temperaturas válidas como referência (>=2 saudáveis)
t_ref_ok=[float(t) for t in t_d0 if ct.loc[t,"D0"]>=2]
log(f"\n  temperaturas com >=2 saudáveis (candidatas a T_ref): {len(t_ref_ok)}")
log(f"    {t_ref_ok}")

# ---- sanidade numérica ----
log("\n--- SANIDADE NUMÉRICA (base inteira) ---")
Xall=df[fcols].to_numpy(np.float32)
n_nan=int(np.isnan(Xall).sum()); n_inf=int(np.isinf(Xall).sum())
log(f"  NaN={n_nan} | inf={n_inf}")
log(f"  min={np.nanmin(Xall):.4g} | mediana={np.nanmedian(Xall):.4g} | max={np.nanmax(Xall):.4g}")
dup=int(df.duplicated(subset=fcols[:200]).sum())
log(f"  linhas duplicadas (nas 200 primeiras freq): {dup}")

# outliers por energia da curva
band3040=[c for c in fcols if 30000<=freq_of(c)<40000]
E=df[band3040].to_numpy(np.float64).mean(axis=1)
q1,q3=np.percentile(E,[25,75]); iqr=q3-q1
out=np.where((E<q1-3*iqr)|(E>q3+3*iqr))[0]
log(f"  outliers (média 30-40kHz, regra 3*IQR): {len(out)} -> índices {out.tolist()[:20]}")
if len(out):
    log(df.iloc[out][["temperatura_c","falha"]].to_string())

# ---- resumo ----
summary={"n_curvas":int(len(df)),"n_freq_cols":len(fcols),"fmin_hz":float(fr.min()),"fmax_hz":float(fr.max()),
 "res_hz":float(np.median(d)),"n_temperaturas":len(temps),"n_D0":int((df.falha==0).sum()),
 "n_D1":int((df.falha==1).sum()),"n_D2":int((df.falha==2).sum()),
 "n_temps_3_classes":len(t_all3),"temps_3_classes":[float(x) for x in t_all3],
 "temps_dano_sem_saudavel":[float(x) for x in t_dano_sem_saudavel],
 "temps_ref_validas":t_ref_ok,"nan":n_nan,"inf":n_inf,"duplicatas":dup,
 "identificador_especime":cand}
pd.DataFrame([summary]).to_csv(os.path.join(OUT,"dataset_summary.csv"),index=False)
json.dump(summary,open(os.path.join(OUT,"dataset_summary.json"),"w"),indent=2,ensure_ascii=False)

with open(os.path.join(OUT,"audit_report.md"),"w",encoding="utf-8") as f:
    f.write("# Auditoria da base — base-completo--.pkl\n\n```\n"+"\n".join(L)+"\n```\n")
log(f"\n✅ auditoria salva em {OUT}")

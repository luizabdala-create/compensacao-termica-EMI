# -*- coding: utf-8 -*-
"""
MODELO FORMA — compensacao focada no FORMATO (shape) da curva compensada
========================================================================
Objetivo do usuario: um modelo cuja curva compensada fique com o MESMO FORMATO
da curva de referencia SAUDAVEL (so remove temperatura, nada mais). AE e RF.
Referencia = mediana saudavel @ T_ref (congelada). Treino SO no saudavel de treino,
aplicado IGUAL as curvas de dano (o modelo nunca ve dano -> nao pode "consertar" dano).

REGRA DE OURO (nao negociavel):
  - Um modelo de FORMA so e melhor se melhora (A) SEM piorar (B).
  - (A) remocao termica  -> nas curvas SAUDAVEIS de teste, shape vs ref melhora?
  - (B) preservacao do dano -> classificacao 0/1/2 antes vs depois NAO pode cair;
        curvas de dano NAO podem virar saudaveis.
  - Explorar varias faixas/objetivos e REPORTAR com honestidade onde ganha e onde
    perde. NUNCA ajustar so para o AE/RF "ganhar" do Park.

Dois compensadores de forma (alem de Original / Park / AE-amp / RF-amp):
  AE_forma : loss = Huber(correcao) + lam_d*Huber(derivada) + lam_c*(1 - corr(compensada, ref))
             o termo de correlacao age sobre a CURVA COMPENSADA reconstruida nas ancoras.
             lam_c=0 recupera o AE atual (amplitude). lam_c escolhido por CV interna (CCDM).
  RF_forma : mesma familia (ExtraTrees, alvo = ref-curva), mas hiperparametros escolhidos
             por CV interna pelo CCDM (forma) em vez de RMSD (amplitude).

Selecao de hiperparametros: UMA vez por banda, por CV interna num subconjunto de
temperaturas de validacao (mesmo espirito da parteB, que usou params fixos por banda).
Cada fold LOTO e RE-TREINADO excluindo a temperatura de teste (sem leakage no ajuste).
Referencia congelada, so saudavel. Temperatura de teste nunca entra no scaler/treino.

Avaliacao (LOTO externo):
  (A) metricas de forma no SAUDAVEL de teste: CCDM, CORR, RMSD, SAM_deg, NRMSE, erro de pico (Hz)
  (B) classificacao 0/1/2 antes/depois (protocolo parteB: FS1-4 x logreg/svm/rfc x bin/multi
      + controle negativo com rotulos embaralhados p/ detectar leakage)
  (C) guarda de dano: distancia (CCDM/RMSD) das curvas D1/D2 a referencia — NAO pode
      colapsar para ~0 (isso seria apagar o dano).

Robusto: pasta timestamp, config JSON, salvamento incremental, resume, try/except por
experimento, seeds fixas, liberacao de memoria. Resolucao de trabalho decimada p/ ~WORK_PTS.

Uso:
  DRY=1  python modelo_forma.py     # valida ponta-a-ponta (1 banda, poucos folds, epocas curtas)
         python modelo_forma.py     # bateria completa (resume automatico)
"""
import os, sys, json, time, gc, traceback, numpy as np, pandas as pd
ROOT = r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
sys.path.insert(0, ROOT)
import pipeline as P
import torch, torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, precision_score, confusion_matrix)

# ---------------------------------------------------------------- config
DRY   = os.environ.get("DRY", "0") == "1"
T_REF = 30.0
SEED  = 42
WORK_PTS = 2000                # resolucao de trabalho (shape metrics finas o bastante; ~2-5 Hz/pt)
OUT   = os.path.join(ROOT, "13_modelo_forma")
os.makedirs(OUT, exist_ok=True)
for sub in ("logs", "metricas", "curvas", "graficos", "config", "checkpoints"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

BANDS   = ["30-40"] if DRY else ["30-40", "40-50", "50-60", "60-70", "70-80", "30-70"]
LAM_C   = [0.0, 1.0] if DRY else [0.0, 0.5, 1.5, 4.0, 8.0]   # peso de forma do AE (0 = AE atual)
AE_EPO  = 120 if DRY else 500
AE_PAT  = 30 if DRY else 70
MAX_FOLDS = 3 if DRY else None
INNER_MAX = 3 if DRY else 5                              # temps de validacao interna (selecao)

# clip_k: limita a correcao termica ao envelope [mu +- k*sd] do DELTA saudavel de treino
# (remove artefatos de sobrecorreccao do termo de forma sem tocar no dano). smooth_win: leve.
AE_BASE = {"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,
           "dropout":0.10,"noise":0.01,"epochs":AE_EPO,"patience":AE_PAT,"lambda_d1":0.3,
           "clip_k":3.5,"smooth_win":5}
PARK_TUNED = {"30-40":{"max_shift_frac":0.05,"nsteps":121,"smooth_win":9},
              "40-50":{"max_shift_frac":0.10,"nsteps":121,"smooth_win":9},
              "50-60":{"max_shift_frac":0.15,"nsteps":121,"smooth_win":9},
              "60-70":{"max_shift_frac":0.25,"nsteps":121,"smooth_win":9},
              "70-80":{"max_shift_frac":0.15,"nsteps":121,"smooth_win":9},
              "30-70":{"max_shift_frac":0.10,"nsteps":121,"smooth_win":9}}
RF_GRID = [dict(n_estimators=200, max_depth=dp, min_samples_leaf=lf, max_features="sqrt",
                smooth_win=sw, input_decim=2)
           for dp in (10, 16) for lf in (1,) for sw in (5, 9)]        # 4 configs
if DRY: RF_GRID = RF_GRID[:2]

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "logs", "run.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

# ---------------------------------------------------------------- base
df, COLS, FR = P.load_base()
T = df["temperatura_c"].to_numpy(float)
y = df["falha"].to_numpy(int)
ybin = (y > 0).astype(int)
FOLDS3 = [t for t in sorted(np.unique(T)) if all(((np.isclose(T, t)) & (y == c)).sum() > 0 for c in (0,1,2))]
FOLDS0 = [t for t in sorted(np.unique(T)) if ((np.isclose(T, t)) & (y == 0)).sum() > 0]

def band_data(banda):
    lo, hi = map(int, banda.split("-"))
    fc, f = P.band(lo, hi, decim=1)
    dec = max(1, round(len(fc)/WORK_PTS))
    fc, f = P.band(lo, hi, decim=dec)
    X = df[fc].to_numpy(np.float64)
    ref, n_ref = P.build_reference(df, fc, T_REF)
    return X, f, ref, n_ref

# ---------------------------------------------------------------- metricas de forma
def peak_freq_err(ycurve, ref, f):
    return float(abs(f[int(np.argmax(ycurve))] - f[int(np.argmax(ref))]))

def shape_metrics(ycurve, ref, f):
    m = P.all_metrics(ycurve, ref)
    m["peak_hz"] = peak_freq_err(ycurve, ref, f)
    return m

# ---------------------------------------------------------------- AE de FORMA
def _pearson_rows(a, b, eps=1e-8):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = torch.sqrt((a*a).sum(dim=1) * (b*b).sum(dim=1) + eps) + eps
    return num / den

def comp_ae_forma(X, T, ref, mask, params, T_ref, lambda_c=0.0, seed=SEED):
    """AE treinado SO no saudavel de treino. lambda_c>0 adiciona perda de FORMA
    (1-corr) sobre a curva COMPENSADA reconstruida nas ancoras vs. ref. Sem leakage."""
    torch.manual_seed(seed); np.random.seed(seed)
    n_pts = X.shape[1]
    dec  = max(1, n_pts // params.get("n_input", 1000))
    n_anch = params.get("n_anchors", 64)
    anchors = np.linspace(0, n_pts-1, n_anch).round().astype(int)

    Xh = X[mask]; Th = T[mask]
    if len(Xh) < 6: raise ValueError("poucas saudaveis para treinar AE")
    D = ref[None, :] - Xh
    xin_tr = Xh[:, ::dec]
    mu_x, sd_x = xin_tr.mean(0), xin_tr.std(0)+1e-8
    d_tr = D[:, anchors]; mu_d, sd_d = d_tr.mean(0), d_tr.std(0)+1e-8

    def prep(Xa, Ta):
        xi = (Xa[:, ::dec]-mu_x)/sd_x
        t  = ((Ta-T_ref)/100.0).reshape(-1,1); dt = (np.abs(Ta-T_ref)/100.0).reshape(-1,1)
        return (torch.tensor(xi,dtype=torch.float32), torch.tensor(t,dtype=torch.float32),
                torch.tensor(dt,dtype=torch.float32))

    xt, tt, dtt = prep(Xh, Th)
    yt = torch.tensor((d_tr-mu_d)/sd_d, dtype=torch.float32)
    Xanch = torch.tensor(Xh[:, anchors], dtype=torch.float32)
    refA  = torch.tensor(ref[anchors], dtype=torch.float32)
    mu_dT = torch.tensor(mu_d, dtype=torch.float32); sd_dT = torch.tensor(sd_d, dtype=torch.float32)

    n = len(xt); idx = np.random.RandomState(seed).permutation(n)
    nv = max(2, int(0.2*n)); vi, ti_ = idx[:nv], idx[nv:]

    model = P.ThermalAE(xt.shape[1], n_anch, params.get("latent",12),
                        params.get("hidden",128), params.get("dropout",0.10))
    opt = torch.optim.AdamW(model.parameters(), lr=params.get("lr",2e-3),
                            weight_decay=params.get("wd",1e-4))
    huber = nn.SmoothL1Loss()
    lam_d = params.get("lambda_d1", 0.3)

    Xtr,Ttr,DTtr,Ytr = xt[ti_],tt[ti_],dtt[ti_],yt[ti_]
    Xv,Tv,DTv,Yv     = xt[vi],tt[vi],dtt[vi],yt[vi]
    XaTr, XaV = Xanch[ti_], Xanch[vi]
    noise = params.get("noise", 0.01)

    def composite(p, Ytar, Xa):
        l = huber(p, Ytar) + lam_d*huber(p[:,1:]-p[:,:-1], Ytar[:,1:]-Ytar[:,:-1])
        if lambda_c > 0:
            comp = Xa + (p*sd_dT + mu_dT)
            l = l + lambda_c*(1.0 - _pearson_rows(comp, refA.expand_as(comp))).mean()
        return l

    best = (np.inf, None); bad = 0
    for ep in range(params.get("epochs", 400)):
        model.train(); opt.zero_grad()
        xin = Xtr + noise*torch.randn_like(Xtr) if noise > 0 else Xtr
        p = model(xin, Ttr, DTtr)
        loss = composite(p, Ytr, XaTr)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(composite(model(Xv, Tv, DTv), Yv, XaV))
        if vl < best[0]-1e-7:
            best = (vl, {k:v.detach().clone() for k,v in model.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= params.get("patience", 60): break
    if best[1] is not None: model.load_state_dict(best[1])
    model.eval()
    xa, ta, dta = prep(X, T)
    with torch.no_grad():
        pa = model(xa, ta, dta).cpu().numpy()
    da = pa*sd_d + mu_d
    ck = params.get("clip_k")               # limita a correcao ao envelope termico do saudavel
    if ck: da = np.clip(da, mu_d - ck*sd_d, mu_d + ck*sd_d)
    full = np.arange(n_pts)
    Y = np.vstack([X[i] + np.interp(full, anchors, da[i]) for i in range(len(X))])
    w = params.get("smooth_win", 1)
    if w > 1: Y = np.vstack([P.moving_average(yy, w) for yy in Y])
    return Y, {"val_loss":best[0], "epochs":ep+1, "lambda_c":lambda_c}

# ---------------------------------------------------------------- ExtraTrees (RF de forma)
def comp_et(X, T, ref, mask, params):
    idc = params.get("input_decim", 2); Xh = X[mask]; Th = T[mask]
    et = ExtraTreesRegressor(n_estimators=params.get("n_estimators",200),
        max_depth=params.get("max_depth",10), min_samples_leaf=params.get("min_samples_leaf",1),
        max_features=params.get("max_features","sqrt"), n_jobs=-1, random_state=0)
    et.fit(P._rf_feats(Xh, Th, "direct", idc), ref[None,:]-Xh)
    Y = X + et.predict(P._rf_feats(X, T, "direct", idc))
    w = params.get("smooth_win", 5)
    if w > 1: Y = np.vstack([P.moving_average(yy, w) for yy in Y])
    return Y, {"n_train":int(len(Xh))}

# ---------------------------------------------------------------- selecao UMA vez por banda
def inner_temps():
    base = [t for t in FOLDS0 if not np.isclose(t, T_REF)]
    if len(base) <= INNER_MAX: return base
    idx = np.linspace(0, len(base)-1, INNER_MAX).round().astype(int)
    return [base[i] for i in sorted(set(idx))]

def _held_scores(X, ref, make, itemps, metrics):
    """Media dos metrics no saudavel held-out, treinando so no restante (saudavel)."""
    acc = {m: [] for m in metrics}
    for tk in itemps:
        ho = np.isclose(T, tk) & (y == 0); keep = (~np.isclose(T, tk)) & (y == 0)
        if ho.sum() == 0 or keep.sum() < 8: continue
        try:
            Y = make(keep); ii = np.where(ho)[0]
            for m in metrics:
                acc[m].append(np.mean([P.all_metrics(Y[i], ref)[m] for i in ii]))
        except Exception:
            for m in metrics: acc[m].append(np.inf)
    return {m: (float(np.mean(v)) if v else np.inf) for m, v in acc.items()}

def select_configs(X, f, ref):
    itemps = inner_temps()
    rf_rows = []
    for g in RF_GRID:
        s = _held_scores(X, ref, lambda k, gg=g: comp_et(X, T, ref, k, gg)[0], itemps, ["RMSD","CCDM"])
        rf_rows.append({"g":g, **s})
    rf_amp   = min(rf_rows, key=lambda r: r["RMSD"])["g"]
    rf_forma = min(rf_rows, key=lambda r: r["CCDM"])["g"]
    ae_scores = {}
    for lc in LAM_C:
        s = _held_scores(X, ref, lambda k, l=lc: comp_ae_forma(X, T, ref, k, AE_BASE, T_REF, l)[0], itemps, ["CCDM"])
        ae_scores[lc] = s["CCDM"]
    lc_best = min(ae_scores, key=ae_scores.get)
    return rf_amp, rf_forma, lc_best, itemps, ae_scores, rf_rows

# ---------------------------------------------------------------- classificacao (protocolo parteB)
def clf_factory(name, seed=0):
    if name == "logreg":
        return Pipeline([("sc",StandardScaler()),("c",LogisticRegression(max_iter=5000,class_weight="balanced",random_state=seed))])
    if name == "svm":
        return Pipeline([("sc",StandardScaler()),("c",SVC(kernel="rbf",class_weight="balanced",random_state=seed))])
    if name == "rfc":
        return Pipeline([("sc",StandardScaler()),("c",RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=-1,random_state=seed))])
    raise ValueError(name)

def feats(Y, ref, fs, tr_mask, te_mask):
    R = Y - ref[None, :]
    if fs in ("FS1", "FS2"):
        M = []
        for i in range(len(Y)):
            m = P.all_metrics(Y[i], ref)
            M.append([m["RMSD"],m["CCDM"]] if fs=="FS1" else
                     [m["RMSD"],m["CCDM"],m["RMSE"],m["MAE"],m["NRMSE"],m["CORR"],m["SAM_deg"]])
        M = np.array(M); return M[tr_mask], M[te_mask]
    if fs == "FS3":
        w = max(1, R.shape[1]//500)
        Rs = np.vstack([P.moving_average(r,51)[::w] for r in R])
        pca = PCA(n_components=min(10, tr_mask.sum()-1), random_state=0).fit(Rs[tr_mask])
        return pca.transform(Rs[tr_mask]), pca.transform(Rs[te_mask])
    if fs == "FS4":
        Fm = []
        for r in R:
            rs = P.moving_average(r, 51)
            idx = np.argsort(-np.abs(rs))[:200]
            Fm.append([rs.max(),rs.min(),np.ptp(rs),np.std(rs),np.mean(np.abs(rs)),
                       float(np.mean(idx)),float(np.std(idx)),
                       float(np.percentile(np.abs(rs),95)),float(np.sum(rs**2))])
        Fm = np.array(Fm); return Fm[tr_mask], Fm[te_mask]
    raise ValueError(fs)

def evaluate(yte, pred, binario):
    d = {"accuracy":accuracy_score(yte,pred),"bal_acc":balanced_accuracy_score(yte,pred),
         "macro_f1":f1_score(yte,pred,average="macro",zero_division=0)}
    if binario:
        cm = confusion_matrix(yte,pred,labels=[0,1])
        d["recall_dano"]=recall_score(yte,pred,pos_label=1,zero_division=0)
        d["taxa_falso_saudavel"]=float(cm[1,0]/max(cm[1].sum(),1))
    else:
        f1 = f1_score(yte,pred,average=None,labels=[0,1,2],zero_division=0)
        for i,c in enumerate([0,1,2]): d[f"f1_D{c}"]=float(f1[i])
    return d

FS_LIST  = ["FS1","FS4"] if DRY else ["FS1","FS2","FS3","FS4"]
CLF_LIST = ["logreg"] if DRY else ["logreg","svm","rfc"]

# ---------------------------------------------------------------- resume/checkpoint
CK_A = os.path.join(OUT, "metricas", "shapeA_saudavel.csv")
CK_C = os.path.join(OUT, "metricas", "guardaC_dano.csv")
CK_B = os.path.join(OUT, "metricas", "classB_dano.csv")
CK_SEL = os.path.join(OUT, "checkpoints", "selecao_por_banda.json")
def done_keys(path, keycols):
    if not os.path.exists(path): return set()
    d = pd.read_csv(path)
    if not len(d): return set()
    return set(map(tuple, d[keycols].astype(str).values.tolist()))
def append_rows(path, rows):
    if not rows: return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

sel_store = json.load(open(CK_SEL)) if os.path.exists(CK_SEL) else {}

cfg = {"DRY":DRY,"T_REF":T_REF,"SEED":SEED,"WORK_PTS":WORK_PTS,"BANDS":BANDS,"LAM_C":LAM_C,
       "AE_BASE":AE_BASE,"RF_GRID_n":len(RF_GRID),"FS_LIST":FS_LIST,"CLF_LIST":CLF_LIST,
       "INNER_MAX":INNER_MAX,"FOLDS3":FOLDS3,"FOLDS0":FOLDS0}
json.dump(cfg, open(os.path.join(OUT,"config","config.json"),"w"), indent=2, default=str)

# ---------------------------------------------------------------- LOOP PRINCIPAL
t0 = time.time()
log(f"=== MODELO FORMA {'(DRY RUN)' if DRY else '(COMPLETO)'} | bandas={BANDS} | lam_c={LAM_C} | work_pts~{WORK_PTS} ===")
kA = done_keys(CK_A, ["banda","T_test","metodo"])
kC = done_keys(CK_C, ["banda","T_test","metodo"])
kB = done_keys(CK_B, ["banda","T_test","metodo","feature_set","clf","task","controle"])

for banda in BANDS:
    X, f, ref, n_ref = band_data(banda)
    if ref is None:
        log(f"  {banda}: SEM referencia @T_ref={T_REF} — pulando"); continue
    log(f"--- banda {banda} | {X.shape[1]} pts | ref de {n_ref} curvas saudaveis @ {T_REF}C ---")
    park_p = PARK_TUNED.get(banda, {"max_shift_frac":0.10,"nsteps":121,"smooth_win":9})

    # selecao UMA vez por banda (com resume)
    if banda in sel_store:
        rf_amp = sel_store[banda]["rf_amp"]; rf_forma = sel_store[banda]["rf_forma"]; lc_best = sel_store[banda]["lc_best"]
        log(f"  (selecao carregada) rf_amp={rf_amp} | rf_forma={rf_forma} | lam_c={lc_best}")
    else:
        rf_amp, rf_forma, lc_best, itemps, ae_scores, rf_rows = select_configs(X, f, ref)
        sel_store[banda] = {"rf_amp":rf_amp,"rf_forma":rf_forma,"lc_best":lc_best,
                            "ae_ccdm_por_lamc":ae_scores,"inner_temps":itemps,
                            "rf_scores":[{"g":r["g"],"RMSD":r["RMSD"],"CCDM":r["CCDM"]} for r in rf_rows]}
        json.dump(sel_store, open(CK_SEL,"w"), indent=2, default=str)
        log(f"  selecao: rf_amp(d{rf_amp['max_depth']},s{rf_amp['smooth_win']}) "
            f"rf_forma(d{rf_forma['max_depth']},s{rf_forma['smooth_win']}) | lam_c*={lc_best} "
            f"| CCDM(AE) por lam_c={ {k:round(v,4) for k,v in ae_scores.items()} }")

    folds = FOLDS3 if not DRY else FOLDS3[:MAX_FOLDS]
    rep_temps = {folds[0], folds[-1]} if len(folds) else set()   # folds p/ salvar curvas de exemplo
    for T_test in folds:
        if np.isclose(T_test, T_REF): continue
        te = np.isclose(T, T_test); tr = ~te; trh = tr & (y == 0)
        i0 = np.where(te & (y == 0))[0]
        idmg = np.where(te & (y > 0))[0]
        if trh.sum() < 8 or len(i0) == 0: continue

        methods = {}
        try:
            methods["Original"] = X.copy()
            methods["Park"]     = P.comp_park(X, ref, f, park_p)[0]
            methods["RF_amp"]   = comp_et(X, T, ref, trh, rf_amp)[0]
            methods["RF_forma"] = comp_et(X, T, ref, trh, rf_forma)[0]
            methods["AE_amp"]   = comp_ae_forma(X, T, ref, trh, AE_BASE, T_REF, 0.0)[0]
            methods["AE_forma"] = comp_ae_forma(X, T, ref, trh, AE_BASE, T_REF, lc_best)[0]
        except Exception as e:
            log(f"  [{banda} T={T_test}] erro compensacao: {e}\n{traceback.format_exc()}"); continue

        # salvar curvas de exemplo (folds representativos) p/ figuras
        if T_test in rep_temps:
            expath = os.path.join(OUT, "curvas", f"ex_{banda}_T{int(round(T_test))}.npz")
            if not os.path.exists(expath):
                d1 = np.where(te & (y==1))[0]; d2 = np.where(te & (y==2))[0]
                ih = int(i0[0])
                ex = {"f": f, "ref": ref, "T_test": float(T_test), "banda": banda,
                      "idx_healthy": ih, "idx_d1": int(d1[0]) if len(d1) else -1,
                      "idx_d2": int(d2[0]) if len(d2) else -1}
                for mname, Y in methods.items():
                    ex[f"Y_{mname}_healthy"] = Y[ih]
                    if len(d1): ex[f"Y_{mname}_d1"] = Y[int(d1[0])]
                    if len(d2): ex[f"Y_{mname}_d2"] = Y[int(d2[0])]
                np.savez_compressed(expath, **ex)

        # (A) saudavel de teste + (C) guarda de dano
        rowsA, rowsC = [], []
        for mname, Y in methods.items():
            kkey = (banda, str(T_test), mname)
            if kkey not in kA:
                for i in i0:
                    rowsA.append({"banda":banda,"T_test":T_test,"metodo":mname,"dano":0,
                                  "lam_c":(lc_best if mname=="AE_forma" else (0.0 if mname=="AE_amp" else np.nan)),
                                  **shape_metrics(Y[i], ref, f)})
            if kkey not in kC:
                for i in idmg:
                    rowsC.append({"banda":banda,"T_test":T_test,"metodo":mname,"dano":int(y[i]),
                                  **shape_metrics(Y[i], ref, f)})
        append_rows(CK_A, rowsA); append_rows(CK_C, rowsC)

        # (B) classificacao antes/depois
        rowsB = []
        for mname, Y in methods.items():
            for fs in FS_LIST:
                try: Xtr, Xte = feats(Y, ref, fs, tr, te)
                except Exception: continue
                for cname in CLF_LIST:
                    for task, yy in [("bin", ybin), ("multi", y)]:
                        if (banda, str(T_test), mname, fs, cname, task, "real") in kB: continue
                        try:
                            c = clf_factory(cname); c.fit(Xtr, yy[tr]); pr = c.predict(Xte)
                            rec = {"banda":banda,"T_test":T_test,"metodo":mname,"feature_set":fs,
                                   "clf":cname,"task":task,"controle":"real"}
                            rec.update(evaluate(yy[te], pr, task=="bin")); rowsB.append(rec)
                            rng = np.random.RandomState(0); ysh = yy[tr].copy(); rng.shuffle(ysh)
                            c2 = clf_factory(cname); c2.fit(Xtr, ysh); pr2 = c2.predict(Xte)
                            rec2 = {"banda":banda,"T_test":T_test,"metodo":mname,"feature_set":fs,
                                    "clf":cname,"task":task,"controle":"shuffled"}
                            rec2.update(evaluate(yy[te], pr2, task=="bin")); rowsB.append(rec2)
                        except Exception:
                            pass
        append_rows(CK_B, rowsB)
        log(f"  [{banda} T={T_test:+.0f}] +A{len(rowsA)} +C{len(rowsC)} +B{len(rowsB)} | {(time.time()-t0)/60:.1f}min")

    del X; gc.collect()

log(f"=== FIM em {(time.time()-t0)/60:.1f} min ===")

# ---------------------------------------------------------------- RESUMO honesto
def safe_read(p): return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
A = safe_read(CK_A); B = safe_read(CK_B); C = safe_read(CK_C)
resumo = {}
if len(A):
    gA = A.groupby("metodo")[["RMSD","CCDM","CORR","SAM_deg","peak_hz"]].mean().round(4)
    resumo["A_forma_saudavel"] = json.loads(gA.to_json())
    log("\n=== (A) FORMA no SAUDAVEL (media; CCDM/RMSD/SAM/peak menor=melhor, CORR maior=melhor) ===")
    log(gA.sort_values("CCDM").to_string())
if len(B):
    real = B[B.controle=="real"]
    gB = real.groupby("metodo")[["bal_acc","macro_f1"]].mean().round(4)
    resumo["B_classificacao"] = json.loads(gB.to_json())
    log("\n=== (B) CLASSIFICACAO de dano (media; maior=melhor) ===")
    log(gB.sort_values("bal_acc", ascending=False).to_string())
    sh = B[B.controle=="shuffled"].groupby("metodo")["bal_acc"].mean().round(4)
    log(f"  controle negativo (embaralhado) bal_acc: {sh.to_dict()}  (esperado ~0.33-0.5; alto=LEAKAGE)")
if len(C):
    gC = C.groupby(["metodo","dano"])[["CCDM","RMSD"]].mean().round(4)
    resumo["C_guarda_dano"] = C.groupby(["metodo","dano"])[["CCDM","RMSD"]].mean().round(4).reset_index().to_dict(orient="records")
    log("\n=== (C) GUARDA DE DANO: distancia das curvas D1/D2 a referencia (NAO pode colapsar p/ ~0) ===")
    log(gC.to_string())
json.dump(resumo, open(os.path.join(OUT,"metricas","resumo.json"),"w"), indent=2, default=str)
log(f"\nOK. Saidas em {OUT}")
print("DONE_MODELO_FORMA", flush=True)

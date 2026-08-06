# -*- coding: utf-8 -*-
"""
PIPELINE UNIFICADA — AE (rede neural) x PARK x RANDOM FOREST x ORIGINAL
=======================================================================
Métodos principais do artigo:
  ORIGINAL : sem compensação (baseline obrigatório)
  PARK     : shift horizontal + offset vertical (clássico)
  RF       : RandomForestRegressor aprendendo a correção térmica (2 formulações)
  AE       : autoencoder NEURAL (PyTorch), treinado SÓ em curvas saudáveis
Auxiliar (ablation, NUNCA chamado de AE):
  TAU_T    : shift térmico τ(T) + modelo térmico de baixo posto (SVD)

ANTI-LEAKAGE (regra dura):
  - protocolo externo = Leave-One-Temperature-Out (LOTO)
  - a temperatura de teste NUNCA entra em: treino do AE/RF, scaler, PCA, SVD,
    seleção de hiperparâmetro, seleção de janela/faixa, treino do classificador.
  - PROTOCOLO A (padrão): a referência é uma curva saudável real em T_ref, declarada
    como calibração física disponível a priori e CONGELADA em todos os folds.
    Se T_test == T_ref o fold é marcado e reportado à parte.
  - PROTOCOLO B (strict): a referência é remontada só com temperaturas de treino.
"""
import os, re, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
import torch, torch.nn as nn

BASE_PATH = r"C:\Users\luize\base-completo--.pkl"

# ============================ DADOS ============================
def freq_of(c):
    m = re.match(r"^f_(\d+(?:\.\d+)?)Hz$", str(c)); return float(m.group(1)) if m else None

_CACHE = {}
def load_base():
    if "df" not in _CACHE:
        df = pd.read_pickle(BASE_PATH).reset_index(drop=True)
        df["temperatura_c"] = pd.to_numeric(df["temperatura_c"], errors="coerce")
        df["falha"] = pd.to_numeric(df["falha"], errors="coerce").astype(int)
        _CACHE["df"] = df
        allf = [(c, freq_of(c)) for c in df.columns if freq_of(c) is not None]
        allf.sort(key=lambda t: t[1])
        _CACHE["cols"] = [c for c, _ in allf]
        _CACHE["freqs"] = np.array([f for _, f in allf])
    return _CACHE["df"], _CACHE["cols"], _CACHE["freqs"]

def band(fmin_khz, fmax_khz, decim=1):
    """Colunas e frequências de uma banda. decim>1 subamostra (para custo)."""
    df, cols, fr = load_base()
    m = (fr >= fmin_khz*1e3) & (fr < fmax_khz*1e3)
    c = [cc for cc, mm in zip(cols, m) if mm][::decim]
    f = fr[m][::decim]
    return c, f

# ============================ MÉTRICAS ============================
def rmsd(y, r): return float(np.sqrt(np.mean((y-r)**2)))
def rmse(y, r): return rmsd(y, r)
def mae(y, r):  return float(np.mean(np.abs(y-r)))
def nrmse(y, r):
    rng = np.ptp(r)+1e-12; return float(rmsd(y, r)/rng)
def corr(y, r):
    y0=y-y.mean(); r0=r-r.mean()
    return float(np.sum(y0*r0)/(np.sqrt(np.sum(y0**2)*np.sum(r0**2))+1e-18))
def ccdm(y, r): return float(1.0-corr(y, r))
def sam_deg(y, r):
    num=float(np.dot(y,r)); den=float(np.linalg.norm(y)*np.linalg.norm(r))+1e-18
    return float(np.degrees(np.arccos(np.clip(num/den,-1,1))))
def all_metrics(y, r):
    return {"RMSD":rmsd(y,r),"CCDM":ccdm(y,r),"RMSE":rmse(y,r),"MAE":mae(y,r),
            "NRMSE":nrmse(y,r),"CORR":corr(y,r),"SAM_deg":sam_deg(y,r)}

def moving_average(a, w):
    a=np.asarray(a,float)
    if w<=1: return a.copy()
    if w%2==0: w+=1
    pad=w//2
    return np.convolve(np.pad(a,(pad,pad),mode="edge"),np.ones(w)/w,mode="valid")[:len(a)]

# ============================ REFERÊNCIA ============================
def build_reference(df, fcols, T_ref, allowed_mask=None):
    """Mediana das curvas saudáveis em T_ref. allowed_mask=None -> PROTOCOLO A (congelada)."""
    m = (df["falha"].to_numpy(int)==0) & np.isclose(df["temperatura_c"].to_numpy(float), T_ref)
    if allowed_mask is not None: m = m & allowed_mask
    if m.sum()==0: return None, 0
    return np.median(df.loc[m, fcols].to_numpy(np.float64), axis=0), int(m.sum())

# ============================ PARK ============================
def shift_interp(x, f, tau):
    return np.interp(f, f+tau, x, left=x[0], right=x[-1])

def park_one(x, ref, f, max_shift_frac=0.10, nsteps=111, smooth_win=1):
    band_w = f[-1]-f[0]; tmax = max_shift_frac*band_w
    best=(np.inf,0.0,0.0)
    for tau in np.linspace(-tmax, tmax, nsteps):
        xs = shift_interp(x, f, tau); dS = np.mean(ref-xs)
        e = np.mean((ref-(xs+dS))**2)
        if e < best[0]: best=(e,tau,dS)
    y = shift_interp(x, f, best[1]) + best[2]
    return moving_average(y, smooth_win), best[1]

def comp_park(X, ref, f, params):
    Y=np.zeros_like(X); taus=np.zeros(len(X))
    for i in range(len(X)):
        Y[i],taus[i]=park_one(X[i],ref,f,params.get("max_shift_frac",0.10),
                              params.get("nsteps",111),params.get("smooth_win",1))
    return Y, {"tau":taus}

# ============================ RANDOM FOREST ============================
def _rf_feats(X, T, mode, input_decim=1):
    """mode='direct': curva(+decimada)+estatísticas+T.
       mode='temponly': SÓ temperatura (ablation: RF não pode ver o dano).
       input_decim>1 subamostra as colunas da curva de entrada do RF (acelera muito;
       a resolução de ~input_decim Hz ainda é fina para os picos). Estatísticas usam
       a curva completa."""
    T = np.asarray(T,float).reshape(-1,1)
    if mode=="temponly":
        return np.hstack([T, T**2])
    mu=X.mean(1,keepdims=True); sd=X.std(1,keepdims=True)
    amp=(X.max(1)-X.min(1)).reshape(-1,1)
    Xin = X[:, ::input_decim] if input_decim>1 else X
    return np.hstack([Xin, mu, sd, amp, T])

def comp_rf(X, T, ref, train_healthy_mask, params, mode="direct"):
    """Treina SÓ em saudáveis de treino: alvo = ref - curva (delta térmico).
    params['input_decim'] subamostra a entrada do RF (default 1)."""
    idc=params.get("input_decim",1)
    Xh=X[train_healthy_mask]; Th=T[train_healthy_mask]
    if len(Xh)<3: raise ValueError("poucas saudáveis para treinar RF")
    rf=RandomForestRegressor(
        n_estimators=params.get("n_estimators",250), max_depth=params.get("max_depth",10),
        min_samples_leaf=params.get("min_samples_leaf",2), min_samples_split=params.get("min_samples_split",4),
        max_features=params.get("max_features","sqrt"), n_jobs=-1, random_state=params.get("seed",0))
    rf.fit(_rf_feats(Xh,Th,mode,idc), ref[None,:]-Xh)
    Y = X + rf.predict(_rf_feats(X,T,mode,idc))
    w=params.get("smooth_win",5)
    if w>1: Y=np.vstack([moving_average(y,w) for y in Y])
    return Y, {"n_train":int(len(Xh))}

# ============================ AUTOENCODER (NEURAL) ============================
class ThermalAE(nn.Module):
    """AE neural: entrada = curva decimada + (T, |T-Tref|); saída = correção térmica
    em `n_anchors` âncoras, interpolada para a resolução plena.
    Treinado SÓ em curvas saudáveis: alvo = ref - curva."""
    def __init__(self, n_in, n_anchors, latent=12, hidden=128, dropout=0.10):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in+2, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.LayerNorm(hidden//2), nn.GELU(),
            nn.Linear(hidden//2, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden//2), nn.GELU(),
            nn.Linear(hidden//2, hidden), nn.GELU(),
            nn.Linear(hidden, n_anchors))
    def forward(self, x, t, dt):
        return self.dec(self.enc(torch.cat([x,t,dt],dim=1)))

def comp_ae(X, T, ref, train_healthy_mask, params, T_ref, seed=42):
    """AE neural treinado só no saudável de TREINO. Sem leakage."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_pts = X.shape[1]
    dec  = max(1, n_pts//params.get("n_input",1000))          # decimação da entrada
    n_anch = params.get("n_anchors",64)
    anchors = np.linspace(0, n_pts-1, n_anch).round().astype(int)

    Xh=X[train_healthy_mask]; Th=T[train_healthy_mask]
    if len(Xh)<6: raise ValueError("poucas saudáveis para treinar AE")
    D = ref[None,:]-Xh                                        # alvo: correção térmica
    # normalização ajustada SÓ no treino
    xin_tr = Xh[:,::dec]
    mu_x, sd_x = xin_tr.mean(0), xin_tr.std(0)+1e-8
    d_tr = D[:,anchors]; mu_d, sd_d = d_tr.mean(0), d_tr.std(0)+1e-8

    def prep(Xa, Ta):
        xi=(Xa[:,::dec]-mu_x)/sd_x
        t=((Ta-T_ref)/100.0).reshape(-1,1); dt=(np.abs(Ta-T_ref)/100.0).reshape(-1,1)
        return (torch.tensor(xi,dtype=torch.float32), torch.tensor(t,dtype=torch.float32),
                torch.tensor(dt,dtype=torch.float32))

    xt,tt,dtt = prep(Xh,Th)
    yt = torch.tensor((d_tr-mu_d)/sd_d, dtype=torch.float32)
    n=len(xt); idx=np.random.RandomState(seed).permutation(n)
    nv=max(2,int(0.2*n)); vi,ti_=idx[:nv],idx[nv:]

    model=ThermalAE(xt.shape[1], n_anch, params.get("latent",12),
                    params.get("hidden",128), params.get("dropout",0.10)).to(dev)
    opt=torch.optim.AdamW(model.parameters(), lr=params.get("lr",2e-3), weight_decay=params.get("wd",1e-4))
    huber=nn.SmoothL1Loss()
    best=(np.inf,None); bad=0
    Xtr,Ttr,DTtr,Ytr = xt[ti_].to(dev),tt[ti_].to(dev),dtt[ti_].to(dev),yt[ti_].to(dev)
    Xv,Tv,DTv,Yv     = xt[vi].to(dev),tt[vi].to(dev),dtt[vi].to(dev),yt[vi].to(dev)
    noise=params.get("noise",0.01)
    for ep in range(params.get("epochs",400)):
        model.train(); opt.zero_grad()
        xin = Xtr + noise*torch.randn_like(Xtr) if noise>0 else Xtr
        p=model(xin,Ttr,DTtr)
        loss=huber(p,Ytr)+params.get("lambda_d1",0.3)*huber(p[:,1:]-p[:,:-1], Ytr[:,1:]-Ytr[:,:-1])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(huber(model(Xv,Tv,DTv),Yv))
        if vl<best[0]-1e-7: best=(vl,{k:v.detach().clone() for k,v in model.state_dict().items()}); bad=0
        else:
            bad+=1
            if bad>=params.get("patience",60): break
    if best[1] is not None: model.load_state_dict(best[1])
    model.eval()
    xa,ta,dta=prep(X,T)
    with torch.no_grad():
        pa=model(xa.to(dev),ta.to(dev),dta.to(dev)).cpu().numpy()
    da=pa*sd_d+mu_d
    full=np.arange(n_pts)
    Y=np.vstack([X[i]+np.interp(full,anchors,da[i]) for i in range(len(X))])
    w=params.get("smooth_win",1)
    if w>1: Y=np.vstack([moving_average(y,w) for y in Y])
    n_par=sum(p.numel() for p in model.parameters())
    return Y, {"val_loss":best[0],"epochs":ep+1,"n_params":int(n_par),"device":dev}

# ============================ AUXILIAR: τ(T) + baixo posto ============================
def comp_tauT(X, T, ref, train_healthy_mask, f, params):
    """Ablation: shift SÓ por temperatura (nenhuma curva alinhada individualmente)."""
    Th=T[train_healthy_mask]; Xh=X[train_healthy_mask]
    temps=np.array(sorted(np.unique(Th)))
    taus=[]
    for tk in temps:
        med=np.median(Xh[np.isclose(Th,tk)],axis=0)
        bw=f[-1]-f[0]; tm=params.get("max_shift_frac",0.14)*bw
        grid=np.linspace(-tm,tm,params.get("nsteps",81)); best=(np.inf,0.0)
        for tau in grid:
            xs=shift_interp(med,f,tau); xs=xs-xs.mean()+ref.mean()
            e=np.mean((xs-ref)**2)
            if e<best[0]: best=(e,tau)
        taus.append(best[1])
    taus=np.array(taus)
    # ajuste robusto τ(T)
    msk=np.ones(len(temps),bool)
    for _ in range(3):
        c=np.polyfit(temps[msk],taus[msk],2); p=np.poly1d(c)
        r=taus-p(temps); md=np.median(np.abs(r-np.median(r)))+1e-9
        msk=np.abs(r-np.median(r))<=3*1.4826*md
        if msk.sum()<4: break
    p=np.poly1d(np.polyfit(temps[msk],taus[msk],2))
    A=np.vstack([shift_interp(X[i],f,float(p(T[i]))) for i in range(len(X))])
    # modelo térmico de baixo posto no resíduo saudável
    Ah=A[train_healthy_mask]; Rh=Ah-ref[None,:]
    tu=np.array(sorted(np.unique(Th)))
    M=np.vstack([np.median(Rh[np.isclose(Th,tk)],axis=0) for tk in tu])
    _,_,Vt=np.linalg.svd(M,full_matrices=False)
    r=min(params.get("rank",8),len(tu),Vt.shape[0]); Vr=Vt[:r]; Cf=M@Vr.T
    Y=np.zeros_like(A)
    for i in range(len(A)):
        mh=np.array([np.interp(T[i],tu,Cf[:,j]) for j in range(r)])@Vr
        y=A[i]-mh; y=y+np.median(ref-y)
        Y[i]=y
    return Y, {"n_outliers_tau":int((~msk).sum())}

# ============================ MONOTONICIDADE ============================
def monotonicity(vals_by_class):
    """vals_by_class: {0:v0,1:v1,2:v2}. Retorna healthy_sep e full_order."""
    if not {0,1,2} <= set(vals_by_class): return None, None
    v0,v1,v2 = vals_by_class[0],vals_by_class[1],vals_by_class[2]
    return bool(v0<min(v1,v2)), bool(v0<v1<v2)

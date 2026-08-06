# -*- coding: utf-8 -*-
"""
GRÁFICOS DE EXEMPLO ADICIONAIS (curvas), para ilustrar visualmente os resultados:
 (1) OVERLAY antes/depois: todas as curvas saudáveis (várias temperaturas) antes da compensação
     (espalhadas pela temperatura) e depois (colapsando na referência), por método — mostra
     visualmente a remoção da variabilidade térmica. LOTO (cada curva compensada por modelo
     treinado sem a sua temperatura).
 (2) Curvas D0/D1/D2 compensadas em bandas/temperaturas adicionais (fora da amostra).
Configs tunadas (bestcfg). Times New Roman.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"pdf.fonttype":42,
 "axes.spines.top":False,"axes.spines.right":False})
COR={"Original":"#7f7f7f","Park":"#2ca02c","RF_direct":"#d62728","AE":"#1f5fd0"}
LBL={"Park":"Park","RF_direct":"RF (direto)","AE":"Autoencoder"}
T_REF=30.0
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda,m,d):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
DEF={"Park":{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5},
     "RF_direct":{"n_estimators":300,"max_depth":10,"input_decim":4,"smooth_win":5},
     "AE":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":384,"lr":1e-3,"epochs":550,"patience":75}}
def save(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n,flush=True)
def clean(ax): ax.grid(alpha=.25,lw=.6)
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
t0=time.time()

# ============ (1) OVERLAY antes/depois (70-80) ============
banda="70-80"; lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF); fk=f/1e3
htemps=sorted([t for t in np.unique(T[y==0]) if (np.isclose(T,t)&(y==0)).sum()>0])
cmap=plt.cm.coolwarm; norm=plt.Normalize(min(htemps),max(htemps))
# compensa cada temperatura saudável em LOTO, guarda uma curva por temperatura
comp_by_t={"Original":{},"Park":{},"AE":{}}
for tk in htemps:
    te=np.isclose(T,tk); trh=(~te)&(y==0); idx0=np.where(te&(y==0))[0]
    if len(idx0)==0 or trh.sum()<8: continue
    k=idx0[0]; comp_by_t["Original"][tk]=X[k]
    try:
        comp_by_t["Park"][tk]=P.comp_park(X,ref,f,cfg(banda,"Park",DEF["Park"]))[0][k]
        comp_by_t["AE"][tk]=P.comp_ae(X,T,ref,trh,cfg(banda,"AE",DEF["AE"]),T_REF,seed=42)[0][k]
    except Exception as e: print("err",tk,e,flush=True)
    print(f"  overlay T={tk} ok | {(time.time()-t0)/60:.1f} min",flush=True)
fig,axes=plt.subplots(1,3,figsize=(16,4.8),dpi=170,sharey=True)
for ax,m,tit in zip(axes,["Original","Park","AE"],["(a) Original (sem compensação)","(b) Park","(c) Autoencoder"]):
    for tk,curve in comp_by_t[m].items():
        ax.plot(fk,curve,lw=.8,alpha=.7,color=cmap(norm(tk)))
    ax.plot(fk,ref,"k--",lw=1.8,label="Referência (30°C)",zorder=6)
    ax.set_title(tit); ax.set_xlabel("Frequência (kHz)"); clean(ax)
axes[0].set_ylabel("Componente real da impedância"); axes[0].legend(frameon=False,loc="upper right")
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
cb=fig.colorbar(sm,ax=axes,fraction=.02,pad=.01); cb.set_label("Temperatura (°C)")
fig.suptitle("Curvas saudáveis de todas as temperaturas — antes e depois da compensação (70–80 kHz)",y=1.02,fontsize=13)
save(fig,"figE_overlay_saudavel")

# ============ (2) curvas D0/D1/D2 extras ============
def fig_curvas(banda,T_show,fname):
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF); fk=f/1e3
    te=np.isclose(T,T_show); trh=(~te)&(y==0)
    comps={"Park":P.comp_park(X,ref,f,cfg(banda,"Park",DEF["Park"]))[0],
           "RF_direct":P.comp_rf(X,T,ref,trh,cfg(banda,"RF_direct",DEF["RF_direct"]),"direct")[0],
           "AE":P.comp_ae(X,T,ref,trh,cfg(banda,"AE",DEF["AE"]),T_REF,seed=42)[0]}
    NM={0:"D0 (saudável)",1:"D1 (massa)",2:"D2 (corte)"}
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),dpi=170,sharey=True)
    for ax,c in zip(axes,[0,1,2]):
        ii=np.where(te&(y==c))[0]
        if len(ii)==0: ax.set_title(f"{NM[c]} — n/d"); continue
        k=ii[0]
        ax.plot(fk,ref,"k--",lw=1.5,label="Ref. (saudável 30°C)",zorder=6)
        ax.plot(fk,X[k],color=COR["Original"],lw=1,alpha=.6,label=f"Original {T_show:.0f}°C")
        for m in ["Park","RF_direct","AE"]: ax.plot(fk,comps[m][k],lw=1.4,color=COR[m],alpha=.9,label=LBL[m])
        ax.set_title(NM[c]); ax.set_xlabel("Frequência (kHz)"); clean(ax)
    axes[0].set_ylabel("Componente real da impedância"); axes[0].legend(frameon=False,fontsize=8,loc="upper right")
    fig.suptitle(f"Curvas compensadas — {banda} kHz — T = {T_show:.0f}°C (fora da amostra)",y=1.0)
    fig.tight_layout(); save(fig,fname)
for banda,Ts in [("30-40",[20.0,-10.0]),("60-70",[70.0,0.0])]:
    for Ts_ in Ts:
        if any(np.isclose(T,Ts_)): fig_curvas(banda,Ts_,f"figE_curvas_{banda}_T{int(Ts_)}")
print(f"\n✅ curvas extras concluídas em {(time.time()-t0)/60:.1f} min",flush=True)

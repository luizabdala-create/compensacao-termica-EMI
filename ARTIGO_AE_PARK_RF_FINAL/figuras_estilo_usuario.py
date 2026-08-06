# -*- coding: utf-8 -*-
"""
Figuras no ESTILO do artigo CONEM do usuário: painéis lado a lado, curvas
Park|RF|AE com mesmo eixo, barras RMSD|CCDM por temperatura×dano. Estilo limpo, serif.
CORRIGIDO: compensa a temperatura específica de cada figura (antes só compensava folds,
deixando as curvas 48/78 °C sem cor). Mais exemplos: D0/D1/D2 e mais bandas/temperaturas.
"""
import os,sys,json,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; FIG=os.path.join(ROOT,"10_figuras_artigo")
sys.path.insert(0,ROOT); import pipeline as P
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":12,"axes.labelsize":13,"axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold",
 "xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":9.5,"pdf.fonttype":42,"ps.fonttype":42,
 "axes.spines.top":False,"axes.spines.right":False})
CM={"Original":"#e78a8a","Park":"#2ca02c","RF_direct":"#d62728","AE":"#1f5fd0"}
LB={"Original":"Original","Park":"Park","RF_direct":"Random Forest","AE":"Autoencoder"}
CDMG={0:"#1f77b4",1:"#ff7f0e",2:"#d62728"}
CD={"Park":["#a6d4a6","#4cae4c","#217821"],"RF_direct":["#f2a3a3","#e05252","#a11515"],
    "AE":["#a9c2f0","#4f7fdd","#123f8f"]}
NOME_DANO={0:"D0 (saudável)",1:"Dano 1 (massa)",2:"Dano 2 (corte)"}
def save(fig,n):
    for e,d in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=d,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n)
bestcfg={}
p=os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")
if os.path.exists(p): bestcfg=json.load(open(p))
from collections import Counter
def cfg(banda,m):
    d={"Park":{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5},
       "RF_direct":{"n_estimators":300,"max_depth":10,"min_samples_leaf":2,"input_decim":8,"smooth_win":5},
       "AE":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"epochs":450,"patience":55}}[m]
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int); T_REF=30.0
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]

_BAND={}
def band_data(banda):
    if banda in _BAND: return _BAND[banda]
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF); _BAND[banda]=(fc,f,X,ref); return _BAND[banda]

def comp_at_temp(banda,T_show,dano):
    """Compensa a curva (dano) em T_show por LOTO (treina em tudo menos T_show). Corrige o bug."""
    fc,f,X,ref=band_data(banda)
    te=np.isclose(T,T_show); trh=(~te)&(y==0)
    ii=np.where(np.isclose(T,T_show)&(y==dano))[0]
    if len(ii)==0: return None
    i=ii[0]
    out={"Original":X[i],"ref":ref,"fk":f/1e3}
    out["Park"]=P.comp_park(X,ref,f,cfg(banda,"Park"))[0][i]
    out["RF_direct"]=P.comp_rf(X,T,ref,trh,cfg(banda,"RF_direct"),"direct")[0][i]
    out["AE"]=P.comp_ae(X,T,ref,trh,cfg(banda,"AE"),T_REF,seed=42)[0][i]
    return out

# ============ curvas D0: Park|RF|AE lado a lado (estilo Fig 4/5) ============
def fig_curvas_d0(banda,T_show,fname):
    d=comp_at_temp(banda,T_show,0)
    if d is None: print("sem D0 em",T_show); return
    fk=d["fk"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),dpi=170,sharey=True)
    for ax,m in zip(axes,["Park","RF_direct","AE"]):
        ax.plot(fk,d["ref"],"--",color="black",lw=1.2,label=f"Referência {int(T_REF)}°C",zorder=2)
        ax.plot(fk,d["Original"],color=CM["Original"],lw=1.2,alpha=.85,label=f"Original {int(T_show)}°C",zorder=1)
        ax.plot(fk,d[m],color=CM[m],lw=1.7,label=LB[m],zorder=3)
        ax.set_xlabel("Frequência (kHz)"); ax.set_title(f"({'abc'[['Park','RF_direct','AE'].index(m)]}) {LB[m]}")
        ax.legend(loc="upper right",frameon=False,fontsize=9)
    axes[0].set_ylabel("Componente real da impedância")
    save(fig,fname)

# ============ curvas por DANO: mostra D0/D1/D2 para um método (preservação) ============
def fig_curvas_dano(banda,T_show,metodo,fname):
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),dpi=170,sharey=True)
    for ax,dano in zip(axes,[0,1,2]):
        d=comp_at_temp(banda,T_show,dano)
        if d is None: ax.set_title(f"{NOME_DANO[dano]} — s/ amostra"); continue
        fk=d["fk"]
        ax.plot(fk,d["ref"],"--",color="black",lw=1.2,label=f"Referência saudável {int(T_REF)}°C",zorder=2)
        ax.plot(fk,d["Original"],color=CM["Original"],lw=1.1,alpha=.8,label=f"Original {int(T_show)}°C",zorder=1)
        ax.plot(fk,d[metodo],color=CDMG[dano],lw=1.7,label=f"{LB[metodo]} compensado",zorder=3)
        ax.set_xlabel("Frequência (kHz)"); ax.set_title(f"({'abc'[dano]}) {NOME_DANO[dano]}")
        ax.legend(loc="upper right",frameon=False,fontsize=8.5)
    axes[0].set_ylabel("Componente real da impedância")
    fig.suptitle(f"Preservação da assinatura de dano — {LB[metodo]} — {banda} kHz, T={int(T_show)}°C",y=1.02,fontsize=12)
    save(fig,fname)

# ============ métricas por temperatura×dano (estilo Fig 6/7) ============
def loto_all(banda):
    fc,f,X,ref=band_data(banda)
    comp={m:np.full_like(X,np.nan) for m in ["Park","RF_direct","AE"]}
    for T_test in FOLDS:
        te=np.isclose(T,T_test); trh=(~te)&(y==0); idx=np.where(te)[0]
        comp["Park"][idx]=P.comp_park(X,ref,f,cfg(banda,"Park"))[0][idx]
        comp["RF_direct"][idx]=P.comp_rf(X,T,ref,trh,cfg(banda,"RF_direct"),"direct")[0][idx]
        comp["AE"][idx]=P.comp_ae(X,T,ref,trh,cfg(banda,"AE"),T_REF,seed=42)[0][idx]
    return fc,f,X,ref,comp
def fig_metrica_dano(banda,fname,titulo):
    fc,f,X,ref,comp=loto_all(banda); temps=FOLDS; methods=["Park","RF_direct","AE"]
    fig,axes=plt.subplots(1,2,figsize=(15,5),dpi=170)
    for ax,metrica in zip(axes,["RMSD","CCDM"]):
        ngrp=len(methods)*3; w=0.9/ngrp; x=np.arange(len(temps)); gi=0
        for m in methods:
            for d in [0,1,2]:
                vals=[]
                for t in temps:
                    ii=np.where(np.isclose(T,t)&(y==d))[0]; fn=P.rmsd if metrica=="RMSD" else P.ccdm
                    vals.append(np.mean([fn(comp[m][i],ref) for i in ii]) if len(ii) else np.nan)
                ax.bar(x+(gi-(ngrp-1)/2)*w,vals,w,color=CD[m][d],edgecolor="k",lw=.3); gi+=1
        ax.set_xticks(x); ax.set_xticklabels([f"{int(t)}" for t in temps]); ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel(metrica); ax.set_title(f"({'a' if metrica=='RMSD' else 'b'}) {metrica}")
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=CD[m][d],edgecolor="k",label=f"{LB[m]} — {NOME_DANO[d]}") for m in methods for d in [0,1,2]]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.14))
    fig.suptitle(titulo,y=1.0,fontsize=12); fig.tight_layout(rect=[0,0.03,1,0.98]); save(fig,fname)

# ============ global D0 barras (estilo Fig 3) ============
def fig_global_d0(banda,fname):
    fc,f,X,ref,comp=loto_all(banda); temps=FOLDS; methods=["Park","RF_direct","AE"]
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),dpi=170)
    for ax,metrica in zip(axes,["RMSD","CCDM"]):
        w=0.8/len(methods); x=np.arange(len(temps))
        for gi,m in enumerate(methods):
            vals=[]
            for t in temps:
                ii=np.where(np.isclose(T,t)&(y==0))[0]; fn=P.rmsd if metrica=="RMSD" else P.ccdm
                vals.append(np.mean([fn(comp[m][i],ref) for i in ii]) if len(ii) else np.nan)
            ax.bar(x+(gi-(len(methods)-1)/2)*w,vals,w,color=CM[m],edgecolor="k",lw=.4,label=LB[m])
        ax.set_xticks(x); ax.set_xticklabels([f"{int(t)}" for t in temps]); ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel(metrica); ax.set_title(f"({'a' if metrica=='RMSD' else 'b'}) {metrica} — saudável")
    axes[1].legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.16))
    fig.tight_layout(); save(fig,fname)

if __name__=="__main__":
    # D0 — 40-50 (banda do artigo do usuário) em 48 e 78 °C
    fig_curvas_d0("40-50",48.0,"figE_curvas_40-50_T48")
    fig_curvas_d0("40-50",78.0,"figE_curvas_40-50_T78")
    # D0 — 70-80 (banda onde o AE domina)
    fig_curvas_d0("70-80",60.0,"figE_curvas_70-80_T60")
    fig_curvas_d0("70-80",-10.0,"figE_curvas_70-80_Tm10")
    # preservação de dano D0/D1/D2 (uma temperatura com as 3 classes)
    fig_curvas_dano("40-50",70.0,"AE","figE_dano_AE_40-50_T70")
    fig_curvas_dano("40-50",70.0,"Park","figE_dano_Park_40-50_T70")
    fig_curvas_dano("70-80",50.0,"AE","figE_dano_AE_70-80_T50")
    # métricas por temperatura×dano
    fig_metrica_dano("40-50","figE_rmsd_ccdm_dano_40-50","Métricas por temperatura e estado estrutural — 40–50 kHz")
    fig_metrica_dano("30-70","figE_rmsd_ccdm_dano_30-70","Métricas por temperatura e estado estrutural — 30–70 kHz (banda larga)")
    fig_global_d0("40-50","figE_global_d0_40-50")
    print("\n✅ figuras estilo usuário (corrigidas + ampliadas)")

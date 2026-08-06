# -*- coding: utf-8 -*-
"""
INTERPOLAÇÃO NOS VÃOS (gap-fill) — treinar só com temperaturas EXTREMAS + CENTRAIS e
testar nas do MEIO (que ficaram nos vãos). Diferente da extrapolação (teste FORA da
faixa, onde o Park estrutura vence), aqui o teste está DENTRO do casco convexo do treino
=> é interpolação legítima, o cenário onde o ML pode ganhar do Park.

Pergunta do usuário: treinando nos extremos+centrais, os métodos acertam as do meio?
O Park perde em alguma banda/temperatura?

Métodos: Original, Park, Random Forest (base), Random Forest otimizado (ExtraTrees), Autoencoder.
Só saudável no treino; referência = mediana saudável @30 (30 sempre no treino). RMSD saudável
nas temperaturas de teste (vãos). Honesto: reporta onde cada um vence, sem maquiar.
Saída: 08_analises_avancadas/interpolacao_gaps.csv + figura figR_interp_gaps.
"""
import os, sys, json, time, numpy as np, pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas")
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13,"axes.titleweight":"bold","figure.titlesize":14,"figure.titleweight":"bold","legend.fontsize":9,
 "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
T_REF=30.0; BANDS=["30-40","60-70","70-80","30-70"]
CM={"Original":"#7f7f7f","Park":"#2ca02c","RF":"#d62728","RFotim":"#ff7f0e","AE":"#1f5fd0"}
LBLD={"Original":"Original","Park":"Park","RF":"Random Forest","RFotim":"RF otimizado","AE":"Autoencoder"}
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda,m,d):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
AED={"n_input":2000,"n_anchors":128,"latent":8,"hidden":384,"lr":1e-3,"epochs":550,"patience":75}
RFD={"n_estimators":300,"max_depth":10,"min_samples_leaf":1,"max_features":"sqrt","smooth_win":5,"input_decim":4}
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
HTEMPS=[t for t in sorted(np.unique(T)) if ((np.isclose(T,t))&(y==0)).sum()>0]  # temps com saudável
print("temperaturas saudáveis:",HTEMPS,flush=True)

def comp_et(X,ref,trh,params):
    idc=params.get("input_decim",4); Xh=X[trh]; Th=T[trh]
    et=ExtraTreesRegressor(n_estimators=params.get("n_estimators",300),max_depth=params.get("max_depth",10),
        min_samples_leaf=params.get("min_samples_leaf",1),max_features=params.get("max_features","sqrt"),n_jobs=-1,random_state=0)
    et.fit(P._rf_feats(Xh,Th,"direct",idc),ref[None,:]-Xh)
    Y=X+et.predict(P._rf_feats(X,T,"direct",idc)); w=params.get("smooth_win",5)
    if w>1: Y=np.vstack([P.moving_average(yy,w) for yy in Y])
    return Y
def band(b):
    lo,hi=map(int,b.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    return df[fc].to_numpy(np.float64),f,P.build_reference(df,fc,T_REF)[0]

def nearest(temps):  # casa alvos aos temps saudáveis disponíveis
    return [min(HTEMPS,key=lambda h:abs(h-t)) for t in temps]
# esquemas de treino: extremos + centrais; teste = vãos do meio (30 sempre no treino p/ referência)
lo,hi=min(HTEMPS),max(HTEMPS)
SCHEMES={
 "3 pontos (extremos+centro)": sorted(set(nearest([lo,T_REF,hi]))),
 "5 pontos (extremos+centrais)": sorted(set(nearest([lo,lo+ (T_REF-lo)/2, T_REF, T_REF+(hi-T_REF)/2, hi]))),
}
for k,v in SCHEMES.items(): print(f"esquema {k}: treino={v}",flush=True)

t0=time.time(); rows=[]; perT=[]
for b in BANDS:
    X,f,ref=band(b)
    for sname,tr_temps in SCHEMES.items():
        trh=np.isin(np.round(T,3),tr_temps)&(y==0)
        te_temps=[t for t in HTEMPS if t not in tr_temps]
        idx=np.where(np.isin(np.round(T,3),te_temps)&(y==0))[0]
        if trh.sum()<6 or len(idx)==0:
            print(f"  pulado {b}|{sname}: trh={trh.sum()} idx={len(idx)}",flush=True); continue
        comps={"Original":X.copy(),
               "Park":P.comp_park(X,ref,f,cfg(b,"Park",{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5}))[0],
               "RF":P.comp_rf(X,T,ref,trh,cfg(b,"RF_direct",RFD),"direct")[0],
               "RFotim":comp_et(X,ref,trh,cfg(b,"RF_direct",RFD)),
               "AE":P.comp_ae(X,T,ref,trh,cfg(b,"AE",AED),T_REF,seed=42)[0]}
        r={"banda":b,"esquema":sname,"n_treino":int(trh.sum()),"n_teste":int(len(idx))}
        for m,Y in comps.items():
            r[m]=float(np.mean([P.rmsd(Y[i],ref) for i in idx]))
            # RMSD por temperatura de teste (para ver onde cada um perde)
            for tt in te_temps:
                ii=np.where(np.isclose(T,tt)&(y==0))[0]
                if len(ii): perT.append({"banda":b,"esquema":sname,"metodo":m,"temp":tt,
                                          "RMSD":float(np.mean([P.rmsd(Y[i],ref) for i in ii]))})
        rows.append(r)
        best_ml=min(r["RF"],r["RFotim"],r["AE"]); park_perde = best_ml < r["Park"]
        print(f"  {b} | {sname}: Park={r['Park']:.2f} RF={r['RF']:.2f} RFotim={r['RFotim']:.2f} AE={r['AE']:.2f} "
              f"Orig={r['Original']:.2f} | Park {'PERDE' if park_perde else 'vence'} p/ ML | {(time.time()-t0)/60:.1f}min",flush=True)

d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"interpolacao_gaps.csv"),index=False)
dpt=pd.DataFrame(perT); dpt.to_csv(os.path.join(OUT,"interpolacao_gaps_por_temp.csv"),index=False)

# ---- figura 1: barras por banda (esquema mais esparso) + por-temperatura ----
sch0=list(SCHEMES)[0]
fig,axes=plt.subplots(1,2,figsize=(15,4.8),dpi=170)
# (a) barras RMSD por banda, esquema esparso
s0=d[d.esquema==sch0]; mets=["Park","RF","RFotim","AE"]; x=np.arange(len(s0)); w=.2
for i,m in enumerate(mets):
    axes[0].bar(x+(i-1.5)*w,s0[m],w,color=CM[m],edgecolor="k",lw=.4,label=LBLD[m])
axes[0].set_xticks(x); axes[0].set_xticklabels(s0.banda); axes[0].set_ylabel("RMSD saudável (temps de teste = vãos)")
axes[0].set_title(f"(a) Interpolação nos vãos — {sch0}"); axes[0].legend(frameon=False,ncol=2,fontsize=8.5)
# (b) RMSD por temperatura (banda 30-70 = a mais difícil), esquema esparso
bb="30-70" if "30-70" in dpt.banda.unique() else BANDS[0]
sp=dpt[(dpt.esquema==sch0)&(dpt.banda==bb)]
for m in mets:
    ss=sp[sp.metodo==m].sort_values("temp")
    axes[1].plot(ss.temp,ss.RMSD,marker="o",color=CM[m],lw=1.8,label=LBLD[m])
for tt in SCHEMES[sch0]: axes[1].axvline(tt,ls=":",color="k",alpha=.25)
axes[1].set_xlabel("Temperatura de teste (°C)"); axes[1].set_ylabel("RMSD saudável")
axes[1].set_title(f"(b) Por temperatura — banda {bb} (linhas verticais = temps de treino)")
axes[1].legend(frameon=False,ncol=2,fontsize=8.5)
fig.suptitle("Interpolação nos vãos: treinar em extremos+centrais, testar nas temperaturas do meio",y=1.02,fontsize=13.5)
fig.tight_layout()
for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"figR_interp_gaps.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
plt.close(fig); print("fig: figR_interp_gaps",flush=True)

# ---- resumo honesto ----
print("\n"+"="*70+"\nRESUMO — PARK PERDE PARA O ML NA INTERPOLAÇÃO DOS VÃOS?\n"+"="*70,flush=True)
for _,r in d.iterrows():
    best_ml=min(r["RF"],r["RFotim"],r["AE"]); who=min(["RF","RFotim","AE"],key=lambda m:r[m])
    marg=(r["Park"]-best_ml)/r["Park"]*100
    print(f"  {r.banda:6s} | {r.esquema:28s}: Park={r['Park']:.2f} vs melhor ML={best_ml:.2f} ({LBLD[who]}) "
          f"-> Park {'PERDE' if best_ml<r['Park'] else 'vence'} ({marg:+.0f}%)",flush=True)
nperde=sum(min(r["RF"],r["RFotim"],r["AE"])<r["Park"] for _,r in d.iterrows())
print(f"\nPark perde para o ML em {nperde}/{len(d)} casos (banda×esquema). "
      f"Média RMSD: Park={d['Park'].mean():.2f}, RF={d['RF'].mean():.2f}, RFotim={d['RFotim'].mean():.2f}, AE={d['AE'].mean():.2f}",flush=True)
print(f"\nOK em {(time.time()-t0)/60:.1f} min",flush=True)
print("DONE_INTERP_GAPS",flush=True)

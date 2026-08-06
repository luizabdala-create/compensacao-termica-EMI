# -*- coding: utf-8 -*-
"""
ANÁLISES AVANÇADAS (novas) — reforço de rigor, sem vazamento:
 (1) DETECÇÃO por índice de dano DI = RMSD(curva compensada, referência saudável): ROC/AUC
     saudável-vs-dano por método e banda. Responde diretamente: a compensação MELHORA a
     detectabilidade do dano? (inclui Original como linha de base). Threshold-free (AUC).
 (2) SEPARABILIDADE de Fisher (FDR) entre D0 e dano, por banda×método (distribuição, não só média).
 (3) INTERVALOS DE CONFIANÇA (bootstrap 95%) do RMSD saudável por método.
 (4) TAMANHO DE EFEITO (delta de Cliff) pareado entre métodos — magnitude, não só p-valor.
 (5) SELETOR ADAPTATIVO por banda (escolhido pela CV interna, sem tocar no teste) vs métodos fixos
     e vs oráculo (teto). Contribuição prática: "melhor de cada banda".
Tudo LOTO, referência congelada (saudável mediana @30°C), configs tunadas (bestcfg).
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try: from sklearn.metrics import roc_auc_score,roc_curve; HAVE_SK=True
except Exception: HAVE_SK=False
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas"); os.makedirs(OUT,exist_ok=True)
CKPC=os.path.join(ROOT,"checkpoints","adv_percurva.csv")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9.5,"pdf.fonttype":42,
 "axes.spines.top":False,"axes.spines.right":False})
CM={"Original":"#7f7f7f","Park":"#2ca02c","RF_direct":"#d62728","AE":"#1f5fd0"}
LB={"Original":"Original","Park":"Park","RF_direct":"Random Forest","AE":"Autoencoder"}
T_REF=30.0; BANDS_PC=["30-40","40-50","60-70","70-80","30-70"]
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda,m,d):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else d
DEF={"Park":{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5},
     "RF_direct":{"n_estimators":300,"max_depth":10,"input_decim":8,"smooth_win":5},
     "AE":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"epochs":450,"patience":55}}
def savef(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n,flush=True)

# ============ PARTE A: DI por curva (recompute) ============
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
if os.path.exists(CKPC):
    pc=pd.read_csv(CKPC); print("DI por-curva do checkpoint:",pc.shape,flush=True)
else:
    rows=[]; t0=time.time()
    for banda in BANDS_PC:
        lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
        X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,T_REF)
        print(f"\n=== {banda} ({len(fc)} pts) ===",flush=True)
        for T_test in FOLDS:
            if np.isclose(T_test,T_REF): continue
            te=np.isclose(T,T_test); trh=(~te)&(y==0); idx=np.where(te)[0]
            if trh.sum()<8: continue
            comps={"Original":X}
            try:
                comps["Park"]=P.comp_park(X,ref,f,cfg(banda,"Park",DEF["Park"]))[0]
                comps["RF_direct"]=P.comp_rf(X,T,ref,trh,cfg(banda,"RF_direct",DEF["RF_direct"]),"direct")[0]
                comps["AE"]=P.comp_ae(X,T,ref,trh,cfg(banda,"AE",DEF["AE"]),T_REF,seed=42)[0]
            except Exception as e: print("  err",banda,T_test,e,flush=True); continue
            for m,Y in comps.items():
                for i in idx:
                    mm=P.all_metrics(Y[i],ref)
                    rows.append({"banda":banda,"T_test":T_test,"metodo":m,"dano":int(y[i]),
                                 "DI_rmsd":mm["RMSD"],"DI_ccdm":mm["CCDM"]})
        print(f"  {banda} ok | {(time.time()-t0)/60:.1f} min | {len(rows)} regs",flush=True)
        pd.DataFrame(rows).to_csv(CKPC,index=False)
    pc=pd.DataFrame(rows); pc.to_csv(CKPC,index=False)
    print(f"\nDI por-curva: {pc.shape} | {(time.time()-t0)/60:.1f} min",flush=True)
pc.to_csv(os.path.join(OUT,"di_por_curva.csv"),index=False)
MET=["Original","Park","RF_direct","AE"]

# ---- (1) ROC/AUC detecção saudável(0) vs dano(1) ----
def auc(sub):
    yv=(sub.dano>0).astype(int).values; s=sub.DI_rmsd.values
    if yv.sum()==0 or yv.sum()==len(yv) or not HAVE_SK: return np.nan
    return roc_auc_score(yv,s)
aucs={m:{} for m in MET}
for m in MET:
    for b in BANDS_PC: aucs[m][b]=auc(pc[(pc.metodo==m)&(pc.banda==b)])
    aucs[m]["global"]=auc(pc[pc.metodo==m])
aucd=pd.DataFrame(aucs).T; aucd.to_csv(os.path.join(OUT,"auc_deteccao.csv"))
print("\n=== AUC de detecção (saudável vs dano), por banda ===\n",aucd.round(3).to_string(),flush=True)
# fig ROC pooled + AUC heatmap
fig,axes=plt.subplots(1,2,figsize=(14,5),dpi=170)
for m in MET:
    s=pc[pc.metodo==m]; yv=(s.dano>0).astype(int).values
    if HAVE_SK and 0<yv.sum()<len(yv):
        fpr,tpr,_=roc_curve(yv,s.DI_rmsd.values); a=roc_auc_score(yv,s.DI_rmsd.values)
        axes[0].plot(fpr,tpr,lw=2.2,color=CM[m],label=f"{LB[m]} (AUC={a:.3f})")
axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=.6); axes[0].set_xlabel("Taxa de falso-positivo (1−especificidade)")
axes[0].set_ylabel("Taxa de verdadeiro-positivo (sensibilidade)"); axes[0].set_title("(a) ROC — detecção de dano pelo índice DI (todas as bandas)")
axes[0].legend(frameon=False,loc="lower right")
bb=[b for b in BANDS_PC]; im=axes[1].imshow(aucd.loc[MET,bb].values.astype(float),cmap="viridis",vmin=0.5,vmax=1.0,aspect="auto")
axes[1].set_xticks(range(len(bb))); axes[1].set_xticklabels(bb,rotation=25,ha="right")
axes[1].set_yticks(range(len(MET))); axes[1].set_yticklabels([LB[m] for m in MET])
for i in range(len(MET)):
    for j in range(len(bb)):
        v=aucd.loc[MET[i],bb[j]]
        if v==v: axes[1].text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8.5,color="w" if v<0.8 else "k")
axes[1].set_title("(b) AUC de detecção por banda × método"); fig.colorbar(im,ax=axes[1],fraction=.046,pad=.04,label="AUC")
fig.tight_layout(); savef(fig,"figA_roc_deteccao")

# ---- (2) Fisher Discriminant Ratio (FDR) D0 vs dano ----
def fdr(sub):
    h=sub[sub.dano==0].DI_rmsd.values; d=sub[sub.dano>0].DI_rmsd.values
    if len(h)<2 or len(d)<2: return np.nan
    return (d.mean()-h.mean())**2/(h.var()+d.var()+1e-9)
fdrd=pd.DataFrame({m:{b:fdr(pc[(pc.metodo==m)&(pc.banda==b)]) for b in BANDS_PC} for m in MET}).T
fdrd.to_csv(os.path.join(OUT,"fisher_separabilidade.csv"))
fig,ax=plt.subplots(figsize=(10,4.6),dpi=170); x=np.arange(len(BANDS_PC)); w=.2
for i,m in enumerate(MET):
    ax.bar(x+(i-1.5)*w,[fdrd.loc[m,b] for b in BANDS_PC],w,color=CM[m],edgecolor="k",lw=.5,label=LB[m])
ax.set_xticks(x); ax.set_xticklabels(BANDS_PC); ax.set_ylabel("Razão de Fisher (D0 vs dano)")
ax.set_title("Separabilidade saudável–dano do índice DI (maior = dano mais destacável)",pad=8)
ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.13))
fig.tight_layout(); savef(fig,"figA_fisher_separabilidade")

# ---- separabilidade: violino DI por classe (banda 70-80) ----
bshow="70-80"; fig,axes=plt.subplots(1,4,figsize=(15,4.2),dpi=170,sharey=True)
for ax,m in zip(axes,MET):
    data=[pc[(pc.metodo==m)&(pc.banda==bshow)&(pc.dano==c)].DI_rmsd.values for c in [0,1,2]]
    parts=ax.violinplot(data,showmeans=True,showextrema=False)
    for pcv,col in zip(parts['bodies'],["#1f77b4","#ff7f0e","#d62728"]): pcv.set_facecolor(col); pcv.set_alpha(.6)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(["D0","D1","D2"]); ax.set_title(LB[m])
axes[0].set_ylabel("Índice de dano DI = RMSD à referência")
fig.suptitle(f"Distribuição do índice de dano por classe — {bshow} kHz (D0 deve ficar abaixo de D1/D2)",y=1.02)
fig.tight_layout(); savef(fig,"figA_violino_DI")

# ============ PARTE B: fase8 (barato) ============
f8=pd.read_csv(os.path.join(ROOT,"02_compensacao","fase8_tuning_ampliado.csv")); f8=f8[f8.metodo!="TAU_T"]
MJ=["AE","Park","RF_direct"]; allb=sorted(f8.banda.unique())
# ---- (3) bootstrap IC95 do RMSD_D0 (pooled) ----
rng=np.random.RandomState(0)
def boot(v,B=2000):
    v=np.asarray(v); m=[np.mean(rng.choice(v,len(v),replace=True)) for _ in range(B)]
    return float(np.mean(v)),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))
ci={m:boot(f8[f8.metodo==m].RMSD_D0.values) for m in MJ+["RF_temponly"]}
cid=pd.DataFrame(ci,index=["media","ic_lo","ic_hi"]).T; cid.to_csv(os.path.join(OUT,"bootstrap_ic_rmsd.csv"))
print("\n=== IC95 bootstrap RMSD saudável (todas as condições) ===\n",cid.round(3).to_string(),flush=True)
fig,ax=plt.subplots(figsize=(8,3.4),dpi=170); order=cid.sort_values("media").index.tolist()
for i,m in enumerate(order):
    r=cid.loc[m]; ax.plot([r.ic_lo,r.ic_hi],[i,i],"-",lw=2.5,color=CM.get(m,"#555"))
    ax.plot(r.media,i,"o",ms=8,color=CM.get(m,"#555"))
    ax.text(r.ic_hi+.03,i,f"{r.media:.2f} [{r.ic_lo:.2f}, {r.ic_hi:.2f}]",va="center",fontsize=9)
ax.set_yticks(range(len(order))); ax.set_yticklabels([LB.get(m,m) for m in order])
ax.set_xlabel("RMSD nas curvas saudáveis (média + IC95% bootstrap)"); ax.set_title("Intervalos de confiança do RMSD saudável (todas as condições agrupadas)")
ax.margins(x=.18); fig.tight_layout(); savef(fig,"figA_bootstrap_ic")

# ---- (4) delta de Cliff pareado ----
def cliff(a,b):
    a=np.asarray(a); b=np.asarray(b); gt=sum((a[:,None]>b[None,:]).sum(1)); lt=sum((a[:,None]<b[None,:]).sum(1))
    return (gt-lt)/(len(a)*len(b))
def mag(d):
    ad=abs(d); return "desprezível" if ad<0.147 else "pequeno" if ad<0.33 else "médio" if ad<0.474 else "grande"
pairs=[("AE","Park"),("RF_direct","Park"),("AE","RF_direct")]; er=[]
for a,b in pairs:
    d=cliff(f8[f8.metodo==a].RMSD_D0.values,f8[f8.metodo==b].RMSD_D0.values)
    er.append({"comparacao":f"{LB[a]} vs {LB[b]}","cliff_delta":round(d,3),"magnitude":mag(d),
               "interpretacao":f"{LB[a]} tem RMSD menor" if d<0 else f"{LB[b]} tem RMSD menor"})
erd=pd.DataFrame(er); erd.to_csv(os.path.join(OUT,"cliff_delta.csv"),index=False)
print("\n=== Tamanho de efeito (delta de Cliff) no RMSD saudável ===\n",erd.to_string(index=False),flush=True)

# ---- (5) seletor adaptativo por banda ----
piv=f8.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean")
cvp=f8.pivot_table(index="banda",columns="metodo",values="cv_inner",aggfunc="mean")  # base de treino p/ escolher
sel_rows=[]
for b in piv.index:
    if not all(m in piv.columns and pd.notna(piv.loc[b,m]) for m in MJ): continue
    # escolha PELA CV INTERNA (treino) — sem tocar no teste; pula bandas sem CV interna disponível
    cand={m:cvp.loc[b,m] for m in MJ if (b in cvp.index and m in cvp.columns and pd.notna(cvp.loc[b,m]))}
    if not cand: continue
    escolhido=min(cand,key=cand.get)
    sel_rows.append({"banda":b,"escolhido_por_cv":escolhido,"rmsd_seletor":piv.loc[b,escolhido],
                     "rmsd_AE":piv.loc[b,"AE"],"rmsd_Park":piv.loc[b,"Park"],"rmsd_RF":piv.loc[b,"RF_direct"],
                     "rmsd_oraculo":min(piv.loc[b,m] for m in MJ)})
seld=pd.DataFrame(sel_rows); seld.to_csv(os.path.join(OUT,"seletor_por_banda.csv"),index=False)
resumo={"seletor(CV)":seld.rmsd_seletor.mean(),"sempre AE":seld.rmsd_AE.mean(),"sempre Park":seld.rmsd_Park.mean(),
        "sempre RF":seld.rmsd_RF.mean(),"oráculo(teto)":seld.rmsd_oraculo.mean()}
print("\n=== SELETOR ADAPTATIVO por banda (RMSD saudável médio) ===")
for k,v in sorted(resumo.items(),key=lambda x:x[1]): print(f"  {k:16s}: {v:.3f}",flush=True)
json.dump(resumo,open(os.path.join(OUT,"seletor_resumo.json"),"w"),indent=2)
fig,ax=plt.subplots(figsize=(8,3.6),dpi=170); it=sorted(resumo.items(),key=lambda x:x[1])
names=[k for k,_ in it]; vals=[v for _,v in it]
cols=["#6a3d9a" if "eletor" in n else "#2ca02c" if "Park" in n else "#d62728" if "RF" in n else "#1f5fd0" if "AE" in n else "#444" for n in names]
ax.barh(range(len(names)),vals,color=cols,edgecolor="k",lw=.5)
for i,v in enumerate(vals): ax.text(v+.01,i,f"{v:.2f}",va="center",fontsize=9)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlabel("RMSD saudável médio (todas as bandas)"); ax.set_title("Seletor adaptativo por banda vs. método fixo (menor = melhor)"); ax.margins(x=.12)
fig.tight_layout(); savef(fig,"figA_seletor")
print("\n✅ análises avançadas concluídas",flush=True)

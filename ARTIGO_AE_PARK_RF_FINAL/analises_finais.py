# -*- coding: utf-8 -*-
"""
ANÁLISES FINAIS DO PARECER (rápidas; usam dados existentes + Park/RF):
 (A) MÉTRICAS DE SEGURANÇA com limiar definido na CV interna: PR-AUC, sensibilidade, especificidade,
     falso-saudável, falso-dano, em um limiar do índice DI escolhido no treino (Youden J).
 (B) ESTATÍSTICAS DO SELETOR: frequência de escolha, regret vs oráculo, margem p/ 2º melhor, estabilidade.
 (C) ROBUSTEZ À REFERÊNCIA: nº de curvas (1/3/5/todas), média vs mediana, referência com ruído.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"08_analises_avancadas")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
 "axes.titlesize":13.5,"axes.titleweight":"bold","figure.titlesize":14.5,"figure.titleweight":"bold","legend.fontsize":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
CM={"Original":"#7f7f7f","Park":"#2ca02c","RF_direct":"#d62728","AE":"#1f5fd0"}
LB={"Original":"Original","Park":"Park","RF_direct":"Random Forest","AE":"Autoencoder"}
MET=["Original","Park","RF_direct","AE"]; T_REF=30.0
def save(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n,flush=True)
t0=time.time()

# ================= (A) MÉTRICAS DE SEGURANÇA (limiar na CV interna) =================
pc=pd.read_csv(os.path.join(OUT,"di_por_curva.csv"))
rows=[]
for m in MET:
    s=pc[pc.metodo==m]; temps=sorted(s.T_test.unique())
    prauc=average_precision_score((s.dano>0).astype(int),s.DI_rmsd) if s.dano.nunique()>1 else np.nan
    # limiar por LOTO: escolhe limiar (Youden) nas outras temperaturas, aplica na de teste
    sens=[]; spec=[]; fh=[]; fd=[]
    for tt in temps:
        tr=s[s.T_test!=tt]; te=s[s.T_test==tt]
        if tr.dano.nunique()<2 or len(te)==0: continue
        # varre limiares no treino, escolhe o de maior Youden J = sens+spec-1
        thr=np.quantile(tr.DI_rmsd,np.linspace(.05,.95,40)); bestj=(-9,thr[0])
        ytr=(tr.dano>0).astype(int).values; dtr=tr.DI_rmsd.values
        for th in thr:
            pred=(dtr>=th).astype(int)
            tp=((pred==1)&(ytr==1)).sum(); tn=((pred==0)&(ytr==0)).sum()
            se=tp/max((ytr==1).sum(),1); sp=tn/max((ytr==0).sum(),1)
            if se+sp-1>bestj[0]: bestj=(se+sp-1,th)
        th=bestj[1]; yte=(te.dano>0).astype(int).values; dte=te.DI_rmsd.values; pred=(dte>=th).astype(int)
        tp=((pred==1)&(yte==1)).sum(); tn=((pred==0)&(yte==0)).sum(); fp=((pred==1)&(yte==0)).sum(); fn=((pred==0)&(yte==1)).sum()
        if (yte==1).sum(): sens.append(tp/(yte==1).sum()); fh.append(fn/(yte==1).sum())
        if (yte==0).sum(): spec.append(tn/(yte==0).sum()); fd.append(fp/(yte==0).sum())
    rows.append({"metodo":m,"PR_AUC":round(prauc,3),"sensibilidade":round(np.mean(sens),3) if sens else np.nan,
                 "especificidade":round(np.mean(spec),3) if spec else np.nan,
                 "falso_saudavel":round(np.mean(fh),3) if fh else np.nan,"falso_dano":round(np.mean(fd),3) if fd else np.nan})
seg=pd.DataFrame(rows); seg.to_csv(os.path.join(OUT,"metricas_seguranca.csv"),index=False)
print("=== SEGURANÇA (limiar na CV interna) ===\n",seg.to_string(index=False),flush=True)
fig,ax=plt.subplots(figsize=(9,4.6),dpi=170); x=np.arange(len(seg)); w=.18
for i,(col,lab) in enumerate([("PR_AUC","PR-AUC ↑"),("sensibilidade","Sensibilidade ↑"),("especificidade","Especificidade ↑"),("falso_saudavel","Falso-saudável ↓")]):
    ax.bar(x+(i-1.5)*w,seg[col],w,edgecolor="k",lw=.4,label=lab)
ax.set_xticks(x); ax.set_xticklabels([LB[m] for m in seg.metodo],rotation=15); ax.set_ylim(0,1.05)
ax.set_title("Métricas de segurança de detecção (limiar escolhido na CV interna, sem tocar no teste)",pad=8)
ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.13)); fig.tight_layout(); save(fig,"figR_seguranca")

# ================= (B) ESTATÍSTICAS DO SELETOR =================
f8=pd.read_csv(os.path.join(ROOT,"02_compensacao","fase8_tuning_ampliado.csv")); f8=f8[f8.metodo!="TAU_T"]; MJ=["AE","Park","RF_direct"]
choices=[]; regrets=[]; margins=[]
for (b,tt),g in f8[f8.metodo.isin(MJ)].groupby(["banda","T_test"]):
    gg=g.set_index("metodo")
    if not set(MJ)<=set(gg.index): continue
    cvs={m:gg.loc[m,"cv_inner"] for m in MJ if pd.notna(gg.loc[m,"cv_inner"])}
    rms={m:gg.loc[m,"RMSD_D0"] for m in MJ}
    if not cvs: continue
    esc=min(cvs,key=cvs.get); orac=min(rms,key=rms.get)
    choices.append(esc); regrets.append(rms[esc]-rms[orac])
    srt=sorted(rms.values()); margins.append(srt[1]-srt[0])
freq={m:round(choices.count(m)/len(choices),3) for m in MJ}
selstat={"frequencia_escolha":freq,"regret_medio":round(float(np.mean(regrets)),4),"regret_max":round(float(np.max(regrets)),3),
         "pct_igual_oraculo":round(float(np.mean(np.array(regrets)<1e-6)),3),"margem_media_2o":round(float(np.mean(margins)),3),"n":len(choices)}
json.dump(selstat,open(os.path.join(OUT,"seletor_stats.json"),"w"),indent=2)
print("\n=== SELETOR (estatísticas) ===\n",json.dumps(selstat,indent=2,ensure_ascii=False),flush=True)

# ================= (C) ROBUSTEZ À REFERÊNCIA =================
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
rob=[]; rng=np.random.RandomState(0)
for b in ["30-40","70-80"]:
    lo,hi=map(int,b.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); m0=np.isclose(T,T_REF)&(y==0); H=X[m0]
    variantes={"1 curva":H[:1].mean(0),"mediana (todas)":np.median(H,0),"média (todas)":H.mean(0),
               "3 curvas":H[:3].mean(0),"5 curvas":H[:min(5,len(H))].mean(0),
               "mediana+ruído 2%":np.median(H,0)*(1+0.02*rng.randn(H.shape[1]))}
    cP={"max_shift_frac":0.1,"nsteps":121,"smooth_win":5}
    for nm,ref in variantes.items():
        errs=[]
        for tt in FOLDS:
            if np.isclose(tt,T_REF): continue
            te=np.isclose(T,tt); i0=np.where(te&(y==0))[0]
            Y=P.comp_park(X,ref,f,cP)[0]
            errs.append(np.mean([P.rmsd(Y[i],ref) for i in i0]))
        rob.append({"banda":b,"referencia":nm,"RMSD":round(float(np.mean(errs)),3)})
    print(f"  robustez ref {b} ok | {(time.time()-t0)/60:.1f}min",flush=True)
dr=pd.DataFrame(rob); dr.to_csv(os.path.join(OUT,"robustez_referencia.csv"),index=False)
print("\n=== ROBUSTEZ À REFERÊNCIA (RMSD saudável, Park) ===\n",dr.to_string(index=False),flush=True)
fig,ax=plt.subplots(figsize=(9,4.4),dpi=170)
order=["1 curva","3 curvas","5 curvas","média (todas)","mediana (todas)","mediana+ruído 2%"]
for b,cc in zip(["30-40","70-80"],["#1f5fd0","#d62728"]):
    s=dr[dr.banda==b].set_index("referencia").reindex(order)
    ax.plot(range(len(order)),s.RMSD,marker="o",lw=2,color=cc,label=f"{b} kHz")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=25,ha="right")
ax.set_ylabel("RMSD saudável"); ax.set_title("Robustez à definição da referência saudável",pad=8); ax.legend(frameon=False)
fig.tight_layout(); save(fig,"figR_robustez_ref")
print(f"\n✅ análises finais concluídas em {(time.time()-t0)/60:.1f} min",flush=True)

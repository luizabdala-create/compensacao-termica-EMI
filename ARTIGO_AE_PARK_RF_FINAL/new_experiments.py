# -*- coding: utf-8 -*-
"""
EXPERIMENTOS NOVOS (deste ciclo) — cada bloco salva CSV/figura próprios, com try/except.
1) CUSTO COMPUTACIONAL (destaque): tempo treino/inferência, nº params AE, tamanho RF, Park sem treino.
2) HISTERESE TÉRMICA (coluna sentido = aquecimento/resfriamento).
3) VAZAMENTO DE TEMPERATURA: prever T da curva compensada (menor R² = mais invariante).
4) REFERÊNCIA DANIFICADA (ground-truth): compensar dano@T -> comparar com dano REAL@Tref.
5) DAMAGE PRESERVATION RATIO (fórmula documentada).
6) CORRELAÇÃO RMSD_saudável <-> Macro-F1 (menor RMSD prediz melhor classificação?).
7) MATRIZ Ttreino x Tteste (classificação) para Original/Park/RF/AE.
8) MÉTRICAS DE PICO (erro de freq/amplitude, taxa de preservação).
9) PCA por dano vs temperatura.
10) WORST-CASE por método.
"""
import os,sys,json,time,traceback,numpy as np,pandas as pd
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score,f1_score,r2_score
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
FIG=os.path.join(ROOT,"10_figuras_artigo")
OUTn={"cost":"01_temperature_compensation","hyst":"01_temperature_compensation","leak":"01_temperature_compensation",
      "dmgref":"06_damage_preservation","dpr":"06_damage_preservation","corr":"10_statistics",
      "matrix":"08_cross_temperature_classification","peak":"01_temperature_compensation","pca":"07_damage_classification","worst":"10_statistics"}
for d in set(OUTn.values()): os.makedirs(os.path.join(ROOT,"results_article",d),exist_ok=True)
def outp(key,fn): return os.path.join(ROOT,"results_article",OUTn[key],fn)
def savef(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n,flush=True)

df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int); sent=df["sentido"].to_numpy(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json"))) if os.path.exists(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")) else {}
def cfg(banda,m,default):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else default
CFG={"Park":{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5},
     "RF_direct":{"n_estimators":300,"max_depth":10,"min_samples_leaf":2,"input_decim":8,"smooth_win":5},
     "AE":{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"epochs":450,"patience":55}}
def bandX(banda):
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    return fc,f,df[fc].to_numpy(np.float64)
def compensar(banda,X,f,T_REF=30.0):
    ref=np.median(X[np.isclose(T,T_REF)&(y==0)],axis=0); out={"Original":X.copy(),"ref":ref}
    for T_test in np.unique(T):
        te=np.isclose(T,T_test); trh=(~te)&(y==0); idx=np.where(te)[0]
        if trh.sum()<8: continue
        for m,fn in [("Park",lambda:P.comp_park(X,ref,f,cfg(banda,"Park",CFG["Park"]))[0]),
                     ("RF_direct",lambda:P.comp_rf(X,T,ref,trh,cfg(banda,"RF_direct",CFG["RF_direct"]),"direct")[0]),
                     ("AE",lambda:P.comp_ae(X,T,ref,trh,cfg(banda,"AE",CFG["AE"]),T_REF,seed=42)[0])]:
            out.setdefault(m,np.full_like(X,np.nan)); out[m][idx]=fn()[idx]
    return out
t0=time.time()

# ===== 1) CUSTO COMPUTACIONAL =====
try:
    print("\n[1] CUSTO COMPUTACIONAL",flush=True)
    banda="40-50"; fc,f,X=bandX(banda); ref=np.median(X[np.isclose(T,30.0)&(y==0)],axis=0)
    trh=(~np.isclose(T,60.0))&(y==0); rows=[]
    # Park (sem treino)
    t=time.time(); Y=P.comp_park(X,ref,f,CFG["Park"])[0]; tp=time.time()-t
    rows.append({"metodo":"Park","treino_s":0.0,"total_%d_curvas_s"%len(X):round(tp,3),"inferencia_ms_por_curva":round(1000*tp/len(X),2),"n_params_ou_tamanho":"—"})
    # RF
    t=time.time(); rf=RandomForestRegressor(n_estimators=300,max_depth=10,min_samples_leaf=2,n_jobs=-1,random_state=0)
    from pipeline import _rf_feats
    rf.fit(_rf_feats(X[trh],T[trh],"direct",8),ref[None,:]-X[trh]); ttr=time.time()-t
    t=time.time(); _=rf.predict(_rf_feats(X,T,"direct",8)); tinf=time.time()-t
    nnodes=sum(e.tree_.node_count for e in rf.estimators_)
    rows.append({"metodo":"RF direto","treino_s":round(ttr,3),"total_%d_curvas_s"%len(X):round(tinf,3),"inferencia_ms_por_curva":round(1000*tinf/len(X),2),"n_params_ou_tamanho":f"{rf.n_estimators} árv., {nnodes} nós"})
    # AE
    import torch
    t=time.time(); Y,info=P.comp_ae(X,T,ref,trh,CFG["AE"],30.0,seed=42); tae=time.time()-t
    rows.append({"metodo":"Autoencoder","treino_s":round(tae,3),"total_%d_curvas_s"%len(X):round(tae,3),"inferencia_ms_por_curva":"incl. no total","n_params_ou_tamanho":f"{info['n_params']} params"})
    d=pd.DataFrame(rows); d.to_csv(outp("cost","custo_computacional.csv"),index=False)
    print(d.to_string(index=False),flush=True)
    fig,ax=plt.subplots(figsize=(7,4.2),dpi=170)
    ax.bar(["Park","RF direto","Autoencoder"],[0.0,ttr,tae],color=["#2ca02c","#d62728","#1f5fd0"],edgecolor="k")
    for i,v in enumerate([0.0,ttr,tae]): ax.text(i,v+0.02,f"{v:.2f}s",ha="center")
    ax.set_ylabel("Tempo de treino (s)"); ax.set_title("Custo computacional — treino por fold (40–50 kHz, CPU)")
    savef(fig,"figN_custo_computacional")
except Exception as e: print("[1] erro:",e,traceback.format_exc()[-300:],flush=True)

# ===== 2) HISTERESE TÉRMICA (sentido) =====
try:
    print("\n[2] HISTERESE TÉRMICA (sentido)",flush=True)
    banda="40-50"; fc,f,X=bandX(banda); rows=[]
    for t_ in sorted(np.unique(T)):
        for d_ in [0,1,2]:
            i1=np.where(np.isclose(T,t_)&(y==d_)&(sent==1))[0]; i2=np.where(np.isclose(T,t_)&(y==d_)&(sent==2))[0]
            if len(i1) and len(i2):
                m1=np.median(X[i1],axis=0); m2=np.median(X[i2],axis=0)
                rows.append({"T":t_,"dano":d_,"RMSD_s1_vs_s2":P.rmsd(m1,m2),"CCDM_s1_vs_s2":P.ccdm(m1,m2)})
    h=pd.DataFrame(rows); h.to_csv(outp("hyst","histerese_sentido.csv"),index=False)
    print(f"  histerese saudável (média RMSD entre sentidos): {h[h.dano==0].RMSD_s1_vs_s2.mean():.3f}",flush=True)
    print(f"  (comparar com magnitude do dano ~5-6): se << dano, histerese é pequena",flush=True)
    fig,ax=plt.subplots(figsize=(9,4.4),dpi=170)
    for d_,c in [(0,"#1f77b4"),(1,"#ff7f0e"),(2,"#d62728")]:
        s=h[h.dano==d_].sort_values("T"); ax.plot(s["T"],s.RMSD_s1_vs_s2,marker="o",color=c,label=f"Dano {d_}")
    ax.set_xlabel("Temperatura (°C)"); ax.set_ylabel("RMSD entre sentidos (aquec. vs resfr.)")
    ax.set_title("Histerese térmica: diferença entre sentidos de varredura"); ax.legend(frameon=False)
    savef(fig,"figN_histerese")
except Exception as e: print("[2] erro:",e,flush=True)

# ===== 3) VAZAMENTO DE TEMPERATURA =====
try:
    print("\n[3] VAZAMENTO DE TEMPERATURA (prever T da curva compensada)",flush=True)
    banda="40-50"; fc,f,X=bandX(banda); comp=compensar(banda,X,f)
    hmask=(y==0); rows=[]
    for m in ["Original","Park","RF_direct","AE"]:
        Y=comp[m]; ok=hmask & ~np.isnan(Y).any(axis=1)
        # LOTO: prever T (regressão RF) da curva compensada saudável
        preds=[]; truth=[]
        for t_ in np.unique(T[ok]):
            te=np.isclose(T,t_)&ok; tr=(~np.isclose(T,t_))&ok
            if tr.sum()<10 or te.sum()==0: continue
            reg=RandomForestRegressor(n_estimators=200,n_jobs=-1,random_state=0)
            reg.fit(Y[tr][:,::5],T[tr]); p=reg.predict(Y[te][:,::5]); preds+=list(p); truth+=list(T[te])
        r2=r2_score(truth,preds); mae=np.mean(np.abs(np.array(truth)-np.array(preds)))
        rows.append({"metodo":m,"R2_prever_T":round(r2,3),"MAE_T_C":round(mae,2)})
    d=pd.DataFrame(rows); d.to_csv(outp("leak","temperature_leakage.csv"),index=False)
    print(d.to_string(index=False),flush=True); print("  (menor R² = mais invariante à temperatura = melhor)",flush=True)
    fig,ax=plt.subplots(figsize=(7,4.2),dpi=170)
    _LBN={"Original":"Original","Park":"Park","RF_direct":"Random Forest","AE":"Autoencoder","ExtraTrees":"Extra Trees","RF_temponly":"RF (só temp.)"}
    ax.bar([_LBN.get(m,m) for m in d.metodo],d.R2_prever_T,color=["#7f7f7f","#2ca02c","#d62728","#1f5fd0"],edgecolor="k")
    ax.set_ylabel("R² ao prever a temperatura"); ax.set_title("Recuperabilidade da temperatura após compensação\n(menor = mais invariante à temperatura)")
    for i,v in enumerate(d.R2_prever_T): ax.text(i,v+.01,f"{v:.2f}",ha="center")
    savef(fig,"figN_temperature_leakage")
except Exception as e: print("[3] erro:",e,traceback.format_exc()[-300:],flush=True)

# ===== 4) REFERÊNCIA DANIFICADA (ground-truth de preservação) =====
try:
    print("\n[4] REFERÊNCIA DANIFICADA (ground-truth)",flush=True)
    banda="40-50"; fc,f,X=bandX(banda); comp=compensar(banda,X,f); rows=[]
    T_REF=30.0
    for d_ in [1,2]:
        gt=np.median(X[np.isclose(T,T_REF)&(y==d_)],axis=0)  # dano REAL na referência (só p/ avaliação)
        for m in ["Original","Park","RF_direct","AE"]:
            Y=comp[m]
            for t_ in [tt for tt in np.unique(T) if not np.isclose(tt,T_REF)]:
                ii=np.where(np.isclose(T,t_)&(y==d_))[0]
                ii=[i for i in ii if not np.isnan(Y[i]).any()]
                if not ii: continue
                yc=np.median(Y[ii],axis=0)
                rows.append({"metodo":m,"dano":d_,"T":t_,"RMSD_vs_dano_ref":P.rmsd(yc,gt),"CCDM_vs_dano_ref":P.ccdm(yc,gt),"CORR":P.corr(yc,gt)})
    dr=pd.DataFrame(rows); dr.to_csv(outp("dmgref","referencia_danificada.csv"),index=False)
    piv=dr.groupby(["metodo","dano"])["RMSD_vs_dano_ref"].mean().unstack().round(3)
    print("RMSD da curva compensada vs dano REAL na referência (menor=preservou melhor a assinatura):",flush=True)
    print(piv.to_string(),flush=True)
except Exception as e: print("[4] erro:",e,traceback.format_exc()[-300:],flush=True)

# ===== 5) DAMAGE PRESERVATION RATIO =====
try:
    print("\n[5] DAMAGE PRESERVATION RATIO",flush=True)
    # DPR = (sep saudável-dano APÓS comp) / (sep saudável-dano ANTES=Original), por dano/temperatura
    banda="40-50"; fc,f,X=bandX(banda); comp=compensar(banda,X,f); ref=comp["ref"]; rows=[]
    for m in ["Park","RF_direct","AE"]:
        for d_ in [1,2]:
            num=[]; den=[]
            for t_ in FOLDS:
                i0=np.where(np.isclose(T,t_)&(y==0))[0]; idd=np.where(np.isclose(T,t_)&(y==d_))[0]
                if not len(i0) or not len(idd): continue
                # antes (Original): distância dano-saudável
                den.append(P.rmsd(np.median(X[idd],axis=0),np.median(X[i0],axis=0)))
                Ym=comp[m]
                if np.isnan(Ym[idd]).any() or np.isnan(Ym[i0]).any(): continue
                num.append(P.rmsd(np.median(Ym[idd],axis=0),np.median(Ym[i0],axis=0)))
            if num and den: rows.append({"metodo":m,"dano":d_,"DPR":round(np.mean(num)/np.mean(den),3)})
    dpr=pd.DataFrame(rows); dpr.to_csv(outp("dpr","damage_preservation_ratio.csv"),index=False)
    print("DPR (≈1 preserva, <1 atenua, >1 amplifica):",flush=True); print(dpr.to_string(index=False),flush=True)
except Exception as e: print("[5] erro:",e,flush=True)

# ===== 6) CORRELAÇÃO RMSD_saudável <-> Macro-F1 =====
try:
    print("\n[6] CORRELAÇÃO RMSD_saudável x Macro-F1 (por banda×método)",flush=True)
    comp8=pd.read_csv(os.path.join(ROOT,"02_compensacao","fase8_tuning_ampliado.csv")); comp8=comp8[comp8.metodo!="TAU_T"]
    rmsd=comp8.groupby(["banda","metodo"]).RMSD_D0.mean().reset_index()
    pb=pd.read_csv(os.path.join(ROOT,"checkpoints","parteB.csv")); pb=pb[(pb.controle=="real")&(pb.task=="multi")]
    f1=pb.groupby(["banda","metodo"]).macro_f1.mean().reset_index()
    mg=rmsd.merge(f1,on=["banda","metodo"])
    from scipy import stats as sps
    r,p=sps.spearmanr(mg.RMSD_D0,mg.macro_f1)
    print(f"  Spearman(RMSD_saudável, Macro-F1) = {r:.3f} (p={p:.3g}) sobre {len(mg)} pares",flush=True)
    print(f"  => {'menor RMSD tende a MAIOR F1' if r<0 else 'NÃO há relação clara: menor RMSD NÃO garante melhor classificação'}",flush=True)
    mg.to_csv(outp("corr","corr_rmsd_f1.csv"),index=False)
    fig,ax=plt.subplots(figsize=(6.5,5),dpi=170)
    cs={"AE":"#1f5fd0","Park":"#2ca02c","RF_direct":"#d62728","Original":"#7f7f7f","RF_temponly":"#9467bd"}
    for m in mg.metodo.unique(): s=mg[mg.metodo==m]; ax.scatter(s.RMSD_D0,s.macro_f1,color=cs.get(m,"k"),label=m,s=60,edgecolor="k")
    ax.set_xlabel("RMSD saudável (compensação)"); ax.set_ylabel("Macro-F1 (classificação)")
    ax.set_title(f"Compensação prediz classificação? Spearman={r:.2f}"); ax.legend(frameon=False,fontsize=8)
    savef(fig,"figN_corr_rmsd_f1")
except Exception as e: print("[6] erro:",e,traceback.format_exc()[-300:],flush=True)

# ===== 7) MATRIZ Ttreino x Tteste (classificação) =====
try:
    print("\n[7] MATRIZ Ttreino x Tteste (classificação multiclasse)",flush=True)
    banda="40-50"; fc,f,X=bandX(banda); comp=compensar(banda,X,f)
    temps=[t for t in FOLDS]
    fig,axes=plt.subplots(1,4,figsize=(19,5),dpi=170)
    for ax,m in zip(axes,["Original","Park","RF_direct","AE"]):
        Y=comp[m]; M=np.full((len(temps),len(temps)),np.nan)
        R=Y-comp["ref"][None,:]; Rs=np.vstack([P.moving_average(r,51) for r in R])
        for i,ttr in enumerate(temps):
            tri=np.isclose(T,ttr)
            if len(np.unique(y[tri]))<3: continue
            clf=Pipeline([("s",StandardScaler()),("c",RandomForestClassifier(n_estimators=200,class_weight="balanced",n_jobs=-1,random_state=0))])
            try: clf.fit(Rs[tri],y[tri])
            except Exception: continue
            for j,tte in enumerate(temps):
                tei=np.isclose(T,tte)
                if tei.sum()==0 or len(np.unique(y[tei]))<2: continue
                M[i,j]=balanced_accuracy_score(y[tei],clf.predict(Rs[tei]))
        im=ax.imshow(M,cmap="viridis",vmin=0.3,vmax=1.0,aspect="auto")
        ax.set_xticks(range(len(temps))); ax.set_xticklabels([f"{int(t)}" for t in temps],rotation=90,fontsize=7)
        ax.set_yticks(range(len(temps))); ax.set_yticklabels([f"{int(t)}" for t in temps],fontsize=7)
        ax.set_title(m); ax.set_xlabel("T teste");
        if m=="Original": ax.set_ylabel("T treino")
        np.savetxt(outp("matrix",f"matriz_{m}.csv"),M,delimiter=",")
    fig.colorbar(im,ax=axes,fraction=.015,pad=.01,label="Balanced accuracy")
    fig.suptitle("Generalização térmica da classificação: T treino × T teste (40–50 kHz)",y=1.02,fontsize=13)
    savef(fig,"figN_matriz_ttreino_tteste")
except Exception as e: print("[7] erro:",e,traceback.format_exc()[-300:],flush=True)

# ===== 8) MÉTRICAS DE PICO =====
try:
    print("\n[8] MÉTRICAS DE PICO",flush=True)
    from scipy.signal import find_peaks
    banda="40-50"; fc,f,X=bandX(banda); comp=compensar(banda,X,f); ref=comp["ref"]; fk=f/1e3
    pk_ref,_=find_peaks(ref,prominence=3,distance=len(ref)//40)
    rows=[]
    for m in ["Original","Park","RF_direct","AE"]:
        Y=comp[m]; ii=np.where((y==0)&~np.isnan(Y).any(axis=1))[0]
        fe=[]; ae=[]; pr=[]
        for i in ii:
            pk,_=find_peaks(Y[i],prominence=3,distance=len(ref)//40)
            if len(pk)==0: pr.append(0); continue
            # casar cada pico da ref ao mais próximo
            errs_f=[]; errs_a=[]
            for p0 in pk_ref:
                j=pk[np.argmin(np.abs(pk-p0))]; errs_f.append(abs(fk[j]-fk[p0])*1000); errs_a.append(abs(Y[i][j]-ref[p0]))
            fe.append(np.mean(errs_f)); ae.append(np.mean(errs_a)); pr.append(min(1.0,len(pk)/max(1,len(pk_ref))))
        rows.append({"metodo":m,"peak_freq_err_Hz":round(np.mean(fe),1),"peak_amp_err":round(np.mean(ae),2),"peak_preservation":round(np.mean(pr),3)})
    dp=pd.DataFrame(rows); dp.to_csv(outp("peak","peak_metrics.csv"),index=False)
    print(dp.to_string(index=False),flush=True)
except Exception as e: print("[8] erro:",e,traceback.format_exc()[-300:],flush=True)

# ===== 9) PCA por dano vs temperatura =====
try:
    print("\n[9] PCA",flush=True)
    banda="40-50"; fc,f,X=bandX(banda); comp=compensar(banda,X,f); ref=comp["ref"]
    fig,axes=plt.subplots(2,4,figsize=(19,9),dpi=170)
    CDM={0:"#1f77b4",1:"#ff7f0e",2:"#d62728"}
    for col,m in enumerate(["Original","Park","RF_direct","AE"]):
        Y=comp[m]; ok=~np.isnan(Y).any(axis=1); R=(Y-ref[None,:])[ok]
        pc=PCA(n_components=2).fit_transform(R); yy=y[ok]; tt=T[ok]
        for d_ in [0,1,2]: axes[0,col].scatter(pc[yy==d_,0],pc[yy==d_,1],c=CDM[d_],s=18,label=f"D{d_}",alpha=.7)
        axes[0,col].set_title(f"{m} — cor=dano");
        sc=axes[1,col].scatter(pc[:,0],pc[:,1],c=tt,cmap="coolwarm",s=18); axes[1,col].set_title("cor=temperatura")
    axes[0,0].legend(frameon=False,fontsize=8); fig.colorbar(sc,ax=axes[1,:],fraction=.01,pad=.01,label="T (°C)")
    fig.suptitle("PCA do resíduo compensado — organização por DANO (cima) vs TEMPERATURA (baixo)",y=1.0,fontsize=13)
    savef(fig,"figN_pca")
except Exception as e: print("[9] erro:",e,traceback.format_exc()[-300:],flush=True)

# ===== 10) WORST-CASE =====
try:
    print("\n[10] WORST-CASE (pior temperatura por método)",flush=True)
    comp8=pd.read_csv(os.path.join(ROOT,"02_compensacao","fase8_tuning_ampliado.csv")); comp8=comp8[comp8.metodo!="TAU_T"]
    rows=[]
    for m in comp8.metodo.unique():
        s=comp8[comp8.metodo==m]; g=s.groupby("T_test").RMSD_D0.mean()
        rows.append({"metodo":m,"RMSD_medio":round(g.mean(),3),"RMSD_pior":round(g.max(),3),"T_pior":g.idxmax(),"RMSD_p90":round(g.quantile(.9),3)})
    w=pd.DataFrame(rows).sort_values("RMSD_pior"); w.to_csv(outp("worst","worst_case.csv"),index=False)
    print(w.to_string(index=False),flush=True)
except Exception as e: print("[10] erro:",e,flush=True)

print(f"\n✅ NEW_EXPERIMENTS concluído em {(time.time()-t0)/60:.1f} min",flush=True)

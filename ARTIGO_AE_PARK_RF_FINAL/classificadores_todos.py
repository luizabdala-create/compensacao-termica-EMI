# -*- coding: utf-8 -*-
"""
CLASSIFICAÇÃO DE DANO — TODOS OS CLASSIFICADORES × TODAS AS FEATURES.
7 classificadores (LogReg, SVM-RBF, SVM-linear, KNN, RandomForest, GradientBoosting, MLP)
× 4 conjuntos de features (FS1 métricas, FS2 métricas completas, FS3 PCA do resíduo,
FS4 features de pico) sobre as curvas compensadas (Original/Park/RF/AE), LOTO, binário
e multiclasse, com controle negativo (rótulos embaralhados). Bandas representativas
(estreitas + largas). Diz qual classificador e qual feature set é melhor por método.
"""
import os,sys,json,time,numpy as np,pandas as pd
from collections import Counter
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; sys.path.insert(0,ROOT); import pipeline as P
from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score,recall_score,confusion_matrix
OUT=os.path.join(ROOT,"04_dano_multiclasse"); CKPT=os.path.join(ROOT,"checkpoints","classificadores.csv")
BANDS=["40-50","60-70","70-80","30-70","30-100"]
bestcfg=json.load(open(os.path.join(ROOT,"checkpoints","fase8_bestcfg.json")))
def cfg(banda,m,default):
    ks=[k for k in bestcfg if k.split("|")[0]==banda and k.split("|")[2]==m]
    return json.loads(Counter([json.dumps(bestcfg[k],sort_keys=True) for k in ks]).most_common(1)[0][0]) if ks else default
def clf(name):
    return {"LogReg":Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=5000,class_weight="balanced"))]),
            "SVM-RBF":Pipeline([("s",StandardScaler()),("c",SVC(kernel="rbf",class_weight="balanced"))]),
            "SVM-lin":Pipeline([("s",StandardScaler()),("c",SVC(kernel="linear",class_weight="balanced"))]),
            "KNN":Pipeline([("s",StandardScaler()),("c",KNeighborsClassifier(n_neighbors=5))]),
            "RForest":Pipeline([("s",StandardScaler()),("c",RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=-1,random_state=0))]),
            "GBoost":Pipeline([("s",StandardScaler()),("c",GradientBoostingClassifier(random_state=0))]),
            "MLP":Pipeline([("s",StandardScaler()),("c",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1500,random_state=0))])}[name]
CLFS=["LogReg","SVM-RBF","SVM-lin","KNN","RForest","GBoost","MLP"]; FSS=["FS1","FS2","FS3","FS4"]
def feats(Y,ref,fs,tr,te):
    R=Y-ref[None,:]
    if fs in("FS1","FS2"):
        M=[]
        for i in range(len(Y)):
            m=P.all_metrics(Y[i],ref); M.append([m["RMSD"],m["CCDM"]] if fs=="FS1" else [m["RMSD"],m["CCDM"],m["RMSE"],m["MAE"],m["NRMSE"],m["CORR"],m["SAM_deg"]])
        M=np.array(M); return M[tr],M[te]
    if fs=="FS3":
        w=max(1,R.shape[1]//500); Rs=np.vstack([P.moving_average(r,51)[::w] for r in R])
        pca=PCA(n_components=min(10,int(tr.sum())-1),random_state=0).fit(Rs[tr]); return pca.transform(Rs[tr]),pca.transform(Rs[te])
    F=[]
    for r in R:
        rs=P.moving_average(r,51); idx=np.argsort(-np.abs(rs))[:200]
        F.append([rs.max(),rs.min(),np.ptp(rs),np.std(rs),np.mean(np.abs(rs)),float(np.mean(idx)),float(np.std(idx)),float(np.percentile(np.abs(rs),95)),float(np.sum(rs**2))])
    F=np.array(F); return F[tr],F[te]
df,_,_=P.load_base(); T=df["temperatura_c"].to_numpy(float); y=df["falha"].to_numpy(int); ybin=(y>0).astype(int)
FOLDS=[t for t in sorted(np.unique(T)) if all(((np.isclose(T,t))&(y==c)).sum()>0 for c in [0,1,2])]
rows=[]; t0=time.time()
for banda in BANDS:
    lo,hi=map(int,banda.split("-")); fc,f=P.band(lo,hi,decim=1); dec=max(1,len(fc)//10000); fc,f=P.band(lo,hi,decim=dec)
    X=df[fc].to_numpy(np.float64); ref,_=P.build_reference(df,fc,30.0)
    print(f"\n=== {banda} ===",flush=True)
    for T_test in FOLDS:
        if np.isclose(T_test,30.0): continue
        te=np.isclose(T,T_test); tr=~te; trh=tr&(y==0)
        M={"Original":X.copy(),"Park":P.comp_park(X,ref,f,cfg(banda,"Park",{"max_shift_frac":0.1,"nsteps":121,"smooth_win":5}))[0],
           "RF_direct":P.comp_rf(X,T,ref,trh,cfg(banda,"RF_direct",{"n_estimators":300,"max_depth":10,"input_decim":8,"smooth_win":5}),"direct")[0],
           "AE":P.comp_ae(X,T,ref,trh,cfg(banda,"AE",{"n_input":2000,"n_anchors":128,"latent":8,"hidden":256,"lr":2e-3,"epochs":450,"patience":55}),30.0,seed=42)[0]}
        for mn,Y in M.items():
            for fs in FSS:
                try: Xtr,Xte=feats(Y,ref,fs,tr,te)
                except Exception: continue
                for cn in CLFS:
                    for task,yy in [("bin",ybin),("multi",y)]:
                        try:
                            c=clf(cn); c.fit(Xtr,yy[tr]); pr=c.predict(Xte)
                            rec={"banda":banda,"T_test":T_test,"metodo":mn,"feature_set":fs,"clf":cn,"task":task,"controle":"real",
                                 "bal_acc":balanced_accuracy_score(yy[te],pr),"macro_f1":f1_score(yy[te],pr,average="macro",zero_division=0)}
                            if task=="bin":
                                cm=confusion_matrix(yy[te],pr,labels=[0,1]); rec["taxa_falso_saudavel"]=float(cm[1,0]/max(cm[1].sum(),1)); rec["recall_dano"]=recall_score(yy[te],pr,pos_label=1,zero_division=0)
                            rows.append(rec)
                            rng=np.random.RandomState(0); ysh=yy[tr].copy(); rng.shuffle(ysh); c2=clf(cn); c2.fit(Xtr,ysh); pr2=c2.predict(Xte)
                            rows.append({"banda":banda,"T_test":T_test,"metodo":mn,"feature_set":fs,"clf":cn,"task":task,"controle":"shuffled","bal_acc":balanced_accuracy_score(yy[te],pr2)})
                        except Exception: pass
        pd.DataFrame(rows).to_csv(CKPT,index=False); print(f"  T={T_test} ok | {(time.time()-t0)/60:.1f} min | {len(rows)} regs",flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(OUT,"classificadores_todos.csv"),index=False)
r=d[d.controle=="real"]
print(f"\n✅ {len(d)} regs | {(time.time()-t0)/60:.1f} min",flush=True)
print("\n=== MELHOR CLASSIFICADOR (média bal_acc sobre bandas/folds/features) ===")
for task in ["bin","multi"]:
    s=r[r.task==task]
    print(f"\n[{task}] por classificador:"); print(s.groupby("clf")["bal_acc"].mean().sort_values(ascending=False).round(3).to_string())
    print(f"[{task}] por feature set:"); print(s.groupby("feature_set")["bal_acc"].mean().sort_values(ascending=False).round(3).to_string())
    print(f"[{task}] melhor pipeline por método:")
    for m in ["Original","Park","RF_direct","AE"]:
        sm=s[s.metodo==m]
        if len(sm):
            g=sm.groupby(["clf","feature_set"])["bal_acc"].mean(); bi=g.idxmax(); print(f"    {m:10s}: {bi[0]}+{bi[1]} = {g.max():.3f}")
sh=d[d.controle=="shuffled"].groupby("task")["bal_acc"].mean().round(3)
print("\ncontrole negativo:",dict(sh))

# ---- figuras de comparação de classificadores ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
FIG=os.path.join(ROOT,"10_figuras_artigo")
LBM={"Original":"Original","Park":"Park","RF_direct":"RF","AE":"AE"}
def savef(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig)
# (a) barras: bal_acc por classificador, binário e multiclasse
fig,axes=plt.subplots(1,2,figsize=(14,4.6),dpi=170)
for ax,task,tt in zip(axes,["bin","multi"],["Binária","Multiclasse"]):
    s=r[r.task==task].groupby("clf")["bal_acc"].mean().reindex(CLFS)
    ax.bar(range(len(CLFS)),s.values,color="#4C72B0",edgecolor="k",lw=.5)
    for i,v in enumerate(s.values): ax.text(i,v+.005,f"{v:.2f}",ha="center",fontsize=8)
    ax.set_xticks(range(len(CLFS))); ax.set_xticklabels(CLFS,rotation=25,ha="right")
    ax.set_ylabel("Acurácia balanceada"); ax.set_title(f"({'a' if task=='bin' else 'b'}) {tt}"); ax.set_ylim(0,1)
fig.suptitle("Comparação de classificadores (média sobre compensadores, features, bandas e folds)",y=1.0,fontsize=12)
fig.tight_layout(); savef(fig,"figC_classificadores")
# (b) heatmap método × classificador (binário)
fig,axes=plt.subplots(1,2,figsize=(14,4.4),dpi=170)
for ax,task,tt in zip(axes,["bin","multi"],["Binária","Multiclasse"]):
    piv=r[r.task==task].pivot_table(index="metodo",columns="clf",values="bal_acc",aggfunc="mean").reindex(["Original","Park","RF_direct","AE"])[CLFS]
    im=ax.imshow(piv.values,cmap="viridis",vmin=0.6,vmax=1.0,aspect="auto")
    ax.set_xticks(range(len(CLFS))); ax.set_xticklabels(CLFS,rotation=30,ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([LBM[m] for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v=piv.values[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7.5,color="w" if v<0.82 else "k")
    ax.set_title(f"({'a' if task=='bin' else 'b'}) {tt}")
fig.colorbar(im,ax=axes,fraction=.02,pad=.02,label="Acurácia balanceada")
fig.suptitle("Compensador × classificador — acurácia balanceada",y=1.0,fontsize=12); savef(fig,"figC_heatmap_metodo_clf")
print("✅ figuras de classificadores geradas")

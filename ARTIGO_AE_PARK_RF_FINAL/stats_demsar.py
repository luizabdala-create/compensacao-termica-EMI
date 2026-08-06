# -*- coding: utf-8 -*-
"""
ESTATÍSTICA (Demšar 2006) — comparação de vários métodos sobre várias condições.
Friedman (omnibus) + pós-teste de Nemenyi (diagrama de diferença crítica) +
Wilcoxon signed-rank pareado com correção de Holm (mais potente que Nemenyi).
Aplicável à compensação (RMSD/CCDM) e à classificação (bal_acc etc.).
Referências: Demšar (2006) JMLR; Friedman (1937); Nemenyi (1963); Wilcoxon (1945); Holm (1979).
"""
import os,sys,json,itertools,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try: from scipy import stats as sps; HAVE=True
except Exception: HAVE=False
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
FIG=os.path.join(ROOT,"10_figuras_artigo"); OUT=os.path.join(ROOT,"09_estatistica")
os.makedirs(FIG,exist_ok=True); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"pdf.fonttype":42})
# q_alpha (Nemenyi, alpha=0.05), Demšar 2006 Tab.5
QA={2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164}
LBL={"Original":"Original","Park":"Park","RF_direct":"RF (direct)","ExtraTrees":"RF otim.","RF_temponly":"RF (temp-only)",
     "AE":"Autoencoder","TAU_T":"tau(T) (aux.)"}

def cd_analysis(df, value_col, cond_cols, method_col="metodo", lower_better=True,
                methods=None, name="cd", title=""):
    """Retorna ranks médios + CD; desenha o diagrama."""
    w=df.pivot_table(index=cond_cols,columns=method_col,values=value_col,aggfunc="mean").dropna()
    if methods: w=w[[m for m in methods if m in w.columns]]
    if len(w)<3 or w.shape[1]<3:
        print(f"[{name}] dados insuficientes"); return None
    N,k=w.shape
    # ranks por condição (1 = melhor)
    ranks=w.rank(axis=1,ascending=lower_better)
    avg=ranks.mean().sort_values()
    CD=QA.get(k,3.2)*np.sqrt(k*(k+1)/(6*N))
    # Friedman
    fr_p=np.nan
    if HAVE:
        fr=sps.friedmanchisquare(*[w[c].values for c in w.columns]); fr_p=fr.pvalue
    # ---- desenho do diagrama CD ----
    fig,ax=plt.subplots(figsize=(9,2.6+0.28*k),dpi=170)
    lo=0.9; hi=k+0.1
    ax.set_xlim(lo,hi); ax.set_ylim(0,1); ax.axis("off")
    y0=0.85
    ax.plot([lo,hi],[y0,y0],"k-",lw=1.2)
    for r in range(1,k+1):
        ax.plot([r,r],[y0,y0+0.03],"k-",lw=1); ax.text(r,y0+0.06,str(r),ha="center",fontsize=9)
    ax.text((lo+hi)/2,y0+0.13,f"Rank médio (1 = melhor) — CD={CD:.2f} (Nemenyi, α=0.05)",ha="center",fontsize=9.5)
    order=list(avg.index)
    n=len(order); ystep=0.5/max(1,n)
    for i,m in enumerate(order):
        r=avg[m]; side=0 if r<(1+k)/2 else 1
        yy=y0-0.08-(i*ystep if side==0 else (n-1-i)*ystep)
        xend=lo if side==0 else hi
        ax.plot([r,r],[y0,yy],"k-",lw=0.9); ax.plot([r,xend],[yy,yy],"k-",lw=0.9)
        ax.text(xend+(-0.03 if side==0 else 0.03),yy,f"{LBL.get(m,m)} ({r:.2f})",
                ha="right" if side==0 else "left",va="center",fontsize=9.5)
    # barras de não-significância (grupos dentro de CD)
    srt=avg.sort_values(); vals=srt.values; names=list(srt.index)
    ybar=y0-0.02; groups=[]
    i=0
    while i<len(vals):
        j=i
        while j+1<len(vals) and vals[j+1]-vals[i]<=CD: j+=1
        if j>i: groups.append((vals[i],vals[j]));
        i+=1
    # remove grupos contidos
    filt=[g for g in groups if not any(g!=h and h[0]<=g[0] and g[1]<=h[1] for h in groups)]
    for gi,(a,b) in enumerate(filt):
        yb=ybar-gi*0.035
        ax.plot([a-0.03,b+0.03],[yb,yb],"-",lw=3,color="crimson",solid_capstyle="round")
    if title: ax.text((lo+hi)/2,0.02,title,ha="center",fontsize=10,style="italic")
    for e,d in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{name}.{e}"),dpi=d,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(f"[{name}] N={N} k={k} | Friedman p={fr_p:.2e} | CD={CD:.3f}")
    print("   ranks:",{m:round(float(avg[m]),3) for m in order})
    return {"avg_rank":{m:float(avg[m]) for m in avg.index},"CD":float(CD),"friedman_p":float(fr_p),"N":int(N),"k":int(k)}

def wilcoxon_holm(df, value_col, cond_cols, methods, method_col="metodo", lower_better=True):
    w=df.pivot_table(index=cond_cols,columns=method_col,values=value_col,aggfunc="mean").dropna()
    methods=[m for m in methods if m in w.columns]
    pairs=list(itertools.combinations(methods,2)); res=[]
    for a,b in pairs:
        diff=(w[a]-w[b]).values
        p=float(sps.wilcoxon(w[a],w[b]).pvalue) if HAVE and np.any(diff!=0) else np.nan
        res.append([a,b,float(diff.mean()),p])
    ps=[r[3] for r in res]; order=np.argsort(ps); m=len(ps); adj=[None]*m; prev=0
    for rk,idx in enumerate(order):
        v=min(1.0,max(prev,(m-rk)*ps[idx])); adj[idx]=v; prev=v
    out=[]
    for (a,b,dm,p),pa in zip(res,adj):
        out.append({"A":LBL.get(a,a),"B":LBL.get(b,b),"dif_med":round(dm,4),"p_wilcoxon":p,"p_holm":pa,
                    "signif":"sim" if (pa==pa and pa<0.05) else "não"})
    return pd.DataFrame(out)

if __name__=="__main__":
    # ---- COMPENSAÇÃO (fase8 tunado) ----
    comp=None
    for p in ["02_compensacao/fase8_tuning_ampliado.csv","checkpoints/fase8_tuning.csv","02_compensacao/comparacao_justa_todos_tunados.csv"]:
        fp=os.path.join(ROOT,p)
        if os.path.exists(fp):
            c=pd.read_csv(fp)
            if c.metodo.nunique()>=3 and len(c)>20: comp=c; print("comp source:",p); break
    if comp is not None: comp=comp[comp.metodo!="TAU_T"]
    MAIN=["AE","Park","RF_direct","ExtraTrees","RF_temponly"]
    rep={}
    if comp is not None:
        rep["rmsd"]=cd_analysis(comp,"RMSD_D0",["banda","T_test"],methods=MAIN,name="figCD_RMSD",
                                title="Compensação térmica (RMSD saudável), por banda×temperatura")
        rep["ccdm"]=cd_analysis(comp,"CCDM_D0",["banda","T_test"],methods=MAIN,name="figCD_CCDM",
                                title="Compensação térmica (CCDM saudável)")
        wilcoxon_holm(comp,"RMSD_D0",["banda","T_test"],MAIN).to_csv(os.path.join(OUT,"wilcoxon_holm_RMSD.csv"),index=False)
        print("\nWilcoxon-Holm RMSD:\n",wilcoxon_holm(comp,"RMSD_D0",["banda","T_test"],MAIN).to_string(index=False))
    # ---- CLASSIFICAÇÃO ----
    pb=None
    for p in ["checkpoints/parteB.csv"]:
        fp=os.path.join(ROOT,p)
        if os.path.exists(fp): pb=pd.read_csv(fp)
    if pb is not None:
        pb=pb[pb.metodo!="TAU_T"]; real=pb[pb.controle=="real"]
        MC=["Original","Park","RF_direct","RF_temponly","AE"]
        for task in ["bin","multi"]:
            s=real[real.task==task]
            rep[f"clf_{task}"]=cd_analysis(s,"bal_acc",["banda","T_test","feature_set","clf"],methods=MC,
                lower_better=False,name=f"figCD_clf_{task}",
                title=f"Classificação {task} (balanced accuracy), por condição")
    json.dump(rep,open(os.path.join(OUT,"demsar_resumo.json"),"w"),indent=2)
    print("\n✅ estatística Demšar salva")

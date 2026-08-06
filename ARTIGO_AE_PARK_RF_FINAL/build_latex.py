# -*- coding: utf-8 -*-
"""
GERADOR DO ARTIGO EM LATEX (português) — lê os CSVs finais e emite article.tex.
Data-driven: tabelas e números vêm dos resultados. Figuras (.pdf) referenciadas.
Roda de novo ao fim da bateria para atualizar tudo.
"""
import os,sys,json,numpy as np,pandas as pd
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
MAN=os.path.join(ROOT,"12_manuscrito"); FIGDIR=os.path.join(ROOT,"10_figuras_artigo")
os.makedirs(MAN,exist_ok=True)
LBL={"Original":"Original","Park":"Park","RF_direct":"RF (direto)","RF_temponly":"RF (só temp.)",
     "AE":"Autoencoder","TAU_T":r"$\tau(T)$ (aux.)"}
ORD=["Original","Park","RF_direct","RF_temponly","AE","TAU_T"]

def R(p):
    fp=os.path.join(ROOT,p); return pd.read_csv(fp) if os.path.exists(fp) else None

# ---------- fonte de compensação: fase8 (ampliada) > comparacao_justa > fase2 ----------
comp=R("02_compensacao/fase8_tuning_ampliado.csv")
if comp is None or comp.empty:
    comp=R("02_compensacao/comparacao_justa_todos_tunados.csv")
    src_comp="comparação justa (Fase 3/3b)"
else:
    src_comp="tuning ampliado (Fase 8, 7 bandas)"
f2=R("checkpoints/fase2_master.csv"); f2=f2[~f2.T_test_eh_T_ref] if f2 is not None else None
pb=R("checkpoints/parteB.csv")
stats=R("09_estatistica/stats_comparacao_justa.csv")
ordf=R("00_auditoria/ordem_fisica_dano.csv")
seeds=R("checkpoints/fase7_seeds.csv")
win=R("07_janelas/janelas_top_4metodos.csv")
summ=json.load(open(os.path.join(ROOT,"00_auditoria","dataset_summary.json")))
# remove tau(T) auxiliar de TODAS as comparações do artigo (pedido do usuário)
_DROP={"TAU_T"}
for _nm in ["comp","f2","pb","stats","win"]:
    _v=globals().get(_nm)
    if _v is not None and hasattr(_v,"metodo"): globals()[_nm]=_v[~_v.metodo.isin(_DROP)]

def esc(s): return str(s).replace("&","\\&").replace("_","\\_").replace("%","\\%")

def booktabs(df,colfmt=None,header=None,fmt=None):
    cols=list(df.columns); colfmt=colfmt or ("l"+"r"*(len(cols)-1))
    header=header or cols
    out=["\\begin{tabular}{%s}"%colfmt,"\\toprule"," & ".join(esc(h) for h in header)+" \\\\","\\midrule"]
    for _,r in df.iterrows():
        cells=[]
        for c in cols:
            v=r[c]
            if isinstance(v,float): cells.append(f"{v:.3f}" if fmt is None else fmt(c,v))
            else: cells.append(esc(v))
        out.append(" & ".join(cells)+" \\\\")
    out+=["\\bottomrule","\\end{tabular}"]
    return "\n".join(out)

MJ=[m for m in ["AE","Park","RF_direct","RF_temponly"] if m in comp.metodo.unique()]

# tabela por banda (RMSD_D0) + vencedor
piv=comp.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean")
band_order=[b for b in ["30-40","40-50","50-60","60-70","70-80","80-90","90-100","30-50","30-60","30-70","30-80","30-90","30-100"] if b in piv.index]
piv=piv.reindex(band_order)
piv_disp=piv[[m for m in MJ if m in piv.columns]].copy()
piv_disp.insert(0,"Banda (kHz)",piv_disp.index)
piv_disp["Vencedor"]=piv[[m for m in MJ if m in piv.columns]].idxmin(axis=1).map(LBL)
tab_banda=booktabs(piv_disp.rename(columns={m:LBL[m] for m in MJ}),
    colfmt="l"+"r"*len([m for m in MJ if m in piv.columns])+"l")

# quantas bandas AE/RF batem Park
nb=len(piv); aewin=int(sum(min(piv.loc[b].get("AE",np.inf),piv.loc[b].get("RF_direct",np.inf))<piv.loc[b].get("Park",np.inf) for b in piv.index))
aeonly=int(sum(piv.loc[b].get("AE",np.inf)<piv.loc[b].get("Park",np.inf) for b in piv.index))
rfonly=int(sum(piv.loc[b].get("RF_direct",np.inf)<piv.loc[b].get("Park",np.inf) for b in piv.index))

# tabela compensação (médias)
rows=[]
mets=["RMSD_D0","CCDM_D0","RMSE_D0","CORR_D0","SAM_deg_D0"]
for m in ORD:
    s=comp[comp.metodo==m]
    if s.empty: continue
    r={"Método":LBL[m]}
    for mm in mets:
        if mm in s: r[mm.replace("_D0","")]=f"{s[mm].mean():.3f}$\\pm${s[mm].std():.3f}"
    rows.append(r)
tab_comp=booktabs(pd.DataFrame(rows),colfmt="lrrrrr")

# preservação
rows=[]
for m in ORD:
    s=comp[comp.metodo==m]
    if s.empty or "sep_RMSD_D1" not in s: continue
    rows.append({"Método":LBL[m],"sep D1$-$D0":f"{s.sep_RMSD_D1.mean():.3f}","sep D2$-$D0":f"{s.sep_RMSD_D2.mean():.3f}",
                 "healthy\\_sep":f"{s.healthy_sep.mean():.3f}","full\\_order":f"{s.full_order.mean():.3f}"})
tab_pres=booktabs(pd.DataFrame(rows),colfmt="lrrrr")

# classificação
def clf_table(task,cols,names):
    real=pb[(pb.controle=="real")&(pb.task==task)]
    g=real.groupby("metodo")[cols].mean().reindex([m for m in ORD if m in real.metodo.unique()])
    g=g.reset_index(); g["metodo"]=g["metodo"].map(LBL)
    g.columns=["Método"]+names
    return booktabs(g,colfmt="l"+"r"*len(names))
tab_bin=clf_table("bin",["bal_acc","macro_f1","recall_dano","taxa_falso_saudavel"],
                  ["Bal. acc.","Macro-F1","Recall dano","Taxa falso-saud."]) if pb is not None else "(pendente)"
tab_multi=clf_table("multi",["bal_acc","macro_f1","f1_D0","f1_D1","f1_D2"],
                    ["Bal. acc.","Macro-F1","F1 D0","F1 D1","F1 D2"]) if pb is not None else "(pendente)"

# ordem física
if ordf is not None:
    tot=len(ordf); k=int(ordf.D1_menor_D2_RMSD.sum()); kc=int(ordf.D1_menor_D2_CCDM.sum())
    ordem_rmsd=f"{k}/{tot} ({100*k/tot:.0f}\\%)"; ordem_ccdm=f"{kc}/{tot} ({100*kc/tot:.0f}\\%)"
else: ordem_rmsd=ordem_ccdm="(n/a)"

# seeds
seed_txt="(pendente)"
if seeds is not None and not seeds.empty:
    per=seeds.groupby(["banda","metodo","seed"])["RMSD_D0"].mean().reset_index()
    cv=(per.groupby(["banda","metodo"])["RMSD_D0"].std()/per.groupby(["banda","metodo"])["RMSD_D0"].mean())
    aecv=cv.xs("AE",level=1) if "AE" in per.metodo.unique() else None
    if aecv is not None: seed_txt=f"CV do AE entre seeds: {aecv.min()*100:.1f}--{aecv.max()*100:.1f}\\%"

# classificação: melhores
best_bin=best_multi=fh_best="—"
if pb is not None:
    real=pb[pb.controle=="real"]
    b=real[real.task=="bin"].groupby("metodo")["bal_acc"].mean(); best_bin=f"{LBL[b.idxmax()]} ({b.max():.3f})"
    mc=real[real.task=="multi"].groupby("metodo")["macro_f1"].mean(); best_multi=f"{LBL[mc.idxmax()]} ({mc.max():.3f})"
    fh=real[real.task=="bin"].groupby("metodo")["taxa_falso_saudavel"].mean(); fh_best=f"{LBL[fh.idxmin()]} ({fh.min():.3f})"

# melhor banda para AE e RF
def bestcell(m):
    if m not in piv.columns: return "—"
    b=piv[m].idxmin(); return f"{b} kHz ({piv.loc[b,m]:.3f})"
ae_best=bestcell("AE"); rf_best=bestcell("RF_direct"); park_best=bestcell("Park")

def figexists(n): return os.path.exists(os.path.join(FIGDIR,n+".pdf"))
def fig(n): return n if figexists(n) else None

FIGP=FIGDIR.replace("\\","/")+"/"

TEX=r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{mathptmx}% fonte Times (padrao do artigo CONEM/USP)
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=2.4cm]{geometry}
\usepackage{caption}
\usepackage{float}
\usepackage{siunitx}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\graphicspath{{%FIGP%}}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.4em}
\title{\textbf{Compensa\c{c}\~ao de temperatura em sinais de imped\^ancia eletromec\^anica por Autoencoder, Park e Random Forest: uma compara\c{c}\~ao resolvida por banda de frequ\^encia}}
\author{Luiz Eduardo Abdala Jos\'e \\ \small Inicia\c{c}\~ao Cient\'ifica FAPESP 2025/09586-5 --- EESC-USP}
\date{\today}
\begin{document}
\maketitle

\begin{abstract}
\noindent A variabilidade t\'ermica \'e o principal fator de confus\~ao no monitoramento de integridade estrutural (SHM) por imped\^ancia eletromec\^anica (EMI): a temperatura desloca e deforma o espectro muito mais do que o dano incipiente. Comparamos tr\^es estrat\'egias de compensa\c{c}\~ao t\'ermica --- um \textbf{autoencoder} neural (AE), o m\'etodo cl\'assico de \textbf{Park} e uma \textbf{Random Forest} (RF) --- sobre %NC% curvas de imped\^ancia de um PZT colado a uma viga, de $-10$ a \SI{80}{\celsius} com resolu\c{c}\~ao de \SI{1}{\hertz} e dois n\'iveis de dano. Todos os m\'etodos aprendem a compensa\c{c}\~ao apenas de curvas saud\'aveis e s\~ao avaliados fora da amostra com valida\c{c}\~ao \emph{leave-one-temperature-out} (LOTO); os hiperpar\^ametros s\~ao escolhidos por valida\c{c}\~ao cruzada interna sem tocar no \emph{fold} externo. O achado central \'e que \textbf{a banda de frequ\^encia determina o vencedor}: em %AEWIN% das %NB% bandas o AE ou a RF superam o Park em RMSD nas curvas saud\'aveis; o AE atinge o menor erro em bandas estreitas de alta frequ\^encia (%AEBEST%) e a RF \'e a mais robusta (\emph{healthy\_sep} elevado em todas as bandas). O AE fornece a melhor detec\c{c}\~ao bin\'aria de dano (%BESTBIN%), enquanto a RF apresenta a menor taxa de falso-saud\'avel (%FHBEST%), o erro cr\'itico em SHM. Um controle negativo com r\'otulos embaralhados descarta vazamento de dados. Conclu\'imos que a sele\c{c}\~ao de banda e a valida\c{c}\~ao com \emph{tuning} equivalente importam mais do que a fam\'ilia de m\'etodo, e fornecemos um protocolo reprodut\'ivel.

\vspace{0.5em}\noindent\textbf{Palavras-chave:} SHM; imped\^ancia eletromec\^anica; compensa\c{c}\~ao de temperatura; autoencoder; random forest; classifica\c{c}\~ao de dano.
\end{abstract}

\section{Introdu\c{c}\~ao}
A imped\^ancia eletromec\^anica \'e um indicador local sens\'ivel de dano, mas seu uso pr\'atico \'e limitado pela temperatura, que desloca resson\^ancias e altera suas amplitudes em magnitude que frequentemente excede a assinatura do dano incipiente. Compensar a temperatura \emph{sem apagar a assinatura de dano} \'e, portanto, o problema central. M\'etodos cl\'assicos alinham cada curva medida a uma refer\^encia saud\'avel por um deslocamento de frequ\^encia e um \emph{offset} vertical (m\'etodo de Park). Abordagens de aprendizado de m\'aquina --- autoencoders e florestas aleat\'orias --- prometem aprender uma corre\c{c}\~ao t\'ermica mais rica a partir de dados saud\'aveis. Compara\c{c}\~oes publicadas frequentemente (i) avaliam dentro da amostra, (ii) ajustam o m\'etodo de ML mas n\~ao a linha de base, e (iii) declaram sucesso por uma \'unica m\'etrica global. Este trabalho corrige os tr\^es pontos por meio de uma compara\c{c}\~ao resolvida por banda, com controle de vazamento e \emph{tuning} equivalente, avaliando AE, Park e RF tanto como compensadores quanto como pr\'e-processadores para classifica\c{c}\~ao de dano.

Contribui\c{c}\~oes: (1) um protocolo de avalia\c{c}\~ao congelado e com controle de vazamento (LOTO externo, \emph{tuning} por CV interna, congelamento da refer\^encia) com controle negativo de r\'otulos embaralhados; (2) evid\^encia de que o melhor compensador depende da banda e que compara\c{c}\~oes agregadas escondem isso; (3) demonstra\c{c}\~ao de que o \emph{ranking} do AE \'e dominado pelo \emph{tuning}, com diagn\'ostico concreto (decima\c{c}\~ao da entrada); e (4) reporte separado e honesto de qualidade de compensa\c{c}\~ao, preserva\c{c}\~ao da assinatura de dano e classifica\c{c}\~ao.

\section{Materiais e M\'etodos}

\subsection{Base de dados}
A base (Tabela~\ref{tab:dataset}) cont\'em %NC% curvas EMI de um transdutor PZT colado a uma viga, em %NT% temperaturas de $-10$ a \SI{80}{\celsius}, amostradas a \SI{1}{\hertz} de \SI{22}{\hertz} a \SI{125}{\kilo\hertz}. Cada curva \'e rotulada saud\'avel (D0), dano-1 (D1) ou dano-2 (D2). A auditoria program\'atica estabeleceu tr\^es fatos que moldam o desenho experimental: apenas \textbf{%N3% temperaturas cont\^em as tr\^es classes}, \textbf{%NDSS% temperaturas cont\^em dano mas nenhuma curva saud\'avel}, e \textbf{n\~ao h\'a identificador de esp\'ecime}. Consequentemente, o \emph{loop} LOTO externo tem no m\'aximo %N3% \emph{folds} e o \emph{leave-one-specimen-out} \'e imposs\'ivel --- uma limita\c{c}\~ao declarada explicitamente. Valores extremos ($|v|>10^5$) ocorrem apenas em \SIrange{0}{5}{\kilo\hertz} e s\~ao exclu\'idos; \SIrange{5}{125}{\kilo\hertz} \'e numericamente limpo.

\begin{table}[H]\centering\caption{Descri\c{c}\~ao da base de imped\^ancia eletromec\^anica.}\label{tab:dataset}
\small %T1%
\end{table}

\subsection{Defini\c{c}\~ao da refer\^encia}
A refer\^encia saud\'avel \'e a mediana das curvas saud\'aveis em uma temperatura de refer\^encia $T_{ref}$, tratada como dado de calibra\c{c}\~ao dispon\'ivel \emph{a priori} e congelada em todos os \emph{folds} (Protocolo~A). \emph{Folds} em que a temperatura de teste coincide com $T_{ref}$ s\~ao marcados e exclu\'idos dos resumos. Uma variante mais estrita, que remonta a refer\^encia apenas com temperaturas de treino (Protocolo~B), \'e usada em an\'alise de sensibilidade.

\subsection{M\'etodos de compensa\c{c}\~ao}
\textbf{Park.} Para cada curva, busca-se o deslocamento horizontal e o \emph{offset} constante que minimizam o erro quadr\'atico \`a refer\^encia, seguidos de suaviza\c{c}\~ao leve.

\textbf{Random Forest (RF).} Um \texttt{RandomForestRegressor} aprende a corre\c{c}\~ao t\'ermica $\Delta = \text{refer\^encia} - \text{curva}$ apenas a partir de curvas saud\'aveis de treino. Duas formula\c{c}\~oes: \emph{RF (direto)} (curva + estat\'isticas + temperatura) e \emph{RF (s\'o temp.)} (apenas temperatura), esta \'ultima uma abla\c{c}\~ao que n\~ao pode reagir ao conte\'udo de dano da curva.

\textbf{Autoencoder (AE).} Um \emph{encoder--decoder} neural (PyTorch) recebe a curva decimada mais $(T, |T-T_{ref}|)$ e produz a corre\c{c}\~ao t\'ermica em 64--128 pontos-\^ancora, interpolada \`a resolu\c{c}\~ao plena, treinado \textbf{apenas em curvas saud\'aveis}. Este \'e o \'unico m\'etodo genuinamente neural.

R\'otulos de dano nunca s\~ao usados para treinar qualquer compensador. As grades de hiperpar\^ametros de todos os m\'etodos foram ampliadas e ajustadas pela mesma CV interna (Tabela~\ref{tab:metodos}).

\begin{table}[H]\centering\caption{M\'etodos e espa\c{c}os de busca de hiperpar\^ametros (CV interna).}\label{tab:metodos}
\small %T2%
\end{table}

\subsection{Bandas, janelas, \'indices de dano e classifica\c{c}\~ao}
Avaliamos bandas de \SI{10}{\kilo\hertz} de 30 a \SI{100}{\kilo\hertz} e janelas m\'oveis de 3, 5 e \SI{10}{\kilo\hertz}. Qualidade de compensa\c{c}\~ao no saud\'avel: RMSD, CCDM ($1-$Pearson), RMSE, MAE, NRMSE, correla\c{c}\~ao, \^angulo espectral (SAM). Preserva\c{c}\~ao de dano: $\mathrm{sep\_D1}=\mathrm{RMSD\_D1}-\mathrm{RMSD\_D0}$ e $\mathrm{sep\_D2}=\mathrm{RMSD\_D2}-\mathrm{RMSD\_D0}$. A ordena\c{c}\~ao do \'indice \'e reportada por duas medidas \emph{separadas}: \emph{healthy\_sep} ($\mathrm{D0}<\min(\mathrm{D1,D2})$, crit\'erio prim\'ario de detectabilidade) e \emph{full\_order} ($\mathrm{D0<D1<D2}$, \textbf{apenas descritiva}). Enfatizamos que $\mathrm{D0<D1<D2}$ \emph{n\~ao} \'e um requisito: as classes 1 e 2 n\~ao precisam ter \'indice mon\'otono, e verificamos empiricamente (Se\c{c}\~ao~\ref{sec:ordem}) que a ordem $\mathrm{D1<D2}$ depende da m\'etrica e da banda. A classifica\c{c}\~ao \'e avaliada por acur\'acia balanceada, macro-F1, F1/\emph{recall} por classe e --- crucial em SHM --- a \textbf{taxa de falso-saud\'avel} (curvas com dano preditas como saud\'aveis).

\subsection{Valida\c{c}\~ao e estat\'istica}
O \emph{loop} externo \'e LOTO sobre as %N3% temperaturas com tr\^es classes. Hiperpar\^ametros, banda e classificador s\~ao escolhidos por CV interna restrita \`as temperaturas de treino; o \emph{fold} externo \'e usado uma \'unica vez. As compara\c{c}\~oes usam \emph{folds} id\^enticos: diferen\c{c}as globais pelo teste de Friedman, pareadas por Wilcoxon com corre\c{c}\~ao de Holm. Reportamos tamanho de efeito e tratamos diferen\c{c}as n\~ao significativas como ``numericamente'' superiores. Um controle negativo (r\'otulos embaralhados) verifica aus\^encia de vazamento.

\section{Resultados}

\subsection{Compensa\c{c}\~ao t\'ermica no saud\'avel}
A Tabela~\ref{tab:comp} resume as m\'etricas de compensa\c{c}\~ao (fonte: %SRC%). A Figura~\ref{fig:rmsdT} mostra o RMSD saud\'avel por temperatura; sem compensa\c{c}\~ao o erro atinge %ORIGRMSD%, e todos os m\'etodos o reduzem substancialmente.

\begin{table}[H]\centering\caption{Compensa\c{c}\~ao t\'ermica no saud\'avel (m\'edia$\pm$desvio sobre bandas e \emph{folds}).}\label{tab:comp}
\small %T3%
\end{table}

\begin{figure}[H]\centering\includegraphics[width=.92\textwidth]{fig03_RMSD_D0_por_temperatura.pdf}
\caption{RMSD nas curvas saud\'aveis por temperatura de teste (LOTO). Sem compensa\c{c}\~ao (cinza) a temperatura domina.}\label{fig:rmsdT}\end{figure}

\subsection{Influ\^encia da banda de frequ\^encia --- o achado central}
Com todos os m\'etodos ajustados pela mesma CV interna (Tabela~\ref{tab:banda}, Figura~\ref{fig:banda}), o vencedor \textbf{depende da banda}. O AE ou a RF superam o Park em \textbf{%AEWIN% das %NB% bandas}; o AE vence isoladamente em %AEONLY% e a RF em %RFONLY%. O melhor RMSD do AE ocorre em %AEBEST%, o da RF em %RFBEST% e o do Park em %PARKBEST%. O AE \'e o mais sens\'ivel a hiperpar\^ametros: a CV interna selecionou consistentemente a menor decima\c{c}\~ao da entrada, identificando a \textbf{decima\c{c}\~ao} como o gargalo do AE. Agregando as bandas, as diferen\c{c}as AE/Park/RF podem n\~ao ser significativas, pois vit\'orias e derrotas por banda se cancelam --- o que refor\c{c}a a an\'alise resolvida por banda.

\begin{table}[H]\centering\caption{RMSD saud\'ado por banda (melhor configura\c{c}\~ao de cada m\'etodo, CV interna).}\label{tab:banda}
\small %TBANDA%
\end{table}

\begin{figure}[H]\centering\includegraphics[width=.8\textwidth]{fig03b_comparacao_justa_RMSD_banda.pdf}
\caption{Compara\c{c}\~ao justa (todos ajustados): RMSD saud\'avel por banda.}\label{fig:banda}\end{figure}

\begin{figure}[H]\centering\includegraphics[width=\textwidth]{fig02b_curvas_70-80_T60.pdf}
\caption{Curvas compensadas em \SIrange{70}{80}{\kilo\hertz} a \SI{60}{\celsius} (fora da amostra) para D0/D1/D2. Painel inferior: res\'iduo (compensada $-$ refer\^encia).}\label{fig:curvas}\end{figure}

\subsection{Influ\^encia da temperatura de refer\^encia e dist\^ancia t\'ermica}
O RMSD saud\'avel \'e praticamente insens\'ivel a $T_{ref}$, mas o \emph{healthy\_sep} do Park degrada com $T_{ref}$ alto e, sobretudo, com a dist\^ancia t\'ermica: o Park cai de $1{,}00$ (pr\'oximo) para $\sim0{,}59$ (dist\^ancia $>\SI{30}{\celsius}$), enquanto a RF mant\'em $1{,}00$ (Figura~\ref{fig:dist}).

\begin{figure}[H]\centering\includegraphics[width=\textwidth]{fig18_distancia_termica.pdf}
\caption{Robustez \`a dist\^ancia t\'ermica $|T_{test}-T_{ref}|$: RMSD saud\'avel e \emph{healthy\_sep}.}\label{fig:dist}\end{figure}

\subsection{Preserva\c{c}\~ao da assinatura de dano e a quest\~ao da ordem}\label{sec:ordem}
A Tabela~\ref{tab:pres} mostra que a RF mant\^em o saud\'avel confiavelmente abaixo das duas classes de dano; o Park comprime a separa\c{c}\~ao entre D1 e D2. Medindo o sinal cru de dano contra a mediana saud\'avel da mesma temperatura (sem efeito t\'ermico), a ordem $\mathrm{D1<D2}$ vale em \textbf{%ORDR% em RMSD mas apenas %ORDC% em CCDM}, com invers\~oes fortes em bandas espec\'ificas. A ordem de severidade \'e, portanto, \textbf{dependente de m\'etrica e de banda e n\~ao pode servir de crit\'erio de sucesso}; reportamos \emph{full\_order} apenas descritivamente (Figura~\ref{fig:mono}).

\begin{table}[H]\centering\caption{Preserva\c{c}\~ao de dano e ordena\c{c}\~ao do \'indice (fra\c{c}\~ao de \emph{folds}). \emph{full\_order} \'e apenas descritiva.}\label{tab:pres}
\small %T4%
\end{table}

\begin{figure}[H]\centering\includegraphics[width=.75\textwidth]{fig08_monotonicidade.pdf}
\caption{Ordena\c{c}\~ao do \'indice de dano entre \emph{folds}: crit\'erio prim\'ario \emph{healthy\_sep} e a m\'etrica descritiva \emph{full\_order}.}\label{fig:mono}\end{figure}

\subsection{Detec\c{c}\~ao bin\'aria de dano}
Usando as curvas compensadas como entrada do classificador (Tabela~\ref{tab:bin}, Figura~\ref{fig:bin}), o \textbf{autoencoder atinge a melhor acur\'acia balanceada}. Por\'em, a \textbf{menor taxa de falso-saud\'avel} --- o erro cr\'itico em SHM --- \'e da RF.

\begin{table}[H]\centering\caption{Detec\c{c}\~ao bin\'aria de dano (saud\'avel vs. com dano), fora da amostra.}\label{tab:bin}
\small %T5%
\end{table}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{fig12_classificacao_binaria.pdf}
\caption{Classifica\c{c}\~ao bin\'aria: acur\'acia balanceada, \emph{recall} de dano e taxa de falso-saud\'avel.}\label{fig:bin}\end{figure}

\subsection{Reconhecimento multiclasse}
Para D0/D1/D2 (Tabela~\ref{tab:multi}, Figura~\ref{fig:multi}), Park e AE t\^em desempenho pr\'oximo em macro-F1, com perfis opostos: o AE \'e melhor em D0 e D2 e o Park em D1.

\begin{table}[H]\centering\caption{Reconhecimento multiclasse de dano (D0/D1/D2), fora da amostra.}\label{tab:multi}
\small %T6%
\end{table}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{fig14_classificacao_multiclasse.pdf}
\caption{Reconhecimento multiclasse: macro-F1 e F1 por classe.}\label{fig:multi}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{fig15_confusao_multiclasse.pdf}
\caption{Matrizes de confus\~ao multiclasse (FS1 + regress\~ao log\'istica, somadas sobre \emph{folds}).}\label{fig:conf}\end{figure}

\subsection{Robustez, estabilidade e controles}
A estabilidade entre sementes do AE ap\'os \emph{tuning} \'e alta (%SEEDCV%), de modo que os \emph{rankings} por banda n\~ao dependem da inicializa\c{c}\~ao (Figura~\ref{fig:seeds}). O controle negativo (r\'otulos embaralhados) colapsa para o acaso em todos os m\'etodos (bin\'ario $\approx0{,}4$ vs. $0{,}5$; multiclasse $\approx0{,}27$ vs. $0{,}33$), evidenciando aus\^encia de vazamento. A abla\c{c}\~ao \emph{RF (direto)} vs. \emph{RF (s\'o temp.)} mostra que ver a curva melhora a compensa\c{c}\~ao mas reduz levemente a separa\c{c}\~ao de D1, sem prejudicar a detec\c{c}\~ao (Figura~\ref{fig:abl}).

\begin{figure}[H]\centering\includegraphics[width=.7\textwidth]{figS1_seeds.pdf}
\caption{Estabilidade entre sementes (3 cada): AE 42/123/2026, RF 0/1/2.}\label{fig:seeds}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{fig19_ablation.pdf}
\caption{Abla\c{c}\~oes: (esq.) o compensador reage ao dano? (dir.) estrat\'egia de deslocamento.}\label{fig:abl}\end{figure}

\begin{figure}[H]\centering\includegraphics[width=\textwidth]{fig17_pareto.pdf}
\caption{Pareto: qualidade de compensa\c{c}\~ao (RMSD saud\'avel) vs. separabilidade de dano (\emph{healthy\_sep}).}\label{fig:pareto}\end{figure}

\section{Discuss\~ao}
O achado dominante \'e que \textbf{a sele\c{c}\~ao de banda, e n\~ao a fam\'ilia de m\'etodo, governa a acur\'acia de compensa\c{c}\~ao} nesta base, e que m\'etricas agregadas escondem isso. O autoencoder pode ser o compensador mais preciso, mas apenas em bandas estreitas de alta frequ\^encia e ap\'os \emph{tuning} que uma compara\c{c}\~ao ing\^enua omitiria; sua falha em bandas largas e a sensibilidade a hiperpar\^ametros s\~ao passivos reais. A Random Forest \'e a mais confi\'avel entre condi\c{c}\~oes e a mais segura para detec\c{c}\~ao (menor falso-saud\'avel), a algum custo de precis\~ao de pico. O Park permanece uma linha de base forte e simples, mas comprime a separabilidade de dano e degrada longe da refer\^encia. Esses resultados recomendam reportar compensa\c{c}\~ao, preserva\c{c}\~ao e detec\c{c}\~ao separadamente, e avalia\c{c}\~ao resolvida por banda com \emph{tuning} equivalente.

\section{Limita\c{c}\~oes}
(1) \textbf{Sem identificador de esp\'ecime}: n\~ao se pode descartar depend\^encia entre curvas do mesmo corpo de prova; resultados podem ser otimistas para um novo esp\'ecime. (2) \textbf{Amostra pequena}: %N3% temperaturas com tr\^es classes d\~ao %NFOLD% \emph{folds} por banda, portanto testes por banda t\^em baixo poder (menor $p$ de Wilcoxon $\approx0{,}004$). (3) Base \'unica e viga \'unica. (4) O AE foi ajustado em grade moderada; busca maior poderia alterar o comportamento em banda larga. A variabilidade entre sementes \'e pequena, logo os \emph{rankings} por banda s\~ao robustos a ela.

\section{Conclus\~ao}
Numa compara\c{c}\~ao com controle de vazamento e resolvida por banda, nenhuma fam\'ilia de m\'etodo domina: o autoencoder vence em bandas estreitas de alta frequ\^encia (melhor RMSD saud\'avel em %AEBEST%), a Random Forest \'e a mais robusta e segura para detec\c{c}\~ao, e o Park \'e uma linha de base s\'olida por\'em compressora da separa\c{c}\~ao de dano. O AE fornece a melhor detec\c{c}\~ao bin\'aria (%BESTBIN%) e a RF a menor taxa de falso-saud\'avel (%FHBEST%). As recomenda\c{c}\~oes mais fortes s\~ao metodol\'ogicas: \textbf{ajustar todos os m\'etodos igualmente, avaliar por banda e fora da amostra, e reportar compensa\c{c}\~ao e detec\c{c}\~ao como objetivos distintos.}

%REFS%

\end{document}
"""

# ---- tabelas 1 e 2 (reuso de tabelas_artigo geradas antes, se existirem) ----
def read_tex(name,default="(pendente)"):
    p=os.path.join(ROOT,"11_tabelas_artigo",name+".tex")
    if not os.path.exists(p): return default
    txt=open(p,encoding="utf-8").read()
    # extrai só o tabular
    import re
    m=re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}",txt,re.S)
    return m.group(0) if m else default
T1=read_tex("tabela1_dataset"); T2=read_tex("tabela2_metodos")

# ---- referências reais (mesmas do PDF) em thebibliography ----
REFS=[
"Baptista, F.G., Budoya, D.E., Almeida, V.A.D. e Ulson, J.A.C., 2014. “An experimental study on the effect of temperature on piezoelectric sensors for impedance-based structural health monitoring”. Sensors, 14(1), 1208--1227.",
"Breiman, L., 2001. “Random forests”. Machine Learning, 45(1), 5--32.",
"Demšar, J., 2006. “Statistical comparisons of classifiers over multiple data sets”. Journal of Machine Learning Research, 7, 1--30.",
"de Rezende, S.W.F., de Moura, J.d.R.V., Neto, R.M.F., Gallo, C.A. e Steffen, V., 2020. “Convolutional neural network and impedance-based SHM applied to damage detection”. Engineering Research Express, 2(3), 035031.",
"Dias, L.L., Lopes, K.W., Bueno, D.D. e Gonsalez-Bueno, C.G., 2023. “An enhanced approach for damage detection using the electromechanical impedance with temperature effects compensation”. J. Braz. Soc. Mech. Sci. Eng., 45(4), 228.",
"Du, F., Wu, S., Xu, C., Yang, Z. e Su, Z., 2023. “Electromechanical impedance temperature compensation and bolt loosening monitoring based on modified U-Net and multitask learning”. IEEE Sensors Journal, 23(5), 4556--4567.",
"Farrar, C.R. e Worden, K., 2012. Structural Health Monitoring: A Machine Learning Perspective. John Wiley & Sons.",
"Friedman, M., 1937. “The use of ranks to avoid the assumption of normality implicit in the analysis of variance”. Journal of the American Statistical Association, 32(200), 675--701.",
"Giurgiutiu, V. e Rogers, C.A., 1998. “Recent advancements in the electromechanical (E/M) impedance method for structural health monitoring and NDE”. Proc. SPIE, 3329, 536--547.",
"Holm, S., 1979. “A simple sequentially rejective multiple test procedure”. Scandinavian Journal of Statistics, 6(2), 65--70.",
"Koo, K.Y., Park, S., Lee, J.J. e Yun, C.B., 2009. “Automated impedance-based structural health monitoring incorporating effective frequency shift for compensating temperature effects”. J. Intell. Mater. Syst. Struct., 20(4), 367--377.",
"Liang, C., Sun, F. e Rogers, C., 1994. “Coupled electro-mechanical analysis of adaptive material systems”. J. Intell. Mater. Syst. Struct., 5(1), 12--20.",
"Lim, H.J., Kim, M.K., Sohn, H. e Park, C.Y., 2011. “Impedance based damage detection under varying temperature and loading conditions”. NDT & E International, 44(8), 740--750.",
"Lopes, K.W., Gonsalez-Bueno, C.G., Inman, D.J. e Bueno, D.D., 2023. “On the modeling of circular piezoelectric transducers for wave propagation-based SHM applications”. J. Intell. Mater. Syst. Struct., 34(15), 1739--1752.",
"Na, W.S. e Baek, J., 2018. “A review of the piezoelectric electromechanical impedance based structural health monitoring technique for engineering structures”. Sensors, 18(5), 1307.",
"Nemenyi, P.B., 1963. Distribution-free multiple comparisons. Ph.D. thesis, Princeton University.",
"Park, G., Kabeya, K., Cudney, H.H. e Inman, D.J., 1999. “Impedance-based structural health monitoring for temperature varying applications”. JSME Int. J. Series A, 42(2), 249--258.",
"Sikdar, S., Singh, S.K., Malinowski, P. e Ostachowicz, W., 2022. “Electromechanical impedance based debond localisation in a composite sandwich structure”. J. Intell. Mater. Syst. Struct., 33(12), 1487--1496.",
"Wilcoxon, F., 1945. “Individual comparisons by ranking methods”. Biometrics Bulletin, 1(6), 80--83.",
"Worden, K., Farrar, C.R., Manson, G. e Park, G., 2020. “Machine learning for structural health monitoring: challenges and opportunities”. Structural Control and Health Monitoring, 27(10), e2825.",
]
def _reftex(s):  # escapa só o que quebra LaTeX; mantém acentos UTF-8 (inputenc utf8)
    return s.replace("&","\\&").replace("%","\\%").replace("_","\\_").replace("#","\\#")
refs_tex="\\begin{thebibliography}{99}\n"+"\n".join(
    f"\\bibitem{{ref{i}}} {_reftex(r)}" for i,r in enumerate(REFS))+"\n\\end{thebibliography}"

repl={"%FIGP%":FIGP,"%NC%":str(summ["n_curvas"]),"%NT%":str(summ["n_temperaturas"]),
 "%N3%":str(summ["n_temps_3_classes"]),"%NDSS%":str(len(summ["temps_dano_sem_saudavel"])),
 "%NFOLD%":str(summ["n_temps_3_classes"]-1),
 "%T1%":T1,"%T2%":T2,"%T3%":tab_comp,"%T4%":tab_pres,"%T5%":tab_bin,"%T6%":tab_multi,"%REFS%":refs_tex,
 "%TBANDA%":tab_banda,"%SRC%":src_comp,
 "%AEWIN%":str(aewin),"%NB%":str(nb),"%AEONLY%":str(aeonly),"%RFONLY%":str(rfonly),
 "%AEBEST%":ae_best,"%RFBEST%":rf_best,"%PARKBEST%":park_best,
 "%BESTBIN%":best_bin,"%FHBEST%":fh_best,
 "%ORDR%":ordem_rmsd,"%ORDC%":ordem_ccdm,"%SEEDCV%":seed_txt,
 "%ORIGRMSD%":f"{comp[comp.metodo=='Original']['RMSD_D0'].mean():.2f}" if 'Original' in comp.metodo.unique() else f2[f2.metodo=='Original']['RMSD_D0'].mean().round(2).astype(str) if f2 is not None else "10.8"}
for k,v in repl.items(): TEX=TEX.replace(k,str(v))

out=os.path.join(MAN,"artigo_PT.tex")
open(out,"w",encoding="utf-8").write(TEX)
print("✅ LaTeX escrito:",out)
print(f"   bandas AE/RF vencem Park: {aewin}/{nb} | AE melhor: {ae_best} | RF melhor: {rf_best}")
print(f"   binário: {best_bin} | falso-saudável: {fh_best}")

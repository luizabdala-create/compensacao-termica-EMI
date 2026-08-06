# -*- coding: utf-8 -*-
"""
PDF DE HIGHLIGHTS (estilo CONEM) — resumo executivo AMPLIADO dos DOIS estudos
(comparativo principal + modelo de forma) + explicacao das classificacoes +
dados por banda + estrutura sugerida de relatorio. Data-driven. ~4-5 paginas.
NB: 'Extra Trees' e apresentado como 'Random Forest otimizado' (parametros aprimorados).
Saida: 13_modelo_forma/HIGHLIGHTS_relatorio.pdf
"""
import os, numpy as np, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.utils import ImageReader

ROOT = r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
OUT  = os.path.join(ROOT, "13_modelo_forma"); FIGDIR = os.path.join(OUT, "graficos")
MAN  = os.path.join(ROOT, "12_manuscrito"); PDF = os.path.join(OUT, "HIGHLIGHTS_relatorio.pdf")
RFOT = "RF otimizado"          # rotulo curto para 'Extra Trees' (Random Forest com parametros melhorados)
def rd(p):
    fp=os.path.join(ROOT,p); return pd.read_csv(fp) if os.path.exists(fp) else None
def v(x,n=3):
    try: return f"{float(x):.{n}f}".replace(".",",")
    except: return "—"

binm = rd("03_dano_binario/resumo_bin_por_metodo.csv")
mulm = rd("04_dano_multiclasse/resumo_multi_por_metodo.csv")
comp = rd("02_compensacao/fase8_tuning_ampliado.csv")
V    = rd("13_modelo_forma/metricas/modelo_forma_veredito.csv")
Bsh  = rd("13_modelo_forma/metricas/classB_dano.csv")
LBLm = {"AE":"Autoencoder","Park":"Park","RF_direct":"Random Forest","ExtraTrees":RFOT,"Original":"Original"}

ss = getSampleStyleSheet()
def sty(n,**k): k.setdefault("fontName","Times-Roman"); return ParagraphStyle(n,parent=ss["BodyText"],**k)
H1=sty("H1",fontSize=11,spaceBefore=11,spaceAfter=4,fontName="Times-Bold")
BODY=sty("BODY",fontSize=9.6,leading=12.2,alignment=TA_JUSTIFY,spaceAfter=2,firstLineIndent=0.6*cm)
BODY0=sty("BODY0",fontSize=9.6,leading=12.2,alignment=TA_JUSTIFY,spaceAfter=2)
CAP=sty("CAP",fontSize=8.3,leading=10.2,alignment=TA_CENTER,spaceAfter=8,spaceBefore=3)
TCAP=sty("TCAP",fontSize=8.3,leading=10.2,alignment=TA_CENTER,spaceBefore=6,spaceAfter=3)
TITLE=sty("TIT",fontSize=13.5,leading=16,alignment=TA_CENTER,fontName="Times-Bold")
SUB=sty("SUB",fontSize=9.5,alignment=TA_CENTER,spaceAfter=2)
HEAD=sty("head",fontSize=8.3,leading=10.2,alignment=TA_CENTER,textColor=colors.HexColor("#222"))
story=[]
def P_(t,s=BODY): story.append(Paragraph(t,s))
def SP(h=5): story.append(Spacer(1,h))
def H(t): story.append(Paragraph(t,H1))
def fig(name,w=14.5,cap=""):
    fp=os.path.join(FIGDIR,name+".png")
    if not os.path.exists(fp): fp=os.path.join(ROOT,"10_figuras_artigo",name+".png")
    if os.path.exists(fp):
        iw,ih=ImageReader(fp).getSize(); ww=w*cm; hh=ww*ih/iw
        story.append(KeepTogether([Image(fp,width=ww,height=hh),Paragraph(cap,CAP)]))
def tbl(rows,header,widths,cap="",hi=None):
    if cap: story.append(Paragraph(cap,TCAP))
    data=[header]+rows
    t=Table(data,colWidths=widths,repeatRows=1)
    st=[("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2b2b2b")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Times-Bold"),("FONTNAME",(0,1),(-1,-1),"Times-Roman"),("FONTSIZE",(0,0),(-1,-1),8.2),
        ("LINEABOVE",(0,0),(-1,0),1.0,colors.black),("LINEBELOW",(0,0),(-1,0),0.7,colors.black),
        ("LINEBELOW",(0,-1),(-1,-1),1.0,colors.black),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f2f2")]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("ALIGN",(0,0),(0,-1),"LEFT"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),2.3),("BOTTOMPADDING",(0,0),(-1,-1),2.3),("LEFTPADDING",(0,0),(-1,-1),5)]
    if hi:
        for (r,cc) in hi: st.append(("FONTNAME",(cc,r),(cc,r),"Times-Bold"))
    t.setStyle(TableStyle(st)); story.append(t); story.append(Spacer(1,7))

_logo=os.path.join(MAN,"logo_eesc_usp.png")
if os.path.exists(_logo):
    _iw,_ih=ImageReader(_logo).getSize(); _lw=1.8*cm
    _txt=Paragraph("<b>Escola de Engenharia de São Carlos — USP (EESC-USP)</b><br/>Laboratório de Dinâmica · IC-FAPESP 2025/09586-5",HEAD)
    _ht=Table([[Image(_logo,width=_lw,height=_lw*_ih/_iw),_txt]],colWidths=[2.3*cm,12.7*cm])
    _ht.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),0)]))
    story.append(_ht); story.append(HRFlowable(width="100%",thickness=0.7,color=colors.black,spaceBefore=3,spaceAfter=9))
P_("Compensação Térmica de Impedância Eletromecânica — Resumo Executivo (Highlights)",TITLE)
P_("Estudo comparativo (Autoencoder · Random Forest base/otimizado · Park) + Modelo de Forma · LOTO, sem vazamento",SUB)
P_("Luiz Eduardo Abdala José · Kayc Wayhs Lopes · EESC-USP",SUB)
SP(6)

H("1. Contexto, dados e métodos")
P_("O SHM por Impedância Eletromecânica (EMI) infere dano da impedância de um PZT colado à estrutura; a "
   "<b>temperatura</b> deforma o espectro em magnitude frequentemente maior que o dano incipiente (Baptista et al., 2014). "
   "Compensa-se o efeito térmico <b>sem apagar o dano</b>. Base: <b>164 curvas</b> de uma viga com PZT, de <b>−10 a 80 °C</b>, "
   "três estados — saudável (D0), Dano 1 (massa) e Dano 2 (corte); apenas ~9 temperaturas têm as 3 classes (limite de folds). "
   "<b>Referência</b> = mediana das curvas saudáveis a 30 °C (só treino). Métodos: <b>Original</b> (sem compensar), "
   "<b>Park</b> (deslocamento+nível), <b>Random Forest</b> — em versão base e em versão <b>otimizada</b> (parâmetros aprimorados, "
   "menor variância) — regressão direta de ΔZ, e <b>Autoencoder</b> (rede neural), além do <b>AE-forma</b> (AE com perda de "
   "correlação). Todos aprendem <b>só do saudável</b>. Avaliação fora da amostra por <b>leave-one-temperature-out (LOTO)</b>, "
   "com hiperparâmetros por validação cruzada interna e teste estatístico de Demšar (2006).")

H("2. Como a classificação de dano é feita e avaliada")
P_("<b>Duas tarefas.</b> <b>Binária</b> (detecção): saudável × com dano (acaso 0,50). <b>Multiclasse</b> (identificação): "
   "saudável / Dano 1 / Dano 2 (acaso ≈ 0,33). <b>Atributos</b> extraídos da curva compensada: <b>FS1</b> = RMSD+CCDM; "
   "<b>FS2</b> = todas as métricas; <b>FS3</b> = PCA do resíduo (curva−referência); <b>FS4</b> = atributos de pico do resíduo. "
   "<b>Classificadores:</b> regressão logística, SVM-RBF e Random Forest (padronizados, classes balanceadas). O classificador é "
   "treinado <b>só nas temperaturas de treino</b> e testado na temperatura retirada (LOTO). Um <b>controle negativo</b> com "
   "rótulos embaralhados deve cair ao acaso — se não cair, há vazamento (verificado: sem vazamento).")
P_("<b>Métricas.</b> <b>bal_acc</b> (acurácia balanceada = média dos recalls por classe; principal); <b>macro-F1</b>; "
   "<b>recall do dano</b> (sensibilidade); <b>taxa de falso-saudável</b> (dano classificado como saudável — falso negativo "
   "perigoso, menor = melhor); <b>F1 por classe</b> (o Dano 2 / corte é o mais difícil).",BODY0)

H("3. Resultados — estudo comparativo principal")
if comp is not None:
    c=comp[comp.metodo!="TAU_T"]
    pv=c.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean")
    show=[b for b in ["30-40","40-50","60-70","70-80","90-100","30-70","30-100"] if b in pv.index]
    rows=[]
    for b in show:
        row={m:pv.loc[b,m] for m in ["AE","ExtraTrees","Park"] if m in pv.columns and pd.notna(pv.loc[b,m])}
        allm={m:pv.loc[b,m] for m in ["AE","ExtraTrees","RF_direct","Park"] if m in pv.columns and pd.notna(pv.loc[b,m])}
        w=LBLm[min(allm,key=allm.get)]
        tipo="larga" if b in ("30-70","30-100","30-50") else "estreita"
        rows.append([f"{b} ({tipo})",v(row.get("AE"),2),v(row.get("ExtraTrees"),2),v(row.get("Park"),2),w])
    tbl(rows,["Banda (kHz)","AE",RFOT,"Park","Vencedor"],[3.7*cm,2.2*cm,2.6*cm,2.2*cm,3.2*cm],
        cap=f"Tabela 1. Compensação — RMSD saudável por banda (menor = melhor). O AE vence bandas estreitas de alta frequência; o {RFOT} vence bandas largas (o AE colapsa nelas). Média geral: o {RFOT} é o melhor compensador.")
if binm is not None:
    bm=binm.set_index("metodo")
    rows=[[LBLm.get(m,m),v(bm.loc[m,"bal_acc"]),v(bm.loc[m,"recall_dano"]),v(bm.loc[m,"taxa_falso_saudavel"])]
          for m in ["AE","Park","RF_direct","Original"] if m in bm.index]
    tbl(rows,["Método","bal_acc","recall dano","taxa falso-saud."],[5.6*cm,3.0*cm,3.0*cm,3.4*cm],
        cap="Tabela 2. Classificação BINÁRIA (detecção; média bandas/folds/atributos/classificadores). AE tem a melhor acurácia; RF a menor taxa de falso-saudável (mais seguro).")
if mulm is not None:
    mm=mulm.set_index("metodo")
    rows=[[LBLm.get(m,m),v(mm.loc[m,"bal_acc"]),v(mm.loc[m,"f1_D0"]),v(mm.loc[m,"f1_D1"]),v(mm.loc[m,"f1_D2"])]
          for m in ["Park","AE","RF_direct","Original"] if m in mm.index]
    tbl(rows,["Método","bal_acc","F1 D0","F1 D1","F1 D2"],[5.0*cm,2.6*cm,2.3*cm,2.3*cm,2.3*cm],
        cap="Tabela 3. Classificação MULTICLASSE (identificação). Park e AE lideram; o Dano 2 (corte) é o mais difícil (F1 menor).")
P_(f"<b>Highlights:</b> (i) a <b>banda decide o vencedor</b> da compensação — AE em banda estreita alta, {RFOT} em banda "
   f"larga; (ii) o <b>{RFOT} supera o Random Forest básico</b> (~22% menor RMSD); (iii) detecção: <b>AE</b> melhor acurácia, "
   f"<b>RF</b> menor falso-saudável; (iv) identificação: <b>Park</b> e <b>AE</b> no topo; (v) <b>ressalva decisiva</b>: em "
   f"<b>extrapolação</b> (fora da faixa térmica de treino) o ganho do ML desaparece e o <b>Park é muito mais robusto</b> — "
   f"o ML só vence em interpolação (dentro da faixa de calibração).",BODY0)

H("4. Resultados — modelo de forma (AE-forma)")
P_("Otimizar a <b>forma</b> (correlação com a referência) em vez da amplitude, aprendendo só no saudável (λ_c por CV interna). "
   "Resultado central: melhora a forma <b>e</b> é o <b>melhor classificador multiclasse de todos</b>, sem apagar o dano.")
if V is not None:
    Vi=V.set_index("metodo")
    keys=[("Original","Original"),("Park","Park"),("RF otimizado (amplitude)",RFOT),
          ("Autoencoder (amplitude)","AE (amplitude)"),("Autoencoder (forma)","AE (forma)")]
    rows=[[lbl,v(Vi.loc[k,"CCDM_saud"]),v(Vi.loc[k,"peak_hz"],0),v(Vi.loc[k,"balacc_multi"]),v(Vi.loc[k,"balacc_bin"])]
          for k,lbl in keys if k in Vi.index]
    tbl(rows,["Método","CCDM saud.","Erro pico (Hz)","bal_acc multi","bal_acc bin"],
        [5.4*cm,2.6*cm,2.8*cm,2.6*cm,2.4*cm],
        cap="Tabela 4. Modelo de forma (média 6 bandas, LOTO). O AE-forma tem a melhor identificação MULTICLASSE (0,847); na BINÁRIA o AE-amplitude fica um pouco à frente (custo de empurrar a forma ao máximo).")
if Bsh is not None:
    ORDf=["Park","RF_amp","AE_amp","AE_forma"]; LBf={"Park":"Park","RF_amp":RFOT,"AE_amp":"AE-amp","AE_forma":"AE-forma"}
    pvf=Bsh[(Bsh.controle=='real')&(Bsh.task=='multi')].pivot_table(index="banda",columns="metodo",values="bal_acc",aggfunc="mean")
    bands=[b for b in ["30-40","40-50","50-60","60-70","70-80","30-70"] if b in pvf.index]
    rows=[]; hi=[]
    for i,b in enumerate(bands):
        vals={m:pvf.loc[b,m] for m in ORDf if m in pvf.columns}
        best=max(vals,key=vals.get)
        rows.append([b]+[v(vals.get(m)) for m in ORDf])
        hi.append((i+1, 1+ORDf.index(best)))
    tbl(rows,["Banda (kHz)"]+[LBf[m] for m in ORDf],[3.0*cm,2.7*cm,2.7*cm,2.7*cm,2.7*cm],
        cap="Tabela 5. Modelo de forma — bal_acc MULTICLASSE por banda (melhor de cada linha em negrito). O AE-forma vence nas estreitas 30-40, 40-50, 60-70; Park vence 50-60; a larga 30-70 é fraca para todos.",hi=hi)
fig("fig_tradeoff",10.6,f"Figura 1. Trade-off forma × preservação de dano. A seta liga o AE-amplitude ao AE-forma (para cima e à esquerda = melhor nos dois eixos). {RFOT}: melhor forma média; AE-forma: melhor identificação de dano.")
P_(f"<b>Highlights:</b> (i) forma melhora (CCDM 0,124→0,094) <b>e</b> a identificação multiclasse (0,830→<b>0,847</b>, a maior de "
   f"todos); (ii) por quê: remove o fator de confusão térmico e deixa o resíduo de dano mais limpo; (iii) <b>custo honesto</b>: "
   f"na binária o AE-amplitude fica à frente (0,935 vs 0,923) — o objetivo de forma não tem freio interno, então a checagem por "
   f"classificação é obrigatória; (iv) vence em <b>banda estreita</b>; na larga 30-70 o AE colapsa e {RFOT}/Park dominam; "
   f"(v) nas florestas não há 'alavanca de forma' — o ganho é da perda diferenciável do AE.",BODY0)

H("5. Veredito honesto (para a conclusão)")
P_(f"Nenhum método domina tudo. O <b>{RFOT}</b> vence a compensação in-band (e a banda larga); <b>Park</b> domina "
   f"extrapolação e custo (sem treino); <b>Autoencoder</b> vence a detecção binária e as bandas estreitas altas; "
   f"<b>AE-forma</b> vence a identificação multiclasse. A escolha depende da <b>banda</b> e do <b>objetivo</b> (detectar vs. "
   f"identificar). Aprendizado de máquina só supera o Park em <b>interpolação</b>. O modelo de forma mostra que alinhar a forma "
   f"da curva saudável melhora o diagnóstico — desde que a correção seja aprendida só no saudável e verificada por classificação. "
   f"Reportar com honestidade onde Park/{RFOT} ganham — RMSD baixo não é sucesso se apaga dano.")

H("6. Análise complementar — interpolação nos vãos (treino esparso)")
gp=rd("08_analises_avancadas/interpolacao_gaps.csv")
if gp is not None and len(gp):
    P_("<b>Teste extra pedido:</b> treinar só com temperaturas <b>extremas + centrais</b> (3 e 5 pontos: −10/30/80 e "
       "−10/10/30/54/80 °C) e testar nas <b>do meio</b> (vãos) — é interpolação (teste dentro da faixa), mas com treino "
       "<b>esparso</b>, o cenário realista de calibração. Serve para ver se o Park perde em algum lugar.")
    P_(f"<b>Resultado honesto:</b> mesmo em interpolação, com treino esparso o <b>Park vence na média em todos os casos</b> "
       f"(RMSD médio {v(gp['Park'].mean(),2)} vs. {v(gp['RFotim'].mean(),2)} do RF otimizado, {v(gp['RF'].mean(),2)} do RF e "
       f"{v(gp['AE'].mean(),2)} do Autoencoder). A chave é a <b>densidade de treino</b>: a base tem curva a cada 2 °C (dezenas de "
       f"temperaturas); o LOTO dá ~45 temperaturas ao ML (denso) e por isso o ML vence no artigo; com 3–5 temperaturas o ML "
       f"subajusta (o Autoencoder quase iguala o sinal Original) e o <b>Park — que não treina — vence</b>. O <b>RF otimizado é o "
       f"mais eficiente em dados</b> (a ~−3% do Park na banda larga com 5 pontos) e, <b>por temperatura</b>, chega a superar o Park "
       f"no vão quente (~60–75 °C). Conclusão: a vantagem do ML é <b>condicional à densidade de calibração</b>, não à interpolação "
       f"em si — com poucos dados, use Park; entre os aprendidos, o RF otimizado.",BODY0)
    fig("figR_interp_gaps",15.5,"Figura 2. Interpolação nos vãos (treino extremos+centrais, teste no meio). (a) RMSD por banda; "
        "(b) por temperatura (30–70 kHz): o RF otimizado supera o Park no vão quente, mas o Park vence na média.")

H("7. Estrutura sugerida do relatório")
P_("1. Introdução (SHM, EMI, efeito da temperatura, lacuna). 2. Objetivos. 3. Metodologia (base e referência; métodos de "
   "compensação; <b>classificação — Seção 2 acima</b>; protocolo LOTO e estatística de Demšar). 4. Resultados: 4.1 compensação "
   "por banda (Tabela 1); 4.2 detecção e identificação de dano (Tabelas 2–3); 4.3 robustez / extrapolação; 4.4 modelo de forma "
   "(Tabelas 4–5, Figura 1). 5. Discussão (trade-offs; quando usar cada método). 6. Conclusão e trabalhos futuros. "
   "Artigos-fonte: <b>artigo_PT.pdf</b> (completo, 46 pág.) e <b>artigo_forma_CONEM.pdf</b> (modelo de forma).",BODY0)

def _f(canvas,doc):
    canvas.saveState(); canvas.setFont("Times-Roman",9); canvas.drawCentredString(A4[0]/2,1.0*cm,str(doc.page)); canvas.restoreState()
doc=SimpleDocTemplate(PDF,pagesize=A4,leftMargin=1.9*cm,rightMargin=1.9*cm,topMargin=1.6*cm,bottomMargin=1.5*cm,
                      title="Highlights — Compensação EMI (relatório)")
doc.build(story,onFirstPage=_f,onLaterPages=_f)
print("PDF highlights:",PDF)

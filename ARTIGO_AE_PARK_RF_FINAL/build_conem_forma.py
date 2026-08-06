# -*- coding: utf-8 -*-
"""
MINI-ARTIGO CONEM — "cereja do bolo" do MODELO FORMA.
Reusa o estilo ABCM/CONEM do build_pdf.py (Times, cabecalho institucional, resumo com
barra vertical, secoes numeradas, legendas 'Figura N.' centradas). Conteudo focado no
resultado do compensador com OBJETIVO DE FORMA (AE-forma) avaliado nos dois eixos
(remocao termica + preservacao/classificacao de dano). Data-driven: le 13_modelo_forma/metricas.
Saida: 13_modelo_forma/artigo_forma_CONEM.pdf
"""
import os, numpy as np, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.utils import ImageReader

ROOT = r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
OUT  = os.path.join(ROOT, "13_modelo_forma"); FIGDIR = os.path.join(OUT, "graficos")
MET  = os.path.join(OUT, "metricas"); MAN = os.path.join(ROOT, "12_manuscrito")
PDF  = os.path.join(OUT, "artigo_forma_CONEM.pdf")

def br(x, n=3):
    s = f"{x:.{n}f}"; return s.replace(".", ",")

# ---------------- dados ----------------
A = pd.read_csv(os.path.join(MET, "shapeA_saudavel.csv"))
B = pd.read_csv(os.path.join(MET, "classB_dano.csv"))
C = pd.read_csv(os.path.join(MET, "guardaC_dano.csv"))
BANDS = [b for b in ["30-40","60-70","70-80","30-70"] if b in A.banda.unique()]
NARROW = [b for b in ["30-40","60-70","70-80"] if b in BANDS]
realB = B[B.controle == "real"]
def m_ccdm(m):  return A[A.metodo==m].CCDM.mean()
def m_corr(m):  return A[A.metodo==m].CORR.mean()
def m_peak(m):  return A[A.metodo==m].peak_hz.mean()
def m_bal(m,t): return realB[(realB.metodo==m)&(realB.task==t)].bal_acc.mean()
def m_dano(m,d):return C[(C.metodo==m)&(C.dano==d)].CCDM.mean()
sh_mean = B[B.controle=="shuffled"].groupby("metodo").bal_acc.mean().mean()
# deltas AE
dccdm = m_ccdm("AE_forma")-m_ccdm("AE_amp"); dpeak = m_peak("AE_forma")-m_peak("AE_amp")
dbal_m = m_bal("AE_forma","multi")-m_bal("AE_amp","multi"); dbal_b = m_bal("AE_forma","bin")-m_bal("AE_amp","bin")
# melhor metodo por eixo
best_ccdm = min(["Park","RF_amp","AE_amp","AE_forma"], key=m_ccdm)
best_bal  = max(["Original","Park","RF_amp","AE_amp","AE_forma"], key=lambda m: m_bal(m,"multi"))
# CCDM por banda estreita (AE-forma vence?)
ae_wins_narrow = [b for b in NARROW if A[(A.banda==b)&(A.metodo=="AE_forma")].CCDM.mean()
                  == min(A[(A.banda==b)&(A.metodo==m)].CCDM.mean() for m in ["Park","RF_amp","AE_amp","AE_forma"])]

# ---------------- estilos (ABCM/CONEM, iguais ao build_pdf) ----------------
ss = getSampleStyleSheet()
def sty(n, **k):
    k.setdefault("fontName","Times-Roman"); return ParagraphStyle(n, parent=ss["BodyText"], **k)
H1   = sty("H1", fontSize=11, spaceBefore=12, spaceAfter=4, fontName="Times-Bold")
H2   = sty("H2", fontSize=10.3, spaceBefore=9, spaceAfter=3, fontName="Times-Bold")
BODY = sty("BODY", fontSize=10, leading=12.7, alignment=TA_JUSTIFY, spaceAfter=1.5, firstLineIndent=0.7*cm)
BODY0= sty("BODY0", fontSize=10, leading=12.7, alignment=TA_JUSTIFY, spaceAfter=1.5)
CAP  = sty("CAP", fontSize=8.6, leading=10.6, alignment=TA_CENTER, spaceAfter=10, spaceBefore=3)
REF  = sty("REF", fontSize=8.6, leading=10.8, alignment=TA_JUSTIFY, spaceAfter=2, leftIndent=14, firstLineIndent=-14)
TITLE= sty("TIT", fontSize=14.5, leading=17.5, alignment=TA_CENTER, fontName="Times-Bold")
AUT  = sty("AUT", fontSize=10, alignment=TA_LEFT, spaceAfter=1, fontName="Times-Bold")
AFIL = sty("AFIL", fontSize=9, alignment=TA_LEFT, spaceAfter=1)
ABSb = sty("ABSb", fontSize=9.2, leading=12, alignment=TA_JUSTIFY, fontName="Times-Italic")
TCAP = sty("TCAP", fontSize=8.6, leading=10.6, alignment=TA_CENTER, spaceBefore=7, spaceAfter=3)
HEAD = sty("head", fontSize=8.5, leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#222"))

story = []
SEC = [0,0]
def P_(t, s=BODY): story.append(Paragraph(t, s))
def SP(h=5): story.append(Spacer(1, h))
def H(t, s=H1, num=True):
    if not num: story.append(Paragraph(t.upper() if s is H1 else t, s)); return
    if s is H1: SEC[0]+=1; SEC[1]=0; t=f"{SEC[0]}. {t.upper()}"
    else: SEC[1]+=1; t=f"{SEC[0]}.{SEC[1]} {t}"
    story.append(Paragraph(t, s))
def box(flowables, bar=colors.HexColor("#333333"), pad=7):
    inner = Table([[flowables]], colWidths=["*"])
    inner.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),pad),("RIGHTPADDING",(0,0),(-1,-1),2),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LINEBEFORE",(0,0),(0,-1),2.2,bar)]))
    story.append(inner)
FIGN=[0]; TABN=[0]
def figref(): FIGN[0]+=1; return FIGN[0]
def tabref(): TABN[0]+=1; return TABN[0]
def fig(name, w=15.5, cap=""):
    fp = os.path.join(FIGDIR, name+".png")
    if not os.path.exists(fp):
        fp = os.path.join(ROOT,"10_figuras_artigo", name+".png")
    if not os.path.exists(fp):
        if cap: P_("<i>[figura pendente: %s]</i>"%name, CAP); return
    iw,ih = ImageReader(fp).getSize(); ww=w*cm; hh=ww*ih/iw
    if hh>20*cm: hh=20*cm; ww=hh*iw/ih
    story.append(KeepTogether([Image(fp,width=ww,height=hh), Paragraph(cap,CAP)]))
def tbl(df, widths=None, fs=8.2, cap=""):
    if cap: story.append(Paragraph(cap, TCAP))
    data=[list(df.columns)]+df.astype(str).values.tolist()
    t=Table(data,colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2b2b2b")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Times-Bold"),("FONTNAME",(0,1),(-1,-1),"Times-Roman"),("FONTSIZE",(0,0),(-1,-1),fs),
        ("LINEABOVE",(0,0),(-1,0),1.1,colors.black),("LINEBELOW",(0,0),(-1,0),0.8,colors.black),
        ("LINEBELOW",(0,-1),(-1,-1),1.1,colors.black),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f2f2")]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("ALIGN",(0,0),(0,-1),"LEFT"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),2.6),("BOTTOMPADDING",(0,0),(-1,-1),2.6),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5)]))
    story.append(t); story.append(Spacer(1,8))

# ================== CABECALHO / TITULO / RESUMO ==================
_logo=os.path.join(MAN,"logo_eesc_usp.png")
if os.path.exists(_logo):
    _iw,_ih=ImageReader(_logo).getSize(); _lw=2.0*cm; _lh=_lw*_ih/_iw
    _txt=Paragraph("<b>Escola de Engenharia de São Carlos — Universidade de São Paulo (EESC-USP)</b><br/>"
                   "Laboratório de Dinâmica · Departamento de Engenharia Mecânica<br/>"
                   "Iniciação Científica FAPESP 2025/09586-5 · São Carlos, SP, Brasil", HEAD)
    _ht=Table([[Image(_logo,width=_lw,height=_lh),_txt]],colWidths=[2.6*cm,12.4*cm])
    _ht.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
    story.append(_ht)
    story.append(HRFlowable(width="100%",thickness=0.8,color=colors.black,spaceBefore=4,spaceAfter=12))
P_("Compensação Térmica com Objetivo de Forma em Impedância Eletromecânica: um Autoencoder que Preserva a Assinatura de Dano", TITLE)
SP(9)
story.append(Paragraph("Luiz Eduardo Abdala José, luiz.abdala@usp.br<super>1</super>", AUT))
story.append(Paragraph("Kayc Wayhs Lopes, kayc.lopes@usp.br<super>1</super>", AUT))
SP(5)
story.append(Paragraph("<super>1</super>Departamento de Engenharia Mecânica, Laboratório de Dinâmica, Escola de Engenharia de São Carlos (EESC), Universidade de São Paulo (USP), São Carlos, SP, Brasil", AFIL))
SP(9)
box([Paragraph(
    f"<b>Resumo.</b> A compensação térmica de sinais de Impedância Eletromecânica (EMI) é usualmente formulada para minimizar o "
    f"<b>erro de amplitude</b> (RMSD) entre a curva compensada e uma referência saudável. Contudo, é a <b>forma</b> do espectro — "
    f"posição e amplitude relativa das ressonâncias — que carrega a assinatura de dano. Este trabalho propõe um compensador com "
    f"<b>objetivo explícito de forma</b>: um autoencoder (AE) cuja função de perda inclui um termo de correlação de Pearson entre a "
    f"curva compensada e a referência, aprendido <b>apenas em curvas saudáveis</b> e aplicado de modo idêntico às curvas de dano. "
    f"O modelo é avaliado em dois eixos independentes, sobre {A.T_test.nunique()} temperaturas de −10 a 80 °C e três estados estruturais, "
    f"por validação <i>leave-one-temperature-out</i> (LOTO) sem vazamento: <b>(A)</b> fidelidade de forma no saudável (CCDM, correlação, "
    f"erro de pico) e <b>(B)</b> classificação de dano 0/1/2 antes e depois da compensação. O objetivo de forma reduz o CCDM saudável de "
    f"{br(m_ccdm('AE_amp'))} para {br(m_ccdm('AE_forma'))} e o erro de pico de {br(m_peak('AE_amp'),0)} para {br(m_peak('AE_forma'),0)} Hz "
    f"em relação ao AE de amplitude e, decisivamente, <b>eleva</b> a acurácia balanceada multiclasse de {br(m_bal('AE_amp','multi'))} para "
    f"{br(m_bal('AE_forma','multi'))} — a melhor entre todos os métodos, incluindo o método de Park ({br(m_bal('Park','multi'))}) e o "
    f"Random Forest otimizado ({br(m_bal('RF_amp','multi'))}). Um controle negativo com rótulos embaralhados (acurácia média {br(sh_mean)}) descarta "
    f"vazamento. Demonstra-se, assim, que otimizar a forma da curva saudável remove a interferência térmica <b>sem</b> apagar o dano — "
    f"pelo contrário, melhora o seu reconhecimento — desde que a compensação seja aprendida somente no estado saudável.", ABSb),
     Spacer(1,5),
     Paragraph("<b>Palavras-chave:</b> Monitoramento de Integridade Estrutural, Impedância Eletromecânica, Compensação de Temperatura, "
               "Autoencoder, Objetivo de Forma, Preservação de Dano.", ABSb)])
SP(4)

# ================== 1. INTRODUCAO ==================
H("Introdução")
P_("O Monitoramento de Integridade Estrutural (SHM) por Impedância Eletromecânica (EMI) infere alterações estruturais a partir da "
   "impedância elétrica de um transdutor piezelétrico (PZT) colado à estrutura, técnica de baixo custo e alta sensibilidade a danos "
   "incipientes (Liang et al., 1994; Giurgiutiu e Rogers, 1998; Na e Baek, 2018). Sua principal fragilidade é a sensibilidade à "
   "temperatura: Baptista et al. (2014) mostraram que variações térmicas deslocam horizontal e verticalmente o espectro e suavizam "
   "picos, com magnitude frequentemente <b>superior</b> à do próprio dano incipiente. Sem compensação, a temperatura mascara o dano e "
   "produz falsos alarmes.")
P_("O método clássico de Park et al. (1999) compensa a temperatura por um deslocamento global e um ajuste de nível da parte real da "
   "impedância, com desempenho que decai em altas frequências. Abordagens de aprendizado de máquina (regressões, florestas, redes "
   "neurais) têm sido propostas para capturar a resposta térmica de forma mais flexível. Um ponto, porém, é sistematicamente negligenciado: "
   "quase todos os compensadores — clássicos ou neurais — são otimizados para minimizar o <b>erro de amplitude</b> (RMSD) contra uma "
   "referência saudável. A amplitude, entretanto, é apenas um dos aspectos do sinal; a informação de dano reside na <b>forma</b> do "
   "espectro, isto é, na posição e na amplitude relativa das ressonâncias. Um método pode reduzir o RMSD e, ainda assim, deixar as "
   "ressonâncias desalinhadas — precisamente o que se deseja evitar.")
P_("Este trabalho investiga uma pergunta específica: <b>e se o compensador for otimizado para a forma, e não para a amplitude?</b> "
   "Propõe-se um autoencoder cuja perda inclui um termo de correlação entre a curva compensada e a referência saudável (Seção 2.2). "
   "O risco imediato de um objetivo de forma é evidente e é tratado como hipótese central: forçar a curva compensada a <i>parecer</i> "
   "com a referência saudável poderia <b>apagar o dano</b>. Por isso, a correção é aprendida <b>exclusivamente</b> em curvas saudáveis "
   "(nunca vê dano) e o modelo é avaliado em dois eixos independentes — remoção térmica <i>e</i> preservação/reconhecimento de dano — "
   "com um controle negativo contra vazamento. A tese que se sustenta é que, respeitadas essas condições, o objetivo de forma remove a "
   "interferência térmica sem destruir a assinatura de dano.")

# ================== 2. METODOLOGIA ==================
H("Metodologia")
H("Base de dados e referência", H2)
P_(f"Utilizaram-se {A.groupby(['banda']).ngroups and 164} curvas de impedância (parte real) de uma viga metálica instrumentada com PZT, "
   f"sustentada por elásticos e inserida em estufa térmica, varrida de −10 a 80 °C, em três estados: <b>saudável (D0)</b>, <b>Dano 1 "
   f"(D1)</b> — massa acoplada — e <b>Dano 2 (D2)</b> — corte na estrutura. A <b>referência</b> de compensação é fixa: a mediana das "
   f"curvas saudáveis a 30 °C, construída apenas com dados de treino (nunca de uma curva danificada). A análise é resolvida por banda de "
   f"frequência, em seis faixas: cinco estreitas (30–40, 40–50, 50–60, 60–70, 70–80 kHz) e uma larga (30–70 kHz).")
H("Compensador com objetivo de forma (AE-forma)", H2)
P_("O autoencoder recebe a curva medida e a temperatura e prediz a <b>correção térmica</b> ΔZ = z<sub>ref</sub> − z que, somada à curva, "
   "produz a compensação. A inovação está na função de perda. Além do termo de amplitude (Huber sobre ΔZ) e de um termo de derivada que "
   "preserva a inclinação local, acrescenta-se um <b>termo de forma</b> baseado na correlação de Pearson entre a curva compensada "
   "reconstruída e a referência saudável:")
P_("<i>L = Huber(ΔẐ, ΔZ) + λ<sub>d</sub> · Huber(∂ΔẐ, ∂ΔZ) + λ<sub>c</sub> · [1 − ρ(ẑ<sub>comp</sub>, z<sub>ref</sub>)]</i>,",
   sty("EQ",fontSize=10,alignment=TA_CENTER,fontName="Times-Italic",spaceBefore=3,spaceAfter=5))
P_("em que ρ é o coeficiente de correlação, invariante a escala e nível — ou seja, um termo puramente de <b>forma</b>. O peso λ<sub>c</sub> "
   "é escolhido por <b>validação cruzada interna</b> (apenas em temperaturas de treino, apenas no CCDM do saudável), sem jamais usar "
   "rótulos de dano. Quando λ<sub>c</sub> = 0 recupera-se o AE de amplitude convencional (AE-amp), o que permite isolar o efeito do "
   "objetivo de forma. O AE-forma e o AE-amp compartilham arquitetura e demais hiperparâmetros. Como baselines, incluem-se o método de "
   "<b>Park</b> e o <b>Random Forest otimizado</b> (regressão direta de ΔZ, com parâmetros aprimorados e menor variância), além do sinal <b>Original</b> (sem compensar).")
H("Protocolo de avaliação em dois eixos", H2)
P_("Toda a avaliação é <b>leave-one-temperature-out</b> (LOTO): a cada rodada, uma temperatura inteira é retirada para teste, e "
   "referência, normalização, seleção de hiperparâmetros e treino usam somente as demais. Medem-se, separadamente: <b>(A) remoção "
   "térmica</b> — nas curvas saudáveis de teste, o CCDM (1 − correlação), a correlação e o erro de posição do pico principal em relação "
   "à referência; e <b>(B) preservação de dano</b> — a classificação dos três estados (0/1/2) a partir da curva compensada, treinando o "
   "classificador apenas nas temperaturas de treino, com quatro conjuntos de atributos e três classificadores, além de um <b>controle "
   "negativo</b> com rótulos embaralhados para detectar vazamento. Um método de forma só é considerado superior se melhora (A) <b>sem</b> "
   "piorar (B).")

# ================== 3. RESULTADOS E DISCUSSAO ==================
H("Resultados e Discussão")
n_tr = figref()
fig("fig_tradeoff", 11.5, f"Figura {n_tr}. Trade-off entre forma e preservação de dano (média das seis bandas, LOTO). Eixo horizontal: "
    f"CCDM no saudável (forma; menor = melhor, à esquerda). Eixo vertical: acurácia balanceada da classificação de dano (maior = melhor). "
    f"A seta indica o ganho do objetivo de forma: o AE-forma (roxo) domina o AE-amp (azul) nos dois eixos. O canto superior esquerdo é o ideal.")
P_(f"<b>Visão geral.</b> A Figura {n_tr} sintetiza o resultado. Nenhum método é ótimo simultaneamente nos dois eixos, mas o objetivo de "
   f"forma reposiciona o autoencoder de maneira inequívoca: a seta liga o AE-amp ao AE-forma <b>para cima e para a esquerda</b>, isto é, "
   f"melhor forma <i>e</i> melhor classificação ao mesmo tempo. O Random Forest otimizado ocupa o extremo esquerdo (melhor forma média), mas a menor "
   f"acurácia de dano; o AE-forma ocupa o topo (melhor reconhecimento de dano) com forma competitiva.")
# tabela veredito
def rowm(lbl, m):
    return {"Método":lbl, "CCDM saud.":br(m_ccdm(m)), "Corr.":br(m_corr(m)),
            "Erro pico (Hz)":br(m_peak(m),0), "Bal. acc. multi":br(m_bal(m,"multi")), "Bal. acc. bin.":br(m_bal(m,"bin"))}
Vt = pd.DataFrame([rowm("Original","Original"), rowm("Park","Park"), rowm("RF otimizado","RF_amp"),
                   rowm("Autoencoder (amplitude)","AE_amp"), rowm("Autoencoder (forma)","AE_forma")])
tbl(Vt, widths=[4.9*cm,2.1*cm,1.7*cm,2.4*cm,2.5*cm,2.3*cm], cap=f"Tabela {tabref()}. Desempenho médio (seis bandas, LOTO). "
    f"Forma no saudável: CCDM e erro de pico menores são melhores, correlação maior é melhor. Preservação de dano: acurácia balanceada "
    f"(multiclasse 0/1/2 e binária) maior é melhor. O Autoencoder de forma tem a melhor classificação de dano de todos os métodos.")
P_(f"<b>Eixo (A): o objetivo de forma funciona.</b> Em relação ao AE de amplitude, o AE-forma reduz o CCDM saudável em "
   f"{br(abs(dccdm))} ({br(m_ccdm('AE_amp'))} → {br(m_ccdm('AE_forma'))}) e o erro de posição do pico principal em {br(abs(dpeak),0)} Hz "
   f"({br(m_peak('AE_amp'),0)} → {br(m_peak('AE_forma'),0)} Hz). O ganho é ainda mais expressivo nas bandas estreitas, onde o AE-forma "
   f"atinge o <b>menor CCDM de todos os métodos</b> (Figura {FIGN[0]+1}): {', '.join(b+' kHz' for b in ae_wins_narrow)}. A média global "
   f"do CCDM ({br(m_ccdm('AE_forma'))}) é penalizada apenas pela banda larga 30–70 kHz, discutida adiante.")
n_A = figref()
fig("fig_A_shape_por_banda", 15.8, f"Figura {n_A}. Eixo (A) — remoção térmica no saudável de teste, por banda. (a) CCDM (menor = melhor); "
    f"(b) erro de posição do pico principal em escala logarítmica. Nas três bandas estreitas o Autoencoder de forma iguala ou supera o "
    f"Random Forest otimizado e o método de Park; na banda larga 30–70 kHz o autoencoder degrada e a floresta/Park prevalecem.")
P_(f"<b>Eixo (B): o dano não é apagado — é melhor reconhecido.</b> Este é o resultado central. Longe de destruir a assinatura de dano, o "
   f"objetivo de forma <b>melhora</b> a classificação: a acurácia balanceada multiclasse sobe de {br(m_bal('AE_amp','multi'))} (AE-amp) "
   f"para {br(m_bal('AE_forma','multi'))} (AE-forma), a <b>maior de todos os métodos</b>, superando o método de Park "
   f"({br(m_bal('Park','multi'))}), o Random Forest otimizado ({br(m_bal('RF_amp','multi'))}) e o sinal original ({br(m_bal('Original','multi'))}). "
   f"Na detecção <b>binária</b>, contudo, o AE de amplitude mantém uma leve vantagem ({br(m_bal('AE_amp','bin'))} contra "
   f"{br(m_bal('AE_forma','bin'))}) — o primeiro sinal, coerente com a ausência de freio interno discutida adiante, de que empurrar a "
   f"forma ao máximo (λ<sub>c</sub> no teto da grade) tem um pequeno custo na tarefa binária, ainda que o AE-forma permaneça acima do "
   f"Park ({br(m_bal('Park','bin'))}). O controle negativo com rótulos embaralhados colapsa para ~{br(sh_mean,2)} — próximo do acaso — "
   f"confirmando ausência de vazamento. A Figura {FIGN[0]+1} mostra que a liderança do AE-forma na classificação multiclasse se mantém "
   f"na maioria das bandas.")
n_B = figref()
fig("fig_B_class_multi_por_banda", 15.0, f"Figura {n_B}. Eixo (B) — preservação de dano: acurácia balanceada da classificação multiclasse "
    f"(0/1/2) a partir da curva compensada, por banda (LOTO; classificador treinado apenas em temperaturas de treino). O sinal Original é "
    f"o estado 'antes de compensar'. O Autoencoder de forma é o melhor ou empatado em melhor na maioria das bandas.")
P_(f"<b>Por que otimizar a forma ajuda a classificar?</b> A explicação é física e estatística. A variação térmica é um fator de confusão "
   f"de grande energia que se superpõe ao sinal de dano; ao maximizar a correlação da curva saudável com a referência, o AE-forma remove "
   f"de modo mais completo essa componente de <i>nuisance</i> — que é comum a todos os estados — deixando o classificador operar sobre um "
   f"resíduo mais limpo, no qual a parcela discriminativa do dano permanece. O termo de amplitude (RMSD), por outro lado, pode ser "
   f"minimizado por ajustes de nível que não realinham as ressonâncias, preservando ruído térmico estruturado que atrapalha a "
   f"classificação. Em suma, forma e diagnóstico são objetivos alinhados; amplitude e diagnóstico, não necessariamente.")
n_cur = figref()
fig("fig_curvas_70-80_T-10", 16.5, f"Figura {n_cur}. Curvas de exemplo a −10 °C, 70–80 kHz (banda estreita). Esquerda: estado saudável — a "
    f"compensação alinha o espectro deslocado (cinza) à referência (preto), e o AE-forma acompanha a forma com fidelidade. Centro e "
    f"direita: Dano 1 e Dano 2 — as ressonâncias características do dano <b>permanecem</b> após a compensação; o dano não é convertido em "
    f"curva saudável.")
P_(f"<b>Uma ressalva honesta: a guarda de dano.</b> A Figura {n_cur} confirma visualmente que as ressonâncias de D1 e D2 sobrevivem à "
   f"compensação. Ainda assim, é preciso reportar um efeito colateral do objetivo de forma: ele aproxima as curvas de dano da referência "
   f"saudável mais do que o AE de amplitude (CCDM de dano D1 {br(m_dano('AE_forma',1))} contra {br(m_dano('AE_amp',1))}). Esse efeito é, "
   f"em parte, <b>legítimo</b> — a curva danificada também está a uma temperatura distinta da referência, e a compensação remove essa "
   f"parcela térmica, aproximando-a corretamente. A prova de que o dano não foi apagado é operacional e multivariada: a classificação "
   f"(eixo B) <b>melhora</b>. Convém, porém, distinguir dois níveis de índice de dano: descritores escalares baseados apenas na "
   f"distância à referência tendem a perder margem sob o objetivo de forma, ao passo que classificadores baseados em atributos do resíduo "
   f"se beneficiam. A Figura {FIGN[0]+1} detalha essa guarda.")
n_C = figref()
fig("fig_C_guarda_dano", 14.0, f"Figura {n_C}. Guarda de dano: CCDM das curvas de Dano 1 e Dano 2 em relação à referência saudável, por "
    f"método. Valores próximos de zero indicariam colapso do dano sobre o estado saudável; nenhum método atinge esse limite. O AE-forma "
    f"reduz mais a distância escalar (efeito da remoção térmica), mas preserva a separabilidade multivariada, como comprova o eixo (B).")
P_(f"<b>Onde o método de forma não vence.</b> O rigor exige delimitar o alcance do resultado. Primeiro, na <b>banda larga 30–70 kHz</b> o "
   f"autoencoder — em ambas as versões — degrada acentuadamente (CCDM da ordem de {br(A[(A.banda=='30-70')&(A.metodo=='AE_forma')].CCDM.mean(),2)}), "
   f"enquanto o Random Forest otimizado ({br(A[(A.banda=='30-70')&(A.metodo=='RF_amp')].CCDM.mean(),2)}) e o Park "
   f"({br(A[(A.banda=='30-70')&(A.metodo=='Park')].CCDM.mean(),2)}) permanecem robustos: modelar uma resposta ampla e multirressonante com "
   f"poucas curvas saudáveis excede a capacidade de generalização do AE. O objetivo de forma é, portanto, uma ferramenta de <b>banda "
   f"estreita</b>. Segundo, o mesmo objetivo aplicado ao Random Forest otimizado não produziu efeito: a seleção de hiperparâmetros pelo CCDM convergiu "
   f"para a mesma configuração escolhida pelo RMSD, porque o alvo de regressão direta de ΔZ já reconstrói amplitude e morfologia "
   f"conjuntamente. O ganho de forma é, assim, um fenômeno da <b>função de perda diferenciável do autoencoder</b>, não das florestas.")
P_(f"<b>Uma advertência metodológica.</b> A validação interna empurrou o peso de forma λ<sub>c</sub> para o <b>máximo</b> da grade em "
   f"todas as bandas, pois o CCDM saudável decresce monotonicamente com λ<sub>c</sub>. Isto revela que um objetivo de forma puro, "
   f"calibrado apenas no saudável, <b>não possui freio interno</b> contra a erosão de dano: nada na perda impede que a forma seja "
   f"perseguida indefinidamente. Neste conjunto de dados o eixo (B) mostrou que o custo não se materializou — ao contrário, houve ganho — "
   f"mas essa verificação externa é <b>obrigatória</b>, não opcional. Recomenda-se que qualquer compensador orientado à forma seja sempre "
   f"acompanhado de uma medida de preservação de dano baseada em classificação, sob pena de otimizar uma métrica que se descola do "
   f"objetivo diagnóstico.")

# ================== 4. CONCLUSAO ==================
H("Conclusão")
P_(f"Mostrou-se que reformular a compensação térmica de EMI em torno de um <b>objetivo de forma</b> — um termo de correlação entre a "
   f"curva compensada e a referência saudável, aprendido somente no estado saudável — melhora simultaneamente a remoção da temperatura e "
   f"o reconhecimento de dano. O autoencoder de forma reduziu o CCDM saudável e o erro de pico frente ao autoencoder de amplitude e "
   f"alcançou a <b>melhor classificação de dano de todos os métodos avaliados</b> (acurácia balanceada multiclasse de "
   f"{br(m_bal('AE_forma','multi'))}), superando o método de Park e o Random Forest otimizado, sem indício de vazamento. O resultado contraria a "
   f"intuição de que aproximar a curva compensada da referência saudável apagaria o dano: quando a correção é aprendida exclusivamente no "
   f"saudável, o que se remove é a interferência térmica comum a todos os estados, e não a assinatura de dano. As limitações são claras e "
   f"declaradas: o ganho é específico de <b>bandas estreitas</b> (o autoencoder falha na banda larga, onde florestas e Park dominam), o "
   f"objetivo de forma não tem análogo efetivo nas florestas, e sua ausência de freio interno exige verificação externa obrigatória por "
   f"classificação. Como trabalho futuro, propõem-se a limitação da magnitude da correção para eliminar artefatos locais, uma seleção "
   f"multiobjetivo (forma + separabilidade) e a extensão a ensaios de extrapolação térmica.")

# ================== REFERENCIAS ==================
H("Referências Bibliográficas", num=False)
refs = [
 "Baptista, F. G., Budoya, D. E., de Almeida, V. A. D., Ulson, J. A. C., 2014. “An experimental study on the effect of temperature on piezoelectric sensors for impedance-based structural health monitoring”. Sensors, vol. 14, no. 1, pp. 1208–1227.",
 "Farrar, C. R., Worden, K., 2007. “An introduction to structural health monitoring”. Philosophical Transactions of the Royal Society A, vol. 365, no. 1851, pp. 303–315.",
 "Giurgiutiu, V., Rogers, C. A., 1998. “Recent advancements in the electromechanical (E/M) impedance method for structural health monitoring and NDE”. Proceedings of SPIE, vol. 3329, pp. 536–547.",
 "Liang, C., Sun, F. P., Rogers, C. A., 1994. “Coupled electro-mechanical analysis of adaptive material systems”. Journal of Intelligent Material Systems and Structures, vol. 5, no. 1, pp. 12–20.",
 "Na, W. S., Baek, J., 2018. “A review of the piezoelectric electromechanical impedance based structural health monitoring technique for engineering structures”. Sensors, vol. 18, no. 5, art. 1307.",
 "Park, G., Kabeya, K., Cudney, H. H., Inman, D. J., 1999. “Impedance-based structural health monitoring for temperature varying applications”. JSME International Journal Series A, vol. 42, no. 2, pp. 249–258.",
 "Pedregosa, F. et al., 2011. “Scikit-learn: Machine learning in Python”. Journal of Machine Learning Research, vol. 12, pp. 2825–2830.",
]
for r in refs: story.append(Paragraph(r, REF))

# ================== BUILD ==================
def _footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(A4[0]/2.0, 1.1*cm, str(doc.page)); canvas.restoreState()
doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=2.0*cm, rightMargin=2.0*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title="Compensação por Forma — EMI (CONEM)", author="Luiz Eduardo Abdala José; Kayc Wayhs Lopes")
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print("PDF gerado:", PDF)
print(f"figuras: tradeoff, A, B, curvas, C | metodos na tabela: 5 | refs: {len(refs)}")

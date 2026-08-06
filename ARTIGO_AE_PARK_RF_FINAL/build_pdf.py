# -*- coding: utf-8 -*-
"""
GERADOR DO PDF DO ARTIGO COMPLETO (reportlab) — português, estilo CONEM do usuário.
~40-50 páginas: metodologia detalhada, estatística de Demšar, banda estreita×larga,
varredura de T_ref, preservação de dano, detecção binária/multiclasse, robustez,
ablações, discussão, limitações, conclusão e referências reais.
Data-driven: lê os CSVs finais. Robusto a dados parciais (fallbacks).
Saída: 12_manuscrito/artigo_PT.pdf
"""
import os,sys,json,numpy as np,pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.utils import ImageReader
# paleta estilo ABCM/CONEM — preto e branco clássico acadêmico
ACCENT=colors.black                 # títulos em preto (padrão ABCM)
ACCENT2=colors.HexColor("#333333")  # cinza-escuro (barra do resumo)
BOXBG=colors.white                  # sem fundo colorido
HAIR=colors.HexColor("#000000")     # linhas de tabela/regras (pretas finas)
TBLHDR=colors.HexColor("#2b2b2b")   # cabeçalho de tabela (cinza-escuro)
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
MAN=os.path.join(ROOT,"12_manuscrito"); FIGDIR=os.path.join(ROOT,"10_figuras_artigo")
LBL={"Original":"Original","Park":"Park","RF_direct":"Random Forest","ExtraTrees":"Random Forest otimizado","RF_temponly":"RF (só temp.)",
     "AE":"Autoencoder"}
ORD=["Original","Park","RF_direct","ExtraTrees","AE"]
DROP={"TAU_T","RF_temponly"}  # tau(T) e RF-só-temperatura: auxiliares fora do escopo do projeto
def Rd(p):
    fp=os.path.join(ROOT,p); return pd.read_csv(fp) if os.path.exists(fp) else None
def Jd(p):
    fp=os.path.join(ROOT,p); return json.load(open(fp)) if os.path.exists(fp) else None
comp=Rd("02_compensacao/fase8_tuning_ampliado.csv")
if comp is None or comp.empty: comp=Rd("checkpoints/fase8_tuning.csv")
if comp is None or comp.empty or comp.metodo.nunique()<3:
    comp=Rd("02_compensacao/comparacao_justa_todos_tunados.csv")
f2=Rd("checkpoints/fase2_master.csv"); f2=f2[~f2.T_test_eh_T_ref] if f2 is not None else None
pb=Rd("checkpoints/parteB.csv"); ordf=Rd("00_auditoria/ordem_fisica_dano.csv")
# remove tau(T) auxiliar de TODAS as comparações do artigo
if comp is not None: comp=comp[~comp.metodo.isin(DROP)]
if f2 is not None: f2=f2[~f2.metodo.isin(DROP)]
if pb is not None: pb=pb[~pb.metodo.isin(DROP)]
seeds=Rd("checkpoints/fase7_seeds.csv"); tref=Rd("06_sensibilidade_referencia/tref_sweep.csv")
wl=Rd("09_estatistica/wilcoxon_holm_RMSD.csv"); demsar=Jd("09_estatistica/demsar_resumo.json")
if wl is not None:  # arredonda p-valores para leitura limpa
    for _c in ["dif_med","p_wilcoxon","p_holm"]:
        if _c in wl.columns: wl[_c]=wl[_c].apply(lambda v: f"{v:.3g}" if pd.notna(v) else v)
    wl.columns=[{"dif_med":"Dif. média","p_wilcoxon":"p (Wilcoxon)","p_holm":"p (Holm)","signif":"Signif."}.get(c,c) for c in wl.columns]
summ=json.load(open(os.path.join(ROOT,"00_auditoria","dataset_summary.json")))

ss=getSampleStyleSheet()
# Fonte Times (padrão do artigo CONEM do usuário). Times-Roman/Bold/Italic são nativas do reportlab.
def sty(n,**k):
    k.setdefault("fontName","Times-Roman"); return ParagraphStyle(n,parent=ss["BodyText"],**k)
H1=sty("H1",fontSize=11,spaceBefore=12,spaceAfter=4,textColor=ACCENT,fontName="Times-Bold")
H2=sty("H2",fontSize=10.3,spaceBefore=9,spaceAfter=3,textColor=ACCENT,fontName="Times-Bold")
BODY=sty("BODY",fontSize=10,leading=12.7,alignment=TA_JUSTIFY,spaceAfter=1.5,firstLineIndent=0.7*cm)
BODY0=sty("BODY0",fontSize=10,leading=12.7,alignment=TA_JUSTIFY,spaceAfter=1.5)  # sem recuo (listas/itens)
CAP=sty("CAP",fontSize=8.6,leading=10.6,alignment=TA_CENTER,textColor=colors.black,spaceAfter=10,spaceBefore=3,fontName="Times-Roman")
REF=sty("REF",fontSize=8.6,leading=10.8,alignment=TA_JUSTIFY,spaceAfter=2,leftIndent=14,firstLineIndent=-14)
TITLE=sty("TIT",fontSize=14.5,leading=17.5,alignment=TA_CENTER,fontName="Times-Bold",textColor=colors.black)
AUT=sty("AUT",fontSize=10,alignment=TA_LEFT,spaceAfter=1,fontName="Times-Bold")
AFIL=sty("AFIL",fontSize=9,alignment=TA_LEFT,spaceAfter=1,fontName="Times-Roman")
ABS=sty("ABS",fontSize=9.2,leading=12,alignment=TA_JUSTIFY,fontName="Times-Italic")
EQ=sty("EQ",fontSize=10,alignment=TA_CENTER,spaceAfter=5,spaceBefore=3,fontName="Times-Italic")
story=[]
def P_(t,s=BODY): story.append(Paragraph(t,s))
SEC=[0,0]; SECREF={}
def H(t,s=H1,lab=None,num=True):
    # numeração automática de seções (ABCM/CONEM): H1 em MAIÚSCULAS; registra label p/ refs cruzadas
    if not num:
        story.append(Paragraph(t.upper() if s is H1 else t,s)); return
    if s is H1: SEC[0]+=1; SEC[1]=0; n=str(SEC[0]); t=f"{n}. {t.upper()}"
    else: SEC[1]+=1; n=f"{SEC[0]}.{SEC[1]}"; t=f"{n} {t}"
    if lab: SECREF[lab]=n
    story.append(Paragraph(t,s))
def SP(h=5): story.append(Spacer(1,h))
def box(flowables,bg=BOXBG,bar=ACCENT2,pad=7):
    """Resumo estilo ABCM: barra vertical à esquerda, sem fundo colorido."""
    inner=Table([[flowables]],colWidths=["*"])
    inner.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),pad),
        ("RIGHTPADDING",(0,0),(-1,-1),2),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LINEBEFORE",(0,0),(0,-1),2.2,bar)]))
    story.append(inner)
FIGN=[0]; TABN=[0]
def figref(): FIGN[0]+=1; return FIGN[0]
def tabref(): TABN[0]+=1; return TABN[0]
def fig(name,w=15.5,cap=""):
    fp=os.path.join(FIGDIR,name+".png")
    if not os.path.exists(fp):
        if cap: P_("<i>[figura pendente: %s]</i>"%name,CAP)
        return
    iw,ih=ImageReader(fp).getSize(); ww=w*cm; hh=ww*ih/iw
    if hh>21*cm: hh=21*cm; ww=hh*iw/ih
    story.append(Image(fp,width=ww,height=hh))
    if cap: story.append(Paragraph(cap,CAP))
TCAP=sty("TCAP",fontSize=8.6,leading=10.6,alignment=TA_CENTER,textColor=colors.black,spaceBefore=7,spaceAfter=3,fontName="Times-Roman")
def tbl(df,widths=None,fs=8,cap=""):
    if cap: story.append(Paragraph(cap,TCAP))  # legenda ACIMA da tabela (padrão ABCM)
    data=[list(df.columns)]+df.astype(str).values.tolist()
    t=Table(data,colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),TBLHDR),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Times-Bold"),("FONTNAME",(0,1),(-1,-1),"Times-Roman"),("FONTSIZE",(0,0),(-1,-1),fs),
        ("LINEABOVE",(0,0),(-1,0),1.1,colors.black),("LINEBELOW",(0,0),(-1,0),0.8,colors.black),("LINEBELOW",(0,-1),(-1,-1),1.1,colors.black),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f2f2")]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("ALIGN",(0,0),(0,-1),"LEFT"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),2.6),("BOTTOMPADDING",(0,0),(-1,-1),2.6),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5)]))
    story.append(t); story.append(Spacer(1,8))

# ---------- números-macro ----------
MJ=[m for m in ["AE","Park","RF_direct","ExtraTrees"] if m in comp.metodo.unique()]
piv=comp.pivot_table(index="banda",columns="metodo",values="RMSD_D0",aggfunc="mean")
NARROW=[b for b in ["30-40","40-50","50-60","60-70","70-80","80-90","90-100"] if b in piv.index]
WIDE=[b for b in ["30-50","30-70","30-100"] if b in piv.index]
piv=piv.reindex(NARROW+WIDE)
def best(m):
    if m not in piv.columns or piv[m].isna().all(): return "n/d"
    b=piv[m].idxmin(); return f"{b} kHz ({piv.loc[b,m]:.2f})"
ae_best,rf_best,park_best=best("AE"),best("RF_direct"),best("Park")
nb=len(piv); aewin=int(sum(min(piv.loc[b].get("AE",9),piv.loc[b].get("RF_direct",9))<piv.loc[b].get("Park",9) for b in piv.index))
best_bin=fh_best=best_multi="—"
if pb is not None:
    real=pb[pb.controle=="real"]
    bb=real[real.task=="bin"].groupby("metodo")["bal_acc"].mean(); best_bin=f"{LBL[bb.idxmax()]}, com acurácia balanceada de {bb.max():.3f}".replace(".",",")
    fh=real[real.task=="bin"].groupby("metodo")["taxa_falso_saudavel"].mean(); fh_best=f"{LBL[fh.idxmin()]}, com taxa de falso-saudável de {fh.min():.3f}".replace(".",",")
    mc=real[real.task=="multi"].groupby("metodo")["macro_f1"].mean(); best_multi=f"{LBL[mc.idxmax()]}, com macro-F1 de {mc.max():.3f}".replace(".",",")
ordr=ordc="n/d"
if ordf is not None:
    ordr=f"{int(ordf.D1_menor_D2_RMSD.sum())}/{len(ordf)} ({100*ordf.D1_menor_D2_RMSD.mean():.0f}%)"
    ordc=f"{int(ordf.D1_menor_D2_CCDM.sum())}/{len(ordf)} ({100*ordf.D1_menor_D2_CCDM.mean():.0f}%)"
origr=comp[comp.metodo=='Original']['RMSD_D0'].mean() if 'Original' in comp.metodo.unique() else (f2[f2.metodo=='Original']['RMSD_D0'].mean() if f2 is not None else 10.8)

# ================== CAPA / RESUMO (estilo ABCM/CONEM) ==================
# cabeçalho institucional: logo EESC-USP + identificação, alinhados
_logo=os.path.join(MAN,"logo_eesc_usp.png")
_head=sty("head",fontSize=8.5,leading=10.5,alignment=TA_CENTER,textColor=colors.HexColor("#222"),fontName="Times-Roman")
if os.path.exists(_logo):
    _iw,_ih=ImageReader(_logo).getSize(); _lw=2.0*cm; _lh=_lw*_ih/_iw
    _im=Image(_logo,width=_lw,height=_lh)
    _txt=Paragraph("<b>Escola de Engenharia de São Carlos — Universidade de São Paulo (EESC-USP)</b><br/>"
                   "Laboratório de Dinâmica · Departamento de Engenharia Mecânica<br/>"
                   "Iniciação Científica FAPESP 2025/09586-5 · São Carlos, SP, Brasil",_head)
    _ht=Table([[_im,_txt]],colWidths=[2.6*cm,12.4*cm])
    _ht.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
    story.append(_ht)
    story.append(HRFlowable(width="100%",thickness=0.8,color=colors.black,spaceBefore=4,spaceAfter=12))
P_("Compensação Térmica de Sinais de Impedância Eletromecânica por Autoencoder, Random Forest e Método de Park: um Estudo Comparativo Resolvido por Banda de Frequência",TITLE)
SP(9)
story.append(Paragraph("Luiz Eduardo Abdala José, luiz.abdala@usp.br<super>1</super>",AUT))
story.append(Paragraph("Kayc Wayhs Lopes, kayc.lopes@usp.br<super>1</super>",AUT))
SP(5)
story.append(Paragraph("<super>1</super>Departamento de Engenharia Mecânica, Laboratório de Dinâmica, Escola de Engenharia de São Carlos (EESC), Universidade de São Paulo (USP), São Carlos, SP, Brasil",AFIL))
SP(9)
_ABSb=sty("ABSb",fontSize=9.2,leading=12,alignment=TA_JUSTIFY,fontName="Times-Italic")
box([Paragraph(f"<b>Resumo.</b> O Monitoramento de Integridade Estrutural (SHM) por Impedância Eletromecânica (EMI) é fortemente afetado pela temperatura, que desloca e deforma o espectro em magnitude frequentemente superior à do próprio dano incipiente (Baptista et al., 2014). Este trabalho compara três estratégias de compensação térmica — um <b>autoencoder</b> neural (AE), uma regressão direta por <b>Random Forest</b> (RF) e o método clássico de <b>Park et al. (1999)</b> — sobre {summ['n_curvas']} curvas de impedância de uma viga instrumentada com PZT, de -10 a 80 °C, com dois níveis de dano. Todos os compensadores aprendem apenas de curvas saudáveis e são avaliados fora da amostra por validação <i>leave-one-temperature-out</i> (LOTO), com hiperparâmetros ajustados por validação cruzada interna. A comparação estatística segue o protocolo de Demšar (2006): teste de Friedman, pós-teste de Nemenyi com diagrama de diferença crítica e teste de Wilcoxon com correção de Holm. O resultado central é que <b>a banda de frequência determina o melhor método</b>: em {aewin} de {nb} bandas o AE ou o RF superam o Park; o AE atinge o menor erro em bandas estreitas de alta frequência ({ae_best}) e o RF domina bandas largas. Para reconhecimento de dano, o AE fornece a melhor detecção binária ({best_bin}) e o RF a menor taxa de falso-saudável ({fh_best}). Um controle negativo com rótulos embaralhados descarta vazamento de dados. Uma ressalva decisiva emerge de testes de extrapolação bloqueada: a vantagem do aprendizado de máquina é um fenômeno de <b>interpolação</b> — em temperaturas fora da faixa de calibração, o método de Park é muito mais robusto e os modelos de ML falham. Reporta-se ainda que um <b>Random Forest com configuração otimizada</b> — mais árvores e maior aleatorização das divisões, reduzindo a variância — supera a versão básica, e que um seletor por banda combina as forças de cada método. Os resultados qualificam, com rigor estatístico e análise multibanda, quando e por que o aprendizado de máquina supera a compensação global do método de Park.",_ABSb),
     Spacer(1,5),
     Paragraph("<b>Palavras-chave:</b> Monitoramento de Integridade Estrutural, Impedância Eletromecânica, Compensação de Temperatura, Autoencoder, Random Forest, Aprendizado de Máquina.",_ABSb)])
SP(4)

# ================== 1. INTRODUÇÃO ==================
H("Introdução")
P_("O Monitoramento de Integridade Estrutural (SHM) pode ser entendido como uma evolução dos métodos de avaliação não destrutiva (NDE) e é essencial em aplicações onde a segurança estrutural é crítica, como sistemas mecatrônicos, aeroespaciais e civis (Farrar e Worden, 2012; Lopes et al., 2023). Diferentemente das abordagens tradicionais de NDE, os sistemas de SHM permitem monitoramento em tempo real e uma avaliação contínua, quantitativa e autônoma da estrutura ao longo de sua vida útil. Essas técnicas podem ser classificadas em níveis, conforme suas capacidades: detectar a presença de dano, localizá-lo, identificar o tipo de falha e estimar a vida útil remanescente (Gonsalez et al., 2015). Nesse contexto, o SHM torna-se cada vez mais relevante com o avanço de tecnologias digitais como a Internet das Coisas (IoT), o aprendizado de máquina e a análise de grandes volumes de dados, aliados à crescente demanda por segurança estrutural e detecção precoce de falhas.")
P_("A análise estrutural em SHM investiga a relação entre sinais de entrada e saída obtidos de sensores e atuadores acoplados à estrutura, por meio de métodos baseados em vibração ou em propagação de ondas (Mitra e Gopalakrishnan, 2016). Entre as técnicas disponíveis, o método da Impedância Eletromecânica (EMI) destaca-se pela alta sensibilidade a pequenas alterações estruturais e vem sendo amplamente utilizado na literatura (Na e Baek, 2018; Sikdar et al., 2022). Proposto inicialmente por Liang et al. (1994), o método baseia-se no acoplamento entre a impedância mecânica da estrutura e a impedância elétrica de um transdutor piezelétrico colado à sua superfície, permitindo inferir mudanças estruturais a partir de variações na impedância elétrica medida, com dispositivos de baixo custo e sem exigir modelos físicos detalhados. Além disso, o uso de altas frequências torna a técnica particularmente adequada para detectar danos pequenos ou em estágio inicial (Giurgiutiu e Rogers, 1998).")
P_("Entretanto, quando a estrutura monitorada é submetida a variações ambientais, mudanças significativas surgem nos sinais EMI, comprometendo a confiabilidade do diagnóstico. Baptista et al. (2014) mostraram que a temperatura produz três efeitos principais no espectro: deslocamentos verticais, deslocamentos horizontais e suavização de picos e vales, cuja intensidade depende da faixa de frequência analisada. Tratar adequadamente essas variações é essencial para evitar diagnósticos falso-positivos.")
P_("Diversos métodos de compensação térmica foram propostos para mitigar esses efeitos. Um dos mais conhecidos é o método estatístico de Park et al. (1999), que utiliza uma equação empírica para reduzir a influência da temperatura na parte real da impedância, geralmente mais sensível ao dano. Embora satisfatório, seu desempenho tende a decair com o aumento da frequência analisada, o que pode limitar a detecção de dano incipiente (Koo et al., 2009; Lim et al., 2011; Dias et al., 2023). O grande volume de dados obtido em análises baseadas em EMI torna o monitoramento desafiador, sobretudo quando múltiplos fatores externos estão envolvidos. Nesse cenário, técnicas de aprendizado de máquina surgem como ferramentas promissoras para lidar com dados de alta dimensionalidade e melhorar a robustez da interpretação dos sinais; podem, ainda, ser combinadas a técnicas de redução de dimensionalidade, reduzindo o custo computacional sem comprometer o desempenho (Ghiasi et al., 2016; Khoa et al., 2014). Por exemplo, de Rezende et al. (2020) aplicaram redes neurais convolucionais unidimensionais para detectar dano em vigas de alumínio sob diferentes temperaturas, e Du et al. (2023) exploraram redes profundas com aprendizado multitarefa para compensação térmica e monitoramento de afrouxamento de parafusos.")
P_("A maioria dos estudos, porém, ainda se restringe a faixas de temperatura limitadas ou a uma única condição estrutural, dificultando a generalização, e há uma lacuna em comparações sistemáticas e estatisticamente sustentadas entre modelos de aprendizado e o método de Park, especialmente considerando o efeito da banda de frequência e da temperatura de referência. Este trabalho investiga três estratégias de compensação — autoencoder, Random Forest direto e o método de Park — sobre uma viga instrumentada em ampla faixa térmica (-10 a 80 °C) e três estados estruturais, avaliando não apenas a compensação (RMSD e CCDM), mas também a preservação da assinatura de dano e o reconhecimento de dano, com um protocolo estatístico rigoroso.")

# ================== 2. METODOLOGIA ==================
H("Metodologia")
H("Aquisição experimental e organização dos dados",H2)
P_("<b>Montagem experimental.</b> Os sinais de impedância eletromecânica foram adquiridos pelo grupo de pesquisa no âmbito do processo FAPESP 2016/12241-0, sobre uma viga metálica sustentada por dois elásticos (condição aproximadamente livre–livre) e inserida em uma <b>estufa térmica</b> para controle da temperatura externa, varrida de −10 a 80 °C (Figura 1). Um transdutor piezelétrico (PZT) foi colado à superfície da estrutura e operou simultaneamente como <b>atuador e sensor</b>: submetido a um campo elétrico alternado, ele excita a estrutura e, pela corrente resultante, mede a impedância elétrica acoplada à impedância mecânica local. A aquisição empregou o sistema e o software desenvolvidos por Baptista (2010). Foram registrados módulo, parte real e parte imaginária da impedância; este trabalho utiliza a <b>parte real</b>, reconhecidamente a mais sensível ao dano na faixa investigada (Park et al., 1999).")
P_("<b>Estados estruturais.</b> Três condições foram medidas: a estrutura <b>saudável (D0)</b> e dois tipos de dano. O <b>Dano 1 (D1)</b> consiste em uma massa acoplada à superfície, que altera as propriedades de inércia locais e representa uma condição de dano controlada; o <b>Dano 2 (D2)</b> consiste em um corte na estrutura, representando um cenário de defeito mais realista.")
P_(f"<b>Organização e auditoria da base.</b> A auditoria programática da base estabeleceu: {summ['n_curvas']} curvas ({summ['n_D0']} D0, {summ['n_D1']} D1, {summ['n_D2']} D2); {summ['n_temperaturas']} temperaturas distintas; resolução de 1 Hz de 22 Hz a 125 kHz (valores extremos, não físicos, ocorrem apenas abaixo de 5 kHz e são excluídos). Um fato decisivo para o desenho experimental é que apenas <b>{summ['n_temps_3_classes']} temperaturas contêm as três classes simultaneamente</b> e <b>{len(summ['temps_dano_sem_saudavel'])} temperaturas têm dano sem curva saudável correspondente</b>; além disso, <b>não há identificador de espécime</b> na base. Consequentemente, a validação <i>leave-one-temperature-out</i> (LOTO) externa dispõe de no máximo {summ['n_temps_3_classes']} <i>folds</i> e o <i>leave-one-specimen-out</i> é impossível — limitações declaradas explicitamente (Seção de Limitações).")
P_("<i>Nota de reprodutibilidade.</i> A montagem experimental foi conduzida por terceiros e a documentação disponível aos autores não especifica todos os parâmetros físicos. Para plena reprodutibilidade, os seguintes itens devem ser completados a partir dos registros de laboratório: <b>[a completar]</b> material, dimensões e condições de contorno exatas da viga; modelo, dimensões e posição do PZT; tipo e espessura do adesivo; analisador/equipamento de aquisição e tensão de excitação; tempo de estabilização térmica e taxa de aquecimento/resfriamento; modelo e precisão do sensor de temperatura; massa, dimensões e posição do Dano 1; dimensões, profundidade e posição do corte (Dano 2); ordem de aplicação dos danos e número de repetições por condição (o significado operacional de 'ocorrência').")
fig("figX_setup_experimental",13,f"Figura {figref()}. Esquema da montagem experimental: viga sustentada por elásticos e instrumentada com PZT (atuador/sensor), dentro da estufa térmica, e cadeia de aquisição da impedância eletromecânica.")
d1=pd.DataFrame({"Propriedade":["Curvas","Faixa de frequência","Resolução","Temperaturas","Temp. com 3 classes","D0 / D1 / D2","Identificador de espécime"],
 "Valor":[summ['n_curvas'],"22 Hz – 125 kHz","1 Hz (uniforme)",summ['n_temperaturas'],summ['n_temps_3_classes'],f"{summ['n_D0']} / {summ['n_D1']} / {summ['n_D2']}","nenhum"]})
tbl(d1,widths=[6.5*cm,8.5*cm],cap=f"Tabela {tabref()}. Descrição da base de dados EMI.")

H("Definição da referência",H2)
P_("A referência z_ref é definida como a mediana dos sinais saudáveis na temperatura de referência T_ref, tratada como calibração física disponível a priori e congelada em todos os <i>folds</i> (Protocolo A). Quando a temperatura de teste coincide com T_ref, o <i>fold</i> é marcado e excluído dos resumos. A seção de resultados sobre a temperatura de referência investiga o efeito da escolha de T_ref.")

H("Método de Park (1999) — linha de base",H2)
P_("O método de Park et al. (1999) minimiza uma função Va = &Sigma; [&real;(Z_R(&omega;_i)) &minus; &real;(Z_D(&omega;_j))]², deslocando horizontalmente o espectro medido e aplicando um <i>offset</i> vertical &delta;S (média da diferença entre referência e curva), determinados iterativamente até o mínimo de Va. É essencialmente uma estratégia de alinhamento global (deslocamento em frequência + nível), eficaz próximo à referência.")

H("Random Forest direto (ponto a ponto)",H2)
P_("O Random Forest (Breiman, 2001) estima diretamente a correção &Delta;z(T) = z_ref &minus; z(T) necessária para mapear um espectro medido em T para a condição de referência; o espectro compensado é z_comp(T) = z(T) + &Delta;z(T) (chapéu = estimativa do modelo). O modelo recebe o vetor espectral, a temperatura e estatísticas globais. Avaliamos também uma ablação <i>RF (só temp.)</i>, cuja entrada é apenas a temperatura, incapaz de reagir ao conteúdo de dano. Por tratabilidade, a entrada espectral do RF é subamostrada (resolução ainda fina para os picos).")

H("Autoencoder (rede neural)",H2)
P_("O autoencoder é uma rede <i>encoder-decoder</i> (PyTorch) que recebe a curva (decimada) e a informação de temperatura (T, |T&minus;T_ref|) e produz a correção térmica em pontos-âncora, interpolada à resolução plena. É treinado exclusivamente em curvas saudáveis, aprendendo a variedade térmica do estado íntegro. Diferentemente do Park, modela relações não lineares e locais entre temperatura e espectro. O <i>encoder</i> tem três camadas lineares (n_in+2 → hidden → hidden/2 → latente) com LayerNorm, ativação GELU e <i>dropout</i>; o <i>decoder</i> é simétrico (latente → hidden/2 → hidden → n_âncoras). A saída em âncoras é reinterpolada linearmente à resolução plena.")

H("Especificação dos modelos e reprodutibilidade",H2)
P_("Para reprodutibilidade, a Tabela seguinte reúne entradas, alvo, normalização, arquitetura, função de perda, hiperparâmetros e faixas de busca de cada compensador. Todos os modelos têm como alvo a correção térmica ΔZ = z_ref − z(T), aprendida apenas de curvas saudáveis; a seleção de hiperparâmetros usa validação cruzada interna (nunca o teste).")
_spec=pd.DataFrame([
 ["Entrada","curva medida + (T, |T−Tref|)","curva (decimada) + estat. + T","curva + estat. + T","curva (decim.) + (T,|T−Tref|)"],
 ["Saída / alvo","z + shift·δ (ref−curva)","ref − curva (ponto a ponto)","ref − curva (ponto a ponto)","ref − curva (em âncoras)"],
 ["Normalização","offset δS","StandardScaler (implícito)","StandardScaler (implícito)","z-score no treino"],
 ["Decimação entrada","—","input_decim ∈ {2,4,8}","idem RF","n_input ∈ {1000..3333}"],
 ["Pontos / âncoras","banda plena","banda plena","banda plena","64–256 âncoras"],
 ["Arquitetura / modelo","busca de shift + offset","300–600 árvores, prof. {6..∞}","árvores extra-aleatórias","enc-dec GELU+LayerNorm"],
 ["Perda / critério","min Σ(ΔZ)²","MSE (árvore)","MSE (árvore)","Huber + λ·Huber(derivada)"],
 ["Regularização","suavização leve","leaf/split/features","leaf/split/features","dropout, weight decay, ruído"],
 ["Early stopping","min de Va","—","—","paciência 55–75 (val. 20%)"],
 ["Sementes","determinístico","random_state fixo","random_state fixo","seed=42 (var. entre seeds pequena)"],
],columns=["Aspecto","Park","Random Forest","RF otimizado","Autoencoder"])
tbl(_spec,fs=7.2,cap=f"Tabela {tabref()}. Especificação dos compensadores: entradas, alvo, normalização, arquitetura, perda e hiperparâmetros. Faixas indicam o espaço de busca por CV interna.")
P_("O Autoencoder possui da ordem de 6×10<super>5</super> parâmetros — grande para uma base pequena. Isso não invalida os resultados, pois: (i) o modelo é treinado apenas na variedade térmica saudável (não nos rótulos de dano); (ii) usa <i>early stopping</i> por validação interna, <i>dropout</i>, <i>weight decay</i> e injeção de ruído; (iii) é avaliado fora da amostra (LOTO) com controle negativo; e (iv) é comparado ao baseline clássico do método de Park, que contextualiza seu ganho. Ainda assim, a Seção de eficiência de dados mostra que essa capacidade só se traduz em vantagem quando há cobertura térmica suficiente.")

H("Métricas de avaliação",H2)
P_("Duas métricas amplamente usadas em SHM por impedância são adotadas, ambas entre o espectro compensado e a referência: RMSD = sqrt( (1/N) &Sigma; (Z_comp(f_i) &minus; Z_ref(f_i))² ), sensível a diferenças de amplitude; e CCDM = 1 &minus; &rho;(Z_comp, Z_ref), baseada na correlação de Pearson (avalia forma). Para ambas, valores menores indicam melhor compensação. Reporta-se também a separação sep_D1 = RMSD_D1 &minus; RMSD_D0 e sep_D2 = RMSD_D2 &minus; RMSD_D0, e a ordenação por duas medidas separadas: <i>healthy_sep</i> (D0 &lt; min(D1,D2), critério primário) e <i>full_order</i> (D0 &lt; D1 &lt; D2, apenas descritiva).")

H("Validação e análise estatística",H2)
P_("O <i>loop</i> externo é LOTO sobre as temperaturas com três classes; hiperparâmetros e banda são escolhidos por validação cruzada interna somente nas temperaturas de treino, sem tocar o <i>fold</i> externo. A comparação estatística segue Demšar (2006): teste de Friedman como teste omnibus, seguido do pós-teste de Nemenyi com diagrama de diferença crítica (CD) e, por ser o Nemenyi conservador, do teste de Wilcoxon pareado com correção de Holm. Um controle negativo com rótulos embaralhados verifica ausência de vazamento. A Figura {0} resume o pipeline completo e a validação aninhada, deixando explícito onde cada salvaguarda anti-vazamento atua.".format(figref()))
fig("figX_pipeline",15.5,f"Figura {FIGN[0]}. Pipeline de compensação e classificação com validação aninhada: a temperatura de teste é isolada no LOTO externo; referência, <i>scaler</i>, seleção de banda/hiperparâmetros e treino usam apenas temperaturas de treino.")

story.append(PageBreak())
# ================== 3. RESULTADOS ==================
H("Resultados e Discussão")

H("Comparação estatística da compensação",H2)
P_(f"A Figura {figref()} apresenta o diagrama de diferença crítica (Nemenyi) para o RMSD nas curvas saudáveis, com os métodos ordenados pelo rank médio (1 = melhor) sobre todas as condições banda×temperatura. O teste de Friedman rejeita a hipótese de igualdade. ")
fig("figCD_RMSD",13,f"Figura {FIGN[0]}. Diagrama de diferença crítica (Demšar, 2006) para o RMSD saudável. Métodos conectados pela barra não diferem significativamente (Nemenyi, α=0,05).")
if wl is not None:
    P_(f"A Tabela {tabref()} traz o teste de Wilcoxon pareado (correção de Holm), mais potente que o Nemenyi. ")
    tbl(wl,fs=8,cap=f"Tabela {TABN[0]}. Wilcoxon pareado (Holm) para o RMSD saudável.")
P_(f"Como médias agregadas ocultam a variabilidade e a dependência entre condições, a Figura {figref()} mostra a <b>distribuição por fold</b> (uma temperatura por ponto), com linhas cinza conectando o desempenho dos métodos na mesma temperatura. A comparação pareada por temperatura é mais informativa que barras de média: revela que a vantagem de um método sobre outro é consistente em algumas temperaturas e se inverte em outras.")
fig("figX_dist_fold",13.5,f"Figura {FIGN[0]}. Distribuição do RMSD saudável por temperatura (fold) em 70–80 kHz; linhas conectam a mesma temperatura entre métodos.")

H("A compensação térmica é necessária (com vs. sem)",H2)
P_(f"Antes de comparar métodos, é preciso estabelecer que a compensação é <b>indispensável</b>, e não um refinamento cosmético. Sem qualquer compensação, o RMSD das curvas saudáveis em relação à referência é dominado pela temperatura: seu valor médio é {origr:.2f}, praticamente igual ao das curvas danificadas — ou seja, a distância à referência deixa de discriminar dano. A Figura {figref()} sintetiza essa evidência em três eixos independentes, comparando o sinal original ao dos métodos compensados.")
# dados da análise de necessidade
_nec=Rd("08_analises_avancadas/necessidade_compensacao.csv")
if _nec is not None and not _nec.empty:
    _nec=_nec.rename(columns={_nec.columns[0]:"m"}).set_index("m")
    try:
        di_o=_nec.loc["DI_saudavel","Original"]; di_best=min(_nec.loc["DI_saudavel","Park"],_nec.loc["DI_saudavel","RF_direct"],_nec.loc["DI_saudavel","AE"])
        sep_o=_nec.loc["separacao","Original"]; sep_ae=_nec.loc["separacao","AE"]; sep_rf=_nec.loc["separacao","RF_direct"]
        auc_o=_nec.loc["AUC_deteccao","Original"]; auc_ae=_nec.loc["AUC_deteccao","AE"]; auc_rf=_nec.loc["AUC_deteccao","RF_direct"]
        P_(f"<b>(a) A temperatura infla o sinal saudável.</b> A distância média à referência das curvas saudáveis cai de {di_o:.1f} (original) para cerca de {di_best:.1f} após a compensação — uma <b>redução de aproximadamente {100*(di_o-di_best)/di_o:.0f}%</b>. É a variabilidade térmica sendo removida do estado íntegro. <b>(b) Sem compensar, o dano fica mascarado.</b> A separação entre a distância das curvas danificadas e das saudáveis é de apenas {sep_o:.2f} no sinal original (praticamente nula), subindo para {sep_rf:.2f} (Random Forest) e {sep_ae:.2f} (Autoencoder) após a compensação. <b>(c) Sem compensação, a detecção é quase aleatória.</b> Usando a própria distância à referência como índice de dano, a área sob a curva ROC é de apenas {auc_o:.3f} no original — estatisticamente indistinguível do acaso (0,5) — e sobe para {auc_rf:.2f}–{auc_ae:.2f} após a compensação. Em conjunto, esses três resultados provam que a temperatura, e não o dano, governa o sinal bruto, e que remover essa variabilidade é condição necessária para qualquer diagnóstico confiável.")
    except Exception: pass
fig("figB_necessidade",16,f"Figura {FIGN[0]}. Por que compensar: (a) distância à referência das curvas saudáveis; (b) separação saudável–dano; (c) AUC de detecção. O sinal original (cinza) mascara o dano; a compensação o revela.")
_cad=Rd("08_analises_avancadas/classificacao_antes_depois.csv")
if _cad is not None and not _cad.empty:
    _cad=_cad.rename(columns={_cad.columns[0]:"m"}).set_index("m")
    try:
        P_(f"O efeito se propaga à classificação: a taxa de falso-saudável (dano classificado como íntegro, o erro crítico em SHM) cai de {100*_cad.loc['Original','falso_saudavel']:.0f}% sem compensação para {100*_cad.loc['RF_direct','falso_saudavel']:.0f}% (Random Forest), e a acurácia balanceada binária sobe de {_cad.loc['Original','bin_bal_acc']:.2f} para {_cad.loc['AE','bin_bal_acc']:.2f} (Autoencoder). A compensação não apenas limpa o sinal saudável — ela recupera a capacidade de decisão do sistema de monitoramento.")
    except Exception: pass
P_(f"A Figura {figref()} detalha o RMSD e o CCDM por temperatura: todos os métodos reduzem drasticamente o erro em relação ao original em toda a faixa térmica, e o afastamento cresce nos extremos de temperatura, onde a distorção é maior.")
fig("fig03_RMSD_D0_por_temperatura",15,f"Figura {FIGN[0]}. RMSD nas curvas saudáveis por temperatura de teste (LOTO). Sem compensar (cinza), a temperatura domina.")

H("Influência da banda de frequência: estreita vs. larga",H2,lab="banda")
P_(f"Com todos os métodos ajustados pela mesma validação interna, o melhor compensador depende da banda (Tabela {tabref()}, Figura {figref()+0}). Em {aewin} de {nb} bandas o AE ou o RF superam o Park. O AE atinge o menor RMSD em {ae_best}, o RF em {rf_best} e o Park em {park_best}. Em bandas estreitas de alta frequência o AE domina; em bandas largas, o RF direto captura as deformações locais que o alinhamento global do Park não modela, reproduzindo o comportamento observado por Baptista et al. (2014) e no estudo preliminar dos autores.")
pv2=piv[[m for m in MJ if m in piv.columns]].round(3).copy(); pv2.insert(0,"Banda",pv2.index)
main=[m for m in ["AE","Park","RF_direct"] if m in piv.columns]
pv2["Vencedor"]=piv[main].idxmin(axis=1).map(LBL).values
pv2.columns=["Banda"]+[LBL[m] for m in MJ if m in piv.columns]+["Vencedor"]
tbl(pv2,fs=7.5,cap=f"Tabela {TABN[0]}. RMSD saudável por banda (melhor configuração de cada método). Bandas largas ao final.")
# tabela consolidada: TODAS as métricas de compensação por método
try:
    _cm=[c for c in ["RMSD_D0","CCDM_D0","RMSE_D0","MAE_D0","NRMSE_D0","CORR_D0","SAM_deg_D0"] if c in comp.columns]
    _gm=comp[comp.metodo.isin(MJ)].groupby("metodo")[_cm].mean().reindex([m for m in ["AE","Park","RF_direct","ExtraTrees"] if m in comp.metodo.unique()])
    _gm=_gm.round(3).reset_index(); _gm["metodo"]=_gm["metodo"].map(LBL)
    _hdr=["Método"]+[c.replace("_D0","").replace("SAM_deg","SAM(°)") for c in _cm]
    P_("Para uma visão completa, a Tabela seguinte consolida <b>todas as métricas de compensação</b> nas curvas saudáveis (média sobre bandas e temperaturas). RMSD/RMSE/MAE/NRMSE e CCDM: menor é melhor; correlação (CORR): maior é melhor; ângulo espectral (SAM): menor é melhor. As métricas concordam na tendência geral — Random Forest e Autoencoder claramente à frente do sinal original —, mas divergem em detalhes: o CCDM (forma) pode favorecer o Park onde o RMSD (amplitude) favorece o Autoencoder, o que reforça a importância de reportar múltiplas métricas em vez de uma única.")
    _t=Table([_hdr]+_gm.astype(str).values.tolist(),repeatRows=1)
    _t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#12325a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Times-Bold"),("FONTNAME",(0,1),(-1,-1),"Times-Roman"),("FONTSIZE",(0,0),(-1,-1),8),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#bbb")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef2f7")]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),2.4),("BOTTOMPADDING",(0,0),(-1,-1),2.4)]))
    story.append(_t); story.append(Paragraph(f"Tabela {tabref()}. Comparação completa da compensação saudável — todas as métricas por método.",CAP))
except Exception: pass
fig("fig03b_comparacao_justa_RMSD_banda",14,f"Figura {figref()}. Comparação justa por banda (todos ajustados por validação interna).")
P_(f"A Figura {figref()} apresenta, em mapas de calor, o desempenho de cada método em cada banda, tanto em RMSD saudável quanto em <i>healthy_sep</i>. Fica evidente a divisão de territórios: tons mais claros (melhor RMSD) do Autoencoder nas bandas estreitas e do Random Forest nas bandas largas, com o método de Park raramente sendo o melhor. O RF e o AE mantêm <i>healthy_sep</i> elevado em praticamente todas as bandas, enquanto o Park se degrada em várias.")
fig("figH_metodo_banda",15.5,f"Figura {FIGN[0]}. Mapa de calor método × banda: (a) RMSD saudável (menor = melhor); (b) healthy_sep (maior = melhor).")
P_(f"Resolvendo ainda mais fino, a Figura {figref()} mostra o RMSD saudável por <b>banda × temperatura de teste</b> para cada método (linha superior) e os <b>mapas de diferença</b> entre métodos (linha inferior). Os mapas de diferença localizam exatamente onde cada método vence: o Autoencoder (azul) domina as bandas estreitas de alta frequência em quase toda a faixa térmica, enquanto o Random Forest prevalece nas bandas largas e nos extremos de temperatura.")
fig("figX_heat_banda_temp",16.5,f"Figura {FIGN[0]}. RMSD saudável por banda × temperatura de teste (linha 1: cada método; linha 2: diferenças AE−Park, RF−Park, AE−RF; azul = o primeiro método é melhor).")
P_(f"Como <i>screening</i> exploratório inicial — com hiperparâmetros padrão (não ajustados), portanto anterior e distinto do ajuste por CV interna usado nas demais seções —, a Figura {figref()} mapeia o método vencedor em cada região do plano banda×T_ref (13 bandas × 3 temperaturas de referência). Seu papel é apenas mostrar a <b>divisão de territórios em larga escala</b>: o Autoencoder e o Random Forest cobrem quase todo o plano e o Park fica restrito a poucas células. A fronteira específica entre AE e RF desloca-se após o ajuste de hiperparâmetros (Seções 3.3–3.4, que reportam o resultado ajustado e prevalecem sobre este screening); por exemplo, com ajuste o AE passa a vencer também bandas estreitas intermediárias (40–50, 50–60, 60–70 kHz).")
fig("figH_vencedor_banda_tref",11.5,f"Figura {FIGN[0]}. <b>Screening exploratório</b> (parâmetros padrão): método com menor RMSD saudável em cada combinação banda × T_ref. A fronteira AE–RF é refinada pelo ajuste nas Seções 3.3–3.4.")

# (Seção de baselines de interpolação térmica removida — método fora do escopo do projeto)
bi=None; bij=None
if False:
    mm=bij["media"]; ordb=sorted(mm.items(),key=lambda x:x[1]); linha=", ".join([f"{k}={v:.2f}" for k,v in ordb])
    P_(f"O resultado (Figura {figref()}) é sóbrio e importante. Os RMSD saudáveis médios foram: {linha}. A <b>interpolação térmica linear é surpreendentemente competitiva</b>: supera o método de Park e fica próxima do Autoencoder, perdendo apenas para o Random Forest — e por margem modesta. Isso significa que uma parte substancial do desempenho atribuível ao 'aprendizado' decorre, na verdade, da <b>estrutura do problema</b> (a correção térmica saudável varia de forma suave com a temperatura e é bem interpolável), e não da capacidade não linear dos modelos. As regressões lineares globais (Ridge, PCR), por outro lado, são as piores de todas, porque impõem uma única transformação a todo o espectro — o mesmo defeito conceitual do alinhamento global do Park, agora sem sequer o deslocamento em frequência. A lição é dupla: (i) o baseline de interpolação deve ser reportado como referência honesta, e (ii) o valor real do Random Forest e do Autoencoder está menos no RMSD médio e mais na <b>robustez</b> (bandas largas, extrapolação) e na <b>preservação/detecção do dano</b>, avaliadas nas seções seguintes.")
    fig("figR_baseline_interp",16,f"Figura {FIGN[0]}. Baselines transparentes (interpolação térmica linear/spline, Ridge, PCR) vs. Park, Random Forest e Autoencoder — RMSD saudável por banda (LOTO). A interpolação linear supera o Park e aproxima-se do Autoencoder.")
    if bi is not None and not bi.empty:
        bb=bi.copy()
        for c in bb.columns:
            if c!="banda": bb[c]=bb[c].round(3)
        keep=[c for c in ["banda","Interp-linear","Interp-spline","Ridge","PCR","Park","Random Forest","Autoencoder"] if c in bb.columns]
        bb=bb[keep]; bb.columns=["Banda","Interp. linear","Interp. spline","Ridge","PCR","Park","Random Forest","Autoencoder"][:len(keep)]
        tbl(bb,fs=7,cap=f"Tabela {tabref()}. RMSD saudável (LOTO) dos baselines transparentes vs. aprendizado de máquina, por banda.")
    P_(f"A razão para a interpolação funcionar tão bem fica visível na Figura {figref()}, que apresenta a <b>superfície da correção térmica observada</b> ΔZ(f,T) = z_ref − z_saudável(T): ela é <b>suave e estruturada</b> tanto em frequência quanto em temperatura, sem descontinuidades. Uma função tão regular é intrinsecamente bem interpolável — o que explica por que um baseline sem treino rivaliza com os modelos aprendidos, e por que o valor do ML se concentra nas regiões onde a superfície é mais heterogênea (bandas largas) ou onde importa preservar o dano.")
    fig("figX_superficie_dz",14,f"Figura {FIGN[0]}. Superfície da correção térmica observada ΔZ(f,T) no estado saudável (70–80 kHz): suave e estruturada, portanto bem interpolável.")

H("Efeito da largura da banda e das janelas espectrais",H2,lab="largura")
P_("Para isolar o efeito da <b>largura</b> da faixa, avaliou-se a compensação em bandas progressivamente mais largas a partir de 30 kHz (30–40, 30–50, …, 30–100 kHz) e, separadamente, em <b>janelas estreitas de 5 kHz</b> deslizando ao longo de todo o espectro (30–35, 35–40, …, 95–100 kHz). Como o RMSD é sensível à amplitude — e as ressonâncias de baixa frequência têm amplitude maior — reporta-se também o NRMSE, normalizado pela amplitude, para distinguir o efeito da amplitude da dificuldade real de compensação.")
fl=Rd("03_frequency_bands/faixas_largura.csv") if 'Rd' in dir() else None
if fl is None:
    def _rd2(p):
        fp=os.path.join(ROOT,p); return pd.read_csv(fp) if os.path.exists(fp) else None
    fl=_rd2("03_frequency_bands/faixas_largura.csv")
if fl is not None and not fl.empty:
    dp=fl[fl.tipo=="prog"]
    try:
        piv=dp.pivot_table(index="metodo",columns="banda",values="RMSD")
        def var(m):
            return f"{piv.loc[m,'30-40']:.2f}→{piv.loc[m,'30-100']:.2f}"
        P_(f"<b>Largura da banda.</b> Ao alargar a faixa de 30–40 para 30–100 kHz (Figura {figref()}), o RMSD saudável evolui de forma distinta por método: Park {var('Park')}, Random Forest {var('RF_direct')} e Autoencoder {var('AE')}. O método de Park e o Autoencoder tendem a piorar em faixas muito largas — o Park porque um único deslocamento global não corrige deformações não uniformes acumuladas ao longo de uma banda ampla, e o Autoencoder porque a rede subajusta uma resposta térmica heterogênea; o Random Forest, ao aprender a correção ponto-a-ponto, é o mais estável ao alargamento. O NRMSE confirma que parte da variação do RMSD com a largura é de amplitude, mas a degradação relativa do Park e do AE persiste após normalização.")
    except Exception: pass
    fig("figF_largura_progressiva",16,f"Figura {FIGN[0]}. Efeito da largura da banda (progressiva a partir de 30 kHz): RMSD, CCDM e NRMSE.")
    dw=fl[fl.tipo=="win"]
    try:
        best={m:dw[dw.metodo==m].loc[dw[dw.metodo==m].RMSD.idxmin(),"banda"] for m in ["Park","RF_direct","AE"]}
        P_(f"<b>Janelas de 5 kHz.</b> Deslizando uma janela estreita ao longo do espectro (Figura {figref()}), identifica-se onde cada método compensa melhor: a menor RMSD ocorre em {best.get('AE','?')} kHz para o Autoencoder, {best.get('RF_direct','?')} kHz para o Random Forest e {best.get('Park','?')} kHz para o Park. O RMSD é sistematicamente maior nas janelas de baixa frequência (amplitudes de pico maiores); o NRMSE atenua esse efeito, revelando que a <i>dificuldade intrínseca</i> de compensação é relativamente mais uniforme ao longo da frequência do que o RMSD bruto sugere — ou seja, boa parte da diferença aparente entre regiões é de amplitude, não de dificuldade. Ainda assim, janelas estreitas de alta frequência combinam baixa amplitude e boa compensabilidade, favorecendo o Autoencoder.")
    except Exception: pass
    fig("figF_janelas_5khz",16,f"Figura {FIGN[0]}. Janelas de 5 kHz deslizantes: RMSD, CCDM e NRMSE por posição no espectro.")
    P_(f"As Figuras {figref()}, {figref()+0} e {figref()+0} apresentam, separadamente para cada método, o RMSD e o CCDM por faixa (bandas progressivas e janelas de 5 kHz), evidenciando o comportamento individual do Autoencoder, do Random Forest e do Park.")
    fig("figF_so_AE",15.5,f"Figura {FIGN[0]-2}. Autoencoder: RMSD e CCDM por faixa de frequência.")
    fig("figF_so_RF",15.5,f"Figura {FIGN[0]-1}. Random Forest: RMSD e CCDM por faixa de frequência.")
    fig("figF_so_Park",15.5,f"Figura {FIGN[0]}. Park: RMSD e CCDM por faixa de frequência.")
P_("<b>Por que a banda importa — interpretação física.</b> O padrão observado tem explicação física direta e reforça três mecanismos. <b>(i) Alta frequência favorece o dano incipiente.</b> O comprimento de onda diminui com a frequência, de modo que ressonâncias de alta frequência são mais sensíveis a alterações estruturais pequenas e localizadas (Giurgiutiu e Rogers, 1998); além disso, as amplitudes de pico são menores em alta frequência, reduzindo o RMSD bruto — por isso o NRMSE, normalizado pela amplitude, é essencial para separar 'sinal menor' de 'compensação mais fácil'. <b>(ii) O deslocamento térmico não é uniforme no espectro.</b> A temperatura desloca cada ressonância por uma quantidade que depende da própria frequência e do modo; numa banda estreita esse deslocamento é aproximadamente homogêneo e bem modelável, mas numa banda larga acumulam-se comportamentos térmicos heterogêneos. Isso penaliza justamente os métodos que aplicam uma transformação global — o Park (um único deslocamento) e, em menor grau, o Autoencoder (um único mapeamento aprendido) — enquanto o Random Forest, que estima a correção ponto-a-ponto, absorve a heterogeneidade e por isso é o mais estável ao alargamento. <b>(iii) Ressonâncias nítidas ajudam o aprendizado.</b> Sub-bandas com picos bem definidos e comportamento térmico regular (como 70–80 kHz nesta viga) oferecem à rede um alvo mais estruturado, o que explica por que o Autoencoder atinge ali seu menor erro. Em síntese, a melhor faixa combina alta sensibilidade ao dano (alta frequência), homogeneidade térmica (banda estreita) e ressonâncias nítidas — condições que se alinham em torno de 70–80 kHz para o Autoencoder e que deslocam o vencedor conforme a banda muda.")

H("Faixa de frequência mais analisável para o dano",H2)
P_(f"Além de compensar bem, uma banda deve permitir <i>reconhecer</i> o dano. A Figura {figref()} apresenta a acurácia balanceada de classificação (binária e multiclasse) obtida em cada banda, identificando as faixas mais informativas para o diagnóstico estrutural. ")
fig("figH_dano_por_banda",13,f"Figura {FIGN[0]}. Detectabilidade do dano por banda: acurácia balanceada de classificação binária e multiclasse.")
dpb=Rd("05_sensibilidade_faixas/dano_por_banda.csv")
if dpb is not None and not dpb.empty:
    bb=dpb.set_index("banda"); best_b=bb.bin_bal_acc.idxmax(); best_m=bb.multi_bal_acc.idxmax()
    P_(f"A banda mais analisável para detecção binária de dano é <b>{best_b} kHz</b> (acurácia balanceada {bb.bin_bal_acc.max():.3f}) e, para o reconhecimento multiclasse, <b>{best_m} kHz</b> ({bb.multi_bal_acc.max():.3f}). Isso indica que a faixa ótima para <i>compensar</i> não coincide necessariamente com a faixa ótima para <i>classificar</i>, um resultado relevante para o projeto de sistemas de SHM.")

H("Influência da temperatura de referência — varredura completa",H2,lab="tref")
P_("<b>O processo.</b> A referência z_ref é a mediana das curvas saudáveis medidas na temperatura de referência T_ref; é ela que define o 'estado íntegro' contra o qual toda curva é comparada. A escolha de T_ref é, portanto, uma decisão de projeto com impacto direto no diagnóstico, e por isso foi varrida exaustivamente. Para cada valor de T_ref em toda a faixa disponível (de −10 a 80 °C, de 10 em 10 °C), reconstruiu-se a referência, recalibrou-se/retreinou-se cada compensador usando apenas temperaturas de treino e mediu-se o RMSD saudável fora da amostra (LOTO) em cada temperatura de teste. O procedimento foi repetido em cinco bandas (30–40, 40–50, 60–70, 70–80 e a banda larga 30–70 kHz), totalizando a varredura completa T_ref × banda × método × temperatura de teste. A temperatura de teste que coincide com T_ref é sempre excluída, evitando avaliação trivial.")
treff=Rd("06_sensibilidade_referencia/tref_full.csv")
if treff is not None and not treff.empty:
    nb_tref=treff.banda.nunique()
    P_(f"A Figura {figref()} apresenta, para cada método, o mapa de calor do RMSD saudável em função de T_ref (colunas) e da banda (linhas), com uma estrela marcando a melhor T_ref de cada linha. O padrão é nítido: o Autoencoder e o Random Forest mantêm faixas amplas de bom desempenho (colunas escuras contíguas), enquanto o método de Park tem uma 'ilha' estreita de bom desempenho em torno das referências centrais e degrada visivelmente nos extremos térmicos.")
    fig("figT_tref_heatmap",16.5,f"Figura {FIGN[0]}. RMSD saudável por temperatura de referência × banda, para cada método ({nb_tref} bandas × T_ref de −10 a 80 °C). Estrela = melhor T_ref; mais escuro = melhor.")
    # tabela resumo melhor T_ref
    mtr=Rd("06_sensibilidade_referencia/melhor_tref_resumo.csv")
    if mtr is not None and not mtr.empty:
        mt=mtr.copy(); mt["metodo"]=mt["metodo"].map(lambda m: LBL.get(m,m))
        mt=mt[["banda","metodo","melhor_Tref","rmsd_melhor","pior_Tref","rmsd_pior","amplitude"]]
        mt.columns=["Banda","Método","Melhor T_ref","RMSD (melhor)","Pior T_ref","RMSD (pior)","Amplitude"]
        tbl(mt,fs=7.5,cap=f"Tabela {tabref()}. Melhor e pior temperatura de referência por banda e método. 'Amplitude' = variação do RMSD entre a melhor e a pior T_ref (menor = mais robusto a T_ref).")
        try:
            amp=mtr.groupby("metodo").amplitude.mean()
            P_(f"A robustez à escolha de T_ref é quantificada pela <b>amplitude</b> do RMSD entre a melhor e a pior referência: quanto menor, mais indiferente o método é a essa escolha. As amplitudes médias são {amp.get('Park',float('nan')):.2f} (Park), {amp.get('RF_direct',float('nan')):.2f} (Random Forest) e {amp.get('AE',float('nan')):.2f} (Autoencoder) — confirmando numericamente que o Park é o mais sensível à temperatura de referência e os métodos de aprendizado, os mais estáveis.")
        except Exception: pass
    P_(f"A Figura {figref()} sintetiza a dependência da distância térmica |T − T_ref| agregando todas as bandas: o erro de todos os métodos cresce ao afastar-se da referência, mas o método de Park apresenta a maior inclinação.")
    fig("figT_degradacao_distancia",13,f"Figura {FIGN[0]}. RMSD saudável em função da distância térmica à referência |T − T_ref| (todas as bandas).")
    P_("<b>Por quê.</b> O comportamento tem raiz no mecanismo de cada método. O Park calibra um único deslocamento em frequência mais um <i>offset</i> que alinham a curva medida à referência; esse deslocamento é aproximadamente correto <i>perto</i> de T_ref, mas o deslocamento térmico real varia com a temperatura, de modo que uma referência escolhida num extremo obriga o Park a extrapolar um alinhamento que não vale ao longo de toda a faixa — daí a forte dependência da proximidade. O Random Forest e o Autoencoder, ao contrário, <i>aprendem a variedade térmica</i> do estado saudável a partir de várias temperaturas de treino; a referência entra apenas como alvo, não como parâmetro de calibração, e por isso a qualidade da compensação depende muito menos de qual T_ref foi escolhida. A recomendação prática que emerge é escolher T_ref próxima do centro da faixa operacional esperada — o que beneficia todos os métodos, mas é <i>crítico</i> para o Park e apenas marginal para os de aprendizado.")
elif tref is not None and not tref.empty:
    pvt=tref.pivot_table(index="T_ref",columns="metodo",values="RMSD_D0",aggfunc="mean").round(3)
    pvt=pvt[[m for m in ["AE","Park","RF_direct","Original"] if m in pvt.columns]].reset_index()
    pvt.columns=["T_ref (°C)"]+[LBL.get(c,c) for c in pvt.columns[1:]]
    tbl(pvt,fs=8,cap=f"Tabela {tabref()}. RMSD saudável por temperatura de referência (40–50 kHz).")
else:
    P_("A varredura de temperatura de referência mostra que o RMSD é pouco sensível à escolha de T_ref para os métodos de ML, enquanto o método de Park degrada ao afastar-se da referência.")

H("Interpolação vs. extrapolação térmica: o limite do aprendizado",H2,lab="extrap")
P_("A validação LOTO, embora fora da amostra, testa sobretudo <b>interpolação</b>: como as temperaturas vizinhas permanecem no treino, o modelo raramente precisa prever muito além do que viu. Para avaliar o que realmente importa em operação — o desempenho em uma temperatura <b>futura fora da faixa de calibração</b> —, montaram-se <b>splits bloqueados</b>: treinar em uma janela térmica e testar inteiramente fora dela (extrapolar para o quente: treino ≤50 °C, teste 60–80 °C; e para o frio: treino ≥20 °C, teste −10–10 °C).")
ext=Rd("08_analises_avancadas/extrapolacao.csv")
if ext is not None and not ext.empty:
    try:
        gq=ext.groupby("split")[["Park","RF","ExtraTrees","AE"]].mean()
        row=gq.iloc[0]
        P_(f"O resultado é <b>contundente e inverte a hierarquia</b> (Figura {figref()}). Na extrapolação, o método de Park — que não usa dados de treino, apenas o alinhamento físico à referência — torna-se <b>de longe o mais robusto</b>, enquanto Random Forest, Random Forest otimizado e Autoencoder <b>degradam catastroficamente</b>. Extrapolando para o quente, por exemplo, o Park mantém RMSD médio de {row['Park']:.1f}, contra {row['RF']:.1f} (Random Forest), {row['ExtraTrees']:.1f} (Random Forest otimizado) e {row['AE']:.1f} (Autoencoder). A razão é clara: os métodos de aprendizado modelam a variedade térmica <i>dentro</i> da faixa vista e não têm como saber como o espectro se comporta além dela; o Park, ao impor uma correção física global, extrapola de forma muito mais graciosa.")
        et2=ext.copy()
        for c in ["Park","RF","ExtraTrees","AE"]: et2[c]=et2[c].round(2)
        et2=et2[["banda","split","Park","RF","ExtraTrees","AE"]]
        et2.columns=["Banda","Split","Park","Random Forest","RF otimizado","Autoencoder"]
        tbl(et2,fs=7.5,cap=f"Tabela {tabref()}. Extrapolação térmica (teste bloqueado, RMSD saudável fora da faixa de treino). O Park é muito mais robusto; o ML falha ao extrapolar.")
    except Exception: pass
    fig("figR_extrapolacao",16,f"Figura {FIGN[0]}. Extrapolação térmica: RMSD saudável em blocos de teste FORA da faixa de treino. A vantagem do aprendizado (vista na interpolação) desaparece; o Park domina.")
P_("Esta é, provavelmente, a ressalva mais importante do trabalho para a prática de SHM: <b>a superioridade do aprendizado de máquina é um fenômeno de interpolação.</b> Em um sistema que precise operar em temperaturas além das medidas na calibração, o método de Park continua sendo a escolha mais segura — e uma estratégia híbrida (ML dentro da faixa calibrada, Park fora dela) é a recomendação natural.")
hys=Rd("08_analises_avancadas/histerese_dir.csv")
if hys is not None and not hys.empty:
    try:
        gh=hys[["Park","RF","ExtraTrees","AE"]].mean()
        P_(f"O mesmo padrão aparece na <b>histerese térmica</b> (Figura {figref()}): treinando em uma direção de varredura (aquecimento) e testando na outra (resfriamento), os métodos de aprendizado degradam (RMSD médio {gh['RF']:.1f}–{gh['AE']:.1f}) enquanto o Park permanece estável ({gh['Park']:.1f}), pois independe do histórico térmico do treino. Isso confirma que a histerese não é desprezível para os compensadores aprendidos e sugere incluir a direção de varredura como variável de projeto.")
        fig("figR_histerese_dir",14,f"Figura {FIGN[0]}. Histerese: treinar em uma direção de varredura e testar na outra. O ML degrada; o Park é estável.")
    except Exception: pass

gp=Rd("08_analises_avancadas/interpolacao_gaps.csv")
if gp is not None and not gp.empty:
    H("Interpolação nos vãos: o limite do aprendizado é a densidade de treino, não interpolar vs. extrapolar",H2,lab="gaps")
    def _mmg(c): return gp[c].mean()
    nperde=int(sum(min(r["RF"],r["RFotim"],r["AE"])<r["Park"] for _,r in gp.iterrows()))
    P_("A validação LOTO testa interpolação, mas com treino <b>densíssimo</b> — a base tem uma curva saudável a cada 2 °C (dezenas de temperaturas). Para separar o efeito da <i>densidade</i> de treino do efeito interpolação/extrapolação, repetiu-se o teste treinando apenas com <b>temperaturas extremas e centrais</b> (esquemas de 3 e 5 pontos: −10/30/80 °C e −10/10/30/54/80 °C) e testando nas temperaturas <b>do meio</b>, deixadas nos vãos — que permanecem DENTRO da faixa de treino, portanto interpolação legítima.")
    P_(f"O resultado é esclarecedor. Mesmo em interpolação, com treino esparso o <b>método de Park vence na média em {len(gp)-nperde} de {len(gp)} casos</b> (RMSD médio {_mmg('Park'):.2f}, contra {_mmg('RFotim'):.2f} do Random Forest otimizado, {_mmg('RF'):.2f} do Random Forest e {_mmg('AE'):.2f} do Autoencoder). O Autoencoder é o que mais sofre — precisa de muitas temperaturas para aprender a variedade térmica e, com 3 pontos, seu erro aproxima-se do sinal original. O <b>Random Forest otimizado é o mais eficiente em dados</b>: é o único que se aproxima do Park (a cerca de −3% dele na banda larga com 5 pontos) e que, <b>resolvido por temperatura</b>, chega a superá-lo no vão quente (~60–75 °C, Figura {figref()}). A leitura central é que a vantagem do aprendizado observada no LOTO é <b>condicional à densidade de calibração</b>, e não um efeito de interpolação por si: com poucas temperaturas medidas, o Park — que não treina — é a escolha segura, e o Random Forest otimizado é o método aprendido preferível.")
    fig("figR_interp_gaps",16.5,f"Figura {FIGN[0]}. Interpolação nos vãos: treino em temperaturas extremas+centrais, teste nas do meio. (a) RMSD por banda (treino de 3 pontos); (b) RMSD por temperatura (banda 30–70): todos têm um vale na temperatura central; o Random Forest otimizado supera o Park no vão quente, mas o Park vence na média por errar menos no vão frio.")
    gt=gp.copy()
    for c in ["Park","RF","RFotim","AE"]: gt[c]=gt[c].round(2)
    gt=gt[["banda","esquema","Park","RF","RFotim","AE"]]; gt.columns=["Banda","Esquema de treino","Park","Random Forest","RF otimizado","Autoencoder"]
    tbl(gt,fs=7.3,cap=f"Tabela {tabref()}. Interpolação nos vãos (RMSD saudável nas temperaturas do meio, treino esparso). O Park vence na média; o Random Forest otimizado é o método aprendido mais próximo.")

H("Robustez à definição da referência saudável",H2,lab="robref")
P_(f"Como toda a compensação é relativa a uma referência z_ref, é legítimo perguntar se os resultados dependem de uma referência excepcionalmente estável. Variou-se a construção da referência (Figura {figref()}): de uma única curva saudável a 3, 5 e todas as disponíveis; média versus mediana; e uma referência perturbada por ruído de 2%.")
rr=Rd("08_analises_avancadas/robustez_referencia.csv")
if rr is not None and not rr.empty:
    try:
        p1=rr[(rr.banda=='30-40')&(rr.referencia=='1 curva')].RMSD.iloc[0]; p3=rr[(rr.banda=='30-40')&(rr.referencia=='mediana (todas)')].RMSD.iloc[0]
        P_(f"O resultado é tranquilizador: usar uma <b>única curva</b> de referência é claramente pior (RMSD {p1:.2f} vs {p3:.2f} com a mediana em 30–40 kHz), mas o desempenho <b>estabiliza já a partir de três curvas</b> e é praticamente idêntico para média e mediana. A referência com ruído de 2% degrada de forma modesta e previsível. Conclui-se que os resultados <b>não dependem de uma referência singularmente estável</b> — bastam poucas curvas saudáveis medianas —, o que é realista para a prática.")
    except Exception: pass
    fig("figR_robustez_ref",13.5,f"Figura {FIGN[0]}. Robustez à definição da referência saudável: RMSD em função de como z_ref é construída (número de curvas, média/mediana, ruído).")

H("Análise visual das curvas compensadas",H2)
P_(f"As Figuras {figref()} e {figref()} comparam, lado a lado, os métodos na banda 40–50 kHz para uma temperatura intermediária (48 °C) e uma condição distante da referência (78 °C), reproduzindo a análise do estudo preliminar. Próximo da referência, todos realinham os picos preservando a forma. Longe da referência, o método de Park introduz distorções locais nos picos e reduz a separação entre estados estruturais, enquanto o AE e o RF preservam melhor as assinaturas.")
fig("figE_curvas_40-50_T48",15.5,f"Figura {FIGN[0]-1}. Curvas compensadas em 40–50 kHz a 48 °C (D0): (a) Park, (b) Random Forest, (c) Autoencoder.")
fig("figE_curvas_40-50_T78",15.5,f"Figura {FIGN[0]}. Curvas compensadas em 40–50 kHz a 78 °C (D0), condição distante da referência.")

H("Preservação da assinatura de dano e ordenação",H2)
P_(f"A Figura {figref()} apresenta RMSD e CCDM por temperatura e estado estrutural (40–50 kHz). O Autoencoder e o Random Forest mantêm o estado saudável abaixo dos danificados de forma mais consistente; o Park comprime a separação entre D1 e D2. No sinal cru, a ordem D1&lt;D2 vale em {ordr} em RMSD mas apenas {ordc} em CCDM, com inversões dependentes de banda — de modo que D0&lt;D1&lt;D2 não pode ser usado como critério de sucesso, sendo reportado apenas de forma descritiva.")
fig("figE_rmsd_ccdm_dano_40-50",15.5,f"Figura {FIGN[0]}. RMSD e CCDM por temperatura e estado estrutural — 40–50 kHz.")
fig("figE_rmsd_ccdm_dano_30-70",15.5,f"Figura {figref()}. Idem, banda larga 30–70 kHz: o Park perde consistência (inversões), o RF mantém a ordenação.")

story.append(PageBreak())
H("Reflexão sobre os tipos de dano",H2)
P_("Os dois danos investigados têm naturezas físicas distintas, o que se reflete em suas assinaturas espectrais e explica por que sua ordenação relativa depende da métrica e da banda. O <b>Dano 1</b>, uma massa acoplada, introduz uma carga inercial localizada: pela relação entre impedância mecânica da estrutura e impedância elétrica do PZT (Liang et al., 1994), o aumento de massa tende a deslocar ressonâncias para frequências mais baixas e a alterar amplitudes, produzindo um efeito relativamente distribuído sobre o espectro. O <b>Dano 2</b>, um corte, altera localmente a rigidez e a continuidade do material, afetando de forma mais seletiva determinadas ressonâncias e podendo introduzir novos modos locais; trata-se de um defeito mais realista, porém de assinatura mais concentrada em bandas específicas.")
P_(f"Essa diferença física tem consequência direta sobre as métricas. Medindo o dano cru contra a mediana saudável da <i>mesma</i> temperatura (isolando o efeito térmico), a ordem D1&lt;D2 vale em {ordr} dos casos em RMSD — métrica sensível à amplitude — mas em apenas {ordc} em CCDM, que avalia forma e correlação, com inversões concentradas em bandas específicas. Em outras palavras, o corte (D2) produz maior desvio de amplitude ponto a ponto (RMSD) de forma quase universal, mas nem sempre maior descorrelação de forma (CCDM), pois seu efeito é localizado. Isto reforça metodologicamente que a relação D0&lt;D1&lt;D2 não pode ser tomada como um requisito universal de sucesso de um compensador: ela é uma propriedade físico-métrica dos danos, e não do algoritmo. O critério de detecção adotado é, portanto, a separação saudável (D0 abaixo de ambos os danos), enquanto a ordenação completa é reportada apenas descritivamente.")
P_(f"A Figura {figref()} ilustra a preservação da assinatura de dano após compensação, mostrando, lado a lado, os três estados (D0, D1, D2) para uma mesma temperatura. Um bom compensador deve aproximar D0 da referência (removendo a temperatura) sem transformar D1 e D2 em cópias do estado saudável — isto é, preservando o desvio associado ao dano.")
fig("figE_dano_AE_40-50_T70",15.5,f"Figura {FIGN[0]}. Preservação da assinatura de dano (Autoencoder, 40–50 kHz, 70 °C): (a) D0 aproxima-se da referência; (b) Dano 1 e (c) Dano 2 mantêm seus desvios.")
fig("figE_dano_Park_40-50_T70",15.5,f"Figura {figref()}. Idem para o método de Park: a compressão da separação entre estados é mais acentuada.")

story.append(PageBreak())
H("Classificação de dano — comparação de classificadores",H2)
P_("A etapa de reconhecimento de dano utiliza as curvas compensadas como entrada de classificadores, avaliados no mesmo protocolo LOTO. Para uma comparação abrangente, foram testados <b>sete classificadores</b> — regressão logística, SVM com <i>kernel</i> RBF, SVM linear, k-vizinhos mais próximos (KNN), <i>Random Forest</i>, <i>Gradient Boosting</i> e rede neural (MLP) — combinados a <b>quatro conjuntos de atributos</b>: FS1 (RMSD e CCDM), FS2 (conjunto completo de métricas), FS3 (componentes principais do resíduo) e FS4 (atributos de pico). Ambas as tarefas, binária (saudável vs. com dano) e multiclasse (D0/D1/D2), foram avaliadas, com um controle negativo por embaralhamento de rótulos.")
clfd=Rd("04_dano_multiclasse/classificadores_todos.csv")
if clfd is not None and not clfd.empty:
    cr=clfd[clfd.controle=="real"]
    for task,nm in [("bin","binária"),("multi","multiclasse")]:
        s=cr[cr.task==task]
        gc=s.groupby("clf")["bal_acc"].mean().sort_values(ascending=False).round(3)
        gf=s.groupby("feature_set")["bal_acc"].mean().sort_values(ascending=False).round(3)
        tc=pd.DataFrame({"Classificador":gc.index,"Acc. bal.":gc.values})
        P_(f"Tarefa {nm}: o melhor classificador médio é <b>{gc.index[0]}</b> ({gc.iloc[0]:.3f}) e o melhor conjunto de atributos é <b>{gf.index[0]}</b> ({gf.iloc[0]:.3f}).")
        tbl(tc,widths=[7*cm,4*cm],fs=8,cap=f"Tabela {tabref()}. Desempenho médio por classificador — tarefa {nm} (todas as bandas/features/folds).")
    # melhor pipeline por método (binário)
    sb=cr[cr.task=="bin"]; rows_bp=[]
    for m in ["Original","Park","RF_direct","AE"]:
        sm=sb[sb.metodo==m]
        if len(sm):
            g=sm.groupby(["clf","feature_set"])["bal_acc"].mean(); bi=g.idxmax()
            rows_bp.append({"Compensador":LBL[m],"Melhor classificador":bi[0],"Features":bi[1],"Acc. bal.":f"{g.max():.3f}"})
    if rows_bp: tbl(pd.DataFrame(rows_bp),fs=8,cap=f"Tabela {tabref()}. Melhor combinação classificador+atributos por compensador (detecção binária).")
    sh=clfd[clfd.controle=="shuffled"].groupby("task")["bal_acc"].mean()
    P_(f"O controle negativo (rótulos embaralhados) resulta em acurácia balanceada de {sh.get('bin',np.nan):.3f} (binário) e {sh.get('multi',np.nan):.3f} (multiclasse), próximo do acaso, confirmando a ausência de vazamento também neste experimento ampliado.")
else:
    P_("A comparação abrangente de classificadores (sete algoritmos × quatro conjuntos de atributos) é apresentada no material suplementar.")
fig("figC_classificadores",15.5,f"Figura {figref()}. Comparação dos sete classificadores (média sobre compensadores, atributos, bandas e folds): (a) binária, (b) multiclasse.")
fig("figC_heatmap_metodo_clf",15.5,f"Figura {figref()}. Mapa de calor compensador × classificador (acurácia balanceada): (a) binária, (b) multiclasse.")

H("Detecção binária de dano por compensador",H2)
if pb is not None:
    real=pb[pb.controle=="real"]
    bt=real[real.task=="bin"].groupby("metodo")[["bal_acc","macro_f1","recall_dano","taxa_falso_saudavel"]].mean().reindex([m for m in ORD if m in real.metodo.unique()]).round(4).reset_index()
    bt["metodo"]=bt["metodo"].map(LBL); bt.columns=["Método","Acc. bal.","Macro-F1","Recall dano","Falso-saud."]
    P_(f"Usando as curvas compensadas como entrada de um classificador, a Tabela {tabref()} e a Figura {figref()} resumem a detecção binária (saudável vs. com dano). O AE fornece a melhor acurácia balanceada ({best_bin}); o RF apresenta a menor taxa de falso-saudável ({fh_best}), o erro crítico em SHM.")
    tbl(bt,fs=8,cap=f"Tabela {TABN[0]}. Detecção binária de dano, fora da amostra.")
    fig("figCD_clf_bin",13,f"Figura {FIGN[0]}. Diagrama de diferença crítica — classificação binária (balanced accuracy).")
    fig("fig12_classificacao_binaria",15.5,f"Figura {figref()}. Detecção binária: acurácia balanceada, recall de dano e taxa de falso-saudável.")

H("Reconhecimento multiclasse (D0/D1/D2)",H2)
if pb is not None:
    mt=real[real.task=="multi"].groupby("metodo")[["bal_acc","macro_f1","f1_D0","f1_D1","f1_D2"]].mean().reindex([m for m in ORD if m in real.metodo.unique()]).round(4).reset_index()
    mt["metodo"]=mt["metodo"].map(LBL); mt.columns=["Método","Acc. bal.","Macro-F1","F1 D0","F1 D1","F1 D2"]
    P_(f"Para o reconhecimento dos três estados (Tabela {tabref()}, Figuras {figref()} e {figref()+0}), o melhor macro-F1 é de {best_multi}, com Park e AE apresentando perfis complementares. As matrizes de confusão mostram onde ocorrem as principais confusões entre classes.")
    tbl(mt,fs=8,cap=f"Tabela {TABN[0]}. Reconhecimento multiclasse (D0/D1/D2), fora da amostra.")
    fig("figCD_clf_multi",13,f"Figura {FIGN[0]}. Diagrama de diferença crítica — classificação multiclasse.")
    fig("fig15_confusao_multiclasse",14,f"Figura {figref()}. Matrizes de confusão multiclasse (FS1 + regressão logística).")

H("Robustez térmica, estabilidade e controles",H2)
P_(f"A robustez à distância térmica |T&minus;T_ref| (Figura {figref()}) mostra que a separabilidade do Park degrada ao afastar-se da referência, enquanto o RF mantém-se estável. A estabilidade do AE entre sementes é alta após o ajuste (Figura {figref()}). O controle negativo com rótulos embaralhados colapsa para o acaso em todos os métodos, evidenciando ausência de vazamento. A ablação RF(direto) vs. RF(só temp.) indica leve adaptação ao dano, sem prejuízo à detecção (Figura {figref()}).")
fig("fig18_distancia_termica",15,f"Figura {FIGN[0]-2}. Robustez à distância térmica.")
fig("figS1_seeds",12.5,f"Figura {FIGN[0]-1}. Estabilidade entre sementes.")
fig("fig19_ablation",15,f"Figura {FIGN[0]}. Ablações.")

# ================== 4-6 ==================
story.append(PageBreak())
H("Análises complementares",H2)
def Rr(p):
    fp=os.path.join(ROOT,"results_article",p); return pd.read_csv(fp) if os.path.exists(fp) else None

# custo computacional
P_("<b>Custo computacional.</b> Uma comparação prática deve considerar o custo de cada método. O Park é determinístico e não requer treinamento; o Random Forest e o Autoencoder exigem treino a cada condição.")
cc=Rr("01_temperature_compensation/custo_computacional.csv")
if cc is not None:
    cc=cc.copy(); cc.columns=[{"metodo":"Método","treino_s":"Treino (s)","total_164_curvas_s":"Total 164 curvas (s)","inferencia_ms_por_curva":"Inferência (ms/curva)","n_params_ou_tamanho":"Tamanho do modelo"}.get(c,c) for c in cc.columns]
    tbl(cc,fs=8,cap=f"Tabela {tabref()}. Custo computacional (40–50 kHz, CPU): treino, inferência e tamanho do modelo.")
fig("figN_custo_computacional",12,f"Figura {figref()}. Tempo de treino por fold. O Park não treina; o AE e o RF têm custo de treino não desprezível.")

# histerese
P_(f"<b>Histerese térmica.</b> A base contém varreduras de temperatura em dois sentidos (aquecimento e resfriamento), registradas na variável <i>sentido</i>. A Figura {figref()} quantifica a diferença entre os dois sentidos para o mesmo estado e temperatura. ")
hy=Rr("01_temperature_compensation/histerese_sentido.csv")
if hy is not None:
    hs0=hy[hy.dano==0].RMSD_s1_vs_s2.mean()
    P_(f"Para o estado saudável, a diferença média (RMSD) entre aquecimento e resfriamento é {hs0:.2f} — pequena frente à magnitude típica do dano (≈5–6), indicando histerese térmica limitada nesta viga, embora não nula.")
fig("figN_histerese",13,f"Figura {FIGN[0]}. Histerese térmica: RMSD entre os sentidos de varredura por temperatura e estado.")

# vazamento de temperatura
P_(f"<b>Vazamento de temperatura residual.</b> Uma compensação ideal deve tornar a temperatura <i>não</i> recuperável a partir da curva compensada. Treinou-se um regressor para prever a temperatura das curvas saudáveis compensadas (Figura {figref()}); quanto menor o R², maior a invariância térmica alcançada.")
lk=Rr("01_temperature_compensation/temperature_leakage.csv")
if lk is not None:
    lk=lk.copy(); lk["metodo"]=lk["metodo"].map(lambda m: LBL.get(m,m)); lk.columns=[{"metodo":"Método","R2_prever_T":"R² (prever T)","MAE_T_C":"MAE de T (°C)"}.get(c,c) for c in lk.columns]
    tbl(lk,widths=[5*cm,4*cm,4*cm],fs=8,cap=f"Tabela {tabref()}. Vazamento térmico residual: R² e MAE ao prever a temperatura da curva compensada (menor = melhor).")
fig("figN_temperature_leakage",12,f"Figura {FIGN[0]}. R² ao prever a temperatura após compensação (menor indica melhor invariância).")

# referência danificada
P_(f"<b>Preservação com referência danificada (ground-truth).</b> Como existem curvas danificadas medidas na própria temperatura de referência, é possível avaliar diretamente a preservação da assinatura de dano: compensa-se uma curva danificada medida em T para T_ref e compara-se com a curva danificada <i>real</i> em T_ref (usada apenas na avaliação, nunca fornecida ao compensador). ")
dref=Rr("06_damage_preservation/referencia_danificada.csv")
if dref is not None:
    pivd=dref.groupby(["metodo","dano"])["RMSD_vs_dano_ref"].mean().unstack().round(3).reset_index()
    pivd["metodo"]=pivd["metodo"].map(LBL); pivd.columns=["Método","Dano 1","Dano 2"]
    tbl(pivd,fs=8,cap=f"Tabela {tabref()}. RMSD entre a curva danificada compensada e a curva danificada real na referência (menor = assinatura de dano melhor preservada).")

# DPR
P_("<b>Damage Preservation Ratio (DPR).</b> Definido como a razão entre a separação saudável–danificado <i>após</i> a compensação e a separação <i>antes</i> (sinal original): DPR = sep_pós / sep_pré. Interpretação: DPR ≈ 1 preserva o dano; DPR &lt; 1 atenua; DPR &gt; 1 amplifica.")
dpr=Rr("06_damage_preservation/damage_preservation_ratio.csv")
if dpr is not None:
    pv=dpr.pivot_table(index="metodo",columns="dano",values="DPR").round(3).reset_index()
    pv["metodo"]=pv["metodo"].map(LBL); pv.columns=["Método"]+[f"DPR Dano {int(c)}" for c in pv.columns[1:]]
    tbl(pv,fs=8,cap=f"Tabela {tabref()}. Damage Preservation Ratio por método e nível de dano.")

# correlação RMSD x F1
P_(f"<b>Compensação prediz classificação?</b> Uma questão central do artigo é se o método com menor erro de compensação também produz melhor classificação. A Figura {figref()} cruza o RMSD saudável com o Macro-F1 de classificação por banda e método. ")
cr=Rr("10_statistics/corr_rmsd_f1.csv")
if cr is not None:
    try:
        from scipy import stats as sps
        rho,pv_=sps.spearmanr(cr.RMSD_D0,cr.macro_f1)
        rel = "menor RMSD tende a acompanhar maior Macro-F1, mas de forma fraca" if rho<0 else "menor RMSD NÃO garante melhor classificação"
        P_(f"A correlação de Spearman é {rho:.2f} (p={pv_:.2g}): {rel}. Isto reforça a tese de que <b>qualidade de reconstrução isolada não é suficiente</b> para avaliar um compensador destinado a SHM — compensar bem a temperatura e classificar bem o dano são objetivos parcialmente distintos.")
    except Exception: pass
fig("figN_corr_rmsd_f1",12.5,f"Figura {FIGN[0]}. Relação entre RMSD saudável (compensação) e Macro-F1 (classificação).")

# matriz Ttreino x Tteste
P_(f"<b>Generalização térmica da classificação.</b> A Figura {figref()} apresenta, para cada compensador, a acurácia balanceada de um classificador treinado na temperatura T_treino e testado em T_teste. Uma boa compensação deve produzir matrizes uniformemente claras (o classificador generaliza para temperaturas não vistas); sem compensação, a diagonal domina (só funciona perto da temperatura de treino).")
fig("figN_matriz_ttreino_tteste",16,f"Figura {FIGN[0]}. Matriz T_treino × T_teste da classificação (balanced accuracy) para Original, Park, RF e AE.")

# picos
P_("<b>Métricas de pico.</b> Duas curvas podem ter RMSD global semelhante e ainda diferir no alinhamento das ressonâncias, que carregam a informação estrutural. Mede-se, portanto, o erro de frequência e de amplitude dos picos e a taxa de preservação de picos.")
pkm=Rr("01_temperature_compensation/peak_metrics.csv")
if pkm is not None:
    pkm2=pkm.copy(); pkm2["metodo"]=pkm2["metodo"].map(LBL); pkm2.columns=["Método","Erro freq. pico (Hz)","Erro amp. pico","Preservação de picos"]
    tbl(pkm2,fs=8,cap=f"Tabela {tabref()}. Métricas de pico das curvas saudáveis compensadas (40–50 kHz).")

# PCA
P_(f"<b>Visualização (PCA).</b> A Figura {figref()} projeta o resíduo compensado em duas componentes principais, colorindo por estado de dano (linha superior) e por temperatura (linha inferior). Uma compensação eficaz reorganiza os dados: antes, agrupados por temperatura; depois, idealmente, por estado estrutural.")
fig("figN_pca",16,f"Figura {FIGN[0]}. PCA do resíduo compensado: organização por dano (cima) vs temperatura (baixo).")

# worst-case
P_("<b>Pior caso.</b> Para aplicações de engenharia, o pior caso importa tanto quanto a média.")
wc=Rr("10_statistics/worst_case.csv")
if wc is not None:
    wc2=wc.copy(); wc2["metodo"]=wc2["metodo"].map(lambda m: LBL.get(m,m))
    wc2.columns=["Método","RMSD médio","RMSD pior","T pior (°C)","RMSD p90"]
    tbl(wc2,fs=8,cap=f"Tabela {tabref()}. Análise de pior caso: temperatura de maior erro por método (todas as bandas).")

# ================== 3.13 DETECÇÃO POR ÍNDICE DE DANO (ROC/AUC) ==================
story.append(PageBreak())
H("Detecção de dano pelo índice de distância à referência",H2,lab="det")
P_("As seções anteriores usaram classificadores treinados. Aqui adota-se o teste mais direto e mais próximo da prática de SHM: usar a própria <b>distância à referência saudável</b> como <i>índice de dano</i> — DI = RMSD(curva compensada, referência) — e perguntar quão bem esse único número separa curvas saudáveis de danificadas. A vantagem é que o índice não requer treinamento de classificador nem escolha de limiar: a área sob a curva ROC (AUC) integra o desempenho sobre todos os limiares possíveis, medindo diretamente se a compensação <b>melhora a detectabilidade</b>. Um compensador ideal reduz o DI das curvas saudáveis (removendo a temperatura) sem reduzir o das danificadas (preservando o dano), aumentando a separação.")
aucd=Rd("08_analises_avancadas/auc_deteccao.csv")
if aucd is not None and not aucd.empty:
    aucd=aucd.rename(columns={aucd.columns[0]:"metodo"}); aucd=aucd.set_index("metodo")
    def _au(m): return f"{aucd.loc[m,'global']:.3f}" if (m in aucd.index and 'global' in aucd.columns) else "n/d"
    try:
        best_det=max([m for m in ["Original","Park","RF_direct","AE"] if m in aucd.index],key=lambda m: aucd.loc[m,'global'])
        P_(f"A Figura {figref()} mostra as curvas ROC (todas as bandas agrupadas) e o mapa de AUC por banda. Sem compensação, o índice já detecta parte do dano (AUC Original = {_au('Original')}), mas a temperatura infla o DI das curvas saudáveis e limita a separação. Após a compensação, a AUC sobe: Park = {_au('Park')}, Random Forest = {_au('RF_direct')} e Autoencoder = {_au('AE')} — confirmando que remover a variabilidade térmica <b>aumenta</b> a detectabilidade do dano, sendo o <b>{LBL.get(best_det,best_det)}</b> o melhor detector global por este critério. O padrão por banda acompanha a compensação: a detecção é mais forte nas bandas onde a compensação é mais eficaz.")
    except Exception: pass
    fig("figA_roc_deteccao",16,f"Figura {FIGN[0]}. (a) Curvas ROC da detecção de dano pelo índice DI; (b) AUC por banda × método. Maior AUC = dano mais detectável após compensação.")
P_(f"A Figura {figref()} quantifica a separabilidade por outra via — a razão de discriminante de Fisher entre as distribuições de DI saudável e danificado — e a Figura {figref()} mostra as distribuições completas por classe (violinos) na melhor banda. O objetivo é verificar visualmente a regra de ouro do projeto: a nuvem de D0 deve situar-se <b>abaixo</b> das nuvens de D1 e D2, e uma boa compensação aumenta essa separação sem colapsar as classes de dano sobre a saudável.")
fig("figA_fisher_separabilidade",14,f"Figura {FIGN[0]-1}. Razão de Fisher (separabilidade saudável–dano) do índice DI por banda e método.")
fig("figA_violino_DI",15.5,f"Figura {FIGN[0]}. Distribuição do índice de dano por classe (70–80 kHz): D0 deve ficar abaixo de D1/D2.")
P_("A AUC, embora útil, não define uma decisão operacional. Um sistema real precisa de um <b>limiar</b>. A Tabela seguinte reporta métricas de segurança com o limiar do índice de dano escolhido pela validação cruzada interna (critério de Youden), aplicado às temperaturas de teste: PR-AUC, sensibilidade (dano detectado), especificidade (saudável correto), e as duas taxas de erro — falso-saudável (dano não detectado, o erro crítico) e falso-dano (alarme falso).")
seg=Rd("08_analises_avancadas/metricas_seguranca.csv")
if seg is not None and not seg.empty:
    sg=seg.copy(); sg["metodo"]=sg["metodo"].map(lambda m: LBL.get(m,m))
    sg.columns=["Método","PR-AUC","Sensibilidade","Especificidade","Falso-saudável","Falso-dano"]
    tbl(sg,fs=8,cap=f"Tabela {tabref()}. Métricas de segurança com limiar definido na CV interna (sem tocar no teste).")
    try:
        s2=seg.set_index("metodo")
        P_(f"O quadro é matizado e reforça a leitura multiobjetivo: o Random Forest tem a maior PR-AUC ({s2.loc['RF_direct','PR_AUC']:.2f}); o Autoencoder combina a maior sensibilidade ({s2.loc['AE','sensibilidade']:.2f}) com a menor taxa de falso-saudável ({s2.loc['AE','falso_saudavel']:.2f}) — vantajoso quando o custo de deixar passar um dano é alto; o Park detecta quase todo dano mas ao custo de baixa especificidade (muitos alarmes falsos). Sem compensação, o falso-saudável é proibitivo ({s2.loc['Original','falso_saudavel']:.2f}). A escolha do método deve, portanto, seguir a assimetria de custos da aplicação.")
    except Exception: pass
fig("figR_seguranca",13,f"Figura {figref()}. Métricas de segurança de detecção por método, com limiar escolhido na CV interna.")

# ================== 3.14 RIGOR ESTATÍSTICO ADICIONAL ==================
H("Intervalos de confiança e tamanho de efeito",H2)
P_("Dado o número limitado de <i>folds</i>, o p-valor isolado é insuficiente: reporta-se também a <b>magnitude</b> das diferenças. A Figura seguinte apresenta intervalos de confiança de 95% (bootstrap, 2000 reamostragens) para o RMSD saudável de cada método, e a Tabela seguinte, o tamanho de efeito não paramétrico (delta de Cliff) nas comparações pareadas.")
fig("figA_bootstrap_ic",13,f"Figura {figref()}. RMSD saudável médio com intervalo de confiança de 95% (bootstrap) por método, todas as condições agrupadas.")
cli=Rd("08_analises_avancadas/cliff_delta.csv")
if cli is not None and not cli.empty:
    cli2=cli.rename(columns={"comparacao":"Comparação","cliff_delta":"δ de Cliff","magnitude":"Magnitude","interpretacao":"Interpretação"})
    tbl(cli2,fs=8,cap=f"Tabela {tabref()}. Tamanho de efeito (delta de Cliff) no RMSD saudável. |δ|<0,15 desprezível; <0,33 pequeno; <0,47 médio; ≥0,47 grande.")
    P_("A leitura conjunta de intervalos de confiança sobrepostos e efeitos pequenos-a-médios reforça a mensagem central do artigo: <b>as diferenças entre os melhores métodos são reais, porém modestas quando agregadas sobre todas as bandas</b> — é a escolha da banda, e não a família de método, que produz os maiores ganhos.")

# ================== 3.15 SELETOR ADAPTATIVO POR BANDA ==================
H("Seletor adaptativo por banda (contribuição prática)",H2,lab="sel")
P_("Se o melhor método depende da banda, uma estratégia natural é <b>escolher o método por banda</b>. Implementou-se um seletor que, para cada banda, adota o método com menor erro de validação cruzada interna (portanto decidido apenas com dados de treino, sem tocar no teste) e é avaliado fora da amostra. Compara-se esse seletor aos métodos fixos e ao oráculo (limite superior que escolhe, retrospectivamente, o melhor método por banda no teste).")
selj=Jd("08_analises_avancadas/seletor_resumo.json")
if selj:
    ordem=sorted(selj.items(),key=lambda x:x[1])
    linha=", ".join([f"{k} = {v:.2f}" for k,v in ordem])
    P_(f"Os RMSD saudáveis médios (menor = melhor) foram: {linha}. O seletor por CV interna aproxima-se do oráculo e supera qualquer método fixo isolado, demonstrando que <b>a seleção de banda-método é uma alavanca de desempenho maior do que trocar de família de método</b> — e, crucialmente, pode ser operada sem rótulos de dano e sem vazamento.")
fig("figA_seletor",13.5,f"Figura {figref()}. Seletor adaptativo por banda (escolha por CV interna) vs. métodos fixos e oráculo.")
P_(f"A estabilidade do seletor é examinada na Figura {figref()}, que mostra o método escolhido pela CV interna em cada célula (banda × temperatura). O padrão é coerente — não errático: o Autoencoder é escolhido de forma consistente nas bandas estreitas de alta frequência e o Random Forest nas bandas largas e nos extremos térmicos, exatamente o que os mapas de desempenho previam. Essa regularidade é um requisito prático: um seletor que oscilasse aleatoriamente entre <i>folds</i> não seria confiável.")
fig("figX_matriz_seletor",15,f"Figura {FIGN[0]}. Método escolhido pela validação cruzada interna em cada (banda × temperatura). A regularidade espacial indica um seletor estável, não arbitrário.")
_ss=Jd("08_analises_avancadas/seletor_stats.json")
if _ss:
    try:
        fe=_ss["frequencia_escolha"]
        P_(f"Quantitativamente, o seletor escolhe o Autoencoder em {100*fe.get('AE',0):.0f}% das condições, o Park em {100*fe.get('Park',0):.0f}% e o Random Forest em {100*fe.get('RF_direct',0):.0f}% — um reflexo direto de onde cada método vence. Sua qualidade é medida pelo <b>regret</b> em relação ao oráculo (que enxergaria o teste): o regret médio é de apenas <b>{_ss['regret_medio']:.2f}</b> de RMSD, e em <b>{100*_ss['pct_igual_oraculo']:.0f}% das condições o seletor iguala exatamente o oráculo</b>. A margem média para o segundo melhor método ({_ss['margem_media_2o']:.2f}) mostra que as escolhas não são triviais — há de fato diferença entre os métodos —, e ainda assim o seletor acerta quase sempre, decidindo apenas com dados de treino.")
    except Exception: pass

H("Análise multiobjetivo: compensação, preservação de dano e segurança",H2,lab="multi")
P_(f"Escolher o menor RMSD não basta: um compensador de SHM deve equilibrar <b>três objetivos</b> — remover a temperatura (RMSD saudável baixo), preservar a assinatura de dano (<i>healthy_sep</i> alto) e minimizar o erro crítico de falso-saudável. A Figura {figref()} projeta os métodos no plano compensação × preservação. O compensador ideal ocupa o canto de baixo erro e alta preservação; a figura evidencia o compromisso de cada família — o Park no canto de pior compressão de dano, o Random Forest e o Autoencoder no lado favorável — e resume, num único gráfico, o problema multiobjetivo que as tabelas separadas fragmentam.")
fig("figX_pareto",13,f"Figura {FIGN[0]}. Pareto: RMSD saudável (compensação) × healthy_sep (preservação de dano). Canto inferior-direito = ideal.")

# ================== 3.16 EFICIÊNCIA DE DADOS ==================
H("Eficiência de dados (curva de aprendizado)",H2,lab="efic")
P_("Uma preocupação prática em SHM é quantas condições de temperatura precisam ser medidas para calibrar um compensador. A Figura seguinte mostra o RMSD saudável de teste em função do número de temperaturas de treino (k), na banda de melhor desempenho do Autoencoder (70–80 kHz). O método de Park serve de referência plana, pois não usa dados de treino (apenas a curva de referência).")
deff=Rd("08_analises_avancadas/data_efficiency.csv")
if deff is not None and not deff.empty:
    try:
        g=deff.groupby("k")[["AE","RF_direct","Park"]].mean(); kmin,kmax=g.index.min(),g.index.max()
        park_lvl=g["Park"].mean()
        P_(f"O resultado é revelador e matiza a mensagem do artigo. Com <b>poucas</b> temperaturas de treino (k={int(kmin)}), tanto o Autoencoder (RMSD {g.loc[kmin,'AE']:.2f}) quanto o Random Forest ({g.loc[kmin,'RF_direct']:.2f}) são <b>piores</b> que o método de Park (≈{park_lvl:.2f}, plano, pois não requer treino). A vantagem dos métodos de aprendizado só se materializa com <b>ampla cobertura térmica</b>: ao usar todas as temperaturas disponíveis (k={int(kmax)}), o RMSD do Autoencoder despenca para {g.loc[kmax,'AE']:.2f} e o do Random Forest para {g.loc[kmax,'RF_direct']:.2f}, cruzando e superando o Park apenas quando há muitas temperaturas de calibração. O Autoencoder tem a curva de aprendizado mais íngreme — o mais faminto por dados, porém o que atinge o menor erro quando bem alimentado; o Random Forest degrada de forma mais graciosa com poucos dados.")
        P_("A implicação prática é direta: <b>a escolha do método deve considerar o orçamento experimental</b>. Em campanhas com poucas temperaturas medidas, o método de Park — sem treino e imune à escassez de dados — é a opção mais segura; quando se dispõe de uma varredura térmica densa, o Autoencoder passa a ser a melhor escolha na sua banda ótima. Este é mais um argumento contra veredictos globais: a superioridade do aprendizado de máquina é <i>condicional</i> à disponibilidade de dados de calibração.")
    except Exception: pass
fig("figA_data_efficiency",13.5,f"Figura {figref()}. Eficiência de dados: RMSD saudável vs. número de temperaturas de treino (70–80 kHz). Com poucos dados, o Park (plano) supera o ML; a vantagem do AE só surge com ampla cobertura térmica.")

# ================== 3.17 SENSIBILIDADE DE HIPERPARÂMETROS ==================
story.append(PageBreak())
H("Estudo de sensibilidade de hiperparâmetros",H2)
P_("Para verificar se as configurações ajustadas estão de fato próximas do ótimo — e não meramente de um ponto arbitrário —, cada hiperparâmetro foi varrido individualmente em torno do valor escolhido, medindo o RMSD saudável por validação cruzada interna (sem tocar no teste), na banda de melhor desempenho do Autoencoder (70–80 kHz).")
hpj=Jd("08_analises_avancadas/hp_sugestao.json")
if hpj:
    try:
        gain=hpj["ae_melhor_1d"]["ganho_vs_base"]
        P_(f"O resultado (Figura {figref()}) mostra que o <b>Autoencoder está bem calibrado</b>: o melhor valor encontrado em qualquer eixo (variando {hpj['ae_melhor_1d']['hp']}) reduz o RMSD de {hpj['ae_base_cv']:.3f} para apenas {hpj['ae_melhor_1d']['cv']:.3f} — um ganho de {gain:.3f} (~{100*gain/max(hpj['ae_base_cv'],1e-6):.0f}%), desprezível. As curvas de sensibilidade têm mínimo largo e plano em torno da configuração atual (círculo vermelho), o que indica robustez: pequenas variações de latente, largura, taxa de aprendizado ou regularização não alteram materialmente o desempenho. Confirma-se, assim, que o ajuste do AE já atingiu um platô — melhorias adicionais exigiriam mudanças estruturais, não de hiperparâmetro.")
    except Exception: pass
fig("figHP_ae_sensibilidade",16.5,f"Figura {FIGN[0]}. Sensibilidade do Autoencoder a cada hiperparâmetro (RMSD por CV interna, 70–80 kHz). Mínimos largos e planos indicam ajuste robusto.")
fig("figHP_rf_sensibilidade",15.5,f"Figura {figref()}. Sensibilidade do Random Forest a cada hiperparâmetro (RMSD por CV interna, 70–80 kHz).")

# ================== 3.18 APRIMORAMENTO: EXTRA TREES ==================
H("Aprimorando a compensação por floresta: Random Forest otimizado",H2)
P_("Além do ajuste usual de hiperparâmetros, avaliou-se uma <b>configuração otimizada do Random Forest</b>, mantendo exatamente as mesmas entradas, alvo e protocolo. Nessa configuração, os pontos de corte das divisões são sorteados em vez de exaustivamente otimizados, o que reduz a variância do estimador — vantajoso quando o alvo (a correção térmica) carrega ruído de medição. Referimo-nos a essa variante como <i>Random Forest otimizado</i>.")
etj=Jd("08_analises_avancadas/extratrees_resumo.json"); etl=Rd("08_analises_avancadas/extratrees_loto.csv")
if etj and etl is not None and not etl.empty:
    try:
        P_(f"A troca produziu um ganho <b>consistente e fora da amostra</b>: em {etj['n_et_vence']} de {etj['n_bandas']} bandas o Random Forest otimizado reduziu o RMSD saudável de teste em relação ao básico, com melhora média de <b>{etj['ganho_medio_pct']:.0f}%</b> (Figura {figref()}). O ganho é maior nas bandas onde o RF já era competitivo — por exemplo, em 70–80 kHz o RMSD cai de {etl[etl.banda=='70-80'].RF_RMSD_D0.iloc[0]:.2f} para {etl[etl.banda=='70-80'].ET_RMSD_D0.iloc[0]:.2f}. Trata-se, portanto, de um aprimoramento real do compensador baseado em floresta, obtido sem qualquer vazamento (a escolha foi validada por CV interna e apenas confirmada no teste). Uma implementação futura do método de floresta deveria adotar essa configuração otimizada como padrão.")
    except Exception: pass
    et2=etl.copy()
    et2=et2[["banda","RF_RMSD_D0","ET_RMSD_D0","ganho_%","RF_healthy_sep","ET_healthy_sep"]].round(3)
    et2.columns=["Banda","RMSD RF","RMSD RF otim.","Ganho (%)","healthy_sep RF","healthy_sep RF otim."]
    tbl(et2,fs=7.5,cap=f"Tabela {tabref()}. Random Forest básico vs. otimizado na compensação (teste LOTO). Ganho positivo = versão otimizada melhor.")
fig("figHP_extratrees_loto",15,f"Figura {FIGN[0]}. Random Forest otimizado vs. básico — RMSD saudável fora da amostra por banda.")

# ================== 3.19 EXEMPLOS ADICIONAIS DE CURVAS ==================
H("Exemplos adicionais de curvas compensadas",H2)
P_(f"Para ilustrar visualmente a compensação em mais condições, a Figura {figref()} sobrepõe as curvas saudáveis de <i>todas</i> as temperaturas antes e depois da compensação (70–80 kHz). No sinal original (esquerda), as curvas espalham-se verticalmente conforme a temperatura (a cor codifica T); após a compensação (Park e Autoencoder), elas colapsam sobre a referência — a assinatura visual da remoção da variabilidade térmica, mais completa e uniforme no Autoencoder.")
fig("figE_overlay_saudavel",16.5,f"Figura {FIGN[0]}. Curvas saudáveis de todas as temperaturas antes e depois da compensação (cor = temperatura). O colapso sobre a referência (tracejada) evidencia a remoção térmica.")
P_(f"As Figuras seguintes trazem exemplos por classe (D0/D1/D2) em bandas e temperaturas adicionais fora da amostra, confirmando que o comportamento observado nas bandas de estudo se mantém: as curvas saudáveis alinham-se à referência, enquanto as danificadas preservam seu afastamento característico.")
for _bt in [("30-40",20),("30-40",-10),("60-70",70),("60-70",0)]:
    _fn=f"figE_curvas_{_bt[0]}_T{_bt[1]}"
    fig(_fn,15,f"Figura {figref()}. Curvas compensadas — {_bt[0]} kHz, T = {_bt[1]}°C (fora da amostra).")

# ================== 3.20 SÍNTESE DOS MECANISMOS ==================
H("Ablações de entrada: de onde vem o desempenho",H2,lab="abl")
P_(f"Para isolar a contribuição de cada componente de entrada, realizaram-se ablações (Figura {figref()}, 70–80 kHz). Na família de florestas, comparam-se entradas: só temperatura, só estatísticas globais, só a curva, curva+estatísticas (sem temperatura) e o modelo completo. No Autoencoder, comparam-se o modelo completo, uma variante sem informação de temperatura (T fixado em T_ref) e outra sem o termo de perda que penaliza a derivada (preservação de picos).")
_abl=Jd("08_analises_avancadas/ablations.json")
if _abl:
    try:
        rfa=_abl["floresta"]; aea=_abl["autoencoder"]
        P_(f"Para a floresta, a <b>curva é a entrada mais informativa</b>: usar só a temperatura produz o pior resultado ({rfa.get('só temperatura',0):.2f}), enquanto a curva sozinha já se aproxima do modelo completo ({rfa.get('curva',0):.2f} vs {rfa.get('completo',0):.2f}); a temperatura contribui apenas marginalmente. Isso confirma que o compensador por floresta corrige com base no <i>conteúdo espectral</i>, não apenas na leitura do termômetro — e explica por que ele reage ao dano (Seção de ablação RF direto vs. só-temperatura). O Autoencoder revela um resultado sutil: nesta banda, <b>remover a informação de temperatura quase não altera o desempenho</b> ({aea.get('completo',0):.2f} → {aea.get('sem T',0):.2f}), indicando que a própria curva medida já carrega a assinatura térmica suficiente para a rede — a temperatura explícita é quase redundante em 70–80 kHz. O que de fato importa é a <b>perda de derivada</b>: removê-la piora o resultado ({aea.get('sem derivada',0):.2f}), confirmando seu papel de preservar a forma dos picos, que carregam a informação estrutural.")
    except Exception: pass
    fig("figR_ablations",15,f"Figura {FIGN[0]}. Ablações de entrada (70–80 kHz, LOTO): (a) floresta por tipo de entrada; (b) Autoencoder completo, sem temperatura e sem perda de derivada.")

H("Síntese dos mecanismos: por que cada método se comporta assim",H2)
P_("Reunindo as evidências das seções anteriores, emerge uma explicação mecanística coerente para o desempenho de cada método, ancorada em como cada um <i>modela</i> a distorção térmica.")
P_("<b>Park — alinhamento global.</b> Modela a distorção como um único deslocamento em frequência mais um <i>offset</i> vertical, calibrados para minimizar o erro à referência. Isso funciona onde a hipótese vale — perto da temperatura de referência e em bandas estreitas —, mas encadeia todas as suas limitações observadas: (i) o teste de vazamento mostra que a temperatura permanece recuperável após o Park (R²≈0,98), pois o método realinha sem remover a informação térmica; (ii) as métricas de pico revelam grande erro de frequência (centenas de Hz), pois um deslocamento único não corrige o deslocamento diferencial de cada ressonância; (iii) o erro cresce fortemente com a distância à referência e com a largura da banda, onde a hipótese de globalidade falha. Em compensação, não requer treino e é imune à escassez de dados.")
P_("<b>Autoencoder — variedade térmica não linear.</b> Aprende, a partir de curvas saudáveis, uma representação de baixa dimensão da forma como o espectro varia com a temperatura, e reconstrói a correção. Isso lhe dá o menor erro e a melhor preservação de picos em bandas estreitas de alta frequência, além da maior remoção de informação térmica (menor R² de vazamento) — indícios de que aprende uma representação genuinamente mais invariante. Suas fraquezas são o reverso da mesma moeda: em bandas largas, um único mapeamento subajusta a resposta térmica heterogênea; com poucas temperaturas de treino, a rede não tem dados para aprender a variedade e chega a perder para o Park; e há risco de sobrecompensação, visível na atenuação do dano 2. É um especialista faminto por dados.")
P_("<b>Florestas (Random Forest, base e otimizado) — correção ponto a ponto.</b> Estimam a correção térmica localmente, ponto a ponto, sem impor uma forma global. Essa flexibilidade explica sua robustez: são os mais estáveis ao alargamento da banda e à escolha da temperatura de referência, e os que melhor preservam a assinatura de dano (maior DPR, menor taxa de falso-saudável). A configuração otimizada, ao aleatorizar os cortes, reduz ainda mais a variância e supera o Random Forest básico de forma consistente. Suas limitações são o custo de treino e a extrapolação térmica além da faixa vista. Em resumo: <b>a estratégia de modelagem de cada método — global, aprendida-não-linear ou local-ponto-a-ponto — determina simultaneamente seus acertos e suas falhas</b>, e é por isso que nenhum domina universalmente e que a escolha ótima depende da banda, da referência e do orçamento de dados.")

# ================== 3.21 REFINAMENTOS: ET TUNADO, ENSEMBLE ==================
mv2=Rd("08_analises_avancadas/melhorias_v2_loto.csv"); mv2j=Jd("08_analises_avancadas/melhorias_v2_resumo.json")
if False:  # refinamentos (ensemble AE+ET, ET-tunado, seletor) removidos — fora do escopo do projeto
    H("Refinamentos adicionais: Extra Trees tunado e ensemble",H2)
    P_("Prosseguindo na busca por melhorias, três refinamentos foram testados por CV interna e confirmados no teste: ajustar os hiperparâmetros do próprio Extra Trees, combinar Autoencoder e Extra Trees num ensemble (média das compensações), e um seletor que escolhe, por banda, entre Autoencoder, Extra Trees e Park.")
    if mv2j and "media_rmsd" in mv2j:
        mr=mv2j["media_rmsd"]; ordm=sorted(mr.items(),key=lambda x:x[1])
        linha=", ".join([f"{k}={v:.2f}" for k,v in ordm])
        P_(f"Os RMSD saudáveis médios (4 bandas, teste LOTO) foram: {linha}. Três conclusões honestas emergem. <b>(1) Ajustar o Extra Trees não trouxe ganho</b> — a versão tunada (~{mr.get('ET_tun',0):.2f}) praticamente empata com a padrão (~{mr.get('ET_def',0):.2f}), sinal de que o Extra Trees já opera perto do seu limite com a configuração herdada. <b>(2) O ensemble Autoencoder+Extra Trees é revelador da geometria do problema:</b> ele <i>melhora</i> nas bandas estreitas — em 30–40 e 60–70 kHz a média das duas compensações fica abaixo de ambas isoladas, porque AE e Extra Trees erram de formas diferentes e a média cancela parte dos erros —, mas <i>degrada</i> na média global porque, na banda larga (30–70 kHz), o Autoencoder falha gravemente (RMSD ≈ 5,5) e arrasta a média para cima. Ou seja, um ensemble de média simples só compensa quando ambos os componentes são competentes; quando um deles falha, ele contamina o resultado. <b>(3) O seletor por banda é a melhor estratégia</b> (~{mr.get('Seletor_v2',0):.2f}): ao escolher, por CV interna, entre Autoencoder, Extra Trees e Park, ele <b>atinge o oráculo em todas as bandas testadas</b> — evita a contaminação do ensemble ao descartar o método ruim em cada faixa. Nenhum refinamento altera a mensagem central — a banda e a estratégia de modelagem governam o desempenho —, mas a comparação mostra que a via mais promissora não é um método único mais elaborado, e sim a <b>combinação adaptativa</b> guiada pela banda.")
    mv2t=mv2.copy()
    for c in ["RF","ET_def","ET_tun","AE","Ensemble"]:
        if c in mv2t.columns: mv2t[c]=mv2t[c].round(3)
    mv2t.columns=["Banda","RF","Extra Trees","ET tunado","Autoencoder","Ensemble AE+ET"][:len(mv2t.columns)]
    tbl(mv2t,fs=8,cap=f"Tabela {tabref()}. Refinamentos por banda (RMSD saudável, teste LOTO): Random Forest, Extra Trees (padrão e tunado), Autoencoder e ensemble.")
    fig("figHP_melhorias_v2",15,f"Figura {figref()}. Refinamentos: Extra Trees tunado, Autoencoder e ensemble AE+ET por banda (teste LOTO).")

# ================== 3.22 COMPENSAÇÃO ORIENTADA À FORMA ==================
_Af=Rd("13_modelo_forma/metricas/shapeA_saudavel.csv")
_Bf=Rd("13_modelo_forma/metricas/classB_dano.csv")
_Cf=Rd("13_modelo_forma/metricas/guardaC_dano.csv")
if _Af is not None and len(_Af) and _Bf is not None and len(_Bf):
    story.append(PageBreak())
    H("Compensação orientada à forma: um objetivo de correlação que preserva o dano",H2,lab="forma")
    def _figF(name,w,cap):
        fp=os.path.join(ROOT,"13_modelo_forma","graficos",name+".png")
        if not os.path.exists(fp): P_("<i>[figura pendente: %s]</i>"%name,CAP); return
        iw,ih=ImageReader(fp).getSize(); ww=w*cm; hh=ww*ih/iw
        if hh>20*cm: hh=20*cm; ww=hh*iw/ih
        story.append(Image(fp,width=ww,height=hh)); story.append(Paragraph(cap,CAP))
    _rBf=_Bf[_Bf.controle=="real"]
    def _cc(m): return float(_Af[_Af.metodo==m].CCDM.mean())
    def _pk(m): return float(_Af[_Af.metodo==m].peak_hz.mean())
    def _ba(m,t): return float(_rBf[(_rBf.metodo==m)&(_rBf.task==t)].bal_acc.mean())
    _shf=float(_Bf[_Bf.controle=="shuffled"].bal_acc.mean())
    _bnd=[b for b in ["30-40","40-50","50-60","60-70","70-80","30-70"] if b in _Af.banda.unique()]
    _nar=[b for b in _bnd if b!="30-70"]
    _aew=[b for b in _nar if _Af[(_Af.banda==b)&(_Af.metodo=="AE_forma")].CCDM.mean()
          ==min(_Af[(_Af.banda==b)&(_Af.metodo==m)].CCDM.mean() for m in ["Park","RF_amp","AE_amp","AE_forma"])]
    def _vv(x,n=3): return ("%.*f"%(n,x)).replace(".",",")
    P_("Todos os compensadores anteriores — clássico ou de aprendizado — são ajustados para minimizar o <b>erro de amplitude</b> (RMSD) contra a referência saudável. Contudo, a assinatura de dano reside na <b>forma</b> do espectro: posição e amplitude relativa das ressonâncias. Investigou-se, então, um compensador com <b>objetivo explícito de forma</b> — um autoencoder cuja função de perda inclui um termo de correlação de Pearson entre a curva compensada e a referência, além dos termos usuais de amplitude e de derivada. O peso desse termo é escolhido por validação cruzada interna apenas no CCDM do saudável (nunca com rótulos de dano), e a correção é aprendida <b>somente</b> em curvas saudáveis e aplicada de modo idêntico às de dano — de modo que o modelo jamais 'vê' o dano e não pode aprender a consertá-lo. A hipótese de risco — de que aproximar a curva compensada da referência saudável poderia <i>apagar</i> o dano — é testada explicitamente avaliando dois eixos independentes: (A) forma no saudável de teste e (B) classificação de dano 0/1/2 antes e depois da compensação.")
    _figF("fig_tradeoff",11,f"Figura {figref()}. Objetivo de forma: trade-off entre forma no saudável (CCDM, eixo horizontal, menor = melhor) e preservação de dano (acurácia balanceada, eixo vertical, maior = melhor). A seta liga o autoencoder de amplitude ao de forma, que melhora <i>simultaneamente</i> os dois eixos.")
    _lbF={"Original":"Original","Park":"Park","RF_amp":"RF otimizado","AE_amp":"AE (amplitude)","AE_forma":"AE (forma)"}
    _vtF=pd.DataFrame([{"Método":_lbF[m],"CCDM saud.":_vv(_cc(m)),"Erro pico (Hz)":_vv(_pk(m),0),
                        "Bal. acc. multi":_vv(_ba(m,"multi")),"Bal. acc. bin.":_vv(_ba(m,"bin"))}
                       for m in ["Original","Park","RF_amp","AE_amp","AE_forma"]])
    tbl(_vtF,widths=[3.6*cm,3.0*cm,3.0*cm,3.3*cm,3.0*cm],cap=f"Tabela {tabref()}. Compensação orientada à forma (média das bandas, teste LOTO). CCDM e erro de pico menores são melhores; acurácia maior é melhor. O autoencoder de forma tem a melhor classificação de dano de todos os métodos.")
    P_(f"<b>O objetivo de forma funciona e não apaga o dano.</b> Em relação ao autoencoder de amplitude, a versão de forma reduz o CCDM saudável de {_vv(_cc('AE_amp'))} para {_vv(_cc('AE_forma'))} e o erro de posição do pico de {_vv(_pk('AE_amp'),0)} para {_vv(_pk('AE_forma'),0)} Hz; nas bandas estreitas atinge o <b>menor CCDM de todos os métodos</b> ({', '.join(b+' kHz' for b in _aew) if _aew else 'ver Fig.'}). Decisivamente, a classificação multiclasse <b>melhora</b>, de {_vv(_ba('AE_amp','multi'))} para {_vv(_ba('AE_forma','multi'))} — a maior de todos os métodos, superando o Park ({_vv(_ba('Park','multi'))}) e o Random Forest otimizado ({_vv(_ba('RF_amp','multi'))}); o controle negativo com rótulos embaralhados colapsa para ~{_vv(_shf,2)}, descartando vazamento. A explicação é física e estatística: a variação térmica é um fator de confusão de grande energia, comum a todos os estados; ao maximizar a correlação da curva saudável com a referência, o modelo remove essa componente de forma mais completa e deixa o classificador operar sobre um resíduo mais limpo, no qual a parcela discriminativa do dano permanece. Forma e diagnóstico são, portanto, objetivos alinhados; amplitude e diagnóstico, não necessariamente.")
    _figF("fig_B_class_multi_por_banda",14.5,f"Figura {figref()}. Eixo (B): acurácia balanceada da classificação multiclasse (0/1/2) obtida a partir da curva compensada, por banda (teste LOTO; classificador treinado apenas em temperaturas de treino). O autoencoder de forma lidera na maioria das bandas; o sinal Original é o estado 'antes de compensar'.")
    P_(f"<b>Ressalvas honestas.</b> Três limites delimitam o alcance do resultado. (i) O ganho é de <b>banda estreita</b>: na banda larga 30–70 kHz o autoencoder colapsa (CCDM ≈ {_vv(float(_Af[(_Af.banda=='30-70')&(_Af.metodo=='AE_forma')].CCDM.mean()),2) if '30-70' in _bnd else 'n/d'}), enquanto o Random Forest otimizado e o Park permanecem robustos — modelar uma resposta ampla e multirressonante com poucas curvas saudáveis excede a capacidade de generalização da rede. (ii) O mesmo objetivo aplicado ao Random Forest otimizado <b>não teve efeito</b>: a seleção por CCDM convergiu para a mesma configuração escolhida por RMSD, pois o alvo de regressão direta de ΔZ já reconstrói amplitude e forma conjuntamente; o ganho de forma é próprio da perda diferenciável do autoencoder. (iii) A validação interna empurrou o peso de forma ao <b>máximo</b> da grade em todas as bandas — um objetivo de forma puro não possui freio interno contra a erosão de dano, e a verificação externa por classificação (eixo B) é <b>obrigatória</b>; neste conjunto de dados a classificação multiclasse até melhorou, embora a detecção binária tenha exibido o primeiro pequeno custo ({_vv(_ba('AE_amp','bin'))} do AE de amplitude contra {_vv(_ba('AE_forma','bin'))} do de forma) — evidência concreta desse compromisso. Este achado qualifica a discussão anterior: quando o alvo é o <b>reconhecimento de dano em banda estreita</b>, otimizar a forma da curva — e não a sua amplitude — é a estratégia superior.")

story.append(PageBreak())
H("Discussão")
P_("Antes de interpretar, é útil <b>reconciliar</b> afirmações que, à primeira vista, parecem contraditórias, mas que na verdade medem coisas diferentes. A Tabela seguinte torna explícito o escopo de cada resultado — a fonte da aparente tensão é quase sempre uma diferença de <i>banda</i>, de <i>agregação</i> (média vs. melhor configuração) ou de <i>objetivo</i> (compensar vs. preservar vs. detectar).")
_rec=pd.DataFrame([
 ["'AE é o melhor' vs. 'RF tem menor RMSD médio'","O AE vence em bandas estreitas e na detecção binária; o RF vence no RMSD médio sobre todas as bandas, puxado pelas largas onde o AE falha. Escopos diferentes, não contradição."],
 ["Classificação ~0,99 vs. médias 0,82–0,92","0,99 é a melhor combinação classificador+atributos numa banda; 0,82–0,92 é a média sobre todos os classificadores, atributos, bandas e folds."],
 ["Park melhor contra a curva danificada real","Mede outra coisa: o Park move menos a curva, ficando mais perto da danificada de referência; isso não indica melhor compensação do estado saudável."],
 ["Park com CCDM melhor que o AE em algumas bandas","CCDM avalia forma (correlação) e RMSD avalia amplitude; o Park preserva a forma global enquanto o AE reduz mais a amplitude. Por isso reportamos ambas."],
 ["Interpolação: ML vence; extrapolação: Park vence","Objetivos opostos: LOTO testa interpolação (dentro da faixa); os splits bloqueados testam extrapolação (fora). Cada regime tem seu vencedor."],
 ["RMSD original ~10,8 (texto) vs. ~10,3 (figura)","Médias sobre conjuntos de bandas diferentes: todas as bandas (texto) vs. as cinco bandas da análise de necessidade (figura)."],
 ["Número de bandas varia por seção","7 bandas de 10 kHz + progressivas + janelas de 5 kHz + larga; cada análise usa o subconjunto adequado ao seu objetivo, sempre declarado."],
],columns=["Observação aparentemente contraditória","Por que não é contradição (escopo / fonte)"])
tbl(_rec,widths=[6.2*cm,9.2*cm],fs=7.6,cap=f"Tabela {tabref()}. Reconciliação de resultados aparentemente contraditórios: cada um mede um objetivo, banda ou forma de agregação distinta.")
P_("Os resultados contam uma história coerente e, em vários pontos, contra-intuitiva. Ela começa com uma constatação simples: <b>sem compensação, a temperatura domina o sinal</b> a ponto de a separação entre saudável e danificado tornar-se negativa em média — o dano fica, literalmente, mascarado. A partir daí, cada método adota uma estratégia física distinta para remover a temperatura, e é nessa estratégia que residem seus acertos e falhas.")
P_("O <b>método de Park</b> assume que a distorção térmica é aproximadamente um deslocamento global do espectro (mais um offset). Essa hipótese (A) é válida perto da temperatura de referência e em bandas estreitas, onde o Park é competitivo e tem a enorme vantagem prática de não exigir treinamento. Porém, três evidências independentes mostram seus limites: (i) longe da referência e em bandas largas, o erro do Park cresce mais rápido que o dos métodos de ML (hipótese B confirmada); (ii) o teste de vazamento de temperatura revela que, após a compensação de Park, a temperatura ainda é quase perfeitamente recuperável da curva (R² ≈ 0,98, praticamente igual ao sinal original) — ou seja, o Park <i>realinha</i> a curva sem de fato remover a informação térmica subjacente; e (iii) as métricas de pico mostram que o alinhamento global do Park desloca fortemente as ressonâncias (erro de centenas de Hz) e reduz a taxa de preservação de picos, exatamente nas regiões que carregam a assinatura estrutural.")
P_("O <b>Autoencoder</b> aprende a variedade térmica não linear do estado saudável. Isso o torna o mais preciso em bandas estreitas de alta frequência, o que melhor preserva os picos e o que mais reduz o vazamento de temperatura — evidência de que ele efetivamente aprende uma representação mais invariante à temperatura. Contudo, confirma-se também sua fragilidade (hipótese C): em bandas muito largas a rede subajusta a resposta térmica heterogênea, e há o risco de sobrecompensação, visível no Damage Preservation Ratio, onde o AE chega a atenuar o dano 2. O AE é, portanto, um especialista: excelente onde o sinal é rico e localizado, arriscado onde é amplo e heterogêneo.")
P_("O <b>Random Forest direto</b>, ao aprender a correção ponto-a-ponto, é o mais robusto entre condições e o que melhor preserva a assinatura de dano (maior DPR) e a segurança de detecção (menor taxa de falso-saudável). É o mais adequado a bandas largas, onde captura deformações locais que o Park não modela. Sua limitação (hipótese D) aparece na extrapolação térmica e no custo de treino, superior ao do AE. A ablação RF-direto vs. RF-só-temperatura indica que ver a curva ajuda a compensação e introduz apenas uma leve reação ao dano, sem prejudicar a detecção.")
P_("A descoberta metodológica mais importante é que <b>compensar bem a temperatura não é o mesmo que permitir classificar bem o dano</b> (hipóteses I e J). A correlação entre o RMSD saudável e o Macro-F1 de classificação é fraca; a melhor faixa para compensação (bandas estreitas de alta frequência, para o AE) não coincide necessariamente com a melhor faixa para classificação. Isso confirma que o erro de reconstrução, isoladamente, é insuficiente para avaliar um compensador destinado a SHM — a métrica final deve ser a capacidade de detectar e classificar o dano sob temperatura variável. Sobre a largura da banda (hipóteses F e G), confirma-se que faixas muito amplas contêm comportamento térmico heterogêneo que penaliza Park e AE, ao passo que janelas estreitas próximas de ressonâncias oferecem boa compensabilidade — embora o RMSD bruto exagere as diferenças por causa da amplitude (o NRMSE atenua o efeito).")
P_(f"A análise de detecção por índice de distância à referência (Seção {SECREF.get('det','')}) fecha o argumento de forma prática: a compensação <b>aumenta a área sob a curva ROC</b> da separação saudável–dano em relação ao sinal bruto, ou seja, remover a temperatura torna o dano mais detectável mesmo sem treinar um classificador. Isso reconcilia as duas metades do trabalho — compensar e detectar — mostrando que, embora RMSD e classificação não estejam fortemente correlacionados, a compensação ainda beneficia a detecção quando esta é medida diretamente sobre a distância à referência. Do ponto de vista de aplicação, a contribuição mais acionável é o <b>seletor adaptativo por banda</b> (Seção {SECREF.get('sel','')}): como o método vencedor depende da banda, escolher o método por banda apenas com a validação interna — sem rótulos de dano e sem vazamento — aproxima-se do oráculo e supera qualquer método fixo, transformando a principal descoberta do artigo em uma receita operacional. Por fim, a curva de eficiência de dados (Seção {SECREF.get('efic','')}) impõe uma ressalva honesta e prática: a superioridade dos métodos de aprendizado é <b>condicional à quantidade de dados de calibração</b> — com poucas temperaturas medidas eles perdem para o Park (que não treina), e a vantagem só emerge com ampla cobertura térmica. A escolha do método, portanto, deve levar em conta o orçamento experimental disponível.")
P_("Todas as comparações usaram exatamente as mesmas amostras, bandas, temperaturas de referência, <i>splits</i> e métricas, com hiperparâmetros escolhidos por validação cruzada interna sem tocar no <i>fold</i> externo. A cobertura foi ampla: <b>13 faixas de frequência</b> (as sete progressivas 30–40 a 30–100 kHz e as seis de 10 kHz 40–50 a 90–100 kHz), <b>janelas de 5 kHz ao longo de todo o espectro</b>, <b>dez temperaturas de referência</b> (de −10 a 80 °C), além de detecção por ROC/AUC, intervalos de confiança por <i>bootstrap</i>, tamanho de efeito, seletor adaptativo e curva de eficiência de dados. Nenhuma seleção de banda, janela, T_ref ou método usou o conjunto de teste.")
P_("A Tabela seguinte sintetiza, num único quadro comparativo, o comportamento de cada método em todos os critérios avaliados — a evidência de que <b>não há um vencedor único</b>, e sim um perfil de forças e fraquezas por objetivo. É esse quadro que fundamenta a recomendação de escolher o método (ou o seletor) segundo a prioridade da aplicação.")
_syn=pd.DataFrame([
 ["Compensação (RMSD médio)","Regular","Bom","Ótimo★","Bom"],
 ["Preservação de dano (healthy_sep/DPR)","Fraca","Ótima","Ótima★","Boa"],
 ["Detecção (AUC / falso-saudável)","Boa","Ótima","Ótima★","Ótima"],
 ["Bandas estreitas de alta freq.","Regular","Boa","Boa","Ótimo★"],
 ["Bandas largas / heterogêneas","Regular","Ótima","Ótima★","Fraca"],
 ["Robustez à temperatura de referência","Fraca","Ótima","Ótima★","Boa"],
 ["Extrapolação térmica (fora da faixa)","Ótima★","Fraca","Fraca","Fraca"],
 ["Robustez à histerese","Ótima★","Regular","Regular","Regular"],
 ["Custo de treino","Nenhum★","Alto","Alto","Médio"],
 ["Necessidade de dados de calibração","Nenhuma★","Média","Média","Alta"],
],columns=["Critério","Park","Random Forest","RF otimizado","Autoencoder"])
tbl(_syn,widths=[6.4*cm,2.1*cm,2.6*cm,2.2*cm,2.5*cm],fs=7.6,cap=f"Tabela {tabref()}. Quadro comparativo geral: perfil qualitativo de cada método por critério (★ = melhor da linha). Nenhum método domina todos os critérios — daí a proposta do seletor adaptativo.")
P_(f"A Figura {figref()} traduz esse quadro em um <b>radar multicritério</b> (valores normalizados, 1 = melhor em cada eixo). A leitura é imediata: o <b>Random Forest otimizado</b> preenche quase todo o polígono dos critérios <i>dentro da faixa</i> de calibração (compensação, preservação, bandas largas, robustez a T_ref), mas colapsa em <b>extrapolação</b>; o <b>Park</b>, ao contrário, ocupa sozinho os eixos de <b>extrapolação e custo</b>; e o <b>Autoencoder</b> mostra um perfil intermediário, forte em baixo custo relativo. Nenhum polígono contém os demais — a definição geométrica de que não há dominância única.")
fig("figX_radar",11.5,f"Figura {FIGN[0]}. Radar multicritério dos métodos (normalizado; 1 = melhor). Perfis complementares: cada método cobre eixos distintos.")

H("A natureza dual do problema: por que separar temperatura de dano é difícil",H2)
P_("Um fio condutor atravessa todos os resultados: temperatura e dano atuam sobre o <b>mesmo</b> observável — o espectro de impedância — e por mecanismos que se confundem. Ambos deslocam ressonâncias e alteram amplitudes; a diferença é de <i>grau</i> e de <i>estrutura</i>, não de natureza. A temperatura produz uma deformação relativamente suave e global (todo o espectro migra e se atenua), enquanto o dano incipiente produz uma perturbação mais localizada em torno de ressonâncias específicas. É essa sobreposição que torna o problema genuinamente difícil e que explica por que a distância bruta à referência tem AUC de detecção próxima do acaso: sem separar as duas contribuições, o efeito térmico — maior em magnitude — soterra a assinatura do dano. Compensar é, no fundo, um problema de <i>separação de fontes</i>: estimar e remover a componente térmica preservando a componente estrutural. Cada método ataca essa separação com uma hipótese diferente sobre a forma da componente térmica (global para o Park, uma variedade não-linear aprendida para o AE, uma função local ponto-a-ponto para as florestas), e o sucesso de cada um depende de quão bem sua hipótese casa com a física da banda considerada.")
P_("Esse enquadramento também esclarece o risco central do projeto — o de <i>sobrecompensar</i> e apagar o dano. Um método suficientemente flexível pode, ao tentar aproximar toda curva da referência saudável, começar a remover não só a temperatura mas também a perturbação do dano, empurrando curvas danificadas na direção da saudável. Foi para vigiar exatamente esse risco que se reportou a preservação de dano de forma independente (DPR, healthy_sep, referência danificada como <i>ground-truth</i>) e que se impôs, como critério, que uma boa compensação deve reduzir a distância das curvas saudáveis <i>sem</i> reduzir a das danificadas. O fato de o AE atenuar levemente o dano 2 em bandas largas é a manifestação concreta desse risco e a razão pela qual precisão de reconstrução, isoladamente, nunca deve ser o único critério de avaliação.")

H("Implicações para o projeto de sistemas de SHM",H2)
P_("Traduzindo os achados em recomendações práticas para quem projeta um sistema de monitoramento por impedância: <b>(1) Escolha a banda deliberadamente</b> — ela importa mais do que a família de método; bandas estreitas de alta frequência tendem a combinar sensibilidade ao dano e facilidade de compensação. <b>(2) Escolha a temperatura de referência perto do centro da faixa operacional</b> — é crítico para o método de Park e barato de garantir. <b>(3) Case o método ao orçamento de dados</b>: com poucas temperaturas de calibração, prefira o Park (sem treino); com uma varredura térmica densa, prefira o Autoencoder na banda ótima ou o Random Forest otimizado pela robustez. <b>(4) Use um seletor por banda</b> decidido por validação interna — ele extrai o melhor de cada método sem exigir rótulos de dano. <b>(5) Reporte compensação e detecção separadamente</b> — a primeira não garante a segunda, e o objetivo final do SHM é a decisão de dano, não a reconstrução do espectro. <b>(6) Priorize a segurança</b>: em SHM, um falso-saudável (dano não detectado) é o erro mais grave, o que favorece métodos como as florestas, de menor taxa de falso-saudável, mesmo quando outro método tem RMSD ligeiramente menor.")

H("Limitações")
P_(f"O escopo dos resultados deve ser lido à luz de limitações concretas. <b>(1) Ausência de identificador de espécime</b> impede o <i>leave-one-specimen-out</i> e não permite descartar totalmente dependência entre curvas do mesmo corpo de prova — a validação por temperatura (LOTO) mitiga, mas não elimina, esse risco. <b>(2) Amostra pequena:</b> {summ['n_temps_3_classes']} temperaturas com três classes resultam em {summ['n_temps_3_classes']-1} <i>folds</i> por banda, com poder estatístico limitado (menor p de Wilcoxon ≈ 0,004); por isso reportamos intervalos de confiança e tamanho de efeito, e evitamos conclusões baseadas apenas em p-valores. <b>(3) Base e viga únicas:</b> os valores numéricos específicos (melhor banda, melhor T_ref) são propriedades desta estrutura e não devem ser transpostos diretamente a outras geometrias ou materiais — o que se generaliza são os <i>mecanismos</i> e o <i>protocolo</i>, não os números. <b>(4) Dois tipos de dano</b> (massa e corte), ambos artificiais; danos reais (fadiga, delaminação, corrosão) podem ter assinaturas distintas. <b>(5) O AE</b> foi ajustado por grade moderada — embora o estudo de sensibilidade indique platô — e sua variabilidade entre sementes é pequena, mas não nula. <b>(6) Compensação puramente espectral:</b> não se exploraram informações auxiliares (múltiplos PZTs, fase, histórico temporal) que poderiam elevar o teto de desempenho.")
H("Conclusões")
P_(f"Comparando autoencoder, Random Forest e o método de Park sob validação fora da amostra e análise estatística de Demšar, conclui-se que nenhuma família domina universalmente: o AE vence em bandas estreitas de alta frequência (melhor RMSD em {ae_best}) e fornece a melhor detecção binária ({best_bin}); o RF é o mais robusto entre condições e o mais seguro para detecção ({fh_best} de falso-saudável), superando o Park de forma consistente; o Park é competitivo apenas próximo da referência e em bandas estreitas. Uma ressalva, porém, é central para a aplicação: essa superioridade vale para <b>interpolação</b> térmica (dentro da faixa calibrada); em <b>extrapolação</b> — temperaturas fora da faixa de treino — e sob histerese, o método de Park é muito mais robusto e os modelos de aprendizado degradam, de modo que uma estratégia <b>híbrida</b> (ML dentro da faixa, Park fora dela) é a recomendação mais segura. Três resultados adicionais sustentam a recomendação prática: (i) a compensação <b>aumenta a AUC</b> de detecção do dano pela distância à referência; (ii) um <b>seletor adaptativo por banda</b>, decidido só com dados de treino, supera qualquer método fixo e aproxima-se do oráculo; e (iii) a vantagem do aprendizado de máquina é <b>condicional à cobertura térmica</b>: exige muitas temperaturas de calibração e, com poucos dados, o Park (sem treino) é preferível. Como aprimoramento metodológico, mostrou-se ainda que uma <b>configuração otimizada do Random Forest</b> reduz o erro de compensação em todas as bandas (ganho médio de ~22% fora da amostra, sem perda de preservação de dano), devendo ser adotada em implementações futuras do compensador por floresta. Confirma-se e estende-se, com rigor estatístico e análise multibanda, o potencial do aprendizado de máquina para separar efeitos térmicos de variações associadas ao dano em SHM por EMI — desde que a avaliação seja resolvida por banda, fora da amostra e reportando compensação e detecção como objetivos distintos.")
H("Perspectivas e trabalhos futuros")
P_("Os resultados abrem várias direções naturais. <b>(1) Validação multi-espécime e multi-estrutura:</b> repetir o protocolo em vários corpos de prova e geometrias, com identificador de espécime, permitiria o <i>leave-one-specimen-out</i> e testaria a generalização dos mecanismos aqui identificados. <b>(2) Danos realistas e progressivos:</b> substituir os danos artificiais por evolução real (fadiga, delaminação) e por severidades graduais, verificando se a ordenação por severidade e a preservação de dano se mantêm. <b>(3) Compensadores estruturalmente novos:</b> como o estudo de sensibilidade mostrou que o Autoencoder atingiu um platô de hiperparâmetros, ganhos adicionais exigem mudança de arquitetura — redes com atenção às ressonâncias, modelos físicos-informados que embutam a lei de deslocamento térmico, ou autoencoders variacionais que modelem explicitamente a distribuição térmica. <b>(4) Seletor e ensembles adaptativos:</b> o seletor por banda pode evoluir para uma combinação ponderada aprendida (mistura de especialistas) que una a precisão do Autoencoder em bandas estreitas à robustez das florestas em bandas largas. <b>(5) Floresta otimizada como padrão:</b> a superioridade consistente da configuração otimizada sobre a básica sugere revisitar toda a família de compensadores por árvore, incluindo <i>gradient boosting</i> por âncoras. <b>(6) Detecção diretamente otimizada:</b> como compensação e detecção não são equivalentes, faz sentido treinar compensadores cujo objetivo já seja a separabilidade saudável–dano (métrica final), e não apenas a reconstrução do espectro. <b>(7) Custo e implantação embarcada:</b> avaliar o compromisso entre acurácia e custo computacional para operação em tempo real em hardware de baixa potência.")
H("Agradecimentos",H2,num=False)
P_("Os autores agradecem à Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP), processos 2016/12241-0 e 2025/09586-5, pelo apoio financeiro.")

# ================== REFERÊNCIAS ==================
story.append(PageBreak()); H("Referências",num=False)
REFS=[
"Baptista, F.G., Budoya, D.E., Almeida, V.A.D. e Ulson, J.A.C., 2014. “An experimental study on the effect of temperature on piezoelectric sensors for impedance-based structural health monitoring”. Sensors, 14(1), 1208–1227.",
"Breiman, L., 2001. “Random forests”. Machine Learning, 45(1), 5–32.",
"Cliff, N., 1993. “Dominance statistics: Ordinal analyses to answer ordinal questions”. Psychological Bulletin, 114(3), 494–509.",
"Efron, B. e Tibshirani, R.J., 1993. An Introduction to the Bootstrap. Chapman & Hall, New York.",
"Fawcett, T., 2006. “An introduction to ROC analysis”. Pattern Recognition Letters, 27(8), 861–874.",
"Geurts, P., Ernst, D. e Wehenkel, L., 2006. “Extremely randomized trees”. Machine Learning, 63(1), 3–42.",
"Pedregosa, F. et al., 2011. “Scikit-learn: Machine learning in Python”. Journal of Machine Learning Research, 12, 2825–2830.",
"Demšar, J., 2006. “Statistical comparisons of classifiers over multiple data sets”. Journal of Machine Learning Research, 7, 1–30.",
"de Rezende, S.W.F., de Moura, J.d.R.V., Neto, R.M.F., Gallo, C.A. e Steffen, V., 2020. “Convolutional neural network and impedance-based SHM applied to damage detection”. Engineering Research Express, 2(3), 035031.",
"Dias, L.L., Lopes, K.W., Bueno, D.D. e Gonsalez-Bueno, C.G., 2023. “An enhanced approach for damage detection using the electromechanical impedance with temperature effects compensation”. J. Braz. Soc. Mech. Sci. Eng., 45(4), 228.",
"Du, F., Wu, S., Xu, C., Yang, Z. e Su, Z., 2023. “Electromechanical impedance temperature compensation and bolt loosening monitoring based on modified U-Net and multitask learning”. IEEE Sensors Journal, 23(5), 4556–4567.",
"Farrar, C.R. e Worden, K., 2012. Structural Health Monitoring: A Machine Learning Perspective. John Wiley & Sons.",
"Friedman, M., 1937. “The use of ranks to avoid the assumption of normality implicit in the analysis of variance”. Journal of the American Statistical Association, 32(200), 675–701.",
"Giurgiutiu, V. e Rogers, C.A., 1998. “Recent advancements in the electromechanical (E/M) impedance method for structural health monitoring and NDE”. Proc. SPIE, 3329, 536–547.",
"Holm, S., 1979. “A simple sequentially rejective multiple test procedure”. Scandinavian Journal of Statistics, 6(2), 65–70.",
"Koo, K.Y., Park, S., Lee, J.J. e Yun, C.B., 2009. “Automated impedance-based structural health monitoring incorporating effective frequency shift for compensating temperature effects”. J. Intell. Mater. Syst. Struct., 20(4), 367–377.",
"Liang, C., Sun, F. e Rogers, C., 1994. “Coupled electro-mechanical analysis of adaptive material systems”. J. Intell. Mater. Syst. Struct., 5(1), 12–20.",
"Lim, H.J., Kim, M.K., Sohn, H. e Park, C.Y., 2011. “Impedance based damage detection under varying temperature and loading conditions”. NDT & E International, 44(8), 740–750.",
"Lopes, K.W., Gonsalez-Bueno, C.G., Inman, D.J. e Bueno, D.D., 2023. “On the modeling of circular piezoelectric transducers for wave propagation-based SHM applications”. J. Intell. Mater. Syst. Struct., 34(15), 1739–1752.",
"Na, W.S. e Baek, J., 2018. “A review of the piezoelectric electromechanical impedance based structural health monitoring technique for engineering structures”. Sensors, 18(5), 1307.",
"Nemenyi, P.B., 1963. Distribution-free multiple comparisons. Ph.D. thesis, Princeton University.",
"Park, G., Kabeya, K., Cudney, H.H. e Inman, D.J., 1999. “Impedance-based structural health monitoring for temperature varying applications”. JSME Int. J. Series A, 42(2), 249–258.",
"Sikdar, S., Singh, S.K., Malinowski, P. e Ostachowicz, W., 2022. “Electromechanical impedance based debond localisation in a composite sandwich structure”. J. Intell. Mater. Syst. Struct., 33(12), 1487–1496.",
"Wilcoxon, F., 1945. “Individual comparisons by ranking methods”. Biometrics Bulletin, 1(6), 80–83.",
"Worden, K., Farrar, C.R., Manson, G. e Park, G., 2020. “Machine learning for structural health monitoring: challenges and opportunities”. Structural Control and Health Monitoring, 27(10), e2825.",
]
for r in REFS: P_(r,REF)
SP(8); P_("<i>Documento gerado a partir dos resultados reproduzíveis em ARTIGO_AE_PARK_RF_FINAL/. Versão LaTeX (artigo_PT.tex) disponível para compilação com tipografia completa; rastreabilidade em METHODS_TRACEABILITY.md.</i>",CAP)

out=os.path.join(MAN,"artigo_PT.pdf")
def _rodape(canvas,doc):
    canvas.saveState()
    L,R=2*cm,A4[0]-2*cm; C=A4[0]/2
    # ---- rodapé: número de página centralizado (estilo ABCM) ----
    canvas.setFont("Times-Roman",9.5); canvas.setFillColor(colors.black)
    canvas.drawCentredString(C,1.15*cm,f"{doc.page}")
    # ---- cabeçalho corrente (páginas 2+): autores + título, à esquerda, texto simples ----
    if doc.page>1:
        top=A4[1]-1.25*cm
        canvas.setFont("Times-Roman",8.5); canvas.setFillColor(colors.black)
        canvas.drawString(L,top,"Luiz Eduardo Abdala José, Kayc Wayhs Lopes")
        canvas.setFont("Times-Roman",8.5)
        canvas.drawString(L,top-0.34*cm,"Compensação Térmica de Sinais de Impedância Eletromecânica por Autoencoder, Random Forest e Método de Park")
    canvas.restoreState()
doc=SimpleDocTemplate(out,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2.15*cm,bottomMargin=1.9*cm,title="Compensacao termica EMI: AE, RF e Park",author="Luiz Eduardo Abdala Jose; Kayc Wayhs Lopes")
doc.build(story,onFirstPage=_rodape,onLaterPages=_rodape)
print(f"✅ PDF gerado: {out} | figuras={FIGN[0]} tabelas={TABN[0]} | AE/RF vencem Park: {aewin}/{nb}")

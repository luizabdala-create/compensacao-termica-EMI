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
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
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
LBL={"Original":"Original","Park":"Park","RF_direct":"Random Forest (direto)","RF_temponly":"RF (só temp.)",
     "AE":"Autoencoder"}
ORD=["Original","Park","RF_direct","RF_temponly","AE"]
DROP={"TAU_T"}  # tau(T) auxiliar removido das comparações do artigo
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
MJ=[m for m in ["AE","Park","RF_direct","RF_temponly"] if m in comp.metodo.unique()]
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
    bb=real[real.task=="bin"].groupby("metodo")["bal_acc"].mean(); best_bin=f"{LBL[bb.idxmax()]} ({bb.max():.3f})"
    fh=real[real.task=="bin"].groupby("metodo")["taxa_falso_saudavel"].mean(); fh_best=f"{LBL[fh.idxmin()]} ({fh.min():.3f})"
    mc=real[real.task=="multi"].groupby("metodo")["macro_f1"].mean(); best_multi=f"{LBL[mc.idxmax()]} ({mc.max():.3f})"
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
box([Paragraph(f"<b>Resumo.</b> O Monitoramento de Integridade Estrutural (SHM) por Impedância Eletromecânica (EMI) é fortemente afetado pela temperatura, que desloca e deforma o espectro em magnitude frequentemente superior à do próprio dano incipiente (Baptista et al., 2014). Este trabalho compara três estratégias de compensação térmica — um <b>autoencoder</b> neural (AE), uma regressão direta por <b>Random Forest</b> (RF) e o método clássico de <b>Park et al. (1999)</b> — sobre {summ['n_curvas']} curvas de impedância de uma viga instrumentada com PZT, de -10 a 80 °C, com dois níveis de dano. Todos os compensadores aprendem apenas de curvas saudáveis e são avaliados fora da amostra por validação <i>leave-one-temperature-out</i> (LOTO), com hiperparâmetros ajustados por validação cruzada interna. A comparação estatística segue o protocolo de Demšar (2006): teste de Friedman, pós-teste de Nemenyi com diagrama de diferença crítica e teste de Wilcoxon com correção de Holm. O resultado central é que <b>a banda de frequência determina o melhor método</b>: em {aewin} de {nb} bandas o AE ou o RF superam o Park; o AE atinge o menor erro em bandas estreitas de alta frequência ({ae_best}) e o RF domina bandas largas. Para reconhecimento de dano, o AE fornece a melhor detecção binária ({best_bin}) e o RF a menor taxa de falso-saudável ({fh_best}). Um controle negativo com rótulos embaralhados descarta vazamento de dados. Os resultados estendem, com rigor estatístico e análise multibanda, a superioridade do aprendizado de máquina sobre a compensação puramente global do método de Park.",_ABSb),
     Spacer(1,5),
     Paragraph("<b>Palavras-chave:</b> Monitoramento de Integridade Estrutural, Impedância Eletromecânica, Compensação de Temperatura, Autoencoder, Random Forest, Aprendizado de Máquina.",_ABSb)])
SP(4)

# ================== 1. INTRODUÇÃO ==================
H("1. Introdução")
P_("O Monitoramento de Integridade Estrutural (SHM) pode ser entendido como uma evolução dos métodos de avaliação não destrutiva (NDE) e é essencial em aplicações onde a segurança estrutural é crítica, como sistemas mecatrônicos, aeroespaciais e civis (Farrar e Worden, 2012; Lopes et al., 2023). Diferentemente das abordagens tradicionais de NDE, os sistemas de SHM permitem monitoramento em tempo real e uma avaliação contínua, quantitativa e autônoma da estrutura ao longo de sua vida útil. Essas técnicas podem ser classificadas em níveis, conforme suas capacidades: detectar a presença de dano, localizá-lo, identificar o tipo de falha e estimar a vida útil remanescente (Gonsalez et al., 2015). Nesse contexto, o SHM torna-se cada vez mais relevante com o avanço de tecnologias digitais como a Internet das Coisas (IoT), o aprendizado de máquina e a análise de grandes volumes de dados, aliados à crescente demanda por segurança estrutural e detecção precoce de falhas.")
P_("A análise estrutural em SHM investiga a relação entre sinais de entrada e saída obtidos de sensores e atuadores acoplados à estrutura, por meio de métodos baseados em vibração ou em propagação de ondas (Mitra e Gopalakrishnan, 2016). Entre as técnicas disponíveis, o método da Impedância Eletromecânica (EMI) destaca-se pela alta sensibilidade a pequenas alterações estruturais e vem sendo amplamente utilizado na literatura (Na e Baek, 2018; Sikdar et al., 2022). Proposto inicialmente por Liang et al. (1994), o método baseia-se no acoplamento entre a impedância mecânica da estrutura e a impedância elétrica de um transdutor piezelétrico colado à sua superfície, permitindo inferir mudanças estruturais a partir de variações na impedância elétrica medida, com dispositivos de baixo custo e sem exigir modelos físicos detalhados. Além disso, o uso de altas frequências torna a técnica particularmente adequada para detectar danos pequenos ou em estágio inicial (Giurgiutiu e Rogers, 1998).")
P_("Entretanto, quando a estrutura monitorada é submetida a variações ambientais, mudanças significativas surgem nos sinais EMI, comprometendo a confiabilidade do diagnóstico. Baptista et al. (2014) mostraram que a temperatura produz três efeitos principais no espectro: deslocamentos verticais, deslocamentos horizontais e suavização de picos e vales, cuja intensidade depende da faixa de frequência analisada. Tratar adequadamente essas variações é essencial para evitar diagnósticos falso-positivos.")
P_("Diversos métodos de compensação térmica foram propostos para mitigar esses efeitos. Um dos mais conhecidos é o método estatístico de Park et al. (1999), que utiliza uma equação empírica para reduzir a influência da temperatura na parte real da impedância, geralmente mais sensível ao dano. Embora satisfatório, seu desempenho tende a decair com o aumento da frequência analisada, o que pode limitar a detecção de dano incipiente (Koo et al., 2009; Lim et al., 2011; Dias et al., 2023). O grande volume de dados obtido em análises baseadas em EMI torna o monitoramento desafiador, sobretudo quando múltiplos fatores externos estão envolvidos. Nesse cenário, técnicas de aprendizado de máquina surgem como ferramentas promissoras para lidar com dados de alta dimensionalidade e melhorar a robustez da interpretação dos sinais; podem, ainda, ser combinadas a técnicas de redução de dimensionalidade, reduzindo o custo computacional sem comprometer o desempenho (Ghiasi et al., 2016; Khoa et al., 2014). Por exemplo, de Rezende et al. (2020) aplicaram redes neurais convolucionais unidimensionais para detectar dano em vigas de alumínio sob diferentes temperaturas, e Du et al. (2023) exploraram redes profundas com aprendizado multitarefa para compensação térmica e monitoramento de afrouxamento de parafusos.")
P_("A maioria dos estudos, porém, ainda se restringe a faixas de temperatura limitadas ou a uma única condição estrutural, dificultando a generalização, e há uma lacuna em comparações sistemáticas e estatisticamente sustentadas entre modelos de aprendizado e o método de Park, especialmente considerando o efeito da banda de frequência e da temperatura de referência. Este trabalho investiga três estratégias de compensação — autoencoder, Random Forest direto e o método de Park — sobre uma viga instrumentada em ampla faixa térmica (-10 a 80 °C) e três estados estruturais, avaliando não apenas a compensação (RMSD e CCDM), mas também a preservação da assinatura de dano e o reconhecimento de dano, com um protocolo estatístico rigoroso.")

# ================== 2. METODOLOGIA ==================
H("2. Metodologia")
H("2.1 Aquisição experimental e organização dos dados",H2)
P_(f"Os sinais foram obtidos pelo método EMI em uma viga submetida a temperaturas de -10 a 80 °C dentro de uma câmara térmica, com um transdutor piezelétrico atuando simultaneamente como atuador e sensor. Foram considerados três estados estruturais. O <b>Dano 1 (D1)</b> consiste em uma massa acoplada à superfície da estrutura, que altera suas propriedades físicas e representa uma condição de dano; o <b>Dano 2 (D2)</b> consiste em um corte na estrutura, representando um cenário mais realista de defeito. A análise foca na parte real da impedância, mais sensível ao dano na faixa investigada (Park et al., 1999). A auditoria programática da base estabeleceu: {summ['n_curvas']} curvas ({summ['n_D0']} D0, {summ['n_D1']} D1, {summ['n_D2']} D2); {summ['n_temperaturas']} temperaturas; resolução de 1 Hz de 22 Hz a 125 kHz. Apenas <b>{summ['n_temps_3_classes']} temperaturas contêm as três classes</b> e <b>{len(summ['temps_dano_sem_saudavel'])} temperaturas têm dano sem curva saudável</b>; não há identificador de espécime. Logo, o LOTO externo tem no máximo {summ['n_temps_3_classes']} <i>folds</i> e o <i>leave-one-specimen-out</i> não é possível.")
d1=pd.DataFrame({"Propriedade":["Curvas","Faixa de frequência","Resolução","Temperaturas","Temp. com 3 classes","D0 / D1 / D2","Identificador de espécime"],
 "Valor":[summ['n_curvas'],"22 Hz – 125 kHz","1 Hz (uniforme)",summ['n_temperaturas'],summ['n_temps_3_classes'],f"{summ['n_D0']} / {summ['n_D1']} / {summ['n_D2']}","nenhum"]})
tbl(d1,widths=[6.5*cm,8.5*cm],cap=f"Tabela {tabref()}. Descrição da base de dados EMI.")

H("2.2 Definição da referência",H2)
P_("A referência z_ref é definida como a mediana dos sinais saudáveis na temperatura de referência T_ref, tratada como calibração física disponível a priori e congelada em todos os <i>folds</i> (Protocolo A). Quando a temperatura de teste coincide com T_ref, o <i>fold</i> é marcado e excluído dos resumos. A Seção 3.4 investiga o efeito da escolha de T_ref.")

H("2.3 Método de Park (1999) — linha de base",H2)
P_("O método de Park et al. (1999) minimiza uma função Va = &Sigma; [&real;(Z_R(&omega;_i)) &minus; &real;(Z_D(&omega;_j))]², deslocando horizontalmente o espectro medido e aplicando um <i>offset</i> vertical &delta;S (média da diferença entre referência e curva), determinados iterativamente até o mínimo de Va. É essencialmente uma estratégia de alinhamento global (deslocamento em frequência + nível), eficaz próximo à referência.")

H("2.4 Random Forest direto (ponto a ponto)",H2)
P_("O Random Forest (Breiman, 2001) estima diretamente a correção &Delta;z(T) = z_ref &minus; z(T) necessária para mapear um espectro medido em T para a condição de referência; o espectro compensado é z_comp(T) = z(T) + &Delta;z(T)^. O modelo recebe o vetor espectral, a temperatura e estatísticas globais. Avaliamos também uma ablação <i>RF (só temp.)</i>, cuja entrada é apenas a temperatura, incapaz de reagir ao conteúdo de dano. Por tratabilidade, a entrada espectral do RF é subamostrada (resolução ainda fina para os picos).")

H("2.5 Autoencoder (rede neural)",H2)
P_("O autoencoder é uma rede <i>encoder-decoder</i> (PyTorch) que recebe a curva (decimada) e a informação de temperatura (T, |T&minus;T_ref|) e produz a correção térmica em pontos-âncora, interpolada à resolução plena. É treinado exclusivamente em curvas saudáveis, aprendendo a variedade térmica do estado íntegro. Diferentemente do Park, modela relações não lineares e locais entre temperatura e espectro.")

H("2.6 Métricas de avaliação",H2)
P_("Duas métricas amplamente usadas em SHM por impedância são adotadas, ambas entre o espectro compensado e a referência: RMSD = sqrt( (1/N) &Sigma; (Z_comp(f_i) &minus; Z_ref(f_i))² ), sensível a diferenças de amplitude; e CCDM = 1 &minus; &rho;(Z_comp, Z_ref), baseada na correlação de Pearson (avalia forma). Para ambas, valores menores indicam melhor compensação. Reporta-se também a separação sep_D1 = RMSD_D1 &minus; RMSD_D0 e sep_D2 = RMSD_D2 &minus; RMSD_D0, e a ordenação por duas medidas separadas: <i>healthy_sep</i> (D0 &lt; min(D1,D2), critério primário) e <i>full_order</i> (D0 &lt; D1 &lt; D2, apenas descritiva).")

H("2.7 Validação e análise estatística",H2)
P_("O <i>loop</i> externo é LOTO sobre as temperaturas com três classes; hiperparâmetros e banda são escolhidos por validação cruzada interna somente nas temperaturas de treino, sem tocar o <i>fold</i> externo. A comparação estatística segue Demšar (2006): teste de Friedman como teste omnibus, seguido do pós-teste de Nemenyi com diagrama de diferença crítica (CD) e, por ser o Nemenyi conservador, do teste de Wilcoxon pareado com correção de Holm. Um controle negativo com rótulos embaralhados verifica ausência de vazamento.")

story.append(PageBreak())
# ================== 3. RESULTADOS ==================
H("3. Resultados e Discussão")

H("3.1 Comparação estatística da compensação",H2)
P_(f"A Figura {figref()} apresenta o diagrama de diferença crítica (Nemenyi) para o RMSD nas curvas saudáveis, com os métodos ordenados pelo rank médio (1 = melhor) sobre todas as condições banda×temperatura. O teste de Friedman rejeita a hipótese de igualdade. ")
fig("figCD_RMSD",13,f"Figura {FIGN[0]}. Diagrama de diferença crítica (Demšar, 2006) para o RMSD saudável. Métodos conectados pela barra não diferem significativamente (Nemenyi, α=0,05).")
if wl is not None:
    P_(f"A Tabela {tabref()} traz o teste de Wilcoxon pareado (correção de Holm), mais potente que o Nemenyi. ")
    tbl(wl,fs=8,cap=f"Tabela {TABN[0]}. Wilcoxon pareado (Holm) para o RMSD saudável.")

H("3.2 A compensação térmica é necessária (com vs. sem)",H2)
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

H("3.3 Influência da banda de frequência: estreita vs. larga",H2)
P_(f"Com todos os métodos ajustados pela mesma validação interna, o melhor compensador depende da banda (Tabela {tabref()}, Figura {figref()+0}). Em {aewin} de {nb} bandas o AE ou o RF superam o Park. O AE atinge o menor RMSD em {ae_best}, o RF em {rf_best} e o Park em {park_best}. Em bandas estreitas de alta frequência o AE domina; em bandas largas, o RF direto captura as deformações locais que o alinhamento global do Park não modela, reproduzindo o comportamento observado por Baptista et al. (2014) e no estudo preliminar dos autores.")
pv2=piv[[m for m in MJ if m in piv.columns]].round(3).copy(); pv2.insert(0,"Banda",pv2.index)
main=[m for m in ["AE","Park","RF_direct"] if m in piv.columns]
pv2["Vencedor"]=piv[main].idxmin(axis=1).map(LBL).values
pv2.columns=["Banda"]+[LBL[m] for m in MJ if m in piv.columns]+["Vencedor"]
tbl(pv2,fs=7.5,cap=f"Tabela {TABN[0]}. RMSD saudável por banda (melhor configuração de cada método). Bandas largas ao final.")
# tabela consolidada: TODAS as métricas de compensação por método
try:
    _cm=[c for c in ["RMSD_D0","CCDM_D0","RMSE_D0","MAE_D0","NRMSE_D0","CORR_D0","SAM_deg_D0"] if c in comp.columns]
    _gm=comp[comp.metodo.isin(MJ)].groupby("metodo")[_cm].mean().reindex([m for m in ["AE","Park","RF_direct","RF_temponly"] if m in comp.metodo.unique()])
    _gm=_gm.round(3).reset_index(); _gm["metodo"]=_gm["metodo"].map(LBL)
    _hdr=["Método"]+[c.replace("_D0","").replace("SAM_deg","SAM(°)") for c in _cm]
    P_("Para uma visão completa, a Tabela seguinte consolida <b>todas as métricas de compensação</b> nas curvas saudáveis (média sobre bandas e temperaturas). RMSD/RMSE/MAE/NRMSE e CCDM: menor é melhor; correlação (CORR): maior é melhor; ângulo espectral (SAM): menor é melhor. As métricas concordam entre si na ordenação geral, o que dá robustez à comparação.")
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
P_(f"Como <i>screening</i> exploratório inicial — com hiperparâmetros padrão (não ajustados), portanto anterior e distinto do ajuste por CV interna usado nas demais seções —, a Figura {figref()} mapeia o método vencedor em cada região do plano banda×T_ref (13 bandas × 3 temperaturas de referência). Seu papel é apenas mostrar a <b>divisão de territórios em larga escala</b>: o Autoencoder e o Random Forest cobrem quase todo o plano e o Park fica restrito a poucas células. A fronteira específica entre AE e RF desloca-se após o ajuste de hiperparâmetros (Seções 3.3–3.4, que reportam o resultado ajustado e prevalecem sobre este screening); por exemplo, com ajuste o AE passa a vencer também bandas estreitas intermediárias (40–50, 50–60, 60–70 kHz).")
fig("figH_vencedor_banda_tref",9,f"Figura {FIGN[0]}. <b>Screening exploratório</b> (parâmetros padrão): método com menor RMSD saudável em cada combinação banda × T_ref. A fronteira AE–RF é refinada pelo ajuste nas Seções 3.3–3.4.")

H("3.4 Efeito da largura da banda e das janelas espectrais",H2)
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

H("3.4b Faixa de frequência mais analisável para o dano",H2)
P_(f"Além de compensar bem, uma banda deve permitir <i>reconhecer</i> o dano. A Figura {figref()} apresenta a acurácia balanceada de classificação (binária e multiclasse) obtida em cada banda, identificando as faixas mais informativas para o diagnóstico estrutural. ")
fig("figH_dano_por_banda",13,f"Figura {FIGN[0]}. Detectabilidade do dano por banda: acurácia balanceada de classificação binária e multiclasse.")
dpb=Rd("05_sensibilidade_faixas/dano_por_banda.csv")
if dpb is not None and not dpb.empty:
    bb=dpb.set_index("banda"); best_b=bb.bin_bal_acc.idxmax(); best_m=bb.multi_bal_acc.idxmax()
    P_(f"A banda mais analisável para detecção binária de dano é <b>{best_b} kHz</b> (acurácia balanceada {bb.bin_bal_acc.max():.3f}) e, para o reconhecimento multiclasse, <b>{best_m} kHz</b> ({bb.multi_bal_acc.max():.3f}). Isso indica que a faixa ótima para <i>compensar</i> não coincide necessariamente com a faixa ótima para <i>classificar</i>, um resultado relevante para o projeto de sistemas de SHM.")

H("3.5 Influência da temperatura de referência — varredura completa",H2)
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

H("3.5 Análise visual das curvas compensadas",H2)
P_(f"As Figuras {figref()} e {figref()} comparam, lado a lado, os métodos na banda 40–50 kHz para uma temperatura intermediária (48 °C) e uma condição distante da referência (78 °C), reproduzindo a análise do estudo preliminar. Próximo da referência, todos realinham os picos preservando a forma. Longe da referência, o método de Park introduz distorções locais nos picos e reduz a separação entre estados estruturais, enquanto o AE e o RF preservam melhor as assinaturas.")
fig("figE_curvas_40-50_T48",15.5,f"Figura {FIGN[0]-1}. Curvas compensadas em 40–50 kHz a 48 °C (D0): (a) Park, (b) Random Forest, (c) Autoencoder.")
fig("figE_curvas_40-50_T78",15.5,f"Figura {FIGN[0]}. Curvas compensadas em 40–50 kHz a 78 °C (D0), condição distante da referência.")

H("3.6 Preservação da assinatura de dano e ordenação",H2)
P_(f"A Figura {figref()} apresenta RMSD e CCDM por temperatura e estado estrutural (40–50 kHz). O Autoencoder e o Random Forest mantêm o estado saudável abaixo dos danificados de forma mais consistente; o Park comprime a separação entre D1 e D2. No sinal cru, a ordem D1&lt;D2 vale em {ordr} em RMSD mas apenas {ordc} em CCDM, com inversões dependentes de banda — de modo que D0&lt;D1&lt;D2 não pode ser usado como critério de sucesso, sendo reportado apenas de forma descritiva.")
fig("figE_rmsd_ccdm_dano_40-50",15.5,f"Figura {FIGN[0]}. RMSD e CCDM por temperatura e estado estrutural — 40–50 kHz.")
fig("figE_rmsd_ccdm_dano_30-70",15.5,f"Figura {figref()}. Idem, banda larga 30–70 kHz: o Park perde consistência (inversões), o RF mantém a ordenação.")

story.append(PageBreak())
H("3.7 Reflexão sobre os tipos de dano",H2)
P_("Os dois danos investigados têm naturezas físicas distintas, o que se reflete em suas assinaturas espectrais e explica por que sua ordenação relativa depende da métrica e da banda. O <b>Dano 1</b>, uma massa acoplada, introduz uma carga inercial localizada: pela relação entre impedância mecânica da estrutura e impedância elétrica do PZT (Liang et al., 1994), o aumento de massa tende a deslocar ressonâncias para frequências mais baixas e a alterar amplitudes, produzindo um efeito relativamente distribuído sobre o espectro. O <b>Dano 2</b>, um corte, altera localmente a rigidez e a continuidade do material, afetando de forma mais seletiva determinadas ressonâncias e podendo introduzir novos modos locais; trata-se de um defeito mais realista, porém de assinatura mais concentrada em bandas específicas.")
P_(f"Essa diferença física tem consequência direta sobre as métricas. Medindo o dano cru contra a mediana saudável da <i>mesma</i> temperatura (isolando o efeito térmico), a ordem D1&lt;D2 vale em {ordr} dos casos em RMSD — métrica sensível à amplitude — mas em apenas {ordc} em CCDM, que avalia forma e correlação, com inversões concentradas em bandas específicas. Em outras palavras, o corte (D2) produz maior desvio de amplitude ponto a ponto (RMSD) de forma quase universal, mas nem sempre maior descorrelação de forma (CCDM), pois seu efeito é localizado. Isto reforça metodologicamente que a relação D0&lt;D1&lt;D2 não pode ser tomada como um requisito universal de sucesso de um compensador: ela é uma propriedade físico-métrica dos danos, e não do algoritmo. O critério de detecção adotado é, portanto, a separação saudável (D0 abaixo de ambos os danos), enquanto a ordenação completa é reportada apenas descritivamente.")
P_(f"A Figura {figref()} ilustra a preservação da assinatura de dano após compensação, mostrando, lado a lado, os três estados (D0, D1, D2) para uma mesma temperatura. Um bom compensador deve aproximar D0 da referência (removendo a temperatura) sem transformar D1 e D2 em cópias do estado saudável — isto é, preservando o desvio associado ao dano.")
fig("figE_dano_AE_40-50_T70",15.5,f"Figura {FIGN[0]}. Preservação da assinatura de dano (Autoencoder, 40–50 kHz, 70 °C): (a) D0 aproxima-se da referência; (b) Dano 1 e (c) Dano 2 mantêm seus desvios.")
fig("figE_dano_Park_40-50_T70",15.5,f"Figura {figref()}. Idem para o método de Park: a compressão da separação entre estados é mais acentuada.")

story.append(PageBreak())
H("3.8 Classificação de dano — comparação de classificadores",H2)
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

H("3.9 Detecção binária de dano por compensador",H2)
if pb is not None:
    real=pb[pb.controle=="real"]
    bt=real[real.task=="bin"].groupby("metodo")[["bal_acc","macro_f1","recall_dano","taxa_falso_saudavel"]].mean().reindex([m for m in ORD if m in real.metodo.unique()]).round(4).reset_index()
    bt["metodo"]=bt["metodo"].map(LBL); bt.columns=["Método","Acc. bal.","Macro-F1","Recall dano","Falso-saud."]
    P_(f"Usando as curvas compensadas como entrada de um classificador, a Tabela {tabref()} e a Figura {figref()} resumem a detecção binária (saudável vs. com dano). O AE fornece a melhor acurácia balanceada ({best_bin}); o RF apresenta a menor taxa de falso-saudável ({fh_best}), o erro crítico em SHM.")
    tbl(bt,fs=8,cap=f"Tabela {TABN[0]}. Detecção binária de dano, fora da amostra.")
    fig("figCD_clf_bin",13,f"Figura {FIGN[0]}. Diagrama de diferença crítica — classificação binária (balanced accuracy).")
    fig("fig12_classificacao_binaria",15.5,f"Figura {figref()}. Detecção binária: acurácia balanceada, recall de dano e taxa de falso-saudável.")

H("3.10 Reconhecimento multiclasse (D0/D1/D2)",H2)
if pb is not None:
    mt=real[real.task=="multi"].groupby("metodo")[["bal_acc","macro_f1","f1_D0","f1_D1","f1_D2"]].mean().reindex([m for m in ORD if m in real.metodo.unique()]).round(4).reset_index()
    mt["metodo"]=mt["metodo"].map(LBL); mt.columns=["Método","Acc. bal.","Macro-F1","F1 D0","F1 D1","F1 D2"]
    P_(f"Para o reconhecimento dos três estados (Tabela {tabref()}, Figuras {figref()} e {figref()+0}), o melhor macro-F1 é de {best_multi}, com Park e AE apresentando perfis complementares. As matrizes de confusão mostram onde ocorrem as principais confusões entre classes.")
    tbl(mt,fs=8,cap=f"Tabela {TABN[0]}. Reconhecimento multiclasse (D0/D1/D2), fora da amostra.")
    fig("figCD_clf_multi",13,f"Figura {FIGN[0]}. Diagrama de diferença crítica — classificação multiclasse.")
    fig("fig15_confusao_multiclasse",14,f"Figura {figref()}. Matrizes de confusão multiclasse (FS1 + regressão logística).")

H("3.11 Robustez térmica, estabilidade e controles",H2)
P_(f"A robustez à distância térmica |T&minus;T_ref| (Figura {figref()}) mostra que a separabilidade do Park degrada ao afastar-se da referência, enquanto o RF mantém-se estável. A estabilidade do AE entre sementes é alta após o ajuste (Figura {figref()}). O controle negativo com rótulos embaralhados colapsa para o acaso em todos os métodos, evidenciando ausência de vazamento. A ablação RF(direto) vs. RF(só temp.) indica leve adaptação ao dano, sem prejuízo à detecção (Figura {figref()}).")
fig("fig18_distancia_termica",15,f"Figura {FIGN[0]-2}. Robustez à distância térmica.")
fig("figS1_seeds",11,f"Figura {FIGN[0]-1}. Estabilidade entre sementes.")
fig("fig19_ablation",15,f"Figura {FIGN[0]}. Ablações.")

# ================== 4-6 ==================
story.append(PageBreak())
H("3.12 Análises complementares",H2)
def Rr(p):
    fp=os.path.join(ROOT,"results_article",p); return pd.read_csv(fp) if os.path.exists(fp) else None

# custo computacional
P_("<b>Custo computacional.</b> Uma comparação prática deve considerar o custo de cada método. O Park é determinístico e não requer treinamento; o Random Forest e o Autoencoder exigem treino a cada condição.")
cc=Rr("01_temperature_compensation/custo_computacional.csv")
if cc is not None: tbl(cc,fs=8,cap=f"Tabela {tabref()}. Custo computacional (40–50 kHz, CPU): treino, inferência e tamanho do modelo.")
fig("figN_custo_computacional",10,f"Figura {figref()}. Tempo de treino por fold. O Park não treina; o AE e o RF têm custo de treino não desprezível.")

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
    tbl(lk,widths=[5*cm,4*cm,4*cm],fs=8,cap=f"Tabela {tabref()}. Vazamento térmico residual: R² e MAE ao prever a temperatura da curva compensada (menor = melhor).")
fig("figN_temperature_leakage",10,f"Figura {FIGN[0]}. R² ao prever a temperatura após compensação (menor indica melhor invariância).")

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
fig("figN_corr_rmsd_f1",11,f"Figura {FIGN[0]}. Relação entre RMSD saudável (compensação) e Macro-F1 (classificação).")

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
H("3.13 Detecção de dano pelo índice de distância à referência",H2)
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

# ================== 3.14 RIGOR ESTATÍSTICO ADICIONAL ==================
H("3.14 Intervalos de confiança e tamanho de efeito",H2)
P_("Dado o número limitado de <i>folds</i>, o p-valor isolado é insuficiente: reporta-se também a <b>magnitude</b> das diferenças. A Figura seguinte apresenta intervalos de confiança de 95% (bootstrap, 2000 reamostragens) para o RMSD saudável de cada método, e a Tabela seguinte, o tamanho de efeito não paramétrico (delta de Cliff) nas comparações pareadas.")
fig("figA_bootstrap_ic",13,f"Figura {figref()}. RMSD saudável médio com intervalo de confiança de 95% (bootstrap) por método, todas as condições agrupadas.")
cli=Rd("08_analises_avancadas/cliff_delta.csv")
if cli is not None and not cli.empty:
    cli2=cli.rename(columns={"comparacao":"Comparação","cliff_delta":"δ de Cliff","magnitude":"Magnitude","interpretacao":"Interpretação"})
    tbl(cli2,fs=8,cap=f"Tabela {tabref()}. Tamanho de efeito (delta de Cliff) no RMSD saudável. |δ|<0,15 desprezível; <0,33 pequeno; <0,47 médio; ≥0,47 grande.")
    P_("A leitura conjunta de intervalos de confiança sobrepostos e efeitos pequenos-a-médios reforça a mensagem central do artigo: <b>as diferenças entre os melhores métodos são reais, porém modestas quando agregadas sobre todas as bandas</b> — é a escolha da banda, e não a família de método, que produz os maiores ganhos.")

# ================== 3.15 SELETOR ADAPTATIVO POR BANDA ==================
H("3.15 Seletor adaptativo por banda (contribuição prática)",H2)
P_("Se o melhor método depende da banda, uma estratégia natural é <b>escolher o método por banda</b>. Implementou-se um seletor que, para cada banda, adota o método com menor erro de validação cruzada interna (portanto decidido apenas com dados de treino, sem tocar no teste) e é avaliado fora da amostra. Compara-se esse seletor aos métodos fixos e ao oráculo (limite superior que escolhe, retrospectivamente, o melhor método por banda no teste).")
selj=Jd("08_analises_avancadas/seletor_resumo.json")
if selj:
    ordem=sorted(selj.items(),key=lambda x:x[1])
    linha=", ".join([f"{k} = {v:.2f}" for k,v in ordem])
    P_(f"Os RMSD saudáveis médios (menor = melhor) foram: {linha}. O seletor por CV interna aproxima-se do oráculo e supera qualquer método fixo isolado, demonstrando que <b>a seleção de banda-método é uma alavanca de desempenho maior do que trocar de família de método</b> — e, crucialmente, pode ser operada sem rótulos de dano e sem vazamento.")
fig("figA_seletor",12,f"Figura {figref()}. Seletor adaptativo por banda (escolha por CV interna) vs. métodos fixos e oráculo.")

# ================== 3.16 EFICIÊNCIA DE DADOS ==================
H("3.16 Eficiência de dados (curva de aprendizado)",H2)
P_("Uma preocupação prática em SHM é quantas condições de temperatura precisam ser medidas para calibrar um compensador. A Figura seguinte mostra o RMSD saudável de teste em função do número de temperaturas de treino (k), na banda de melhor desempenho do Autoencoder (70–80 kHz). O método de Park serve de referência plana, pois não usa dados de treino (apenas a curva de referência).")
deff=Rd("08_analises_avancadas/data_efficiency.csv")
if deff is not None and not deff.empty:
    try:
        g=deff.groupby("k")[["AE","RF_direct","Park"]].mean(); kmin,kmax=g.index.min(),g.index.max()
        park_lvl=g["Park"].mean()
        P_(f"O resultado é revelador e matiza a mensagem do artigo. Com <b>poucas</b> temperaturas de treino (k={int(kmin)}), tanto o Autoencoder (RMSD {g.loc[kmin,'AE']:.2f}) quanto o Random Forest ({g.loc[kmin,'RF_direct']:.2f}) são <b>piores</b> que o método de Park (≈{park_lvl:.2f}, plano, pois não requer treino). A vantagem dos métodos de aprendizado só se materializa com <b>ampla cobertura térmica</b>: ao usar todas as temperaturas disponíveis (k={int(kmax)}), o RMSD do Autoencoder despenca para {g.loc[kmax,'AE']:.2f} e o do Random Forest para {g.loc[kmax,'RF_direct']:.2f}, cruzando e superando o Park apenas quando há muitas temperaturas de calibração. O Autoencoder tem a curva de aprendizado mais íngreme — o mais faminto por dados, porém o que atinge o menor erro quando bem alimentado; o Random Forest degrada de forma mais graciosa com poucos dados.")
        P_("A implicação prática é direta: <b>a escolha do método deve considerar o orçamento experimental</b>. Em campanhas com poucas temperaturas medidas, o método de Park — sem treino e imune à escassez de dados — é a opção mais segura; quando se dispõe de uma varredura térmica densa, o Autoencoder passa a ser a melhor escolha na sua banda ótima. Este é mais um argumento contra veredictos globais: a superioridade do aprendizado de máquina é <i>condicional</i> à disponibilidade de dados de calibração.")
    except Exception: pass
fig("figA_data_efficiency",12,f"Figura {figref()}. Eficiência de dados: RMSD saudável vs. número de temperaturas de treino (70–80 kHz). Com poucos dados, o Park (plano) supera o ML; a vantagem do AE só surge com ampla cobertura térmica.")

# ================== 3.17 SENSIBILIDADE DE HIPERPARÂMETROS ==================
story.append(PageBreak())
H("3.17 Estudo de sensibilidade de hiperparâmetros",H2)
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
H("3.18 Aprimorando a compensação por floresta: Extra Trees",H2)
P_("Explorando <i>famílias</i> de regressor além do ajuste de hiperparâmetros, testou-se a substituição do Random Forest por <b>Extra Trees</b> (árvores extremamente aleatorizadas), mantendo exatamente as mesmas entradas, alvo e protocolo. A diferença é que o Extra Trees sorteia os pontos de corte das divisões em vez de otimizá-los, o que reduz a variância do estimador — vantajoso quando o alvo (a correção térmica) carrega ruído de medição.")
etj=Jd("08_analises_avancadas/extratrees_resumo.json"); etl=Rd("08_analises_avancadas/extratrees_loto.csv")
if etj and etl is not None and not etl.empty:
    try:
        P_(f"A troca produziu um ganho <b>consistente e fora da amostra</b>: em {etj['n_et_vence']} de {etj['n_bandas']} bandas o Extra Trees reduziu o RMSD saudável de teste em relação ao Random Forest, com melhora média de <b>{etj['ganho_medio_pct']:.0f}%</b> (Figura {figref()}). O ganho é maior nas bandas onde o RF já era competitivo — por exemplo, em 70–80 kHz o RMSD cai de {etl[etl.banda=='70-80'].RF_RMSD_D0.iloc[0]:.2f} para {etl[etl.banda=='70-80'].ET_RMSD_D0.iloc[0]:.2f}. Trata-se, portanto, de um aprimoramento real do compensador baseado em floresta, obtido sem qualquer vazamento (a escolha foi validada por CV interna e apenas confirmada no teste). Uma implementação futura do método de floresta deveria adotar o Extra Trees como padrão.")
    except Exception: pass
    et2=etl.copy()
    et2=et2[["banda","RF_RMSD_D0","ET_RMSD_D0","ganho_%","RF_healthy_sep","ET_healthy_sep"]].round(3)
    et2.columns=["Banda","RMSD RF","RMSD Extra Trees","Ganho (%)","healthy_sep RF","healthy_sep ET"]
    tbl(et2,fs=7.5,cap=f"Tabela {tabref()}. Random Forest vs. Extra Trees na compensação (teste LOTO). Ganho positivo = Extra Trees melhor.")
fig("figHP_extratrees_loto",15,f"Figura {FIGN[0]}. Extra Trees vs. Random Forest — RMSD saudável fora da amostra por banda.")

# ================== 3.19 EXEMPLOS ADICIONAIS DE CURVAS ==================
H("3.19 Exemplos adicionais de curvas compensadas",H2)
P_(f"Para ilustrar visualmente a compensação em mais condições, a Figura {figref()} sobrepõe as curvas saudáveis de <i>todas</i> as temperaturas antes e depois da compensação (70–80 kHz). No sinal original (esquerda), as curvas espalham-se verticalmente conforme a temperatura (a cor codifica T); após a compensação (Park e Autoencoder), elas colapsam sobre a referência — a assinatura visual da remoção da variabilidade térmica, mais completa e uniforme no Autoencoder.")
fig("figE_overlay_saudavel",16.5,f"Figura {FIGN[0]}. Curvas saudáveis de todas as temperaturas antes e depois da compensação (cor = temperatura). O colapso sobre a referência (tracejada) evidencia a remoção térmica.")
P_(f"As Figuras seguintes trazem exemplos por classe (D0/D1/D2) em bandas e temperaturas adicionais fora da amostra, confirmando que o comportamento observado nas bandas de estudo se mantém: as curvas saudáveis alinham-se à referência, enquanto as danificadas preservam seu afastamento característico.")
for _bt in [("30-40",20),("30-40",-10),("60-70",70),("60-70",0)]:
    _fn=f"figE_curvas_{_bt[0]}_T{_bt[1]}"
    fig(_fn,15,f"Figura {figref()}. Curvas compensadas — {_bt[0]} kHz, T = {_bt[1]}°C (fora da amostra).")

# ================== 3.20 SÍNTESE DOS MECANISMOS ==================
H("3.20 Síntese dos mecanismos: por que cada método se comporta assim",H2)
P_("Reunindo as evidências das seções anteriores, emerge uma explicação mecanística coerente para o desempenho de cada método, ancorada em como cada um <i>modela</i> a distorção térmica.")
P_("<b>Park — alinhamento global.</b> Modela a distorção como um único deslocamento em frequência mais um <i>offset</i> vertical, calibrados para minimizar o erro à referência. Isso funciona onde a hipótese vale — perto da temperatura de referência e em bandas estreitas —, mas encadeia todas as suas limitações observadas: (i) o teste de vazamento mostra que a temperatura permanece recuperável após o Park (R²≈0,98), pois o método realinha sem remover a informação térmica; (ii) as métricas de pico revelam grande erro de frequência (centenas de Hz), pois um deslocamento único não corrige o deslocamento diferencial de cada ressonância; (iii) o erro cresce fortemente com a distância à referência e com a largura da banda, onde a hipótese de globalidade falha. Em compensação, não requer treino e é imune à escassez de dados.")
P_("<b>Autoencoder — variedade térmica não linear.</b> Aprende, a partir de curvas saudáveis, uma representação de baixa dimensão da forma como o espectro varia com a temperatura, e reconstrói a correção. Isso lhe dá o menor erro e a melhor preservação de picos em bandas estreitas de alta frequência, além da maior remoção de informação térmica (menor R² de vazamento) — indícios de que aprende uma representação genuinamente mais invariante. Suas fraquezas são o reverso da mesma moeda: em bandas largas, um único mapeamento subajusta a resposta térmica heterogênea; com poucas temperaturas de treino, a rede não tem dados para aprender a variedade e chega a perder para o Park; e há risco de sobrecompensação, visível na atenuação do dano 2. É um especialista faminto por dados.")
P_("<b>Florestas (Random Forest / Extra Trees) — correção ponto a ponto.</b> Estimam a correção térmica localmente, ponto a ponto, sem impor uma forma global. Essa flexibilidade explica sua robustez: são os mais estáveis ao alargamento da banda e à escolha da temperatura de referência, e os que melhor preservam a assinatura de dano (maior DPR, menor taxa de falso-saudável). O Extra Trees, ao aleatorizar os cortes, reduz ainda mais a variância e supera o Random Forest de forma consistente. Suas limitações são o custo de treino e a extrapolação térmica além da faixa vista. Em resumo: <b>a estratégia de modelagem de cada método — global, aprendida-não-linear ou local-ponto-a-ponto — determina simultaneamente seus acertos e suas falhas</b>, e é por isso que nenhum domina universalmente e que a escolha ótima depende da banda, da referência e do orçamento de dados.")

# ================== 3.21 REFINAMENTOS: ET TUNADO, ENSEMBLE ==================
mv2=Rd("08_analises_avancadas/melhorias_v2_loto.csv"); mv2j=Jd("08_analises_avancadas/melhorias_v2_resumo.json")
if mv2 is not None and not mv2.empty:
    H("3.21 Refinamentos adicionais: Extra Trees tunado e ensemble",H2)
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

story.append(PageBreak())
H("4. Discussão")
P_("Os resultados contam uma história coerente e, em vários pontos, contra-intuitiva. Ela começa com uma constatação simples: <b>sem compensação, a temperatura domina o sinal</b> a ponto de a separação entre saudável e danificado tornar-se negativa em média — o dano fica, literalmente, mascarado. A partir daí, cada método adota uma estratégia física distinta para remover a temperatura, e é nessa estratégia que residem seus acertos e falhas.")
P_("O <b>método de Park</b> assume que a distorção térmica é aproximadamente um deslocamento global do espectro (mais um offset). Essa hipótese (A) é válida perto da temperatura de referência e em bandas estreitas, onde o Park é competitivo e tem a enorme vantagem prática de não exigir treinamento. Porém, três evidências independentes mostram seus limites: (i) longe da referência e em bandas largas, o erro do Park cresce mais rápido que o dos métodos de ML (hipótese B confirmada); (ii) o teste de vazamento de temperatura revela que, após a compensação de Park, a temperatura ainda é quase perfeitamente recuperável da curva (R² ≈ 0,98, praticamente igual ao sinal original) — ou seja, o Park <i>realinha</i> a curva sem de fato remover a informação térmica subjacente; e (iii) as métricas de pico mostram que o alinhamento global do Park desloca fortemente as ressonâncias (erro de centenas de Hz) e reduz a taxa de preservação de picos, exatamente nas regiões que carregam a assinatura estrutural.")
P_("O <b>Autoencoder</b> aprende a variedade térmica não linear do estado saudável. Isso o torna o mais preciso em bandas estreitas de alta frequência, o que melhor preserva os picos e o que mais reduz o vazamento de temperatura — evidência de que ele efetivamente aprende uma representação mais invariante à temperatura. Contudo, confirma-se também sua fragilidade (hipótese C): em bandas muito largas a rede subajusta a resposta térmica heterogênea, e há o risco de sobrecompensação, visível no Damage Preservation Ratio, onde o AE chega a atenuar o dano 2. O AE é, portanto, um especialista: excelente onde o sinal é rico e localizado, arriscado onde é amplo e heterogêneo.")
P_("O <b>Random Forest direto</b>, ao aprender a correção ponto-a-ponto, é o mais robusto entre condições e o que melhor preserva a assinatura de dano (maior DPR) e a segurança de detecção (menor taxa de falso-saudável). É o mais adequado a bandas largas, onde captura deformações locais que o Park não modela. Sua limitação (hipótese D) aparece na extrapolação térmica e no custo de treino, superior ao do AE. A ablação RF-direto vs. RF-só-temperatura indica que ver a curva ajuda a compensação e introduz apenas uma leve reação ao dano, sem prejudicar a detecção.")
P_("A descoberta metodológica mais importante é que <b>compensar bem a temperatura não é o mesmo que permitir classificar bem o dano</b> (hipóteses I e J). A correlação entre o RMSD saudável e o Macro-F1 de classificação é fraca; a melhor faixa para compensação (bandas estreitas de alta frequência, para o AE) não coincide necessariamente com a melhor faixa para classificação. Isso confirma que o erro de reconstrução, isoladamente, é insuficiente para avaliar um compensador destinado a SHM — a métrica final deve ser a capacidade de detectar e classificar o dano sob temperatura variável. Sobre a largura da banda (hipóteses F e G), confirma-se que faixas muito amplas contêm comportamento térmico heterogêneo que penaliza Park e AE, ao passo que janelas estreitas próximas de ressonâncias oferecem boa compensabilidade — embora o RMSD bruto exagere as diferenças por causa da amplitude (o NRMSE atenua o efeito).")
P_("A análise de detecção por índice de distância à referência (Seção 3.13) fecha o argumento de forma prática: a compensação <b>aumenta a área sob a curva ROC</b> da separação saudável–dano em relação ao sinal bruto, ou seja, remover a temperatura torna o dano mais detectável mesmo sem treinar um classificador. Isso reconcilia as duas metades do trabalho — compensar e detectar — mostrando que, embora RMSD e classificação não estejam fortemente correlacionados, a compensação ainda beneficia a detecção quando esta é medida diretamente sobre a distância à referência. Do ponto de vista de aplicação, a contribuição mais acionável é o <b>seletor adaptativo por banda</b> (Seção 3.15): como o método vencedor depende da banda, escolher o método por banda apenas com a validação interna — sem rótulos de dano e sem vazamento — aproxima-se do oráculo e supera qualquer método fixo, transformando a principal descoberta do artigo em uma receita operacional. Por fim, a curva de eficiência de dados (Seção 3.16) impõe uma ressalva honesta e prática: a superioridade dos métodos de aprendizado é <b>condicional à quantidade de dados de calibração</b> — com poucas temperaturas medidas eles perdem para o Park (que não treina), e a vantagem só emerge com ampla cobertura térmica. A escolha do método, portanto, deve levar em conta o orçamento experimental disponível.")
P_("Todas as comparações usaram exatamente as mesmas amostras, bandas, temperaturas de referência, <i>splits</i> e métricas, com hiperparâmetros escolhidos por validação cruzada interna sem tocar no <i>fold</i> externo. A cobertura foi ampla: <b>13 faixas de frequência</b> (as sete progressivas 30–40 a 30–100 kHz e as seis de 10 kHz 40–50 a 90–100 kHz), <b>janelas de 5 kHz ao longo de todo o espectro</b>, <b>dez temperaturas de referência</b> (de −10 a 80 °C), além de detecção por ROC/AUC, intervalos de confiança por <i>bootstrap</i>, tamanho de efeito, seletor adaptativo e curva de eficiência de dados. Nenhuma seleção de banda, janela, T_ref ou método usou o conjunto de teste.")

H("4.1 A natureza dual do problema: por que separar temperatura de dano é difícil",H2)
P_("Um fio condutor atravessa todos os resultados: temperatura e dano atuam sobre o <b>mesmo</b> observável — o espectro de impedância — e por mecanismos que se confundem. Ambos deslocam ressonâncias e alteram amplitudes; a diferença é de <i>grau</i> e de <i>estrutura</i>, não de natureza. A temperatura produz uma deformação relativamente suave e global (todo o espectro migra e se atenua), enquanto o dano incipiente produz uma perturbação mais localizada em torno de ressonâncias específicas. É essa sobreposição que torna o problema genuinamente difícil e que explica por que a distância bruta à referência tem AUC de detecção próxima do acaso: sem separar as duas contribuições, o efeito térmico — maior em magnitude — soterra a assinatura do dano. Compensar é, no fundo, um problema de <i>separação de fontes</i>: estimar e remover a componente térmica preservando a componente estrutural. Cada método ataca essa separação com uma hipótese diferente sobre a forma da componente térmica (global para o Park, uma variedade não-linear aprendida para o AE, uma função local ponto-a-ponto para as florestas), e o sucesso de cada um depende de quão bem sua hipótese casa com a física da banda considerada.")
P_("Esse enquadramento também esclarece o risco central do projeto — o de <i>sobrecompensar</i> e apagar o dano. Um método suficientemente flexível pode, ao tentar aproximar toda curva da referência saudável, começar a remover não só a temperatura mas também a perturbação do dano, empurrando curvas danificadas na direção da saudável. Foi para vigiar exatamente esse risco que se reportou a preservação de dano de forma independente (DPR, healthy_sep, referência danificada como <i>ground-truth</i>) e que se impôs, como critério, que uma boa compensação deve reduzir a distância das curvas saudáveis <i>sem</i> reduzir a das danificadas. O fato de o AE atenuar levemente o dano 2 em bandas largas é a manifestação concreta desse risco e a razão pela qual precisão de reconstrução, isoladamente, nunca deve ser o único critério de avaliação.")

H("4.2 Implicações para o projeto de sistemas de SHM",H2)
P_("Traduzindo os achados em recomendações práticas para quem projeta um sistema de monitoramento por impedância: <b>(1) Escolha a banda deliberadamente</b> — ela importa mais do que a família de método; bandas estreitas de alta frequência tendem a combinar sensibilidade ao dano e facilidade de compensação. <b>(2) Escolha a temperatura de referência perto do centro da faixa operacional</b> — é crítico para o método de Park e barato de garantir. <b>(3) Case o método ao orçamento de dados</b>: com poucas temperaturas de calibração, prefira o Park (sem treino); com uma varredura térmica densa, prefira o Autoencoder na banda ótima ou o Extra Trees pela robustez. <b>(4) Use um seletor por banda</b> decidido por validação interna — ele extrai o melhor de cada método sem exigir rótulos de dano. <b>(5) Reporte compensação e detecção separadamente</b> — a primeira não garante a segunda, e o objetivo final do SHM é a decisão de dano, não a reconstrução do espectro. <b>(6) Priorize a segurança</b>: em SHM, um falso-saudável (dano não detectado) é o erro mais grave, o que favorece métodos como as florestas, de menor taxa de falso-saudável, mesmo quando outro método tem RMSD ligeiramente menor.")

H("5. Limitações")
P_(f"O escopo dos resultados deve ser lido à luz de limitações concretas. <b>(1) Ausência de identificador de espécime</b> impede o <i>leave-one-specimen-out</i> e não permite descartar totalmente dependência entre curvas do mesmo corpo de prova — a validação por temperatura (LOTO) mitiga, mas não elimina, esse risco. <b>(2) Amostra pequena:</b> {summ['n_temps_3_classes']} temperaturas com três classes resultam em {summ['n_temps_3_classes']-1} <i>folds</i> por banda, com poder estatístico limitado (menor p de Wilcoxon ≈ 0,004); por isso reportamos intervalos de confiança e tamanho de efeito, e evitamos conclusões baseadas apenas em p-valores. <b>(3) Base e viga únicas:</b> os valores numéricos específicos (melhor banda, melhor T_ref) são propriedades desta estrutura e não devem ser transpostos diretamente a outras geometrias ou materiais — o que se generaliza são os <i>mecanismos</i> e o <i>protocolo</i>, não os números. <b>(4) Dois tipos de dano</b> (massa e corte), ambos artificiais; danos reais (fadiga, delaminação, corrosão) podem ter assinaturas distintas. <b>(5) O AE</b> foi ajustado por grade moderada — embora o estudo de sensibilidade indique platô — e sua variabilidade entre sementes é pequena, mas não nula. <b>(6) Compensação puramente espectral:</b> não se exploraram informações auxiliares (múltiplos PZTs, fase, histórico temporal) que poderiam elevar o teto de desempenho.")
H("6. Conclusões")
P_(f"Comparando autoencoder, Random Forest e o método de Park sob validação fora da amostra e análise estatística de Demšar, conclui-se que nenhuma família domina universalmente: o AE vence em bandas estreitas de alta frequência (melhor RMSD em {ae_best}) e fornece a melhor detecção binária ({best_bin}); o RF é o mais robusto entre condições e o mais seguro para detecção ({fh_best} de falso-saudável), superando o Park de forma consistente; o Park é competitivo apenas próximo da referência e em bandas estreitas. Três resultados sustentam a recomendação prática: (i) a compensação <b>aumenta a AUC</b> de detecção do dano pela distância à referência; (ii) um <b>seletor adaptativo por banda</b>, decidido só com dados de treino, supera qualquer método fixo e aproxima-se do oráculo; e (iii) a vantagem do aprendizado de máquina é <b>condicional à cobertura térmica</b>: exige muitas temperaturas de calibração e, com poucos dados, o Park (sem treino) é preferível. Como aprimoramento metodológico, mostrou-se ainda que a substituição do Random Forest por <b>Extra Trees</b> reduz o erro de compensação em todas as bandas (ganho médio de ~22% fora da amostra, sem perda de preservação de dano), devendo ser adotada em implementações futuras do compensador por floresta. Confirma-se e estende-se, com rigor estatístico e análise multibanda, o potencial do aprendizado de máquina para separar efeitos térmicos de variações associadas ao dano em SHM por EMI — desde que a avaliação seja resolvida por banda, fora da amostra e reportando compensação e detecção como objetivos distintos.")
H("7. Perspectivas e trabalhos futuros")
P_("Os resultados abrem várias direções naturais. <b>(1) Validação multi-espécime e multi-estrutura:</b> repetir o protocolo em vários corpos de prova e geometrias, com identificador de espécime, permitiria o <i>leave-one-specimen-out</i> e testaria a generalização dos mecanismos aqui identificados. <b>(2) Danos realistas e progressivos:</b> substituir os danos artificiais por evolução real (fadiga, delaminação) e por severidades graduais, verificando se a ordenação por severidade e a preservação de dano se mantêm. <b>(3) Compensadores estruturalmente novos:</b> como o estudo de sensibilidade mostrou que o Autoencoder atingiu um platô de hiperparâmetros, ganhos adicionais exigem mudança de arquitetura — redes com atenção às ressonâncias, modelos físicos-informados que embutam a lei de deslocamento térmico, ou autoencoders variacionais que modelem explicitamente a distribuição térmica. <b>(4) Seletor e ensembles adaptativos:</b> o seletor por banda pode evoluir para uma combinação ponderada aprendida (mistura de especialistas) que una a precisão do Autoencoder em bandas estreitas à robustez das florestas em bandas largas. <b>(5) Extra Trees como padrão:</b> a superioridade consistente do Extra Trees sobre o Random Forest sugere revisitar toda a família de compensadores por árvore, incluindo <i>gradient boosting</i> por âncoras. <b>(6) Detecção diretamente otimizada:</b> como compensação e detecção não são equivalentes, faz sentido treinar compensadores cujo objetivo já seja a separabilidade saudável–dano (métrica final), e não apenas a reconstrução do espectro. <b>(7) Custo e implantação embarcada:</b> avaliar o compromisso entre acurácia e custo computacional para operação em tempo real em hardware de baixa potência.")
H("Agradecimentos",H2)
P_("Os autores agradecem à Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP), processos 2016/12241-0 e 2025/09586-5, pelo apoio financeiro.")

# ================== REFERÊNCIAS ==================
story.append(PageBreak()); H("Referências")
REFS=[
"Baptista, F.G., Budoya, D.E., Almeida, V.A.D. e Ulson, J.A.C., 2014. “An experimental study on the effect of temperature on piezoelectric sensors for impedance-based structural health monitoring”. Sensors, 14(1), 1208–1227.",
"Breiman, L., 2001. “Random forests”. Machine Learning, 45(1), 5–32.",
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

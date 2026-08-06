# -*- coding: utf-8 -*-
"""
COMPÊNDIO DE TABELAS — documento dedicado (PDF) com TODAS as tabelas do estudo,
incluindo comparações consolidadas de COMPENSAÇÃO (todas as métricas) e de CLASSIFICAÇÃO
(todos os métodos). Mesma identidade visual do artigo (Times, cabeçalho/rodapé EESC-USP).
Saída: 12_manuscrito/artigo_TABELAS.pdf
"""
import os,json,numpy as np,pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.utils import ImageReader
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; MAN=os.path.join(ROOT,"12_manuscrito")
ACCENT=colors.HexColor("#12325a"); HAIR=colors.HexColor("#c9d4e5")
LBL={"Original":"Original","Park":"Park","RF_direct":"Random Forest","RF_temponly":"RF (só temp.)","AE":"Autoencoder","TAU_T":"tau(T)"}
def Rd(p):
    fp=os.path.join(ROOT,p); return pd.read_csv(fp) if os.path.exists(fp) else None
def Rr(p):
    fp=os.path.join(ROOT,"results_article",p); return pd.read_csv(fp) if os.path.exists(fp) else None
def Jd(p):
    fp=os.path.join(ROOT,p); return json.load(open(fp)) if os.path.exists(fp) else None
ss=getSampleStyleSheet()
def sty(n,**k): k.setdefault("fontName","Times-Roman"); return ParagraphStyle(n,parent=ss["BodyText"],**k)
H1=sty("H1",fontSize=12.5,spaceBefore=10,spaceAfter=3,textColor=ACCENT,fontName="Times-Bold")
BODY=sty("BODY",fontSize=9.2,leading=12.4,alignment=TA_JUSTIFY,spaceAfter=3)
CAP=sty("CAP",fontSize=8,leading=10,alignment=TA_JUSTIFY,textColor=colors.HexColor("#444"),spaceAfter=10,spaceBefore=2,fontName="Times-Italic")
TITLE=sty("TIT",fontSize=17,leading=20,alignment=TA_CENTER,fontName="Times-Bold")
AUT=sty("AUT",fontSize=10,alignment=TA_CENTER,spaceAfter=2)
story=[]; TABN=[0]
def tabref(): TABN[0]+=1; return TABN[0]
def H(t): story.append(Paragraph(t,H1)); story.append(HRFlowable(width="100%",thickness=1.1,color=ACCENT,spaceBefore=1.5,spaceAfter=5,lineCap="round"))
def P_(t): story.append(Paragraph(t,BODY))
def tbl(df,widths=None,fs=8,cap="",hdr=None):
    if df is None or (hasattr(df,"empty") and df.empty):
        story.append(Paragraph(f"<i>[dados indisponíveis: {cap}]</i>",CAP)); return
    cols=list(hdr) if hdr else list(df.columns)
    data=[cols]+df.astype(str).values.tolist()
    t=Table(data,colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),ACCENT),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Times-Bold"),("FONTNAME",(0,1),(-1,-1),"Times-Roman"),("FONTSIZE",(0,0),(-1,-1),fs),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#bbb")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef2f7")]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),2.3),("BOTTOMPADDING",(0,0),(-1,-1),2.3)]))
    story.append(t)
    if cap: story.append(Paragraph(cap,CAP))

# ================== CAPA ==================
_logo=os.path.join(MAN,"logo_eesc_usp.png")
if os.path.exists(_logo):
    iw,ih=ImageReader(_logo).getSize(); lw=2.7*cm; lh=lw*ih/iw
    im=Image(_logo,width=lw,height=lh); im.hAlign="CENTER"; story.append(im); story.append(Spacer(1,6))
P_("")
story.append(Paragraph("Compêndio de Tabelas",TITLE)); story.append(Spacer(1,4))
story.append(Paragraph("Compensação térmica em impedância eletromecânica por Autoencoder, Random Forest e método de Park",AUT))
story.append(Paragraph("EESC-USP — Iniciação Científica FAPESP 2025/09586-5",AUT)); story.append(Spacer(1,10))
P_("Este documento reúne, num só lugar, todas as tabelas do estudo — descrição da base, hiperparâmetros, comparações completas de compensação e de classificação, sensibilidade à temperatura de referência, preservação de dano, detecção, rigor estatístico e custo computacional. Salvo indicação, os valores são médias fora da amostra (LOTO), com hiperparâmetros ajustados por validação cruzada interna.")
story.append(Spacer(1,6))

# ================== 1. BASE ==================
H("1. Base de dados e métodos")
try:
    summ=json.load(open(os.path.join(ROOT,"00_auditoria","dataset_summary.json")))
    d1=pd.DataFrame({"Propriedade":["Curvas","Faixa de frequência","Resolução","Temperaturas","Temp. com 3 classes","D0 / D1 / D2","Identificador de espécime"],
     "Valor":[summ['n_curvas'],"22 Hz – 125 kHz","1 Hz","%d"%summ['n_temperaturas'],summ['n_temps_3_classes'],
              f"{summ['n_D0']} / {summ['n_D1']} / {summ['n_D2']}","nenhum"]})
    tbl(d1,widths=[6.5*cm,8.5*cm],cap=f"Tabela {tabref()}. Descrição da base de dados EMI.")
except Exception as e: P_(f"[base: {e}]")

# ================== 2. COMPENSAÇÃO — TODAS AS MÉTRICAS, TODOS OS MÉTODOS ==================
H("2. Comparação completa da compensação térmica (todos os métodos)")
P_("Média, sobre todas as bandas e temperaturas de teste, de todas as métricas de compensação nas curvas saudáveis. RMSD, RMSE, MAE e NRMSE: menor é melhor. CCDM: menor é melhor (1−correlação). CORR: maior é melhor. SAM: menor é melhor (ângulo espectral, graus).")
f8=Rd("02_compensacao/fase8_tuning_ampliado.csv")
if f8 is not None:
    f8=f8[f8.metodo!="TAU_T"]
    cols=["RMSD_D0","CCDM_D0","RMSE_D0","MAE_D0","NRMSE_D0","CORR_D0","SAM_deg_D0"]
    cols=[c for c in cols if c in f8.columns]
    g=f8.groupby("metodo")[cols].mean().reindex([m for m in ["Original","Park","RF_direct","RF_temponly","AE"] if m in f8.metodo.unique()])
    g=g.round(3).reset_index(); g["metodo"]=g["metodo"].map(LBL)
    hdr=["Método"]+[c.replace("_D0","").replace("SAM_deg","SAM(°)") for c in cols]
    tbl(g,fs=8,hdr=hdr,cap=f"Tabela {tabref()}. Compensação saudável — todas as métricas por método (média sobre bandas e temperaturas).")
# por banda (RMSD e CCDM)
if f8 is not None:
    for met,nm in [("RMSD_D0","RMSD"),("CCDM_D0","CCDM")]:
        pv=f8.pivot_table(index="banda",columns="metodo",values=met,aggfunc="mean")
        order=["AE","Park","RF_direct","RF_temponly"]; pv=pv[[m for m in order if m in pv.columns]].round(3)
        pv=pv.reset_index(); pv.columns=["Banda"]+[LBL[m] for m in order if m in f8.metodo.unique()]
        tbl(pv,fs=7.5,cap=f"Tabela {tabref()}. {nm} saudável por banda e método.")

# ================== 3. CLASSIFICAÇÃO — TODOS OS MÉTODOS ==================
H("3. Comparação completa de classificação de dano (todos os métodos)")
pb=Rd("checkpoints/parteB.csv")
if pb is not None:
    pb=pb[(pb.metodo!="TAU_T")&(pb.controle=="real")]; MB=[m for m in ["Original","Park","RF_direct","RF_temponly","AE"] if m in pb.metodo.unique()]
    b=pb[pb.task=="bin"].groupby("metodo")[["bal_acc","macro_f1","recall_dano","taxa_falso_saudavel"]].mean().reindex(MB).round(4).reset_index()
    b["metodo"]=b["metodo"].map(LBL); b.columns=["Método","Acur. bal.","Macro-F1","Recall dano","Taxa falso-saudável"]
    tbl(b,fs=8,cap=f"Tabela {tabref()}. Detecção binária de dano (saudável vs. com dano), fora da amostra.")
    m=pb[pb.task=="multi"].groupby("metodo")[["bal_acc","macro_f1","f1_D0","f1_D1","f1_D2"]].mean().reindex(MB).round(4).reset_index()
    m["metodo"]=m["metodo"].map(LBL); m.columns=["Método","Acur. bal.","Macro-F1","F1 D0","F1 D1","F1 D2"]
    tbl(m,fs=8,cap=f"Tabela {tabref()}. Reconhecimento multiclasse (D0/D1/D2), fora da amostra.")
# classificadores × features (custo/benefício)
cl=Rd("04_dano_multiclasse/classificadores_todos.csv")
if cl is not None and not cl.empty:
    r=cl[cl.controle=="real"] if "controle" in cl.columns else cl
    for task,nm in [("bin","binária"),("multi","multiclasse")]:
        s=r[r.task==task]
        if len(s):
            gg=s.groupby("clf")["bal_acc"].mean().sort_values(ascending=False).round(3).reset_index()
            gg.columns=["Classificador","Acur. bal. média"]
            tbl(gg,widths=[7*cm,5*cm],fs=8,cap=f"Tabela {tabref()}. Desempenho médio por classificador — tarefa {nm} (média sobre compensadores, features, bandas e folds).")

# ================== 4. NECESSIDADE + DETECÇÃO ==================
H("4. Necessidade da compensação e detecção")
nec=Rd("08_analises_avancadas/necessidade_compensacao.csv")
if nec is not None:
    nec2=nec.rename(columns={nec.columns[0]:"Métrica"});
    nec2.columns=["Métrica"]+[LBL.get(c,c) for c in nec2.columns[1:]]
    tbl(nec2.round(3),fs=8,cap=f"Tabela {tabref()}. Necessidade da compensação: distância à referência (saudável e dano), separação e AUC de detecção. Original = sem compensação.")
auc=Rd("08_analises_avancadas/auc_deteccao.csv")
if auc is not None:
    auc2=auc.rename(columns={auc.columns[0]:"metodo"}); auc2["metodo"]=auc2["metodo"].map(lambda m: LBL.get(m,m))
    auc2.columns=["Método"]+list(auc2.columns[1:]); tbl(auc2.round(3),fs=8,cap=f"Tabela {tabref()}. AUC de detecção de dano pelo índice de distância à referência, por banda.")

# ================== 5. T_ref ==================
H("5. Sensibilidade à temperatura de referência")
mtr=Rd("06_sensibilidade_referencia/melhor_tref_resumo.csv")
if mtr is not None:
    mt=mtr.copy(); mt["metodo"]=mt["metodo"].map(lambda m: LBL.get(m,m))
    mt.columns=["Banda","Método","Melhor T_ref","RMSD (melhor)","Pior T_ref","RMSD (pior)","Amplitude"]
    tbl(mt,fs=7.5,cap=f"Tabela {tabref()}. Melhor/pior temperatura de referência por banda e método (amplitude = sensibilidade a T_ref).")

# ================== 6. PRESERVAÇÃO DE DANO ==================
H("6. Preservação da assinatura de dano")
dpr=Rr("06_damage_preservation/damage_preservation_ratio.csv")
if dpr is not None:
    pv=dpr.pivot_table(index="metodo",columns="dano",values="DPR").round(3).reset_index()
    pv["metodo"]=pv["metodo"].map(lambda m: LBL.get(m,m)); pv.columns=["Método"]+[f"DPR Dano {int(c)}" for c in pv.columns[1:]]
    tbl(pv,fs=8,cap=f"Tabela {tabref()}. Damage Preservation Ratio (DPR = separação pós/pré). ≈1 preserva; <1 atenua; >1 amplifica.")
pkm=Rr("01_temperature_compensation/peak_metrics.csv")
if pkm is not None:
    pk=pkm.copy(); pk["metodo"]=pk["metodo"].map(lambda m: LBL.get(m,m)); pk.columns=["Método","Erro freq. pico (Hz)","Erro amp. pico","Preservação de picos"]
    tbl(pk,fs=8,cap=f"Tabela {tabref()}. Métricas de pico das curvas saudáveis compensadas (menor erro e maior preservação = melhor).")

# ================== 7. RIGOR ESTATÍSTICO ==================
H("7. Rigor estatístico")
wl=Rd("09_estatistica/wilcoxon_holm_RMSD.csv")
if wl is not None: tbl(wl,fs=8,cap=f"Tabela {tabref()}. Wilcoxon pareado (correção de Holm) para o RMSD saudável.")
ic=Rd("08_analises_avancadas/bootstrap_ic_rmsd.csv")
if ic is not None:
    ic2=ic.rename(columns={ic.columns[0]:"metodo"}); ic2["metodo"]=ic2["metodo"].map(lambda m: LBL.get(m,m)); ic2.columns=["Método","Média","IC 2,5%","IC 97,5%"]
    tbl(ic2.round(3),fs=8,cap=f"Tabela {tabref()}. Intervalos de confiança de 95% (bootstrap) do RMSD saudável.")
cli=Rd("08_analises_avancadas/cliff_delta.csv")
if cli is not None:
    cli.columns=["Comparação","δ de Cliff","Magnitude","Interpretação"]; tbl(cli,fs=8,cap=f"Tabela {tabref()}. Tamanho de efeito (delta de Cliff) no RMSD saudável.")

# ================== 8. SELETOR + CUSTO + PIOR CASO ==================
H("8. Seletor adaptativo, custo e pior caso")
sj=Jd("08_analises_avancadas/seletor_resumo.json")
if sj:
    sd=pd.DataFrame(sorted(sj.items(),key=lambda x:x[1]),columns=["Estratégia","RMSD saudável médio"]); sd["RMSD saudável médio"]=sd["RMSD saudável médio"].round(3)
    tbl(sd,widths=[8*cm,5*cm],fs=8,cap=f"Tabela {tabref()}. Seletor adaptativo por banda (escolha por CV interna) vs. métodos fixos e oráculo.")
cc=Rr("01_temperature_compensation/custo_computacional.csv")
if cc is not None: tbl(cc,fs=8,cap=f"Tabela {tabref()}. Custo computacional (CPU): treino, inferência e tamanho do modelo.")
lk=Rr("01_temperature_compensation/temperature_leakage.csv")
if lk is not None:
    lk2=lk.copy(); lk2["metodo"]=lk2["metodo"].map(lambda m: LBL.get(m,m)); lk2.columns=["Método","R² prever T","MAE T (°C)"]
    tbl(lk2,fs=8,cap=f"Tabela {tabref()}. Vazamento térmico residual: recuperabilidade da temperatura após a compensação (menor R² = mais invariante).")
wc=Rr("10_statistics/worst_case.csv")
if wc is not None:
    wc2=wc.copy(); wc2["metodo"]=wc2["metodo"].map(lambda m: LBL.get(m,m)); wc2.columns=["Método","RMSD médio","RMSD pior","T pior (°C)","RMSD p90"]
    tbl(wc2,fs=8,cap=f"Tabela {tabref()}. Análise de pior caso por método (todas as bandas).")

# ================== 9. HIPERPARÂMETROS E APRIMORAMENTO ==================
H("9. Estudo de hiperparâmetros e aprimoramento (Extra Trees)")
etl=Rd("08_analises_avancadas/extratrees_loto.csv")
if etl is not None:
    et2=etl[["banda","RF_RMSD_D0","ET_RMSD_D0","ganho_%","RF_healthy_sep","ET_healthy_sep"]].round(3)
    et2.columns=["Banda","RMSD RF","RMSD Extra Trees","Ganho (%)","healthy_sep RF","healthy_sep ET"]
    tbl(et2,fs=8,cap=f"Tabela {tabref()}. Random Forest vs. Extra Trees na compensação (teste LOTO). Extra Trees vence em todas as bandas, sem perda de preservação de dano.")
reg=Rd("08_analises_avancadas/regressores_alternativos.csv")
if reg is not None:
    reg.columns=["Banda","RMSD RF (CV)","RMSD Extra Trees (CV)","Melhor"]
    tbl(reg,fs=8,cap=f"Tabela {tabref()}. Comparação de regressores por CV interna (validação da escolha, sem tocar no teste).")

out=os.path.join(MAN,"artigo_TABELAS.pdf")
def _deco(canvas,doc):
    canvas.saveState(); L,R=2*cm,A4[0]-2*cm
    canvas.setFont("Times-Italic",7.5); canvas.setFillColor(colors.HexColor("#777"))
    canvas.drawString(L,1.05*cm,"EESC-USP · FAPESP 2025/09586-5 — Compêndio de Tabelas");
    canvas.setFont("Times-Roman",8.5); canvas.setFillColor(ACCENT); canvas.drawRightString(R,1.03*cm,f"{doc.page}")
    canvas.setStrokeColor(HAIR); canvas.setLineWidth(0.5); canvas.line(L,1.3*cm,R,1.3*cm)
    if doc.page>1:
        top=A4[1]-1.15*cm; canvas.setFont("Times-Italic",7.5); canvas.setFillColor(colors.HexColor("#888"))
        canvas.drawString(L,top,"Compêndio de Tabelas — compensação térmica em EMI"); canvas.setFont("Times-Bold",7.5); canvas.setFillColor(ACCENT); canvas.drawRightString(R,top,"EESC-USP")
        canvas.setStrokeColor(HAIR); canvas.setLineWidth(0.5); canvas.line(L,top-0.12*cm,R,top-0.12*cm)
    canvas.restoreState()
doc=SimpleDocTemplate(out,pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2.15*cm,bottomMargin=1.9*cm,title="Compendio de Tabelas - EMI")
doc.build(story,onFirstPage=_deco,onLaterPages=_deco)
print(f"OK PDF de tabelas: {out} | {TABN[0]} tabelas")

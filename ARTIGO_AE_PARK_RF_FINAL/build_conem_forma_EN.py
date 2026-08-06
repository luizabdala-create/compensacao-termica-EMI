# -*- coding: utf-8 -*-
"""
CONEM MINI-PAPER (ENGLISH) — the "cherry on top" of the SHAPE MODEL.
Same ABCM/CONEM layout as build_pdf.py; English text; loads English figures from
13_modelo_forma/graficos_en/. Data-driven from 13_modelo_forma/metricas.
Output: 13_modelo_forma/artigo_forma_CONEM_EN.pdf
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
OUT  = os.path.join(ROOT, "13_modelo_forma"); FIGDIR = os.path.join(OUT, "graficos_en")
MET  = os.path.join(OUT, "metricas"); MAN = os.path.join(ROOT, "12_manuscrito")
PDF  = os.path.join(OUT, "artigo_forma_CONEM_EN.pdf")

def nf(x, n=3): return f"{x:.{n}f}"

A = pd.read_csv(os.path.join(MET, "shapeA_saudavel.csv"))
B = pd.read_csv(os.path.join(MET, "classB_dano.csv"))
C = pd.read_csv(os.path.join(MET, "guardaC_dano.csv"))
BANDS = [b for b in ["30-40","40-50","50-60","60-70","70-80","30-70"] if b in A.banda.unique()]
NARROW = [b for b in BANDS if b != "30-70"]
realB = B[B.controle == "real"]
def m_ccdm(m):  return A[A.metodo==m].CCDM.mean()
def m_corr(m):  return A[A.metodo==m].CORR.mean()
def m_peak(m):  return A[A.metodo==m].peak_hz.mean()
def m_bal(m,t): return realB[(realB.metodo==m)&(realB.task==t)].bal_acc.mean()
def m_dano(m,d):return C[(C.metodo==m)&(C.dano==d)].CCDM.mean()
sh_mean = B[B.controle=="shuffled"].groupby("metodo").bal_acc.mean().mean()
dccdm = m_ccdm("AE_forma")-m_ccdm("AE_amp"); dpeak = m_peak("AE_forma")-m_peak("AE_amp")
ae_wins_narrow = [b for b in NARROW if A[(A.banda==b)&(A.metodo=="AE_forma")].CCDM.mean()
                  == min(A[(A.banda==b)&(A.metodo==m)].CCDM.mean() for m in ["Park","RF_amp","AE_amp","AE_forma"])]

ss = getSampleStyleSheet()
def sty(n, **k):
    k.setdefault("fontName","Times-Roman"); return ParagraphStyle(n, parent=ss["BodyText"], **k)
H1   = sty("H1", fontSize=11, spaceBefore=12, spaceAfter=4, fontName="Times-Bold")
H2   = sty("H2", fontSize=10.3, spaceBefore=9, spaceAfter=3, fontName="Times-Bold")
BODY = sty("BODY", fontSize=10, leading=12.7, alignment=TA_JUSTIFY, spaceAfter=1.5, firstLineIndent=0.7*cm)
CAP  = sty("CAP", fontSize=8.6, leading=10.6, alignment=TA_CENTER, spaceAfter=10, spaceBefore=3)
REF  = sty("REF", fontSize=8.6, leading=10.8, alignment=TA_JUSTIFY, spaceAfter=2, leftIndent=14, firstLineIndent=-14)
TITLE= sty("TIT", fontSize=14.5, leading=17.5, alignment=TA_CENTER, fontName="Times-Bold")
AUT  = sty("AUT", fontSize=10, alignment=TA_LEFT, spaceAfter=1, fontName="Times-Bold")
AFIL = sty("AFIL", fontSize=9, alignment=TA_LEFT, spaceAfter=1)
ABSb = sty("ABSb", fontSize=9.2, leading=12, alignment=TA_JUSTIFY, fontName="Times-Italic")
TCAP = sty("TCAP", fontSize=8.6, leading=10.6, alignment=TA_CENTER, spaceBefore=7, spaceAfter=3)
HEAD = sty("head", fontSize=8.5, leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#222"))

story = []; SEC = [0,0]
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
    if not os.path.exists(fp): fp = os.path.join(OUT,"graficos", name+".png")   # fallback PT
    if not os.path.exists(fp):
        if cap: P_("<i>[figure pending: %s]</i>"%name, CAP); return
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

# ---------- header / title / abstract ----------
_logo=os.path.join(MAN,"logo_eesc_usp.png")
if os.path.exists(_logo):
    _iw,_ih=ImageReader(_logo).getSize(); _lw=2.0*cm; _lh=_lw*_ih/_iw
    _txt=Paragraph("<b>São Carlos School of Engineering — University of São Paulo (EESC-USP)</b><br/>"
                   "Dynamics Laboratory · Department of Mechanical Engineering<br/>"
                   "FAPESP Undergraduate Research 2025/09586-5 · São Carlos, SP, Brazil", HEAD)
    _ht=Table([[Image(_logo,width=_lw,height=_lh),_txt]],colWidths=[2.6*cm,12.4*cm])
    _ht.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
    story.append(_ht)
    story.append(HRFlowable(width="100%",thickness=0.8,color=colors.black,spaceBefore=4,spaceAfter=12))
P_("Shape-Oriented Thermal Compensation of Electromechanical Impedance Signals: an Autoencoder that Preserves the Damage Signature", TITLE)
SP(9)
story.append(Paragraph("Luiz Eduardo Abdala José, luiz.abdala@usp.br<super>1</super>", AUT))
story.append(Paragraph("Kayc Wayhs Lopes, kayc.lopes@usp.br<super>1</super>", AUT))
SP(5)
story.append(Paragraph("<super>1</super>Department of Mechanical Engineering, Dynamics Laboratory, São Carlos School of Engineering (EESC), University of São Paulo (USP), São Carlos, SP, Brazil", AFIL))
SP(9)
box([Paragraph(
    f"<b>Abstract.</b> Thermal compensation of Electromechanical Impedance (EMI) signals is usually formulated to minimize the "
    f"<b>amplitude error</b> (RMSD) between the compensated curve and a healthy reference. Yet it is the <b>shape</b> of the spectrum — "
    f"the position and relative amplitude of the resonances — that carries the damage signature. This work proposes a compensator with an "
    f"<b>explicit shape objective</b>: an autoencoder (AE) whose loss includes a Pearson-correlation term between the compensated curve and "
    f"the reference, learned <b>from healthy curves only</b> and applied identically to damaged curves. The model is assessed on two "
    f"independent axes, over {A.T_test.nunique()} temperatures from -10 to 80 C and three structural states, by leakage-free "
    f"<i>leave-one-temperature-out</i> (LOTO) validation: <b>(A)</b> shape fidelity on healthy curves (CCDM, correlation, peak error) and "
    f"<b>(B)</b> damage classification 0/1/2 before and after compensation. The shape objective lowers the healthy CCDM from "
    f"{nf(m_ccdm('AE_amp'))} to {nf(m_ccdm('AE_forma'))} and the peak error from {nf(m_peak('AE_amp'),0)} to {nf(m_peak('AE_forma'),0)} Hz "
    f"relative to the amplitude AE and, decisively, <b>increases</b> the multiclass balanced accuracy from {nf(m_bal('AE_amp','multi'))} to "
    f"{nf(m_bal('AE_forma','multi'))} — the highest among all methods, including Park's method ({nf(m_bal('Park','multi'))}) and the optimized Random Forest "
    f"({nf(m_bal('RF_amp','multi'))}). A negative control with shuffled labels (mean accuracy {nf(sh_mean)}) rules out leakage. We thus show "
    f"that optimizing the shape of the healthy curve removes the thermal interference <b>without</b> erasing damage — it actually improves "
    f"its recognition — provided that the compensation is learned from the healthy state alone.", ABSb),
     Spacer(1,5),
     Paragraph("<b>Keywords:</b> Structural Health Monitoring, Electromechanical Impedance, Temperature Compensation, Autoencoder, "
               "Shape Objective, Damage Preservation.", ABSb)])
SP(4)

# 1. INTRODUCTION
H("Introduction")
P_("Structural Health Monitoring (SHM) based on Electromechanical Impedance (EMI) infers structural changes from the electrical "
   "impedance of a piezoelectric (PZT) transducer bonded to the structure, a low-cost technique highly sensitive to incipient damage "
   "(Liang et al., 1994; Giurgiutiu and Rogers, 1998; Na and Baek, 2018). Its main weakness is sensitivity to temperature: Baptista et "
   "al. (2014) showed that thermal variations shift the spectrum horizontally and vertically and smooth its peaks, often with a magnitude "
   "<b>larger</b> than that of incipient damage itself. Without compensation, temperature masks damage and produces false alarms.")
P_("The classical method of Park et al. (1999) compensates temperature by a global horizontal shift and a level adjustment of the real "
   "part of the impedance, with performance that degrades at high frequencies. Machine-learning approaches (regressions, forests, neural "
   "networks) have been proposed to capture the thermal response more flexibly. One point, however, is systematically overlooked: almost "
   "every compensator — classical or neural — is optimized to minimize the <b>amplitude error</b> (RMSD) against a healthy reference. "
   "Amplitude, though, is only one aspect of the signal; damage information lies in the <b>shape</b> of the spectrum, i.e., in the position "
   "and relative amplitude of the resonances. A method may lower the RMSD and still leave the resonances misaligned — precisely what one "
   "wants to avoid.")
P_("This work investigates a specific question: <b>what if the compensator is optimized for shape rather than amplitude?</b> We propose an "
   "autoencoder whose loss includes a correlation term between the compensated curve and the healthy reference (Section 2.2). The immediate "
   "risk of a shape objective is evident and is treated as the central hypothesis: forcing the compensated curve to <i>look like</i> the "
   "healthy reference could <b>erase the damage</b>. For this reason the correction is learned <b>exclusively</b> on healthy curves (it "
   "never sees damage) and the model is evaluated on two independent axes — thermal removal <i>and</i> damage preservation/recognition — "
   "with a negative control against leakage. The thesis we defend is that, under these conditions, the shape objective removes the thermal "
   "interference without destroying the damage signature.")

# 2. METHODOLOGY
H("Methodology")
H("Dataset and reference", H2)
P_(f"We use 164 impedance curves (real part) from a PZT-instrumented metallic beam, suspended by elastic bands inside a thermal chamber, "
   f"swept from -10 to 80 C, in three states: <b>healthy (D0)</b>, <b>Damage 1 (D1)</b> — an added mass — and <b>Damage 2 (D2)</b> — a cut "
   f"in the structure. The compensation <b>reference</b> is fixed: the median of the healthy curves at 30 C, built from training data only "
   f"(never from a damaged curve). The analysis is resolved by frequency band, over {len(BANDS)} representative ranges: {len(NARROW)} narrow "
   f"({', '.join(b+' kHz' for b in NARROW)}) and one wide (30-70 kHz).")
H("Shape-oriented compensator (AE-shape)", H2)
P_("The autoencoder receives the measured curve and the temperature and predicts the <b>thermal correction</b> DeltaZ = z_ref - z, which, "
   "added to the curve, yields the compensation. The novelty is in the loss function. Besides the amplitude term (Huber on DeltaZ) and a "
   "derivative term that preserves the local slope, a <b>shape term</b> based on the Pearson correlation between the reconstructed "
   "compensated curve and the healthy reference is added:")
P_("<i>L = Huber(ΔẐ, ΔZ) + λ<sub>d</sub> · Huber(∂ΔẐ, ∂ΔZ) + λ<sub>c</sub> · [1 − ρ(ẑ<sub>comp</sub>, z<sub>ref</sub>)]</i>,",
   sty("EQ",fontSize=10,alignment=TA_CENTER,fontName="Times-Italic",spaceBefore=3,spaceAfter=5))
P_("where ρ is the correlation coefficient, invariant to scale and offset — hence a purely <b>shape</b> term. The weight "
   "λ<sub>c</sub> is selected by <b>inner cross-validation</b> (on training temperatures only, on the healthy CCDM only), never using "
   "damage labels. When &#955;<sub>c</sub> = 0 the conventional amplitude AE (AE-amplitude) is recovered, which isolates the effect of the "
   "shape objective. AE-shape and AE-amplitude share architecture and all other hyperparameters. The thermal correction is bounded to the "
   "healthy training envelope (clip), which removes local over-correction artifacts without touching damage. As baselines we include "
   "<b>Park</b>'s method and the <b>optimized Random Forest</b> (direct regression of DeltaZ, with improved parameters and lower variance), plus the <b>Original</b> (uncompensated) signal.")
H("Two-axis evaluation protocol", H2)
P_("All evaluation is <b>leave-one-temperature-out</b> (LOTO): in each round a whole temperature is held out for testing, and reference, "
   "scaling, hyperparameter selection and training use only the remaining ones. We measure separately: <b>(A) thermal removal</b> — on the "
   "healthy test curves, the CCDM (1 - correlation), the correlation, and the main-peak position error relative to the reference; and "
   "<b>(B) damage preservation</b> — classification of the three states (0/1/2) from the compensated curve, with the classifier trained on "
   "training temperatures only, using four feature sets and three classifiers, plus a <b>negative control</b> with shuffled labels to detect "
   "leakage. A shape method is deemed superior only if it improves (A) <b>without</b> worsening (B).")

# 3. RESULTS AND DISCUSSION
H("Results and Discussion")
n_tr = figref()
fig("fig_tradeoff", 11.5, f"Figure {n_tr}. Shape vs. damage-preservation trade-off (mean of the frequency bands, LOTO). Horizontal axis: "
    f"healthy CCDM (shape; lower = better, to the left). Vertical axis: balanced accuracy of damage classification (higher = better). The "
    f"arrow shows the gain from the shape objective: AE-shape (purple) dominates AE-amplitude (blue) on both axes. The upper-left corner is ideal.")
P_(f"<b>Overview.</b> Figure {n_tr} summarizes the result. No method is optimal on both axes simultaneously, but the shape objective "
   f"repositions the autoencoder unambiguously: the arrow links AE-amplitude to AE-shape <b>upward and to the left</b>, i.e., better shape "
   f"<i>and</i> better classification at the same time. The optimized Random Forest occupies the far left (best mean shape) but the lowest damage accuracy; "
   f"AE-shape occupies the top (best damage recognition) with competitive shape.")
def rowm(lbl, m):
    return {"Method":lbl, "CCDM (healthy)":nf(m_ccdm(m)), "Corr.":nf(m_corr(m)),
            "Peak err. (Hz)":nf(m_peak(m),0), "Bal. acc. multi":nf(m_bal(m,"multi")), "Bal. acc. bin.":nf(m_bal(m,"bin"))}
Vt = pd.DataFrame([rowm("Original","Original"), rowm("Park","Park"), rowm("Optimized RF","RF_amp"),
                   rowm("Autoencoder (amplitude)","AE_amp"), rowm("Autoencoder (shape)","AE_forma")])
tbl(Vt, widths=[4.9*cm,2.3*cm,1.7*cm,2.4*cm,2.5*cm,2.3*cm], cap=f"Table {tabref()}. Mean performance (bands, LOTO). Shape on healthy: "
    f"lower CCDM and peak error are better, higher correlation is better. Damage preservation: balanced accuracy (multiclass 0/1/2 and "
    f"binary), higher is better. The shape autoencoder has the best damage classification of all methods.")
P_(f"<b>Axis (A): the shape objective works.</b> Relative to the amplitude AE, AE-shape reduces the healthy CCDM by {nf(abs(dccdm))} "
   f"({nf(m_ccdm('AE_amp'))} to {nf(m_ccdm('AE_forma'))}) and the main-peak position error by {nf(abs(dpeak),0)} Hz "
   f"({nf(m_peak('AE_amp'),0)} to {nf(m_peak('AE_forma'),0)} Hz). The gain is even larger in the narrow bands, where AE-shape reaches the "
   f"<b>lowest CCDM of all methods</b> (Figure {FIGN[0]+1}): {', '.join(b+' kHz' for b in ae_wins_narrow)}. The global CCDM mean "
   f"({nf(m_ccdm('AE_forma'))}) is penalized only by the wide 30-70 kHz band, discussed below.")
n_A = figref()
fig("fig_A_shape_por_banda", 15.8, f"Figure {n_A}. Axis (A) - thermal removal on the healthy test curves, by band. (a) CCDM (lower = better); "
    f"(b) main-peak position error on a logarithmic scale. In the narrow bands the shape autoencoder matches or beats the optimized Random Forest and Park's "
    f"method; in the wide 30-70 kHz band the autoencoder degrades and the forests/Park prevail.")
P_(f"<b>Axis (B): damage is not erased - it is better recognized.</b> This is the central result. Far from destroying the damage signature, "
   f"the shape objective <b>improves</b> classification: the multiclass balanced accuracy rises from {nf(m_bal('AE_amp','multi'))} "
   f"(AE-amplitude) to {nf(m_bal('AE_forma','multi'))} (AE-shape), the <b>highest of all methods</b>, surpassing Park's method "
   f"({nf(m_bal('Park','multi'))}), the optimized Random Forest ({nf(m_bal('RF_amp','multi'))}) and the original signal ({nf(m_bal('Original','multi'))}). "
   f"In <b>binary</b> detection, however, the amplitude AE retains a slight edge ({nf(m_bal('AE_amp','bin'))} against "
   f"{nf(m_bal('AE_forma','bin'))}) — the first sign, consistent with the absence of an internal brake discussed below, that pushing shape "
   f"to the maximum (λ<sub>c</sub> at the top of the grid) has a small cost on the binary task, although AE-shape stays above Park "
   f"({nf(m_bal('Park','bin'))}). The negative control with shuffled labels collapses to about {nf(sh_mean,2)} - near chance - confirming "
   f"the absence of leakage. Figure {FIGN[0]+1} shows that AE-shape's lead in multiclass classification holds across most bands.")
n_B = figref()
fig("fig_B_class_multi_por_banda", 15.0, f"Figure {n_B}. Axis (B) - damage preservation: balanced accuracy of multiclass (0/1/2) "
    f"classification from the compensated curve, by band (LOTO; classifier trained on training temperatures only). The Original signal is the "
    f"'before-compensation' state. The shape autoencoder is the best or tied-best in most bands.")
P_(f"<b>Why does optimizing shape help classification?</b> The explanation is both physical and statistical. Thermal variation is a "
   f"high-energy confounder superimposed on the damage signal; by maximizing the correlation of the healthy curve with the reference, "
   f"AE-shape more completely removes this <i>nuisance</i> component - common to all states - letting the classifier operate on a cleaner "
   f"residual in which the discriminative damage part remains. The amplitude term (RMSD), by contrast, can be minimized by level adjustments "
   f"that do not realign the resonances, preserving structured thermal noise that hinders classification. In short, shape and diagnosis are "
   f"aligned objectives; amplitude and diagnosis are not necessarily so.")
n_cur = figref()
_cur = f"fig_curvas_70-80_T-10"
fig(_cur, 16.5, f"Figure {n_cur}. Example curves at -10 C, 70-80 kHz (narrow band). Left: healthy state - the compensation aligns the shifted "
    f"spectrum (gray) with the reference (black), and AE-shape follows the shape faithfully. Center and right: Damage 1 and Damage 2 - the "
    f"damage-characteristic resonances <b>remain</b> after compensation; damage is not turned into a healthy curve.")
P_(f"<b>An honest caveat: the damage guard.</b> Figure {n_cur} visually confirms that the D1 and D2 resonances survive compensation. Still, "
   f"a side effect of the shape objective must be reported: it brings damaged curves closer to the healthy reference than the amplitude AE "
   f"does (damage-D1 CCDM {nf(m_dano('AE_forma',1))} against {nf(m_dano('AE_amp',1))}). This effect is partly <b>legitimate</b> - the "
   f"damaged curve is also at a temperature different from the reference, and compensation removes that thermal part, correctly bringing it "
   f"closer. The proof that damage is not erased is operational and multivariate: classification (axis B) <b>improves</b>. It is worth "
   f"distinguishing two kinds of damage index: scalar descriptors based only on the distance to the reference tend to lose margin under the "
   f"shape objective, whereas classifiers based on residual features benefit. Figure {FIGN[0]+1} details this guard.")
n_C = figref()
fig("fig_C_guarda_dano", 14.0, f"Figure {n_C}. Damage guard: CCDM of the Damage-1 and Damage-2 curves relative to the healthy reference, by "
    f"method. Values near zero would indicate collapse of damage onto the healthy state; no method reaches that limit. AE-shape reduces the "
    f"scalar distance more (thermal-removal effect) but preserves multivariate separability, as axis (B) proves.")
P_(f"<b>Where the shape method does not win.</b> Rigor requires bounding the result. First, in the <b>wide 30-70 kHz band</b> the autoencoder "
   f"- in both versions - degrades sharply (CCDM around {nf(A[(A.banda=='30-70')&(A.metodo=='AE_forma')].CCDM.mean(),2)}), while the optimized Random Forest "
   f"({nf(A[(A.banda=='30-70')&(A.metodo=='RF_amp')].CCDM.mean(),2)}) and Park "
   f"({nf(A[(A.banda=='30-70')&(A.metodo=='Park')].CCDM.mean(),2)}) remain robust: modeling a broad, multi-resonant response with few healthy "
   f"curves exceeds the AE's generalization capacity. The shape objective is therefore a <b>narrow-band</b> tool. Second, the same objective "
   f"applied to the optimized Random Forest produced no effect: hyperparameter selection by CCDM converged to the same configuration chosen by RMSD, because "
   f"the direct DeltaZ regression target already reconstructs amplitude and morphology jointly. The shape gain is thus a phenomenon of the "
   f"autoencoder's <b>differentiable loss</b>, not of the forests.")
P_(f"<b>A methodological warning.</b> Inner validation drove the shape weight &#955;<sub>c</sub> to the <b>maximum</b> of the grid in every "
   f"band, since the healthy CCDM decreases monotonically with &#955;<sub>c</sub>. This reveals that a pure shape objective, calibrated on "
   f"healthy data only, has <b>no internal brake</b> against damage erosion: nothing in the loss prevents shape from being pursued "
   f"indefinitely. On this dataset axis (B) showed that the cost did not materialize - on the contrary, there was a gain - but this external "
   f"check is <b>mandatory</b>, not optional. Any shape-oriented compensator should always be paired with a classification-based measure of "
   f"damage preservation, lest one optimize a metric that drifts away from the diagnostic goal.")

# 4. CONCLUSION
H("Conclusion")
P_(f"We showed that reformulating EMI thermal compensation around a <b>shape objective</b> - a correlation term between the compensated "
   f"curve and the healthy reference, learned from the healthy state only - simultaneously improves temperature removal and damage "
   f"recognition. The shape autoencoder reduced the healthy CCDM and the peak error relative to the amplitude autoencoder and reached the "
   f"<b>best damage classification of all evaluated methods</b> (multiclass balanced accuracy of {nf(m_bal('AE_forma','multi'))}), surpassing "
   f"Park's method and the optimized Random Forest, with no sign of leakage. The result counters the intuition that bringing the compensated curve closer to "
   f"the healthy reference would erase damage: when the correction is learned exclusively on healthy data, what is removed is the thermal "
   f"interference common to all states, not the damage signature. The limitations are clear and stated: the gain is specific to <b>narrow "
   f"bands</b> (the autoencoder fails in the wide band, where forests and Park dominate), the shape objective has no effective analogue in "
   f"the forests, and its lack of an internal brake demands mandatory external verification by classification. As future work, we propose "
   f"bounding the correction magnitude to remove local artifacts, a multi-objective selection (shape + separability), and extension to "
   f"thermal extrapolation tests.")

# REFERENCES
H("References", num=False)
refs = [
 "Baptista, F. G., Budoya, D. E., de Almeida, V. A. D., Ulson, J. A. C., 2014. \u201cAn experimental study on the effect of temperature on piezoelectric sensors for impedance-based structural health monitoring\u201d. Sensors, vol. 14, no. 1, pp. 1208-1227.",
 "Farrar, C. R., Worden, K., 2007. \u201cAn introduction to structural health monitoring\u201d. Philosophical Transactions of the Royal Society A, vol. 365, no. 1851, pp. 303-315.",
 "Giurgiutiu, V., Rogers, C. A., 1998. \u201cRecent advancements in the electromechanical (E/M) impedance method for structural health monitoring and NDE\u201d. Proceedings of SPIE, vol. 3329, pp. 536-547.",
 "Liang, C., Sun, F. P., Rogers, C. A., 1994. \u201cCoupled electro-mechanical analysis of adaptive material systems\u201d. Journal of Intelligent Material Systems and Structures, vol. 5, no. 1, pp. 12-20.",
 "Na, W. S., Baek, J., 2018. \u201cA review of the piezoelectric electromechanical impedance based structural health monitoring technique for engineering structures\u201d. Sensors, vol. 18, no. 5, art. 1307.",
 "Park, G., Kabeya, K., Cudney, H. H., Inman, D. J., 1999. \u201cImpedance-based structural health monitoring for temperature varying applications\u201d. JSME International Journal Series A, vol. 42, no. 2, pp. 249-258.",
 "Pedregosa, F. et al., 2011. \u201cScikit-learn: Machine learning in Python\u201d. Journal of Machine Learning Research, vol. 12, pp. 2825-2830.",
]
for r in refs: story.append(Paragraph(r, REF))

def _footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(A4[0]/2.0, 1.1*cm, str(doc.page)); canvas.restoreState()
doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=2.0*cm, rightMargin=2.0*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title="Shape-Oriented EMI Compensation (CONEM)", author="Luiz Eduardo Abdala Jose; Kayc Wayhs Lopes")
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print("PDF (EN):", PDF)

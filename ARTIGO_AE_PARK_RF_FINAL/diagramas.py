# -*- coding: utf-8 -*-
"""
DIAGRAMAS ESQUEMÁTICOS (pedido do revisor):
 (1) figX_setup_experimental — viga + elásticos + PZT + estufa térmica + cadeia de aquisição.
 (2) figX_pipeline — fluxograma do pipeline e da validação aninhada (anti-vazamento).
Desenho vetorial (matplotlib). Times New Roman. Sem dados inventados: apenas o que consta do
projeto de IC (estufa, viga em elásticos, PZT atuador/sensor, software Baptista 2010).
"""
import os,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,FancyArrowPatch,Circle
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; FIG=os.path.join(ROOT,"10_figuras_artigo")
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":11,"pdf.fonttype":42})
def save(fig,n):
    for e,dp in [("png",600),("pdf",None)]: fig.savefig(os.path.join(FIG,f"{n}.{e}"),dpi=dp,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("fig:",n,flush=True)

# ================= (1) SETUP EXPERIMENTAL =================
fig,ax=plt.subplots(figsize=(11,5.2),dpi=170); ax.set_xlim(0,11); ax.set_ylim(0,5.2); ax.axis("off")
# estufa térmica (caixa grande)
ax.add_patch(FancyBboxPatch((0.4,0.6),6.2,4.1,boxstyle="round,pad=0.02,rounding_size=0.12",lw=1.8,ec="#444",fc="#f4f6fb"))
ax.text(3.5,4.45,"Estufa térmica  (−10 a 80 °C)",ha="center",fontsize=11,style="italic",color="#333")
# viga sustentada por 2 elásticos
vy=2.4; ax.add_patch(Rectangle((1.4,vy-0.13),4.2,0.26,fc="#c9c9c9",ec="k",lw=1.2))
ax.text(3.5,vy-0.55,"viga metálica",ha="center",fontsize=10)
# elásticos (molas) nas pontas — ziguezague
import numpy as np
for x0 in [1.4,5.6]:
    yy=np.linspace(vy,vy+1.55,14); xx=x0+0.13*np.array([0,1,-1,1,-1,1,-1,1,-1,1,-1,0,0,0])
    ax.plot(xx,yy,color="#666",lw=1.2)
    ax.plot([x0-0.25,x0+0.25],[vy+1.55,vy+1.55],color="#666",lw=2)  # topo fixo
ax.text(1.0,vy+1.7,"elásticos",ha="center",fontsize=9,color="#555")
# PZT colado na viga
ax.add_patch(Rectangle((3.25,vy+0.13),0.5,0.22,fc="#1f5fd0",ec="k",lw=1))
ax.text(3.5,vy+0.55,"PZT\n(atuador/sensor)",ha="center",fontsize=8.5,color="#1f5fd0")
# fio saindo da estufa para o sistema de aquisição
ax.plot([3.5,3.5,7.0,7.0],[vy+0.35,3.7,3.7,3.15],color="#333",lw=1.3)
# sistema de aquisição
ax.add_patch(FancyBboxPatch((7.1,2.05),3.4,1.15,boxstyle="round,pad=0.02,rounding_size=0.08",lw=1.6,ec="#444",fc="#eef2f7"))
ax.text(8.8,2.8,"Sistema de aquisição",ha="center",fontsize=10.5,weight="bold")
ax.text(8.8,2.4,"analisador de impedância\n(software: Baptista, 2010)",ha="center",fontsize=8.5,color="#333")
# saída: espectro Z(f)
ax.add_patch(FancyBboxPatch((7.1,0.5),3.4,1.15,boxstyle="round,pad=0.02,rounding_size=0.08",lw=1.4,ec="#888",fc="white"))
fx=np.linspace(7.35,10.25,120); fz=1.0+0.32*np.exp(-((fx-8.1)/0.14)**2)+0.28*np.exp(-((fx-9.2)/0.12)**2)
ax.plot(fx,fz-0.05,color="#d62728",lw=1.2); ax.text(8.8,1.4,"Impedância Z(f,T)",ha="center",fontsize=9)
ax.add_patch(FancyArrowPatch((8.8,2.02),(8.8,1.68),arrowstyle="-|>",mutation_scale=14,lw=1.3,color="#333"))
# estados estruturais (legenda à direita inferior)
ax.text(3.5,0.15,"Estados:  D0 saudável   ·   D1 massa acoplada   ·   D2 corte",ha="center",fontsize=9.5,color="#222")
ax.set_title("Montagem experimental do ensaio de impedância eletromecânica sob temperatura controlada",fontsize=12,pad=6)
save(fig,"figX_setup_experimental")

# ================= (2) FLUXOGRAMA DO PIPELINE =================
fig,ax=plt.subplots(figsize=(12.5,6.6),dpi=170); ax.set_xlim(0,12.5); ax.set_ylim(0,6.6); ax.axis("off")
def caixa(x,y,w,h,txt,fc,ec="#333",fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.07",lw=1.4,ec=ec,fc=fc))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs)
def seta(p0,p1,color="#333"):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=13,lw=1.3,color=color))
BL="#eaf0fb"; GR="#e9f5ea"; OR="#fdf0e3"; GY="#eee"
# coluna externa: LOTO
caixa(0.2,5.6,3.0,0.8,"Dados: curvas EMI\n(temperatura × dano)",GY)
caixa(0.2,4.4,3.0,0.8,"LOTO externo:\nsepara 1 temperatura\n(teste), nunca vista",OR,fs=8.5)
seta((1.7,5.6),(1.7,5.2))
# nested CV
caixa(4.0,5.6,4.3,0.8,"Temperaturas de TREINO",BL)
caixa(4.0,4.4,4.3,0.8,"CV interna: escolhe banda +\nhiperparâmetros (só treino)",BL,fs=8.5)
caixa(4.0,3.2,4.3,0.8,"Referência z_ref = mediana\nsaudável @ T_ref (congelada)",GR,fs=8.5)
caixa(4.0,2.0,4.3,0.8,"Treina compensador\n(só curvas saudáveis)",GR,fs=8.5)
caixa(4.0,0.8,4.3,0.8,"Compensa curvas de teste →\nextrai features → classifica",BL,fs=8.5)
seta((6.15,5.6),(6.15,5.2)); seta((6.15,4.4),(6.15,4.0)); seta((6.15,3.2),(6.15,2.8)); seta((6.15,2.0),(6.15,1.6))
seta((3.2,4.8),(4.0,4.8))  # treino -> nested
# avaliação externa
caixa(9.1,2.0,3.2,0.8,"Avaliação EXTERNA\nno fold de teste",OR,fs=9)
caixa(9.1,0.8,3.2,0.8,"Métricas: RMSD/CCDM,\nAUC, F1, falso-saudável",GY,fs=8.5)
seta((8.3,1.2),(9.1,1.2)); seta((10.7,2.0),(10.7,1.6))
seta((1.7,4.4),(1.7,1.2)); seta((1.7,1.2),(4.0,1.2))
# nota anti-vazamento
ax.text(6.25,6.35,"Pipeline de compensação + classificação com validação aninhada (anti-vazamento)",ha="center",fontsize=12,weight="bold")
ax.text(6.25,0.25,"Regra de ouro: a temperatura de teste nunca entra no treino, no scaler, na CV interna nem na seleção de banda/método.",
        ha="center",fontsize=8.6,style="italic",color="#a00")
save(fig,"figX_pipeline")
print("✅ diagramas concluídos",flush=True)

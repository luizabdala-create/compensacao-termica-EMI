# -*- coding: utf-8 -*-
"""ORQUESTRADOR FINAL — experimentos novos + documentos + regenerar tudo + PDF final."""
import os,time,subprocess,traceback
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; PY=r"C:\Users\luize\anaconda3\python.exe"
env=dict(os.environ); env["MPLBACKEND"]="Agg"
def run(s,d,to=30000):
    t=time.time(); print(f"\n{'='*70}\n>>> {d}\n{'='*70}",flush=True)
    try:
        r=subprocess.run([PY,"-u",os.path.join(ROOT,s)],env=env,capture_output=True,text=True,timeout=to)
        print("\n".join((r.stdout or "").splitlines()[-25:]),flush=True)
        if r.returncode!=0: print("[ERRO]\n"+"\n".join((r.stderr or "").splitlines()[-20:]),flush=True)
        print(f"<<< {d} — {(time.time()-t)/60:.1f} min",flush=True); return r.returncode==0
    except Exception as e: print("[EXC]",d,e,flush=True); return False
STEPS=[("new_experiments.py","Experimentos novos (custo/histerese/vazamento/ref.danificada/DPR/corr/matriz/pico/PCA/worst)",40000),
       ("band_tref_analysis.py","Heatmaps (re-run, dados atualizados)",8000),
       ("stats_demsar.py","Estatística Demšar",8000),
       ("figuras_final.py","Figuras finais",8000),
       ("figuras_estilo_usuario.py","Figuras estilo usuário (curvas/dano)",12000),
       ("figuras_artigo2.py","Figuras curvas/confusão/classificação",8000),
       ("tabelas_artigo.py","Tabelas",8000),
       ("gerar_documentos.py","Documentos (metodologia/índice/best/falhas/interpretação/20 respostas)",8000),
       ("build_pdf.py","PDF do artigo COMPLETO",8000),
       ("build_latex.py","LaTeX do artigo",8000)]
t0=time.time(); res={}
for s,d,to in STEPS: res[s]=run(s,d,to)
try:
    import fitz; doc=fitz.open(os.path.join(ROOT,"12_manuscrito","artigo_PT.pdf"))
    outd=os.path.join(ROOT,"12_manuscrito","preview"); os.makedirs(outd,exist_ok=True)
    for i in range(doc.page_count): doc[i].get_pixmap(dpi=110).save(os.path.join(outd,f"p{i+1}.png"))
    print(f"\nPDF FINAL: {doc.page_count} paginas",flush=True)
except Exception as e: print("preview:",e,flush=True)
print(f"\n{'#'*70}\nSUMÁRIO FINAL ({(time.time()-t0)/60:.1f} min)",flush=True)
for s,ok in res.items(): print(f"  {'OK ' if ok else 'FALHOU'} {s}",flush=True)
print("✅ ARTIGO COMPLETO CONCLUÍDO",flush=True)

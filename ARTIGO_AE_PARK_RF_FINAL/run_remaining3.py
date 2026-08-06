# -*- coding: utf-8 -*-
"""
ORQUESTRADOR 3 — expansão final do artigo.
classificadores_todos (7 clf) -> tref_full (T_ref completo) -> band_tref_analysis (heatmaps)
-> stats_demsar -> figuras_final -> figuras_artigo2 -> tabelas_artigo -> build_pdf -> build_latex -> render.
Log + tolerância a erro por etapa. (figuras_estilo_usuario já rodou.)
"""
import os,sys,time,subprocess,traceback
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; PY=r"C:\Users\luize\anaconda3\python.exe"
env=dict(os.environ); env["MPLBACKEND"]="Agg"
def run(script,desc,timeout=20000):
    t=time.time(); print(f"\n{'='*70}\n>>> {desc}\n{'='*70}",flush=True)
    try:
        r=subprocess.run([PY,"-u",os.path.join(ROOT,script)],env=env,capture_output=True,text=True,timeout=timeout)
        print("\n".join((r.stdout or "").splitlines()[-18:]),flush=True)
        if r.returncode!=0: print(f"[ERRO rc={r.returncode}]\n"+"\n".join((r.stderr or '').splitlines()[-18:]),flush=True)
        print(f"<<< {desc} — {(time.time()-t)/60:.1f} min",flush=True); return r.returncode==0
    except Exception as e:
        print(f"[EXC] {desc}: {e}",flush=True); return False
STEPS=[("classificadores_todos.py","Classificação — 7 classificadores x 4 features",30000),
       ("tref_full.py","Varredura completa de T_ref (10 refs x 2 bandas)",30000),
       ("band_tref_analysis.py","Heatmaps banda/T_ref + faixa mais analisável",8000),
       ("stats_demsar.py","Estatística Demšar (sem tau(T))",8000),
       ("figuras_final.py","Figuras finais",8000),
       ("figuras_artigo2.py","Figuras curvas/confusão/classificação",8000),
       ("tabelas_artigo.py","Tabelas 1-9",8000),
       ("build_pdf.py","PDF do artigo (completo)",8000),
       ("build_latex.py","LaTeX do artigo",8000)]
t0=time.time(); res={}
for s,d,to in STEPS: res[s]=run(s,d,to)
try:
    import fitz
    doc=fitz.open(os.path.join(ROOT,"12_manuscrito","artigo_PT.pdf"))
    outd=os.path.join(ROOT,"12_manuscrito","preview"); os.makedirs(outd,exist_ok=True)
    for i in range(doc.page_count): doc[i].get_pixmap(dpi=110).save(os.path.join(outd,f"p{i+1}.png"))
    print(f"\nPDF: {doc.page_count} paginas",flush=True)
except Exception as e: print("preview:",e,flush=True)
print(f"\n{'#'*70}\nSUMÁRIO ({(time.time()-t0)/60:.1f} min)",flush=True)
for s,ok in res.items(): print(f"  {'OK ' if ok else 'FALHOU'} {s}",flush=True)
print("✅ CONCLUÍDO",flush=True)

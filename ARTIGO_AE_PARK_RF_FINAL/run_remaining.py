# -*- coding: utf-8 -*-
"""
ORQUESTRADOR FINAL — roda tudo o que falta em sequência, com log e tolerância a erro.
Ordem: fase9 (bandas largas + T_ref) -> parteB_v2 (classificação 7 bandas) ->
stats_demsar -> figuras (estilo usuário, finais, artigo2, fase1) -> build_pdf ->
build_latex -> render preview do PDF. Ao fim imprime SUMÁRIO.
"""
import os,sys,time,subprocess,traceback
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"
PY=r"C:\Users\luize\anaconda3\python.exe"
LOG=os.path.join(ROOT,"logs","run_remaining.log")
env=dict(os.environ); env["MPLBACKEND"]="Agg"
def run(script,desc):
    t=time.time(); print(f"\n{'='*70}\n>>> {desc}\n{'='*70}",flush=True)
    try:
        r=subprocess.run([PY,"-u",os.path.join(ROOT,script)],env=env,capture_output=True,text=True,timeout=20000)
        tail="\n".join((r.stdout or "").splitlines()[-15:])
        print(tail,flush=True)
        if r.returncode!=0:
            print(f"[ERRO rc={r.returncode}] stderr:\n"+"\n".join((r.stderr or '').splitlines()[-15:]),flush=True)
        print(f"<<< {desc} — {(time.time()-t)/60:.1f} min",flush=True)
        return r.returncode==0
    except Exception as e:
        print(f"[EXCEÇÃO] {desc}: {e}\n{traceback.format_exc()[-500:]}",flush=True); return False

STEPS=[("fase9_wide_tref.py","FASE 9 — bandas largas + varredura T_ref"),
       ("parteB_v2.py","PARTE B v2 — classificação 7 bandas"),
       ("stats_demsar.py","Estatística Demšar (Friedman+Nemenyi CD+Wilcoxon-Holm)"),
       ("figuras_final.py","Figuras finais (7 bandas tunadas)"),
       ("figuras_estilo_usuario.py","Figuras no estilo do usuário (lado a lado)"),
       ("figuras_artigo2.py","Figuras: curvas/confusão/classificação (tunadas)"),
       ("tabelas_artigo.py","Tabelas 1-9"),
       ("build_pdf.py","PDF do artigo (reportlab)"),
       ("build_latex.py","LaTeX do artigo (.tex)")]
t0=time.time(); results={}
for s,d in STEPS: results[s]=run(s,d)

# render preview do PDF
print("\n>>> render preview do PDF",flush=True)
try:
    import fitz
    pdf=os.path.join(ROOT,"12_manuscrito","artigo_PT.pdf"); doc=fitz.open(pdf)
    outdir=os.path.join(ROOT,"12_manuscrito","preview"); os.makedirs(outdir,exist_ok=True)
    for i in range(doc.page_count): doc[i].get_pixmap(dpi=110).save(os.path.join(outdir,f"p{i+1}.png"))
    print(f"PDF: {doc.page_count} paginas renderizadas",flush=True)
except Exception as e: print("erro preview:",e,flush=True)

print(f"\n{'#'*70}\nSUMÁRIO FINAL ({(time.time()-t0)/60:.1f} min)\n{'#'*70}",flush=True)
for s,ok in results.items(): print(f"  {'OK ' if ok else 'FALHOU'} {s}",flush=True)
print("\n✅ ORQUESTRADOR CONCLUÍDO",flush=True)

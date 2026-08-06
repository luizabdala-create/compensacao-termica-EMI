# -*- coding: utf-8 -*-
"""Acabamento: re-roda figuras (idioma corrigido + novas de banda) e reconstrói PDF/LaTeX."""
import os,time,subprocess
ROOT=r"C:\Users\luize\IC_EMI\ARTIGO_AE_PARK_RF_FINAL"; PY=r"C:\Users\luize\anaconda3\python.exe"
env=dict(os.environ); env["MPLBACKEND"]="Agg"; env["PYTHONIOENCODING"]="utf-8"; env["PYTHONUTF8"]="1"
def run(s,d,to=20000):
    t=time.time(); print(f"\n>>> {d}",flush=True)
    try:
        r=subprocess.run([PY,"-u",os.path.join(ROOT,s)],env=env,capture_output=True,text=True,timeout=to)
        print("\n".join((r.stdout or "").splitlines()[-8:]),flush=True)
        if r.returncode!=0: print("[ERRO]\n"+"\n".join((r.stderr or "").splitlines()[-12:]),flush=True)
        print(f"<<< {(time.time()-t)/60:.1f} min",flush=True); return r.returncode==0
    except Exception as e: print("[EXC]",e,flush=True); return False
STEPS=[("figuras_artigo.py","fig03-20 (idioma PT)"),("figuras_artigo2.py","curvas/classif (idioma PT)"),
       ("fig_seeds.py","seeds (idioma PT)"),("figuras_final.py","finais"),
       ("figuras_estilo_usuario.py","estilo usuário",12000),("band_tref_analysis.py","heatmaps"),
       ("analise_faixas_largura.py","banda/janelas (redraw c/ Times New Roman)",6000),
       ("stats_demsar.py","Demšar"),("tabelas_artigo.py","tabelas"),
       ("gerar_documentos.py","documentos"),("build_pdf.py","PDF"),("build_latex.py","LaTeX")]
res={}
for st in STEPS:
    s,d=st[0],st[1]; to=st[2] if len(st)>2 else 20000; res[s]=run(s,d,to)
try:
    import fitz; doc=fitz.open(os.path.join(ROOT,"12_manuscrito","artigo_PT.pdf"))
    outd=os.path.join(ROOT,"12_manuscrito","preview"); os.makedirs(outd,exist_ok=True)
    for i in range(doc.page_count): doc[i].get_pixmap(dpi=110).save(os.path.join(outd,f"p{i+1}.png"))
    print(f"\nPDF: {doc.page_count} paginas",flush=True)
except Exception as e: print("preview:",e,flush=True)
print("\nSUMÁRIO:",flush=True)
for s,ok in res.items(): print(f"  {'OK ' if ok else 'FALHOU'} {s}",flush=True)
print("✅ ACABAMENTO CONCLUÍDO",flush=True)

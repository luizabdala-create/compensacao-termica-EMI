# Compensação Térmica de Sinais de Impedância Eletromecânica com Aprendizado de Máquina

Notebooks da pesquisa de Iniciação Científica **FAPESP 2025/09586-5** (Laboratório de Dinâmica,
Departamento de Engenharia Mecânica, EESC-USP), sobre a **compensação do efeito da temperatura
em sinais de Impedância Eletromecânica (EMI)** por meio de aprendizado de máquina e a
**detecção e classificação de danos** em uma viga instrumentada com um transdutor PZT.

## Métodos

- **Park** — método estatístico clássico (deslocamento em frequência + deslocamento vertical), usado como referência.
- **Random Forest** — regressão que estima, ponto a ponto, a correção térmica do espectro.
- **Autoencoder** — rede neural codificador–decodificador (versões de amplitude e de forma).

A avaliação separa três objetivos: qualidade da compensação (RMSD, CCDM), preservação da
assinatura de dano e desempenho na classificação (detecção e identificação).

## Conteúdo

- `notebooks/` — notebooks Jupyter usados no desenvolvimento (compensação por Park, Random Forest
  e Autoencoder, cálculo dos índices RMSD e CCDM, curvas comparativas e classificação de dano);
  a subpasta `notebooks/testes/` contém os testes exploratórios.
- `ARTIGO_AE_PARK_RF_FINAL/` — pipeline de análise em Python, organizado por etapa: auditoria da
  base (`00_auditoria`), compensação térmica (`02_compensacao`), classificação de dano
  (`03_dano_binario`, `04_dano_multiclasse`), estudos por faixa e por temperatura de referência
  (`05_`–`07_`), ablações e análises complementares (`08_`, `09_`), figuras e tabelas
  (`10_`–`12_`) e o modelo de forma (`13_modelo_forma`).
- `scripts_iniciais/` — versões iniciais do desenvolvimento.
- `graficos_relatorio.py` — reproduz os gráficos de resultados do relatório.

## Ambiente

Python 3 (Anaconda), com `pandas`, `numpy`, `scikit-learn`, `torch` (CPU) e `matplotlib`.

## Dados

A base de dados experimental **não está incluída** neste repositório. Os sinais foram adquiridos
no âmbito do processo FAPESP nº 2016/12241-0 e podem ser disponibilizados mediante solicitação,
respeitadas as diretrizes de gestão de dados da FAPESP e do grupo de pesquisa.

## Autoria

- **Luiz Eduardo Abdala José** — bolsista (luiz.abdala@usp.br)
- **Prof. Dr. Kayc Wayhs Lopes** — orientador

"""
JobMatch AI - Pipeline Principal
=================================
Projeto Final: Desenvolvendo um Projeto E2E de Machine Learning
"""

import kagglehub
from datasets import load_dataset
import pandas as pd
import os

# 1. DOWNLOAD DOS DATASETS

print("=" * 50)
print("📥 Baixando datasets...")
print("=" * 50)

# LinkedIn Job Postings (Kaggle) — 124k vagas reais
print("\n[1/3] LinkedIn Job Postings...")
path_postings = kagglehub.dataset_download("arshkon/linkedin-job-postings")
print(f"  ✅ Salvo em: {path_postings}")

# Job Skill Set (Kaggle) — habilidades por cargo
print("\n[2/3] Job Skill Set Dataset...")
path_skillset = kagglehub.dataset_download("batuhanmutlu/job-skill-set")
print(f"  ✅ Salvo em: {path_skillset}")

# Resume-JD-Match (HuggingFace) — pares currículo/vaga rotulados
print("\n[3/3] Resume-JD-Match (HuggingFace)...")
dataset_match = load_dataset("facehuggerapoorv/resume-jd-match")
print(f"  ✅ Splits disponíveis: {list(dataset_match.keys())}")

# 2. CARREGAMENTO DOS DADOS

print("\n" + "=" * 50)
print("📂 Carregando arquivos em DataFrames...")
print("=" * 50)

# --- Resume-JD-Match ---
df_match = pd.DataFrame(dataset_match["train"])
print(f"\n[Resume-JD-Match]")
print(f"  Linhas: {len(df_match):,} | Colunas: {list(df_match.columns)}")
print(df_match.head(2))

# --- LinkedIn Job Postings ---
# Listar arquivos disponíveis no path
print(f"\n[LinkedIn Job Postings] Arquivos disponíveis:")
for root, dirs, files in os.walk(path_postings):
    for f in files:
        full = os.path.join(root, f)
        print(f"  {full}")

# Carregar o arquivo principal de postagens
postings_csv = os.path.join(path_postings, "postings.csv")
if os.path.exists(postings_csv):
    df_postings = pd.read_csv(postings_csv)
    print(f"\n  Linhas: {len(df_postings):,} | Colunas: {list(df_postings.columns)}")
    print(df_postings.head(2))
else:
    # Tentar encontrar o CSV principal
    csvs = [f for f in os.listdir(path_postings) if f.endswith(".csv")]
    print(f"  CSVs encontrados: {csvs}")
    df_postings = pd.read_csv(os.path.join(path_postings, csvs[0]))

# --- Job Skill Set ---
print(f"\n[Job Skill Set] Arquivos disponíveis:")
for root, dirs, files in os.walk(path_skillset):
    for f in files:
        full = os.path.join(root, f)
        print(f"  {full}")

skillset_files = [f for f in os.listdir(path_skillset) if f.endswith(".csv")]
df_skills = pd.read_csv(os.path.join(path_skillset, skillset_files[0]))
print(f"\n  Linhas: {len(df_skills):,} | Colunas: {list(df_skills.columns)}")
print(df_skills.head(2))

# 3. EDA

print("\n" + "=" * 50)
print("🔍 Exploração Rápida dos Dados")
print("=" * 50)

# Distribuição Fit / No Fit
print(f"\n[Resume-JD-Match] Distribuição 'label':")
print(df_match['label'].value_counts())

# Colunas relevantes do LinkedIn
colunas_uteis_postings = [
    "job_id", "title", "description", "company_name",
    "location", "min_salary", "max_salary", "med_salary",
    "skills_desc", "work_type"
]
colunas_disponiveis = [c for c in colunas_uteis_postings if c in df_postings.columns]
print(f"\n[LinkedIn] Colunas selecionadas: {colunas_disponiveis}")
df_postings_clean = df_postings[colunas_disponiveis].copy()

# Valores nulos
print(f"\n[LinkedIn] Nulos por coluna:")
print(df_postings_clean.isnull().sum())

print("\n✅ Pipeline de carregamento concluída com sucesso!")
print("\nPróximos passos:")
print("  → preprocessing.py  : limpeza e TF-IDF")
print("  → classifier.py     : modelo Fit/No Fit")
print("  → recommender.py    : similaridade cosseno + Top-5")
print("  → salary_model.py   : regressão de salário")
print("  → app.py            : interface Streamlit")
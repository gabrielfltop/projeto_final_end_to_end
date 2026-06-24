"""
JobMatch AI - Pipeline Principal
=================================
Script de ingestão e orquestração do projeto E2E.
Deve ser executado uma única vez para baixar os dados,
processar e treinar todos os modelos.

Ordem de execução:
  1. Download dos 3 datasets (Kaggle + HuggingFace)
  2. Preprocessing (limpeza, TF-IDF, geração dos CSVs em artifacts/)
  3. Classifier (MLP Fit/No Fit)
  4. Recommender (matriz TF-IDF das vagas)
  5. Salary Model (Ridge de estimativa salarial)

Após rodar este script, inicie o app com:
  streamlit run app.py
"""

import os

import kagglehub
import pandas as pd
from datasets import load_dataset

from preprocessing import run_preprocessing
from classifier    import run_classifier
from recommender   import create_jobs_matrix
from salary_model  import run_salary_model

# ── 1. DOWNLOAD DOS DATASETS ─────────────────────────────────────────────────────

print("=" * 50)
print("📥 Baixando datasets...")
print("=" * 50)

# LinkedIn Job Postings (Kaggle) — ~124k vagas reais
print("\n[1/3] LinkedIn Job Postings...")
path_postings = kagglehub.dataset_download("arshkon/linkedin-job-postings")
print(f"  ✅ {path_postings}")

# Job Skill Set (Kaggle) — habilidades por cargo
print("\n[2/3] Job Skill Set Dataset...")
path_skillset = kagglehub.dataset_download("batuhanmutlu/job-skill-set")
print(f"  ✅ {path_skillset}")

# Resume-JD-Match (HuggingFace) — pares currículo/vaga com label de fit
print("\n[3/3] Resume-JD-Match (HuggingFace)...")
dataset_match = load_dataset("facehuggerapoorv/resume-jd-match")
print(f"  ✅ Splits: {list(dataset_match.keys())}")

# ── 2. CARREGAMENTO ──────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("📂 Carregando arquivos em DataFrames...")
print("=" * 50)

df_match    = pd.DataFrame(dataset_match["train"])

# Localiza o CSV principal de postagens
postings_csv = os.path.join(path_postings, "postings.csv")
if not os.path.exists(postings_csv):
    csvs = [f for f in os.listdir(path_postings) if f.endswith(".csv")]
    postings_csv = os.path.join(path_postings, csvs[0])
df_postings = pd.read_csv(postings_csv)

skillset_files = [f for f in os.listdir(path_skillset) if f.endswith(".csv")]
df_skills      = pd.read_csv(os.path.join(path_skillset, skillset_files[0]))

print(f"  Resume-JD-Match : {len(df_match):,} linhas")
print(f"  Job Postings    : {len(df_postings):,} linhas")
print(f"  Job Skill Set   : {len(df_skills):,} linhas")

# ── 3. PREPROCESSING ─────────────────────────────────────────────────────────────

df_match_clean, df_postings_clean, df_skills_clean, _, _ = run_preprocessing(
    df_match, df_postings, df_skills
)

# ── 4. CLASSIFIER ────────────────────────────────────────────────────────────────

run_classifier(n_iter=10, cv=3, force_train=False, do_eval=True)

# ── 5. RECOMMENDER ───────────────────────────────────────────────────────────────

create_jobs_matrix()

# ── 6. SALARY MODEL ──────────────────────────────────────────────────────────────

run_salary_model(force_train=False, do_eval=True)

# ── CONCLUSÃO ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("✅ Pipeline completa! Artefatos salvos em /artifacts/")
print("   Inicie o app com: streamlit run app.py")
print("=" * 50)
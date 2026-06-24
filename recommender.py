"""
JobMatch AI - Recommender
============================
Recomenda vagas com base na similaridade cosseno entre o currículo
do candidato e as descrições das vagas LinkedIn (ambos em TF-IDF).

Fluxo:
  1. create_jobs_matrix() → vetoriza todas as vagas e salva a matriz (roda uma vez)
  2. load_jobs_matrix()   → carrega df + vectorizer + matriz em runtime
  3. vectorize_resume()   → transforma o currículo no mesmo espaço vetorial
  4. calculate_similarities() → retorna índices e scores das Top-N vagas
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = "artifacts"


def create_jobs_matrix():
    """
    Vetoriza todas as vagas do postings_clean.csv e salva a matriz esparsa.
    Executado uma única vez via __main__ após o preprocessing.
    """
    print("[recommender] Vetorizando vagas...")

    df         = pd.read_csv(os.path.join(OUTPUT_DIR, "postings_clean.csv"))
    vectorizer = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_jobs.pkl"))
    matrix     = vectorizer.transform(df["job_text"])

    joblib.dump(matrix, os.path.join(OUTPUT_DIR, "postings_vec_matrix.pkl"))
    print(f"  ✅ Matriz salva: {matrix.shape[0]:,} vagas × {matrix.shape[1]:,} features")


def load_jobs_matrix():
    """
    Carrega o DataFrame de vagas, o vetorizador TF-IDF e a matriz pré-computada.
    Retorna (df, vectorizer, matrix).
    """
    print("[recommender] Carregando vagas...")

    df         = pd.read_csv(os.path.join(OUTPUT_DIR, "postings_clean.csv"))
    vectorizer = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_jobs.pkl"))
    matrix     = joblib.load(os.path.join(OUTPUT_DIR, "postings_vec_matrix.pkl"))

    return df, vectorizer, matrix


def vectorize_resume(resume: str, vectorizer) -> object:
    """Transforma o texto do currículo em vetor TF-IDF (mesmo espaço das vagas)."""
    return vectorizer.transform([resume])


def calculate_similarities(resume_vec, postings_matrix, n: int = 5):
    """
    Calcula a similaridade cosseno entre o currículo e todas as vagas.
    Retorna (índices, scores) das Top-N vagas em ordem decrescente de relevância.
    """
    sim     = cosine_similarity(resume_vec, postings_matrix).flatten()
    top_idx = np.argsort(sim)[-n:][::-1]
    return top_idx, sim[top_idx]


# ── EXECUÇÃO DIRETA ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    create_jobs_matrix()

    # Teste rápido com um currículo do dataset
    resume_sample = pd.read_csv(os.path.join(OUTPUT_DIR, "match_clean.csv")).loc[44, "resume_text"]
    df, vec, mat  = load_jobs_matrix()
    r_vec         = vectorize_resume(resume_sample, vec)
    top_idx, scores = calculate_similarities(r_vec, mat)

    print("\nTop-5 vagas recomendadas:")
    print(df.iloc[top_idx][["title", "company_name"]].assign(score=scores).to_string())
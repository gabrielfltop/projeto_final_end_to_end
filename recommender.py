"""
JobMatch AI - Recommender
============================
Responsável por:

"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

OUTPUT_DIR = "artifacts"

def create_jobs_matrix():
    print("[recommender] Vetorizando vagas...")

    df         = pd.read_csv(os.path.join(OUTPUT_DIR, "postings_clean.csv"))
    vectorizer = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_jobs.pkl"))
    matrix_path = os.path.join(OUTPUT_DIR, "postings_vec_matrix.pkl")

    job_texts = df["job_text"]
    vec_matrix = vectorizer.transform(job_texts)
    joblib.dump(vec_matrix, matrix_path)

def load_jobs_matrix():
    print("[recommender] Carregando vagas...")

    df         = pd.read_csv(os.path.join(OUTPUT_DIR, "postings_clean.csv"))
    vectorizer = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_jobs.pkl"))
    matrix_path = os.path.join(OUTPUT_DIR, "postings_vec_matrix.pkl")

    vec_matrix = joblib.load(matrix_path)

    return df, vectorizer, vec_matrix

def vectorize_resume(resume, vectorizer):
    print("[recommender] Vetorizando currículo...")

    return vectorizer.transform([resume])

def calculate_similarities(resume_vec, postings_matrix, n=5):
    print("[recommender] Calculando similaridades...")

    sim = cosine_similarity(postings_matrix, resume_vec).flatten()
    top_idx = np.argsort(sim)[-n:][::-1]

    return top_idx, sim[top_idx]

def run_recommender(resume_text):
    df, vectorizer, postings_matrix = load_jobs_matrix()

    resume_vec = vectorize_resume(resume_text, vectorizer)

    top_idx, scores = calculate_similarities(
        resume_vec,
        postings_matrix,
        n=5
    )

    results = df.iloc[top_idx][["title", "description"]].copy()
    results["score"] = scores

    print("Currículo:", resume_text[:200])
    print()
    print(results)

if __name__ == "__main__":

    resume_text = pd.read_csv(os.path.join(OUTPUT_DIR, "match_clean.csv")).loc[44, 'resume_text']
    run_recommender(resume_text)
"""
JobMatch AI - Preprocessing
============================
Responsável por:
  1. Parsear o dataset Resume-JD-Match (separar JD e currículo do texto único)
  2. Limpar e normalizar textos (JD, currículo, vagas LinkedIn)
  3. Encodar labels (Fit / No Fit)
  4. Vetorizar textos com TF-IDF
  5. Preparar os datasets LinkedIn e Job Skill Set para uso nos modelos
"""

import ast
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)
MIN_SALARY_VALID   = 10_000   # filtra salários implausíveis (valores anuais em USD)
MAX_SALARY_VALID   = 500_000


# ── 1. PARSEAR RESUME-JD-MATCH ───────────────────────────────────────────────────

def parse_match_text(text: str) -> tuple[str, str]:
    """
    O dataset tem um formato fixo:
      'For the given job description <<JD>> the resume: <<RESUME>>. The result is, LABEL'

    Retorna (jd_text, resume_text). Retorna ('', '') se o parse falhar.
    """
    try:
        jd_match     = re.search(r"job description\s*<<(.+?)>>\s*the resume", text, re.IGNORECASE | re.DOTALL)
        resume_match = re.search(r"the resume:\s*<<(.+?)>>\.",                 text, re.IGNORECASE | re.DOTALL)
        jd     = jd_match.group(1).strip()     if jd_match     else ""
        resume = resume_match.group(1).strip() if resume_match else ""
        return jd, resume
    except Exception:
        return "", ""


def load_match_dataset(df_match: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe o df bruto do HuggingFace e retorna com colunas separadas:
      jd_text | resume_text | label | label_encoded
    """
    print("[preprocessing] Parseando Resume-JD-Match...")

    parsed            = df_match["text"].apply(parse_match_text)
    df_match          = df_match.copy()
    df_match["jd_text"]     = parsed.apply(lambda x: x[0])
    df_match["resume_text"] = parsed.apply(lambda x: x[1])

    # Remove linhas onde o parse falhou
    mask     = (df_match["jd_text"] != "") & (df_match["resume_text"] != "")
    df_match = df_match[mask].reset_index(drop=True)

    label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 1}
    df_match["label_encoded"] = df_match["label"].map(label_map)

    print(f"  ✅ {len(df_match):,} pares válidos | Distribuição:")
    print(df_match["label"].value_counts().to_string())
    return df_match


# ── 2. LIMPEZA DE TEXTO ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Pipeline de limpeza aplicado a todos os textos do projeto:
      - Lowercase
      - Remove cabeçalhos comuns de currículo (summary, profile...)
      - Remove URLs, e-mails e telefones
      - Remove caracteres especiais (mantém letras, números e espaços)
      - Colapsa espaços múltiplos
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"^(summary|professional\s*summary|profile)\s*", "", text)
    text = re.sub(r"http\S+|www\.\S+",       " ", text)  # URLs
    text = re.sub(r"\S+@\S+",               " ", text)  # e-mails
    text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)  # telefones
    text = re.sub(r"[^a-z0-9\s]",           " ", text)  # caracteres especiais
    text = re.sub(r"\s+",                   " ", text)  # espaços múltiplos
    return text.strip()


def clean_dataframe_texts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Aplica clean_text em múltiplas colunas de um DataFrame."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    return df


# ── 3. VETORIZAÇÃO TF-IDF ────────────────────────────────────────────────────────

def build_tfidf_vectorizer(texts: pd.Series, name: str) -> TfidfVectorizer:
    """
    Treina um vetorizador TF-IDF e salva em artifacts/{name}.pkl.
    Retorna o vetorizador já fittado.
    """
    print(f"[preprocessing] Treinando TF-IDF '{name}'...")

    vectorizer = TfidfVectorizer(
        max_features  = TFIDF_MAX_FEATURES,
        ngram_range   = TFIDF_NGRAM_RANGE,
        sublinear_tf  = True,       # log(TF) reduz peso de palavras muito frequentes
        strip_accents = "unicode",
        analyzer      = "word",
        stop_words    = "english",
        min_df        = 2,          # ignora termos que aparecem em menos de 2 docs
    )
    vectorizer.fit(texts.dropna())

    path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
    joblib.dump(vectorizer, path)
    print(f"  ✅ Salvo em {path} | Vocabulário: {len(vectorizer.vocabulary_):,} termos")
    return vectorizer


def build_combined_tfidf(df_match: pd.DataFrame) -> TfidfVectorizer:
    """
    TF-IDF para o classificador Fit/No Fit.
    Combina JD + currículo num único texto para capturar a relação entre os dois.
    """
    combined = df_match["jd_text"] + " " + df_match["resume_text"]
    return build_tfidf_vectorizer(combined, name="tfidf_classifier")


def build_jobs_tfidf(df_postings: pd.DataFrame) -> TfidfVectorizer:
    """
    TF-IDF separado para as vagas do LinkedIn — usado no recomendador e no salary model.
    """
    job_texts = df_postings["title"].fillna("") + " " + df_postings["description"].fillna("")
    return build_tfidf_vectorizer(job_texts, name="tfidf_jobs")


# ── 4. PREPARAR LINKEDIN POSTINGS ────────────────────────────────────────────────

def prepare_postings(df_postings: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e prepara o dataset de vagas LinkedIn:
      - Seleciona colunas relevantes
      - Remove vagas sem descrição
      - Cria coluna 'job_text' combinada (title + description)
      - Consolida colunas de salário em uma única coluna 'salary'
      - Filtra salários fora do intervalo plausível
    """
    print("[preprocessing] Preparando LinkedIn Job Postings...")

    cols = [
        "job_id", "title", "description", "company_name", "location",
        "min_salary", "max_salary", "med_salary", "formatted_work_type",
        "formatted_experience_level", "normalized_salary", "work_type",
    ]
    df = df_postings[[c for c in cols if c in df_postings.columns]].copy()
    df = df.dropna(subset=["description"]).reset_index(drop=True)

    df["job_text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df = clean_dataframe_texts(df, ["job_text", "title", "description"])

    # Consolida colunas de salário priorizando mediana > máximo > mínimo > normalizado
    def consolidate_salary(row):
        for col in ["med_salary", "max_salary", "min_salary", "normalized_salary"]:
            if col in row and pd.notna(row[col]) and row[col] > 0:
                return row[col]
        return np.nan

    df["salary"] = df.apply(consolidate_salary, axis=1)
    salary_mask  = df["salary"].isna() | (
        (df["salary"] >= MIN_SALARY_VALID) & (df["salary"] <= MAX_SALARY_VALID)
    )
    df = df[salary_mask].reset_index(drop=True)

    n_salary = df["salary"].notna().sum()
    print(f"  ✅ {len(df):,} vagas | {n_salary:,} com salário ({n_salary/len(df)*100:.1f}%)")
    return df


# ── 5. PREPARAR JOB SKILL SET ────────────────────────────────────────────────────

def prepare_skillset(df_skills: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa o Job Skill Set Dataset:
      - Parseia 'job_skill_set' de string para lista Python
      - Cria 'skills_list' (lista normalizada em lowercase)
      - Cria 'skills_clean' (string espaço-separada, usada no TF-IDF)
      - Cria 'description_clean' (descrição limpa da vaga)
    """
    print("[preprocessing] Preparando Job Skill Set...")

    df = df_skills.copy()

    def parse_skills(val):
        try:
            skills = ast.literal_eval(val)
            return [s.strip().lower() for s in skills if isinstance(s, str)]
        except Exception:
            return []

    df["skills_list"]       = df["job_skill_set"].apply(parse_skills)
    df["skills_clean"]      = df["skills_list"].apply(lambda lst: " ".join(lst))
    df["description_clean"] = df["job_description"].apply(clean_text)

    print(f"  ✅ {len(df):,} cargos com skill sets")
    return df


# ── 6. PIPELINE COMPLETA ─────────────────────────────────────────────────────────

def run_preprocessing(df_match, df_postings, df_skills):
    """
    Executa toda a pipeline de preprocessing e salva os artefatos em artifacts/.

    Retorna
    -------
    df_match_clean    : dataset de classificação (Fit/No Fit), pronto para o classifier
    df_postings_clean : vagas LinkedIn, prontas para o recommender e salary model
    df_skills_clean   : skill sets por cargo, usados pelo skills_analyzer
    vec_classifier    : TF-IDF fittado para o classifier
    vec_jobs          : TF-IDF fittado para o recommender e salary model
    """
    print("\n" + "=" * 50)
    print("⚙️  Iniciando Preprocessing...")
    print("=" * 50)

    df_match_clean    = load_match_dataset(df_match)
    df_match_clean    = clean_dataframe_texts(df_match_clean, ["jd_text", "resume_text"])
    df_postings_clean = prepare_postings(df_postings)
    df_skills_clean   = prepare_skillset(df_skills)

    vec_classifier = build_combined_tfidf(df_match_clean)
    vec_jobs       = build_jobs_tfidf(df_postings_clean)

    df_match_clean.to_csv(   os.path.join(OUTPUT_DIR, "match_clean.csv"),    index=False)
    df_postings_clean.to_csv(os.path.join(OUTPUT_DIR, "postings_clean.csv"), index=False)
    df_skills_clean.to_csv(  os.path.join(OUTPUT_DIR, "skills_clean.csv"),   index=False)

    print("\n✅ Preprocessing concluído!")
    print(f"  Arquivos salvos em /{OUTPUT_DIR}/")
    print(f"    match_clean.csv    → {len(df_match_clean):,} linhas")
    print(f"    postings_clean.csv → {len(df_postings_clean):,} linhas")
    print(f"    skills_clean.csv   → {len(df_skills_clean):,} linhas")

    return df_match_clean, df_postings_clean, df_skills_clean, vec_classifier, vec_jobs


# ── EXECUÇÃO DIRETA ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import kagglehub
    from datasets import load_dataset

    print("Carregando datasets brutos...")
    path_postings = kagglehub.dataset_download("arshkon/linkedin-job-postings")
    path_skillset = kagglehub.dataset_download("batuhanmutlu/job-skill-set")
    dataset_match = load_dataset("facehuggerapoorv/resume-jd-match")

    df_match    = pd.DataFrame(dataset_match["train"])
    df_postings = pd.read_csv(f"{path_postings}/postings.csv")
    df_skills   = pd.read_csv(f"{path_skillset}/all_job_post.csv")

    run_preprocessing(df_match, df_postings, df_skills)
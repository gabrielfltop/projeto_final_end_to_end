"""
JobMatch AI - Skills Analyzer
==============================
Responsável por extrair e comparar skills de vagas com o perfil do candidato.

Estratégias de extração (usadas em cascata por _resolve_skills no app.py):
  1. Lookup por job_id     → direto do skills_clean.csv (mais preciso)
  2. Busca por título      → encontra vagas com título similar no dataset
  3. Extração heurística   → regex sobre a descrição da vaga (~50 termos)
"""

import ast
import re


# Padrões técnicos para extração heurística da descrição da vaga.
# Cobre linguagens, frameworks, cloud, ferramentas e soft skills comuns.
_TECH_PATTERNS = re.compile(
    r'\b('
    r'python|java|javascript|typescript|sql|nosql|scala|rust|c\+\+|c#|'
    r'machine learning|deep learning|nlp|computer vision|llm|'
    r'tensorflow|pytorch|scikit.learn|keras|xgboost|'
    r'aws|azure|gcp|docker|kubernetes|terraform|spark|hadoop|airflow|kafka|'
    r'pandas|numpy|matplotlib|seaborn|plotly|'
    r'react|angular|vue|node\.?js|django|flask|fastapi|spring|'
    r'postgresql|mysql|mongodb|redis|elasticsearch|snowflake|bigquery|'
    r'git|ci/cd|devops|mlops|agile|scrum|'
    r'excel|power bi|tableau|looker|'
    r'communication|leadership|teamwork|problem.solving|analytical'
    r')',
    re.IGNORECASE,
)


def _search_by_title(df_skills, title: str) -> set:
    """
    Busca no df_skills linhas cujo job_title compartilha palavras
    significativas (len > 3) com o título da vaga. Usa a coluna
    'skills_list' (lista Python serializada) para montar o set de skills.

    Retorna set vazio se nenhuma correspondência for encontrada.
    """
    if not title or not isinstance(title, str):
        return set()

    # Filtra palavras curtas (artigos, preposições) para reduzir falsos positivos
    keywords = [w.lower() for w in title.split() if len(w) > 3]
    if not keywords:
        return set()

    mask = df_skills["job_title"].str.lower().apply(
        lambda x: any(w in str(x) for w in keywords)
    )

    matched = set()
    for _, row in df_skills[mask].head(3).iterrows():
        # Usa 'skills_list' (ex: "['python', 'sql']"), não 'skills_clean' (texto plano)
        raw = row.get("skills_list", "")
        if isinstance(raw, str) and raw.startswith("["):
            try:
                matched.update(ast.literal_eval(raw))
            except Exception:
                pass

    return matched


def _extract_from_description(description: str) -> set:
    """
    Extrai termos técnicos da descrição da vaga via regex.
    Último recurso quando não há correspondência por ID ou título.

    Retorna set de strings em lowercase.
    """
    if not description or not isinstance(description, str):
        return set()

    found = _TECH_PATTERNS.findall(description)
    return {s.lower().strip() for s in found}


def split_skills(job_skills, resume_text: str):
    """
    Divide as skills de uma vaga em duas listas com base no currículo:
      - present : skills mencionadas no currículo (candidato já tem)
      - missing : skills ausentes no currículo (gap de desenvolvimento)

    Parâmetros
    ----------
    job_skills  : set ou list de strings com as skills da vaga
    resume_text : texto do currículo (limpo ou não)

    Retorna
    -------
    (present, missing) — ambas listas ordenadas de strings
    """
    resume_lower = resume_text.lower()
    present = sorted(s for s in job_skills if s.lower() in resume_lower)
    missing = sorted(s for s in job_skills if s.lower() not in resume_lower)
    return present, missing
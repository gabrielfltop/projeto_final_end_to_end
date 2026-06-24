"""
JobMatch AI - Interface Streamlit
===================================
Frontend principal do projeto. Integra:
  - classifier.py      → predição de Fit / No Fit (MLP + TF-IDF)
  - recommender.py     → Top-5 vagas por similaridade cosseno
  - salary_model.py    → estimativa de faixa salarial (Ridge)
  - skills_analyzer.py → skills presentes/ausentes por vaga (3 estratégias)
"""

import ast
import os

import joblib
import pandas as pd
import streamlit as st

from preprocessing   import clean_text
from classifier      import load_model as load_classifier_model
from recommender     import load_jobs_matrix, vectorize_resume, calculate_similarities
from salary_model    import load_model as load_salary_model, predict_salary
from skills_analyzer import split_skills, _search_by_title, _extract_from_description

# ── Constantes ──────────────────────────────────────────────────────────────────
OUTPUT_DIR  = "artifacts"
LABEL_NAMES = {0: "🔴 No Fit", 1: "🟡 Potential Fit", 2: "🟢 Good Fit"}
LABEL_COLOR = {0: "#ef4444",   1: "#f59e0b",           2: "#22c55e"}

# ── Configuração da página ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d0d0d; color: #e5e5e5; }

  .hero-title {
    font-size: 3rem; font-weight: 800;
    color: #39ff14; letter-spacing: -1px; margin-bottom: 0;
  }
  .hero-sub { font-size: 1.1rem; color: #9ca3af; margin-top: 4px; margin-bottom: 2rem; }

  .result-card {
    background: #1a1a1a; border: 1px solid #2d2d2d;
    border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
  }
  .result-card h4     { margin: 0 0 4px 0; color: #f3f4f6; font-size: 1rem; }
  .result-card .company { color: #9ca3af; font-size: 0.85rem; margin-bottom: 8px; }
  .result-card .score-bar-wrap {
    background: #2d2d2d; border-radius: 99px; height: 6px; margin-bottom: 8px;
  }
  .result-card .score-bar { background: #39ff14; height: 6px; border-radius: 99px; }
  .result-card .meta  { font-size: 0.8rem; color: #6b7280; }

  .fit-badge {
    display: inline-block; padding: 4px 14px;
    border-radius: 99px; font-weight: 700; font-size: 0.95rem;
  }

  .skills-section { margin-top: 12px; border-top: 1px solid #2d2d2d; padding-top: 10px; }
  .skills-label   { font-size: 0.78rem; font-weight: 700; margin-bottom: 4px; }
  .skill-tag {
    display: inline-block; padding: 2px 10px; border-radius: 99px;
    font-size: 0.76rem; font-weight: 600; margin: 2px 3px 2px 0;
  }
  .skill-present  { background: #22c55e1a; color: #22c55e; border: 1px solid #22c55e44; }
  .skill-missing  { background: #ef44441a; color: #ef4444; border: 1px solid #ef444444; }
  .coverage-bar-wrap {
    background: #2d2d2d; border-radius: 99px; height: 4px;
    margin: 6px 0 10px 0; width: 100%;
  }
  .coverage-bar   { background: #39ff14; height: 4px; border-radius: 99px; }

  .stTextArea textarea {
    background: #1a1a1a !important; color: #e5e5e5 !important;
    border: 1px solid #2d2d2d !important; border-radius: 10px !important;
    font-size: 0.9rem !important;
  }
  .stButton > button {
    background: #39ff14 !important; color: #0d0d0d !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 8px !important; padding: 0.6rem 2rem !important;
    font-size: 1rem !important; width: 100%;
  }
  .stButton > button:hover { background: #2ecc11 !important; }

  hr { border-color: #2d2d2d; }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cache de modelos ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Carrega e cacheia todos os modelos/vetorizadores necessários para o app."""
    classifier, _                   = load_classifier_model()
    clf_vectorizer                  = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_classifier.pkl"))
    df_jobs, rec_vec, jobs_mat      = load_jobs_matrix()
    salary_model, _, salary_encoder = load_salary_model()
    salary_vec                      = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_jobs.pkl"))
    return classifier, clf_vectorizer, df_jobs, rec_vec, jobs_mat, salary_model, salary_encoder, salary_vec


@st.cache_resource(show_spinner=False)
def load_skills_data():
    """
    Carrega o skills_clean.csv e monta o lookup {job_id: set(skills)}.
    Usa a coluna 'skills_list' (formato de lista Python) para montar os sets.
    Cacheado separadamente dos modelos para evitar recarregamentos desnecessários.
    """
    skills_path = os.path.join(OUTPUT_DIR, "skills_clean.csv")
    if not os.path.exists(skills_path):
        return {}, None

    df_skills = pd.read_csv(skills_path)
    lookup = {}
    for _, row in df_skills.iterrows():
        # 'skills_list' contém listas Python serializadas: "['python', 'sql', ...]"
        raw = row.get("skills_list", "[]")
        try:
            skills_set = set(ast.literal_eval(raw)) if isinstance(raw, str) else set()
        except Exception:
            skills_set = set()
        lookup[row["job_id"]] = skills_set

    return lookup, df_skills


# ── Helpers de renderização ──────────────────────────────────────────────────────

def _tags_html(skills: list, css_class: str) -> str:
    """Gera HTML de skill tags coloridas. Limita a 12 para não poluir o card."""
    if not skills:
        return '<span style="color:#6b7280;font-size:0.76rem;">nenhuma</span>'
    return " ".join(
        f'<span class="skill-tag {css_class}">{s}</span>'
        for s in skills[:12]
    )


def _build_skills_block(job_skills: set, resume_clean: str, source: str = "") -> str:
    """
    Renderiza o bloco de skills de uma vaga com:
      - Barra de cobertura (% de skills que o candidato já tem)
      - Tags verdes para skills presentes no currículo
      - Tags vermelhas para skills ausentes (gap)
      - Badge discreto indicando a fonte das skills
    """
    if not job_skills:
        return ""

    present, missing = split_skills(job_skills, resume_clean)
    total    = len(job_skills)
    coverage = int(len(present) / total * 100) if total else 0

    source_badge = (
        f'<span style="color:#6b7280;font-size:0.72rem;margin-left:8px;">fonte: {source}</span>'
        if source else ""
    )

    return f"""
    <div class="skills-section">
      <div style="display:flex; align-items:center; margin-bottom:2px;">
        <span style="font-size:0.85rem; font-weight:700; color:#e5e5e5;">🧠 Skills</span>
        <span style="color:#39ff14; font-size:0.8rem; font-weight:700; margin-left:8px;">{coverage}% match</span>
        {source_badge}
      </div>
      <div class="coverage-bar-wrap">
        <div class="coverage-bar" style="width:{coverage}%;"></div>
      </div>
      <div style="margin-bottom:6px;">
        <div class="skills-label" style="color:#22c55e;">✅ Você já tem ({len(present)})</div>
        {_tags_html(present, "skill-present")}
      </div>
      <div>
        <div class="skills-label" style="color:#ef4444;">🚀 Gap — falta desenvolver ({len(missing)})</div>
        {_tags_html(missing, "skill-missing")}
      </div>
    </div>
    """


def _resolve_skills(job_id, title, description, skills_lookup, df_skills):
    """
    Obtém as skills de uma vaga usando cascata de 3 estratégias:
      1. Lookup direto por job_id  → mais preciso, usa dados do dataset
      2. Busca por título similar  → cobre vagas fora do overlap de IDs
      3. Extração da descrição     → fallback heurístico via regex

    Retorna (set_de_skills, fonte) onde fonte é uma string descritiva.
    """
    # Estratégia 1: lookup direto por job_id (IDs em comum entre os datasets)
    if job_id in skills_lookup and skills_lookup[job_id]:
        return skills_lookup[job_id], "dataset"

    # Estratégia 2: busca por palavras do título no skills_clean.csv
    if df_skills is not None:
        by_title = _search_by_title(df_skills, title)
        if by_title:
            return by_title, "título similar"

    # Estratégia 3: extração heurística de ~50 termos técnicos da descrição
    by_desc = _extract_from_description(description)
    if by_desc:
        return by_desc, "descrição"

    return set(), ""


# ── Layout ───────────────────────────────────────────────────────────────────────

st.markdown('<p class="hero-title">JobMatch AI 🎯</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Cole seu currículo abaixo e descubra as vagas mais aderentes ao seu perfil.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

col_input, _ = st.columns([2, 1])
with col_input:
    resume_input = st.text_area(
        "**Seu currículo / perfil profissional**",
        height=260,
        placeholder=(
            "Ex: Engenheiro de Software com 5 anos de experiência em Python, "
            "machine learning, APIs REST, AWS e liderança de times ágeis..."
        ),
    )
    job_input = st.text_area(
        "**A vaga que você quer descobrir fit** (opcional)",
        height=200,
    )
    run_btn = st.button("Analisar perfil →")


# ── Lógica principal ─────────────────────────────────────────────────────────────

if run_btn:
    if not resume_input.strip():
        st.warning("Por favor, descreva seu perfil antes de analisar.")
        st.stop()

    with st.spinner("Carregando modelos..."):
        try:
            (classifier, clf_vectorizer,
             df_jobs, rec_vec, jobs_mat,
             salary_model, salary_encoder,
             salary_vec) = load_all_models()
        except Exception as e:
            st.error(f"Erro ao carregar artefatos: {e}")
            st.info("Certifique-se de ter rodado o pipeline completo antes de abrir o app.")
            st.stop()

    resume_clean             = clean_text(resume_input)
    skills_lookup, df_skills = load_skills_data()

    # ── 1. Análise da vaga inserida pelo usuário (opcional) ───────────────────
    if job_input.strip():
        st.markdown("---")
        st.markdown("## 🔎 Análise da vaga escolhida")

        job_clean = clean_text(job_input)
        combined  = job_clean + " " + resume_clean
        X_input   = clf_vectorizer.transform([combined])
        label     = int(classifier.predict(X_input)[0])

        badge_color = LABEL_COLOR[label]
        badge_text  = LABEL_NAMES[label]

        st.markdown(f"""
        <div class="result-card">
            <h4>📌 Classificação da vaga</h4>
            <span class="fit-badge"
                style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}55;">
                {badge_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── 2. Top-5 Recomendações ────────────────────────────────────────────────
    with st.spinner("Calculando recomendações..."):
        resume_vec      = vectorize_resume(resume_clean, rec_vec)
        top_idx, scores = calculate_similarities(resume_vec, jobs_mat, n=5)
        top_jobs        = df_jobs.iloc[top_idx].copy().reset_index(drop=True)
        top_jobs["score"] = scores

    st.markdown("## 🏆 Top-5 Vagas Recomendadas")

    for i, row in top_jobs.iterrows():
        job_id    = row.get("job_id",        None)
        title     = row.get("title",         "Vaga sem título")
        company   = row.get("company_name",  "—")
        jd_text   = str(row.get("job_text",  row.get("description", "")))
        score_pct = int(row["score"] * 100)

        # Classificação Fit (combina JD + currículo, mesmo esquema do treino)
        X_clf  = clf_vectorizer.transform([resume_clean + " " + clean_text(jd_text)])
        label  = int(classifier.predict(X_clf)[0])

        badge_color = LABEL_COLOR[label]
        badge_text  = LABEL_NAMES[label]

        # Skills com cascata de 3 estratégias
        job_skills, source = _resolve_skills(job_id, title, jd_text, skills_lookup, df_skills)
        skills_block       = _build_skills_block(job_skills, resume_clean, source)

        # Salário: tenta modelo preditivo, cai para valor do dataset se falhar
        salary_html = ""
        if salary_model is not None and salary_encoder is not None:
            try:
                salary_est = predict_salary(row, salary_model, salary_vec, salary_encoder)
                if salary_est > 0:
                    salary_html = f'<span class="meta">💰 Salário estimado: <b>${salary_est:,.0f}/ano</b></span>'
            except Exception:
                pass
        if not salary_html:
            raw_salary = row.get("salary", 0)
            if pd.notna(raw_salary) and raw_salary > 0:
                salary_html = f'<span class="meta">💰 Salário: <b>${raw_salary:,.0f}/ano</b></span>'

        salary_sep = "&nbsp;·&nbsp;" + salary_html if salary_html else ""

        # Nota: o </div> de fechamento fica na mesma linha que {skills_block}
        # para evitar que o Streamlit renderize o tag como texto literal.
        st.markdown(f"""
        <div class="result-card">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
              <h4>#{i+1} — {title}</h4>
              <span class="company">🏢 {company}</span>
            </div>
            <span class="fit-badge"
              style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}55;">
              {badge_text}
            </span>
          </div>
          <div class="score-bar-wrap">
            <div class="score-bar" style="width:{score_pct}%;"></div>
          </div>
          <span class="meta">Similaridade: <b>{score_pct}%</b></span>
          {salary_sep}
          {skills_block}</div>
        """, unsafe_allow_html=True)

        with st.expander(f"Ver descrição — {title}"):
            desc = str(row.get("description", ""))
            st.write(desc[:1500] + ("..." if len(desc) > 1500 else ""))

    st.markdown("---")
    st.caption("JobMatch AI · Projeto Final E2E Machine Learning")
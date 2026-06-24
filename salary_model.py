"""
JobMatch AI - Salary Model
============================
Estima o salário anual de uma vaga combinando:
  - TF-IDF da descrição/título da vaga (features textuais)
  - OneHotEncoding de work_type, experience_level e location (features categóricas)

Modelo: Ridge Regression (robusto a multicolinearidade, eficiente com features esparsas).

Uso pelo app: apenas load_model() e predict_salary() são chamadas em runtime.
"""

import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

OUTPUT_DIR = "artifacts"
MODEL_PATH = os.path.join(OUTPUT_DIR, "salary_model.pkl")

# Colunas categóricas usadas como features adicionais ao TF-IDF
CATEG_COLS = ["formatted_work_type", "formatted_experience_level", "location"]


# ── DADOS ────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Carrega apenas as vagas com salário informado."""
    print("[salary_model] Carregando dados...")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "postings_clean.csv"))
    df = df.dropna(subset=["salary"])
    return df[["job_text"] + CATEG_COLS + ["salary"]].copy()


def prepare_features(df: pd.DataFrame, vectorizer, encoder):
    """
    Combina TF-IDF do texto da vaga com features categóricas codificadas.
    Retorna matriz esparsa (scipy) pronta para o modelo.
    """
    text_features   = vectorizer.transform(df["job_text"])
    categ_features  = encoder.transform(df[CATEG_COLS].fillna("unknown"))
    return hstack([text_features, categ_features])


# ── TREINO ───────────────────────────────────────────────────────────────────────

def train_model():
    """
    Treina o Ridge com os dados disponíveis e retorna
    (model, X_test, y_test, encoder, rmse).
    """
    vectorizer = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_jobs.pkl"))
    df         = load_data()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    encoder.fit(df[CATEG_COLS].fillna("unknown"))

    X = prepare_features(df, vectorizer, encoder)
    y = df["salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Ridge()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    print(f"  ✅ RMSE: ${rmse:,.0f} | R²: {r2:.4f}")

    return model, X_test, y_test, encoder, rmse


# ── CARREGAR MODELO ──────────────────────────────────────────────────────────────

def load_model():
    """
    Carrega o modelo salvo em salary_model.pkl.
    Retorna (model, rmse, encoder) ou (None, inf, None) se não existir.
    Chamada pelo app em runtime — não re-treina.
    """
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data["model"], data["rmse"], data["encoder"]
    return None, float("inf"), None


# ── PIPELINE COMPLETA ────────────────────────────────────────────────────────────

def run_salary_model(force_train: bool = False, do_eval: bool = False):
    """
    Treina ou carrega o modelo de salário, salvando apenas se RMSE melhorar.
    Retorna o melhor modelo disponível.
    """
    print("\n" + "=" * 50)
    print("💰 Iniciando Salary Model...")
    print("=" * 50)

    best_model, best_rmse, encoder = load_model()

    if not force_train and best_model is not None:
        print("[salary_model] Modelo existente carregado.")
    else:
        print("[salary_model] Treinando novo modelo...")
        candidate_model, _, _, encoder, candidate_rmse = train_model()

        if candidate_rmse < best_rmse:
            print("[salary_model] Novo modelo é melhor → salvando")
            best_model = candidate_model
            best_rmse  = candidate_rmse
            joblib.dump(
                {"model": best_model, "rmse": best_rmse, "encoder": encoder},
                MODEL_PATH,
            )
        else:
            print("[salary_model] Modelo atual é melhor → descartando novo")

    if do_eval:
        print(f"  RMSE: ${best_rmse:,.0f}/ano")

    return best_model


# ── INFERÊNCIA ───────────────────────────────────────────────────────────────────

def predict_salary(row, model, vectorizer, encoder) -> float:
    """
    Estima o salário anual para uma vaga (Series ou dict).
    Retorna 0.0 se a predição for negativa.
    """
    df = pd.DataFrame([{
        "job_text":                   row.get("job_text", ""),
        "formatted_work_type":        row.get("formatted_work_type",        "unknown"),
        "formatted_experience_level": row.get("formatted_experience_level", "unknown"),
        "location":                   row.get("location",                   "unknown"),
    }])
    X = prepare_features(df, vectorizer, encoder)
    return float(max(0.0, model.predict(X)[0]))


# ── EXECUÇÃO DIRETA ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_salary_model(force_train=True, do_eval=True)
"""
JobMatch AI - Classifier
============================
Treina e avalia um MLP para classificar o fit candidato/vaga em 2 classes:
  0 → No Fit | 1 → Fit

A entrada do modelo é o TF-IDF da concatenação (JD + currículo), capturando
a relação semântica entre os dois textos.

Uso pelo app: apenas load_model() é chamada em runtime.
As funções de treino (run_classifier, run_random_search) são executadas
uma única vez via __main__ para gerar o artefato mlp_best.pkl.
"""

import os

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

OUTPUT_DIR  = "artifacts"
LABEL_NAMES = {0: "No Fit", 1: "Fit"}
MODEL_PATH  = os.path.join(OUTPUT_DIR, "mlp_best.pkl")


# ── 1. DADOS ─────────────────────────────────────────────────────────────────────

def load_data():
    """
    Carrega match_clean.csv e o vetorizador TF-IDF já treinado.
    Retorna o split treino/teste pronto para uso.
    """
    print("[classifier] Carregando dados...")

    df         = pd.read_csv(os.path.join(OUTPUT_DIR, "match_clean.csv"))
    vectorizer = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_classifier.pkl"))

    # Mesmo esquema usado em runtime no app: JD + currículo concatenados
    X_raw = df["jd_text"].fillna("") + " " + df["resume_text"].fillna("")
    X     = vectorizer.transform(X_raw)
    y     = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"  ✅ {X.shape[0]:,} amostras | {X.shape[1]:,} features")
    print(f"  Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


# ── 2. OTIMIZAÇÃO ────────────────────────────────────────────────────────────────

def run_random_search(X_train, y_train, n_iter: int = 10, cv: int = 3):
    """
    Otimiza hiperparâmetros do MLP com RandomizedSearchCV (métrica: F1).
    Retorna o objeto RandomizedSearchCV fittado.
    """
    print(f"\n[classifier] RandomizedSearchCV (n_iter={n_iter}, cv={cv})...")

    param_dist = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "activation":         ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01, 0.0001],
        "alpha":              [0.0001, 0.001, 0.01],
    }

    search = RandomizedSearchCV(
        estimator          = MLPClassifier(max_iter=1000),
        param_distributions= param_dist,
        n_iter             = n_iter,
        cv                 = cv,
        n_jobs             = -1,
        verbose            = 2,
        random_state       = 42,
        scoring            = "f1",
    )
    search.fit(X_train, y_train)

    print(f"\n  ✅ Melhores parâmetros: {search.best_params_}")
    print(f"     Melhor score (CV):  {search.best_score_:.4f}")
    return search


# ── 3. CARREGAR MODELO ───────────────────────────────────────────────────────────

def load_model():
    """
    Carrega o modelo salvo em mlp_best.pkl.
    Retorna (model, score) ou (None, -1) se o artefato não existir.
    Chamada pelo app em runtime — não re-treina.
    """
    if os.path.exists(MODEL_PATH):
        print("[classifier] Carregando modelo...")
        data = joblib.load(MODEL_PATH)
        return data["model"], data["score"]
    return None, -1


# ── 4. AVALIAÇÃO ─────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, label: str = "Modelo"):
    """Imprime o classification report completo (precision, recall, F1)."""
    print(f"\n[classifier] Avaliação — {label}:")
    y_pred = model.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
    ))

# ── 5. PIPELINE COMPLETA ─────────────────────────────────────────────────────────

def run_classifier(n_iter: int = 10, cv: int = 3, force_train: bool = False, do_eval: bool = False):
    """
    Pipeline completa de treino:
      1. Carrega dados
      2. Carrega modelo existente ou treina novo via RandomizedSearchCV
      3. Salva apenas se o novo modelo superar o anterior (F1)
      4. Avalia no conjunto de teste (se do_eval=True)

    Retorna o melhor modelo disponível.
    """
    print("\n" + "=" * 50)
    print("🤖 Iniciando Classifier...")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()
    best_model, best_score           = load_model()

    if not force_train and best_model is not None:
        print("[classifier] Modelo existente carregado.")
    else:
        print("[classifier] Treinando novo modelo...")
        search          = run_random_search(X_train, y_train, n_iter=n_iter, cv=cv)
        candidate_model = search.best_estimator_
        candidate_score = search.best_score_

        if candidate_score > best_score:
            print("[classifier] Novo modelo é melhor → salvando")
            best_model = candidate_model
            best_score = candidate_score
            joblib.dump({"model": best_model, "score": best_score}, MODEL_PATH)
        else:
            print("[classifier] Modelo atual é melhor → descartando novo")

    if do_eval:
        evaluate(best_model, X_test, y_test, label="MLP otimizado")

    return best_model

# ── EXECUÇÃO DIRETA ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_classifier(n_iter=10, cv=3, force_train=False, do_eval=True)
"""
JobMatch AI - Classifier
============================
Responsável por:
  1. Carregar e vetorizar os dados do Resume-JD-Match
  2. Rodar baseline com múltiplos modelos (LR, RF, SVM, MLP)
  3. Otimizar o melhor modelo com RandomizedSearchCV
  4. Avaliar com precision, recall e F1 macro (3 classes)
  5. Salvar o modelo final em artifacts/
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import os

OUTPUT_DIR = "artifacts"
LABEL_NAMES = {0: "No Fit", 1: "Potential Fit", 2: "Good Fit"}
MODEL_PATH = os.path.join(OUTPUT_DIR, "mlp_best.pkl")

# 1. CARREGAR E PREPARAR DADOS

def load_data():
    """
    Carrega o dataset processado e o vetorizador TF-IDF.
    Retorna X (matriz esparsa), y (labels) e o split treino/teste.
    """
    print("[classifier] Carregando dados...")

    df         = pd.read_csv(os.path.join(OUTPUT_DIR, "match_clean.csv"))
    vectorizer = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_classifier.pkl"))

    X_raw = (df["jd_text"].fillna("") + " " + df["resume_text"].fillna(""))
    X     = vectorizer.transform(X_raw)
    y     = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"  ✅ {X.shape[0]:,} amostras | {X.shape[1]:,} features")
    print(f"  Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test

# 2. RANDOMIZEDSEARCHCV

def run_random_search(X_train, y_train, n_iter=10, cv=3):
    """
    Otimiza o MLP com RandomizedSearchCV.
    Retorna o melhor estimador encontrado.
    """
    print(f"\n[classifier] RandomizedSearchCV (n_iter={n_iter}, cv={cv})...")

    param_dist = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "activation":         ["relu", "tanh"],
        "solver":             ["adam", "sgd"],
        "learning_rate_init": [0.001, 0.01, 0.0001],
        "alpha":              [0.0001, 0.001, 0.01],
    }

    random_search = RandomizedSearchCV(
        estimator=MLPClassifier(max_iter=1000),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        scoring="f1_macro"
    )
    random_search.fit(X_train, y_train)

    print(f"\n  ✅ Melhores parâmetros: {random_search.best_params_}")
    print(f"    ✅ Melhor score (CV):   {random_search.best_score_:.4f}")

    return random_search

def load_model():
    if os.path.exists(MODEL_PATH):
        print("[classifier] Carregando modelo...")

        data = joblib.load(MODEL_PATH)
        return data["model"], data["score"]
    return None, -1

# 3. AVALIAÇÃO FINAL

def evaluate(model, X_test, y_test, label: str = "Modelo"):
    """Imprime o classification report completo do modelo."""
    print(f"\n[classifier] Avaliação final — {label}:")

    y_pred = model.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    ))

# 4. PIPELINE COMPLETA

def run_classifier(n_iter=10, cv=3, force_train=False, do_eval=False):
    """
    Executa o pipeline completo:
      1. Carrega dados
      2. Carrega ou cria o melhor modelo
      3. Avalia o modelo final

    Retorna o melhor modelo treinado.
    """
    print("\n" + "=" * 50)
    print("🤖 Iniciando Classifier...")
    print("=" * 50)

    # 1. Dados
    X_train, X_test, y_train, y_test = load_data()

    # 2. Tenta carregar modelo existente
    best_model, best_score = load_model()

    if not force_train and best_model is not None:
        print("[classifier] Modelo existente carregado.")
    else:
        print("[classifier] Treinando novo modelo...")
        search = run_random_search(X_train, y_train, n_iter=n_iter, cv=cv)
        
        candidate_model = search.best_estimator_
        candidate_score = search.best_score_

        if candidate_score > best_score:
            print("[classifier] Novo modelo é melhor → salvando")
            best_model = candidate_model
            best_score = candidate_score
            joblib.dump({"model": best_model, "score": best_score}, MODEL_PATH)
        else:
            print("[classifier] Modelo atual é melhor → descartando novo")

    # 3. Avaliação
    if do_eval:
        evaluate(best_model, X_test, y_test, label="MLP otimizado")

    return best_model

# EXECUÇÃO DIRETA

if __name__ == "__main__":
    run_classifier(n_iter=15, cv=5, force_train=True, do_eval=True)
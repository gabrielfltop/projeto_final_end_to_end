# JobMatch AI 🎯

Projeto Final da disciplina **Desenvolvendo um Projeto End-to-End de Machine Learning**.

Sistema de recomendação inteligente de vagas que analisa a compatibilidade entre um currículo e ofertas de emprego do LinkedIn, classificando o fit, recomendando as Top-5 vagas mais aderentes e estimando faixa salarial.

---

## Funcionalidades

- **Classificação Fit / No Fit** — MLP treinado sobre pares currículo/vaga rotulados
- **Recomendação Top-5** — similaridade cosseno entre o perfil do candidato e 124k vagas reais do LinkedIn
- **Análise de vaga específica** — cole uma vaga e veja o fit + similaridade com seu perfil
- **Estimativa salarial** — Ridge Regression com features textuais e categóricas
- **Análise de skills** — quais skills você já tem e quais faltam para as vagas recomendadas

---

## Datasets

| Dataset | Fonte | Descrição |
|---|---|---|
| LinkedIn Job Postings 2023-2024 | Kaggle | 124k vagas reais |
| Resume-JD-Match | HuggingFace | Pares currículo/vaga rotulados |
| Job Skill Set | Kaggle | Skills exigidas por cargo |

---

## Estrutura do Projeto

```
├── main.py             # Download e carregamento dos datasets
├── preprocessing.py    # Limpeza, TF-IDF e preparação dos dados
├── classifier.py       # Modelo de classificação Fit/No Fit (MLP)
├── recommender.py      # Recomendação por similaridade cosseno
├── salary_model.py     # Estimativa salarial (Ridge Regression)
├── skills_analyzer.py  # Análise de skills presentes e ausentes
├── app.py              # Interface Streamlit
└── artifacts/          # Modelos e dados processados (não versionado)
```

---

## Como Rodar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Rodar o pipeline completo (primeira vez)

```bash
python main.py          # download dos datasets
python preprocessing.py # limpeza e vetorização
python classifier.py    # treina o classificador
python recommender.py   # gera a matriz de vagas
python salary_model.py  # treina o modelo de salário
```

### 3. Iniciar o app

```bash
streamlit run app.py
```

---

## Modelos e Métricas

### Classifier — MLP (Multi-Layer Perceptron)

- **Problema:** classificação binária (Fit / No Fit)
- **Features:** TF-IDF (5000 features, unigramas + bigramas) da concatenação JD + currículo
- **Otimização:** RandomizedSearchCV (F1 macro)
- **Métricas reportadas:**
  - **Precision** — dos classificados como Fit, quantos realmente são
  - **Recall** — dos que são Fit, quantos o modelo identificou
  - **F1 Score (macro)** — média harmônica entre precision e recall, balanceada entre classes
  - **Accuracy** — acurácia geral

### Salary Model — Ridge Regression

- **Problema:** regressão (estimativa de salário anual em USD)
- **Features:** TF-IDF do texto da vaga + OneHotEncoding de `work_type`, `experience_level` e `location`
- **Métricas reportadas:**
  - **RMSE** (Root Mean Squared Error) — erro médio em dólares; quanto menor, melhor
  - **R²** — proporção da variância do salário explicada pelo modelo (0 a 1; quanto maior, melhor)

---

## Técnicas de ML Utilizadas

- **NLP com TF-IDF** — vetorização de textos de currículos e vagas
- **MLP Classifier** — rede neural para classificação Fit/No Fit
- **Similaridade Cosseno** — ranking e recomendação de vagas
- **Ridge Regression** — estimativa de faixa salarial
- **RandomizedSearchCV** — otimização de hiperparâmetros do classifier
- **OneHotEncoding** — encoding de features categóricas para o modelo de salário
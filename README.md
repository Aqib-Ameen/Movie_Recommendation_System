# Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)
![pytest](https://img.shields.io/badge/Tests-52%20passed-brightgreen?style=flat-square&logo=pytest)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

A full-stack, modular movie recommendation system built in Python supporting **Collaborative Filtering**, **Content-Based Filtering**, and a **Hybrid Model**, all served via a **FastAPI** REST API.

---

## Features

| Feature | Details |
|---|---|
| Collaborative Filtering | User-Based CF, Item-Based CF, SVD Matrix Factorization |
| Content-Based Filtering | TF-IDF on genres/tags + Cosine Similarity |
| Hybrid Model | Weighted blend (α × CF + (1-α) × CB) with cold-start handling |
| REST API | FastAPI with auto-generated Swagger docs at `/docs` |
| Evaluation | RMSE, MAE, Precision@K, Recall@K, NDCG@K, Coverage |
| Testing | pytest + httpx for unit and integration tests |

---

## Project Structure

```
Recommendation system/
│
├── data/
│   ├── raw/            # Original CSVs (ratings.csv, movies.csv)
│   └── processed/      # Cleaned, encoded data (auto-generated)
│
├── models/saved/       # Serialized model files (auto-generated)
│
├── notebooks/
│   └── 01_eda.ipynb    # Exploratory Data Analysis
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Load & validate datasets
│   │   └── preprocessor.py     # Cleaning, encoding, train/test split
│   ├── models/
│   │   ├── collaborative.py    # User-Based, Item-Based CF + SVD
│   │   ├── content_based.py    # TF-IDF + Cosine Similarity
│   │   └── hybrid.py           # Weighted hybrid + cold-start logic
│   ├── evaluation/
│   │   └── metrics.py          # All evaluation metrics
│   └── api/
│       ├── main.py             # FastAPI app entry point
│       ├── routes.py           # API endpoints
│       └── schemas.py          # Pydantic models
│
├── tests/                      # pytest test suite
├── config.yaml                 # All hyperparameters & paths
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your dataset

Download [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) and place the files in `data/raw/`:
- `ratings.csv` — columns: `userId, movieId, rating, timestamp`
- `movies.csv` — columns: `movieId, title, genres`

### 3. Train models

```python
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.collaborative import CollaborativeFilter
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender
import yaml

config = yaml.safe_load(open("config.yaml"))
loader = DataLoader(config)
ratings, movies = loader.load()

prep = Preprocessor(config)
train_df, test_df = prep.fit_transform(ratings)

cf = CollaborativeFilter(config)
cf.fit(train_df)

cb = ContentBasedFilter(config)
cb.fit(movies, ratings)

hybrid = HybridRecommender(config, cf, cb)
```

### 4. Start the API

```bash
uvicorn src.api.main:app --reload
```

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Redoc:      [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/recommend/{user_id}?n=10` | Top-N recommendations for a user |
| `GET` | `/similar/{item_id}?n=10` | Items similar to a given item |
| `POST` | `/rate` | Submit a new rating |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

Edit `config.yaml` to tune hyperparameters:

```yaml
model:
  hybrid_alpha: 0.6    # 1.0 = pure CF, 0.0 = pure CB
  svd_factors: 50
  knn_k: 20
```

---

## Evaluation Metrics

- **RMSE / MAE** — rating prediction accuracy
- **Precision@K** — fraction of top-K recommendations that are relevant
- **Recall@K** — fraction of relevant items that appear in top-K
- **NDCG@K** — normalized discounted cumulative gain (ranking quality)
- **Coverage** — % of catalogue the model can recommend

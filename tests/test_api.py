"""
tests/test_api.py
-----------------
Integration tests for the FastAPI REST API using httpx test client.
No live server is needed — uses ASGI TestClient.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, state
from src.models.collaborative import UserBasedCF
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG = {
    "model": {
        "knn_k": 3,
        "knn_min_k": 1,
        "tfidf_max_features": 100,
        "tfidf_ngram_range": [1, 1],
        "hybrid_alpha": 0.6,
        "cold_start_threshold": 2,
    },
    "data": {},
    "api": {"top_n": 5},
}

TRAIN_DF = pd.DataFrame(
    {
        "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "movieId": [10, 20, 30, 10, 20, 40, 10, 30, 40],
        "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0],
    }
)

MOVIES_DF = pd.DataFrame(
    {
        "movieId": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "title": [f"Movie {i}" for i in range(10)],
        "genres": [
            "Action|Adventure",
            "Comedy|Romance",
            "Action|Thriller",
            "Drama|Romance",
            "Comedy|Drama",
            "Thriller|Action",
            "Sci-Fi|Adventure",
            "Horror|Thriller",
            "Animation|Comedy",
            "Documentary|Drama",
        ],
    }
)


@pytest.fixture(autouse=True)
def setup_state():
    """Populate AppState with lightweight trained models before each test."""
    cf = UserBasedCF(CONFIG)
    cf.fit(TRAIN_DF)

    cb = ContentBasedFilter(CONFIG)
    cb.fit(MOVIES_DF, TRAIN_DF)

    hybrid = HybridRecommender(CONFIG, cf, cb, all_ratings=TRAIN_DF)

    state.cf = cf
    state.cb = cb
    state.hybrid = hybrid
    state.ratings_df = TRAIN_DF.copy()
    state.movies_df = MOVIES_DF.copy()
    state.config = CONFIG

    yield

    # Teardown — reset state
    state.cf = None
    state.cb = None
    state.hybrid = None
    state.ratings_df = None
    state.movies_df = None


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# /recommend/{user_id}
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_valid_user_returns_200(self, client):
        response = client.get("/recommend/1")
        assert response.status_code == 200

    def test_response_contains_recommendations(self, client):
        data = client.get("/recommend/1").json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    def test_recommendations_have_correct_fields(self, client):
        data = client.get("/recommend/1").json()
        for item in data["recommendations"]:
            assert "movie_id" in item
            assert "score" in item

    def test_n_parameter_respected(self, client):
        data = client.get("/recommend/1?n=2").json()
        assert len(data["recommendations"]) <= 2

    def test_unknown_user_returns_404(self, client):
        response = client.get("/recommend/99999")
        assert response.status_code == 404

    def test_model_selection_cf(self, client):
        response = client.get("/recommend/1?model=cf")
        assert response.status_code == 200

    def test_model_selection_cb(self, client):
        response = client.get("/recommend/1?model=cb")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /similar/{item_id}
# ---------------------------------------------------------------------------

class TestSimilarItems:
    def test_valid_item_returns_200(self, client):
        response = client.get("/similar/10")
        assert response.status_code == 200

    def test_response_contains_similar_items(self, client):
        data = client.get("/similar/10").json()
        assert "similar_items" in data

    def test_unknown_item_returns_404(self, client):
        response = client.get("/similar/99999")
        assert response.status_code == 404

    def test_n_parameter(self, client):
        data = client.get("/similar/10?n=2").json()
        assert len(data["similar_items"]) <= 2


# ---------------------------------------------------------------------------
# /rate
# ---------------------------------------------------------------------------

class TestRate:
    def test_valid_rating_returns_200(self, client):
        response = client.post(
            "/rate",
            json={"user_id": 1, "movie_id": 10, "rating": 4.5},
        )
        assert response.status_code == 200

    def test_response_echoes_rating(self, client):
        payload = {"user_id": 1, "movie_id": 10, "rating": 3.0}
        data = client.post("/rate", json=payload).json()
        assert data["user_id"] == 1
        assert data["movie_id"] == 10
        assert data["rating"] == 3.0

    def test_invalid_rating_too_low(self, client):
        response = client.post(
            "/rate",
            json={"user_id": 1, "movie_id": 10, "rating": 0.0},
        )
        assert response.status_code == 422   # Pydantic validation error

    def test_invalid_rating_too_high(self, client):
        response = client.post(
            "/rate",
            json={"user_id": 1, "movie_id": 10, "rating": 6.0},
        )
        assert response.status_code == 422

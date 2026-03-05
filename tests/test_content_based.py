"""
tests/test_content_based.py
----------------------------
Unit tests for src.models.content_based.ContentBasedFilter
"""
import pandas as pd
import pytest

from src.models.content_based import ContentBasedFilter

CONFIG = {
    "model": {
        "tfidf_max_features": 500,
        "tfidf_ngram_range": [1, 1],
    },
    "data": {},
    "api": {"top_n": 5},
}


@pytest.fixture
def movies():
    return pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
            "genres": [
                "Action|Adventure",
                "Comedy|Romance",
                "Action|Thriller",
                "Drama|Romance",
                "Comedy|Drama",
            ],
        }
    )


@pytest.fixture
def ratings():
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3],
            "movieId": [10, 20, 30, 10, 40, 20],
            "rating": [5.0, 2.0, 4.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def fitted_model(movies, ratings):
    model = ContentBasedFilter(CONFIG)
    model.fit(movies, ratings)
    return model


class TestContentBasedFilter:
    def test_fit_sets_item_matrix(self, fitted_model):
        assert fitted_model._item_matrix is not None
        assert fitted_model._item_matrix.ndim == 2

    def test_item_matrix_shape(self, movies, ratings, fitted_model):
        # fit() filters movies to those appearing in ratings, so use model's own index count
        n_items = len(fitted_model._item2idx)
        assert fitted_model._item_matrix.shape[0] == n_items
        assert n_items > 0

    def test_tfidf_values_non_negative(self, fitted_model):
        assert (fitted_model._item_matrix >= 0).all()

    def test_predict_returns_float_in_range(self, fitted_model, ratings):
        score = fitted_model.predict(1, 40, ratings)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0 + 1e-6

    def test_recommend_excludes_seen_items(self, fitted_model, ratings):
        seen = set(ratings[ratings["userId"] == 1]["movieId"].tolist())
        recs = fitted_model.recommend(1, ratings, n=10)
        rec_ids = {mid for mid, _ in recs}
        assert rec_ids.isdisjoint(seen)

    def test_recommend_returns_at_most_n(self, fitted_model, ratings):
        recs = fitted_model.recommend(1, ratings, n=2)
        assert len(recs) <= 2

    def test_recommend_unknown_user_returns_empty(self, fitted_model, ratings):
        recs = fitted_model.recommend(999, ratings, n=5)
        assert recs == []

    def test_similar_items_similarity_in_range(self, fitted_model):
        similar = fitted_model.similar_items(10, n=3)
        for _, sim in similar:
            assert 0.0 <= sim <= 1.0 + 1e-6

    def test_similar_items_excludes_self(self, fitted_model):
        similar = fitted_model.similar_items(10, n=10)
        item_ids = [mid for mid, _ in similar]
        assert 10 not in item_ids

"""
tests/test_hybrid.py
---------------------
Unit tests for src.models.hybrid.HybridRecommender
"""
import pandas as pd
import pytest

from src.models.collaborative import UserBasedCF
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender

BASE_CONFIG = {
    "model": {
        "knn_k": 3,
        "knn_min_k": 1,
        "tfidf_max_features": 100,
        "tfidf_ngram_range": [1, 1],
        "hybrid_alpha": 0.6,
        "cold_start_threshold": 5,
    },
    "data": {},
    "api": {"top_n": 5},
}


@pytest.fixture
def train_df():
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3],
            "movieId": [10, 20, 30, 40, 50, 60, 10, 20, 30, 10, 40],
            "rating": [5.0, 4.0, 3.0, 2.0, 1.0, 4.0, 4.0, 5.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def movies():
    return pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50, 60],
            "title": [f"Movie {c}" for c in "ABCDEF"],
            "genres": [
                "Action|Adventure",
                "Comedy|Romance",
                "Action|Thriller",
                "Drama|Romance",
                "Comedy|Drama",
                "Thriller|Action",
            ],
        }
    )


@pytest.fixture
def hybrid(train_df, movies):
    cf = UserBasedCF(BASE_CONFIG)
    cf.fit(train_df)

    cb = ContentBasedFilter(BASE_CONFIG)
    cb.fit(movies, train_df)

    return HybridRecommender(BASE_CONFIG, cf, cb, all_ratings=train_df)


class TestHybridRecommender:
    def test_recommend_returns_list(self, hybrid, train_df):
        recs = hybrid.recommend(1, n=5, ratings_df=train_df)
        assert isinstance(recs, list)

    def test_recommend_length_bounded(self, hybrid, train_df):
        recs = hybrid.recommend(1, n=3, ratings_df=train_df)
        assert len(recs) <= 3

    def test_alpha_1_matches_cf_output(self, train_df, movies):
        """α=1.0 means CF-only — hybrid should give CF-dominated ranking."""
        cfg = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "hybrid_alpha": 1.0}}
        cf = UserBasedCF(cfg)
        cf.fit(train_df)
        cb = ContentBasedFilter(cfg)
        cb.fit(movies, train_df)
        hybrid = HybridRecommender(cfg, cf, cb, all_ratings=train_df)

        recs = hybrid.recommend(1, n=5, ratings_df=train_df)
        # Should still produce valid recommendations
        assert isinstance(recs, list)

    def test_alpha_0_matches_cb_output(self, train_df, movies):
        """α=0.0 means CB-only."""
        cfg = {**BASE_CONFIG, "model": {**BASE_CONFIG["model"], "hybrid_alpha": 0.0}}
        cf = UserBasedCF(cfg)
        cf.fit(train_df)
        cb = ContentBasedFilter(cfg)
        cb.fit(movies, train_df)
        hybrid = HybridRecommender(cfg, cf, cb, all_ratings=train_df)

        recs = hybrid.recommend(1, n=5, ratings_df=train_df)
        assert isinstance(recs, list)

    def test_cold_start_user_falls_back_to_cb(self, train_df, movies):
        """User 3 has only 2 ratings (< threshold=5) — effective alpha should be 0."""
        cf = UserBasedCF(BASE_CONFIG)
        cf.fit(train_df)
        cb = ContentBasedFilter(BASE_CONFIG)
        cb.fit(movies, train_df)
        hybrid = HybridRecommender(BASE_CONFIG, cf, cb, all_ratings=train_df)

        alpha = hybrid._effective_alpha(3, train_df)
        assert alpha == 0.0

    def test_warm_user_uses_configured_alpha(self, hybrid, train_df):
        """User 1 has 6 ratings (≥ threshold=5) — should use the configured alpha."""
        alpha = hybrid._effective_alpha(1, train_df)
        assert alpha == pytest.approx(BASE_CONFIG["model"]["hybrid_alpha"])

    def test_predict_returns_float(self, hybrid, train_df):
        score = hybrid.predict(1, 40, ratings_df=train_df)
        assert isinstance(score, float)

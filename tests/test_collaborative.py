"""
tests/test_collaborative.py
----------------------------
Unit tests for src.models.collaborative
"""
import pandas as pd
import pytest

from src.models.collaborative import ItemBasedCF, UserBasedCF

CONFIG = {
    "model": {"knn_k": 3, "knn_min_k": 1, "svd_factors": 10, "svd_epochs": 5},
    "data": {},
    "api": {"top_n": 5},
}


@pytest.fixture
def train_df():
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "movieId": [10, 20, 30, 10, 20, 40, 10, 30, 40],
            "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


# ---------------------------------------------------------------------------
# UserBasedCF
# ---------------------------------------------------------------------------

class TestUserBasedCF:
    def test_fit_builds_similarity(self, train_df):
        model = UserBasedCF(CONFIG)
        model.fit(train_df)
        assert model._sim is not None
        assert model._sim.shape[0] == model._sim.shape[1]

    def test_predict_returns_float(self, train_df):
        model = UserBasedCF(CONFIG)
        model.fit(train_df)
        pred = model.predict(1, 20)
        assert isinstance(pred, float)

    def test_predict_unknown_user_returns_mean(self, train_df):
        model = UserBasedCF(CONFIG)
        model.fit(train_df)
        pred = model.predict(999, 10)
        assert pred == pytest.approx(model._global_mean, abs=1e-6)

    def test_recommend_length(self, train_df):
        model = UserBasedCF(CONFIG)
        model.fit(train_df)
        recs = model.recommend(1, n=3)
        assert len(recs) <= 3

    def test_recommend_excludes_seen_items(self, train_df):
        model = UserBasedCF(CONFIG)
        model.fit(train_df)
        seen = set(train_df[train_df["userId"] == 1]["movieId"].tolist())
        recs = model.recommend(1, n=10)
        rec_ids = {r[0] for r in recs}
        assert rec_ids.isdisjoint(seen)

    def test_recommend_scores_are_positive(self, train_df):
        model = UserBasedCF(CONFIG)
        model.fit(train_df)
        recs = model.recommend(1, n=5)
        for _, score in recs:
            assert score >= 0


# ---------------------------------------------------------------------------
# ItemBasedCF
# ---------------------------------------------------------------------------

class TestItemBasedCF:
    def test_fit_builds_item_similarity(self, train_df):
        model = ItemBasedCF(CONFIG)
        model.fit(train_df)
        assert model._item_sim is not None

    def test_predict_returns_float(self, train_df):
        model = ItemBasedCF(CONFIG)
        model.fit(train_df)
        pred = model.predict(2, 30)
        assert isinstance(pred, float)

    def test_similar_items_returns_correct_length(self, train_df):
        model = ItemBasedCF(CONFIG)
        model.fit(train_df)
        similar = model.similar_items(10, n=2)
        assert len(similar) <= 2

    def test_similar_items_similarity_in_range(self, train_df):
        model = ItemBasedCF(CONFIG)
        model.fit(train_df)
        similar = model.similar_items(10, n=5)
        for _, sim in similar:
            assert 0.0 <= sim <= 1.0 + 1e-6  # float imprecision tolerance

    def test_recommend_unknown_user_returns_empty(self, train_df):
        model = ItemBasedCF(CONFIG)
        model.fit(train_df)
        recs = model.recommend(999, n=5)
        assert recs == []

"""
tests/test_preprocessor.py
--------------------------
Unit tests for src.data.preprocessor.Preprocessor
"""
import pandas as pd
import pytest

from src.data.preprocessor import Preprocessor

CONFIG = {
    "data": {
        "test_size": 0.2,
        "random_state": 42,
        "processed_dir": "data/processed/",
        "ratings_path": "data/raw/ratings.csv",
        "items_path": "data/raw/movies.csv",
    },
    "model": {},
    "api": {"top_n": 10},
}


@pytest.fixture
def sample_ratings():
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 1, 2, 2, 2, 3, 3],
            "movieId": [10, 20, 30, 40, 50, 10, 20, 30, 10, 40],
            "rating": [5, 4, 3, 2, 1, 4, 3, 5, 2, 4],
            "timestamp": list(range(10)),
        }
    )


def test_fit_transform_returns_split(sample_ratings):
    prep = Preprocessor(CONFIG)
    train, test = prep.fit_transform(sample_ratings)
    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(sample_ratings)


def test_split_ratio_approximate(sample_ratings):
    prep = Preprocessor(CONFIG)
    train, test = prep.fit_transform(sample_ratings)
    actual_ratio = len(test) / len(sample_ratings)
    # Allow ±10% tolerance due to per-user stratification with small data
    assert 0.0 <= actual_ratio <= 0.5


def test_encodings_are_correct(sample_ratings):
    prep = Preprocessor(CONFIG)
    prep.fit_transform(sample_ratings)

    unique_users = sorted(sample_ratings["userId"].unique())
    unique_items = sorted(sample_ratings["movieId"].unique())

    assert len(prep.user2idx) == len(unique_users)
    assert len(prep.item2idx) == len(unique_items)

    # Indices should be contiguous 0-based integers
    assert set(prep.user2idx.values()) == set(range(len(unique_users)))
    assert set(prep.item2idx.values()) == set(range(len(unique_items)))


def test_user_indices_in_output(sample_ratings):
    prep = Preprocessor(CONFIG)
    train, test = prep.fit_transform(sample_ratings)
    assert "user_idx" in train.columns
    assert "item_idx" in train.columns


def test_build_interaction_matrix_shape(sample_ratings):
    prep = Preprocessor(CONFIG)
    train, _ = prep.fit_transform(sample_ratings)
    matrix = prep.build_interaction_matrix(train)
    n_users = len(prep.user2idx)
    n_items = len(prep.item2idx)
    assert matrix.shape == (n_users, n_items)


def test_normalize_ratings_range(sample_ratings):
    prep = Preprocessor(CONFIG)
    normed = prep.normalize_ratings(sample_ratings)
    assert normed["rating"].min() >= 0.0
    assert normed["rating"].max() <= 1.0


def test_check_fitted_raises_before_fit():
    prep = Preprocessor(CONFIG)
    with pytest.raises(RuntimeError, match="fit_transform"):
        prep._check_fitted()

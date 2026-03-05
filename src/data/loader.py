"""
src/data/loader.py
------------------
Loads and validates the MovieLens-style CSV datasets.

Expected CSV schemas
--------------------
ratings.csv : userId (int), movieId (int), rating (float), timestamp (int)
movies.csv  : movieId (int), title (str), genres (str)  -- genres pipe-separated
users.csv   : userId (int), gender (str), age (int), occupation (int)  [optional]
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate raw CSVs for the recommendation system."""

    RATINGS_REQUIRED = {"userId", "movieId", "rating"}
    MOVIES_REQUIRED = {"movieId", "title", "genres"}

    def __init__(self, config: dict) -> None:
        self.config = config
        self.ratings_path = Path(config["data"]["ratings_path"])
        self.items_path = Path(config["data"]["items_path"])
        self.users_path = Path(config["data"].get("users_path", ""))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load ratings and movies DataFrames.

        Returns
        -------
        ratings : pd.DataFrame
        movies  : pd.DataFrame
        """
        logger.info("Loading ratings from %s", self.ratings_path)
        ratings = self._load_csv(self.ratings_path, self.RATINGS_REQUIRED)
        ratings = self._cast_ratings(ratings)

        logger.info("Loading movies from %s", self.items_path)
        movies = self._load_csv(self.items_path, self.MOVIES_REQUIRED)
        movies = self._cast_movies(movies)

        self._validate_referential_integrity(ratings, movies)

        logger.info(
            "Loaded %d ratings · %d movies · %d unique users",
            len(ratings),
            len(movies),
            ratings["userId"].nunique(),
        )
        return ratings, movies

    def load_users(self) -> Optional[pd.DataFrame]:
        """Optionally load user metadata (returns None if file absent)."""
        if not self.users_path or not self.users_path.exists():
            logger.debug("No users file found at %s — skipping", self.users_path)
            return None
        logger.info("Loading users from %s", self.users_path)
        return self._load_csv(self.users_path, {"userId"})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: Path, required_cols: set) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Download MovieLens 100K from https://grouplens.org/datasets/movielens/100k/ "
                "and place the CSV files in data/raw/."
            )
        df = pd.read_csv(path)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"File '{path}' is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )
        return df

    @staticmethod
    def _cast_ratings(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Handle tstamp vs timestamp alias
        if "tstamp" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"tstamp": "timestamp"})
            
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        df["rating"] = df["rating"].astype(float)
        
        cols_to_check = ["userId", "movieId", "rating"]
        null_mask = df[cols_to_check].isnull().any(axis=1)
        if null_mask.any():
            logger.warning("Dropping %d rows with null values in ratings", null_mask.sum())
            df = df[~null_mask]
        # Clamp ratings to [0.5, 5.0] (MovieLens range)
        df["rating"] = df["rating"].clip(0.5, 5.0)
        return df.reset_index(drop=True)

    @staticmethod
    def _cast_movies(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["movieId"] = df["movieId"].astype(int)
        df["title"] = df["title"].astype(str).str.strip()
        df["genres"] = df["genres"].fillna("").astype(str)
        return df.drop_duplicates(subset="movieId").reset_index(drop=True)

    @staticmethod
    def _validate_referential_integrity(
        ratings: pd.DataFrame, movies: pd.DataFrame
    ) -> None:
        orphan_movies = set(ratings["movieId"].unique()) - set(movies["movieId"].unique())
        if orphan_movies:
            logger.warning(
                "%d movieIds appear in ratings but not in movies catalogue — they will be excluded",
                len(orphan_movies),
            )


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def load_from_config(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data using a YAML config file path."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return DataLoader(config).load()

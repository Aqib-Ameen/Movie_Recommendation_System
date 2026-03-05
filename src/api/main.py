import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global application state (populated at startup)
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    cf: Optional[object] = None       # SVDModel | UserBasedCF | ItemBasedCF
    cb: Optional[object] = None       # ContentBasedFilter
    hybrid: Optional[object] = None   # HybridRecommender
    ratings_df: Optional[pd.DataFrame] = None
    movies_df: Optional[pd.DataFrame] = None
    config: dict = field(default_factory=dict)


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan context manager (replaces deprecated on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load/train models at startup."""
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    if not Path(config_path).exists():
        logger.warning("config.yaml not found at %s — running in no-model mode", config_path)
    else:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        state.config = cfg

        model_dir = Path(cfg["api"].get("model_dir", "models/saved/"))
        cf_path = model_dir / "cf_model.pkl"
        cb_path = model_dir / "cb_model.pkl"

        if cf_path.exists() and cb_path.exists():
            logger.info("Loading pre-trained models from %s …", model_dir)
            try:
                import joblib
                state.cf = joblib.load(cf_path)
                state.cb = joblib.load(cb_path)
                logger.info("Models loaded successfully.")
            except Exception as exc:
                logger.error("Failed to load models: %s — will attempt training.", exc)
                _train_models(cfg)
        else:
            logger.info("No saved models found — training from raw data …")
            _train_models(cfg)

        if state.cf is not None and state.cb is not None:
            from src.models.hybrid import HybridRecommender
            state.hybrid = HybridRecommender(cfg, state.cf, state.cb, state.ratings_df)
            logger.info("HybridRecommender initialised.")

    yield   # app runs here
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="🎬 Recommendation System API",
    description=(
        "A hybrid recommendation engine combining Collaborative Filtering "
        "(SVD Matrix Factorisation) and Content-Based Filtering (TF-IDF + Cosine Similarity).\n\n"
        "**Endpoints:**\n"
        "- `GET /recommend/{user_id}` — personalised top-N recommendations\n"
        "- `GET /similar/{item_id}` — similar items (content-based)\n"
        "- `POST /rate` — submit a new rating\n"
        "- `GET /health` — liveness probe"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def _train_models(cfg: dict) -> None:
    """Train CF and CB models from scratch using raw CSV data."""
    try:
        from src.data.loader import DataLoader
        from src.data.preprocessor import Preprocessor
        from src.models.collaborative import SVDModel
        from src.models.content_based import ContentBasedFilter

        loader = DataLoader(cfg)
        ratings, movies = loader.load()
        state.ratings_df = ratings
        state.movies_df = movies

        prep = Preprocessor(cfg)
        train_df, _ = prep.fit_transform(ratings)

        # Train SVD
        cf = SVDModel(cfg)
        cf.fit(train_df)
        state.cf = cf

        # Train CB
        cb = ContentBasedFilter(cfg)
        cb.fit(movies, ratings)
        state.cb = cb

        # Persist models
        model_dir = Path(cfg["api"].get("model_dir", "models/saved/"))
        model_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(cf, model_dir / "cf_model.pkl")
        joblib.dump(cb, model_dir / "cb_model.pkl")
        logger.info("Models trained and saved to %s", model_dir)

    except FileNotFoundError as exc:
        logger.error(
            "Cannot train models — dataset not found: %s\n"
            "Please add ratings.csv and movies.csv to data/raw/ and restart.",
            exc,
        )
    except Exception as exc:
        logger.exception("Unexpected error during model training: %s", exc)


def _use_defaults() -> None:
    """Set state to empty when config is absent (allows API to start for testing)."""
    logger.warning("Running in no-model mode. /recommend and /similar will return 503.")

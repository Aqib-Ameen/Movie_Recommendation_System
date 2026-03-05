import logging
import yaml
from pathlib import Path
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.collaborative import SVDModel
from src.models.content_based import ContentBasedFilter
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Load Data
    logger.info("Loading data...")
    loader = DataLoader(config)
    ratings, movies = loader.load()
    
    # 3. Preprocess
    logger.info("Preprocessing data...")
    prep = Preprocessor(config)
    train_df, test_df = prep.fit_transform(ratings)
    prep.save()
    
    # 4. Train Collaborative Filter (SVD)
    # Using a subset for faster training if needed, but let's try full first
    # Note: If scikit-surprise is missing, this will fail. 
    # Fallback to UserBasedCF if SVD fails.
    try:
        logger.info("Training SVD Model...")
        cf = SVDModel(config)
        cf.fit(train_df)
    except Exception as e:
        logger.warning(f"SVD Training failed or scikit-surprise missing: {e}. Falling back to UserBasedCF...")
        from src.models.collaborative import UserBasedCF
        cf = UserBasedCF(config)
        cf.fit(train_df)
    
    # 5. Train Content-Based Filter
    logger.info("Training Content-Based Filter...")
    cb = ContentBasedFilter(config)
    cb.fit(movies, ratings)
    
    # 6. Save Models
    model_dir = Path(config["api"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(cf, model_dir / "cf_model.pkl")
    joblib.dump(cb, model_dir / "cb_model.pkl")
    logger.info(f"Models saved to {model_dir}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()

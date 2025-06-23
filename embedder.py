from sentence_transformers import SentenceTransformer
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self):
        try:
            from config import CONFIG
            model_path = CONFIG["embedder_model_path"]
            model_name = CONFIG["embedder_model_name"]
            if os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                self.model = SentenceTransformer(model_path)
            else:
                logger.warning(f"Model path {model_path} not found, downloading {model_name}")
                self.model = SentenceTransformer(model_name)
            logger.info("Embedder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Embedder: {e}")
            raise

    def embed(self, texts):
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.info(f"Embedded {len(texts) if isinstance(texts, list) else 1} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise
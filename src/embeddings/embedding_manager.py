from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import logger

class EmbeddingManager:
    """
    Manages the initialization and usage of semantic embedding models.
    """

    SUPPORTED_MODELS = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "bge": "BAAI/bge-small-en-v1.5"
    }

    def __init__(self, model_key: str = "minilm"):
        """
        Initializes the embedding model.

        Args:
            model_key: Key for the supported model ('minilm' or 'bge').
        """
        self.model_key = model_key.lower()
        if self.model_key not in self.SUPPORTED_MODELS:
            logger.warning(f"Model key '{model_key}' not found. Defaulting to 'minilm'.")
            self.model_key = "minilm"

        self.model_name = self.SUPPORTED_MODELS[self.model_key]
        logger.info(f"Loading embedding model: {self.model_name}")

        try:

            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Returns the LangChain embedding object."""
        return self.embeddings

    def get_model_name(self) -> str:
        """Returns the HuggingFace model name."""
        return self.model_name

    def get_embedding_dimension(self) -> int:
        """
        Dynamically determines and caches the embedding dimension.
        """
        if not hasattr(self, "_dimension") or self._dimension is None:
            test_embedding = self.embeddings.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension


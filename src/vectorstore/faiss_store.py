import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from src.utils.logger import logger

class FAISSStore:
    """FAISS-based persistent vector store wrapper."""
    
    def __init__(self, embeddings: Embeddings, persist_directory: str = "data/vectorstores/faiss/"):
        self.embeddings = embeddings
        self.persist_directory = Path(persist_directory)
        self.vector_store: Optional[FAISS] = None

    def _ensure_dir(self):
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    def add_documents(self, documents: List[Document]) -> None:
        """Adds documents to the FAISS index."""
        if not documents:
            return
            
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
            
        logger.info(f"Added {len(documents)} documents to FAISS.")

    def save(self) -> None:
        """Saves the FAISS index to disk."""
        if self.vector_store is not None:
            self._ensure_dir()
            self.vector_store.save_local(str(self.persist_directory))
            logger.info(f"Saved FAISS index to {self.persist_directory}")
        else:
            logger.warning("No FAISS index to save.")

    def load(self) -> bool:
        """Loads the FAISS index from disk. Returns True if successful."""
        index_file = self.persist_directory / "index.faiss"
        if not index_file.exists():
            logger.warning("FAISS index not found on disk.")
            return False
            
        try:
            # allow_dangerous_deserialization is needed in newer versions of LangChain community
            self.vector_store = FAISS.load_local(str(self.persist_directory), self.embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded FAISS index successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """
        Searches for similar documents. 
        Returns a list of (Document, score) tuples.
        Note: FAISS typically returns L2 distance. Lower is more similar.
        """
        if self.vector_store is None:
            return []
        
        # return docs and scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def reset(self) -> None:
        """Deletes the FAISS index from memory and disk."""
        self.vector_store = None
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
            logger.info("Deleted FAISS index from disk.")

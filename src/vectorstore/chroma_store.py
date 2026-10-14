import shutil
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from src.utils.logger import logger

class ChromaStore:
    """ChromaDB-based persistent vector store wrapper."""

    def __init__(self, embeddings: Embeddings, persist_directory: str = "data/vectorstores/chroma/"):
        self.embeddings = embeddings
        self.persist_directory = str(Path(persist_directory).absolute())
        self.collection_name = "study_documents"
        self.vector_store: Optional[Chroma] = None

    def add_documents(self, documents: List[Document]) -> None:
        """Adds documents to the Chroma collection and auto-persists."""
        if not documents:
            return

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,

                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            self.vector_store.add_documents(documents)

        logger.info(f"Added {len(documents)} documents to Chroma.")

    def save(self) -> None:
        """In recent Chroma versions, auto-persistence handles this. We keep method for interface compat."""
        logger.info("Chroma automatically persists data to disk.")

    def load(self) -> bool:
        """Loads the Chroma index from disk."""
        db_path = Path(self.persist_directory) / "chroma.sqlite3"
        if not db_path.exists():
            logger.warning("Chroma DB not found on disk.")
            return False

        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logger.info("Loaded Chroma DB successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Chroma DB: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """
        Searches for similar documents. 
        Returns a list of (Document, score) tuples.
        Note: Chroma typically uses Cosine distance/similarity or L2 depending on config.
        """
        if self.vector_store is None:
            return []

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def reset(self) -> None:
        """Deletes the Chroma collection and directory."""
        if self.vector_store is not None:

            try:
                self.vector_store.delete_collection()
            except Exception as e:
                logger.error(f"Error deleting Chroma collection: {e}")

        self.vector_store = None

        path = Path(self.persist_directory)
        if path.exists():
            shutil.rmtree(path)
            logger.info("Deleted Chroma DB from disk.")

from typing import List, Dict, Any, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.vectorstore.faiss_store import FAISSStore
from src.vectorstore.chroma_store import ChromaStore
from src.utils.logger import logger

class VectorStoreManager:
    """
    Factory-like manager to handle multiple vector store providers (FAISS, Chroma).
    """
    
    def __init__(self, provider: str, embeddings: Embeddings, persist_directory_base: str = "data/vectorstores"):
        self.provider = provider.lower()
        self.embeddings = embeddings
        
        if self.provider == "faiss":
            path = f"{persist_directory_base}/faiss/"
            self.store = FAISSStore(embeddings, persist_directory=path)
        elif self.provider == "chroma":
            path = f"{persist_directory_base}/chroma/"
            self.store = ChromaStore(embeddings, persist_directory=path)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

    def raw_chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """Converts raw dictionary chunks from Phase 2 into LangChain Documents."""
        documents = []
        for chunk in chunks:
            # Preserve source, page, chunk_id
            metadata = {
                "source": chunk.get("source", "unknown"),
                "page": chunk.get("page", 0),
                "chunk_id": chunk.get("chunk_id", -1)
            }
            documents.append(Document(page_content=chunk["text"], metadata=metadata))
        return documents

    def add_documents(self, documents: List[Document]) -> None:
        """Adds documents to the underlying store."""
        self.store.add_documents(documents)

    def save(self) -> None:
        """Saves the underlying store."""
        self.store.save()

    def load(self) -> bool:
        """Loads the underlying store."""
        return self.store.load()

    def similarity_search(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """Performs similarity search."""
        return self.store.similarity_search(query, k=k)

    def reset(self) -> None:
        """Resets/deletes the underlying store."""
        self.store.reset()

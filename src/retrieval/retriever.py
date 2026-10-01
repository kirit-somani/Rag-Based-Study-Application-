from typing import List, Optional
from langchain_core.documents import Document
from src.vectorstore.vector_store_manager import VectorStoreManager

class Retriever:
    """
    Handles similarity search queries against a VectorStoreManager.
    """
    def __init__(self, vector_store: VectorStoreManager, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
        
    def retrieve_with_scores(self, query: str, source_filter: Optional[str] = None) -> List[tuple[Document, float]]:
        """
        Retrieves the top_k relevant documents for the query.
        
        Args:
            query: The processed query string.
            source_filter: Optional specific filename to restrict search.
            
        Returns:
            List of (Document, raw_score) tuples.
        """
        # VectorStoreManager's underlying stores return (Doc, score)
        # Note: In a production app, we would push the source_filter down into the FAISS/Chroma search kwargs.
        # For simplicity and cross-compatibility in this educational project, we retrieve more and filter post-retrieval 
        # if the underlying abstraction doesn't easily expose unified filtering yet.
        # But we'll try to pass filtering kwargs if supported.
        
        filter_kwargs = {}
        if source_filter and source_filter != "All Documents":
            if self.vector_store.provider == "chroma":
                filter_kwargs = {"filter": {"source": source_filter}}
            elif self.vector_store.provider == "faiss":
                # FAISS uses a dict or callable for filtering in similarity_search
                filter_kwargs = {"filter": {"source": source_filter}}

        try:
            if filter_kwargs and hasattr(self.vector_store.store.vector_store, "similarity_search_with_score"):
                results = self.vector_store.store.vector_store.similarity_search_with_score(
                    query, 
                    k=self.top_k, 
                    **filter_kwargs
                )
                return results
            else:
                # Fallback if filters fail
                return self.vector_store.similarity_search(query, k=self.top_k)
        except Exception as e:
            # Fallback for simpler retrieval without filters
            results = self.vector_store.similarity_search(query, k=self.top_k * 3)
            if source_filter and source_filter != "All Documents":
                results = [r for r in results if r[0].metadata.get("source") == source_filter][:self.top_k]
            else:
                results = results[:self.top_k]
            return results

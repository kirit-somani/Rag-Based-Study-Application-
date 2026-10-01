from typing import List, Dict, Any
from langchain_core.documents import Document
import math

def normalize_score(raw_score: float, provider: str) -> float:
    """
    Normalizes vector database scores to a 0.0 - 1.0 relevance scale.
    
    Args:
        raw_score: The score returned by the vector store.
        provider: 'faiss' or 'chroma'
        
    Returns:
        Float between 0.0 and 1.0 where 1.0 is highly relevant.
    """
    # FAISS typically returns L2 distance (0 = identical, higher = less similar)
    if provider.lower() == "faiss":
        # A simple normalization for L2 distance: 1 / (1 + distance)
        return 1.0 / (1.0 + raw_score)
        
    # Chroma typically returns Cosine distance (0 = identical, 1 = orthogonal)
    # Cosine similarity = 1 - cosine_distance
    elif provider.lower() == "chroma":
        # Assume cosine distance for default Chroma config
        similarity = 1.0 - raw_score
        return max(0.0, min(1.0, similarity))
        
    return raw_score

def filter_by_threshold(results: List[tuple[Document, float]], provider: str, threshold: float = None) -> List[Dict[str, Any]]:
    """
    Normalizes scores and filters out results below the given threshold.
    
    Args:
        results: List of (Document, raw_score) tuples.
        provider: Vector database provider name.
        threshold: Minimum normalized relevance score to keep.
        
    Returns:
        List of dictionaries containing text, metadata, and normalized score.
    """
    filtered_results = []
    
    for doc, raw_score in results:
        norm_score = normalize_score(raw_score, provider)
        
        if threshold is None or norm_score >= threshold:
            filtered_results.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0),
                "chunk_id": doc.metadata.get("chunk_id", -1),
                "score": round(norm_score, 4)
            })
            
    # Sort by score descending
    filtered_results.sort(key=lambda x: x["score"], reverse=True)
    return filtered_results

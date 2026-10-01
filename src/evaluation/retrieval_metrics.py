from typing import List, Dict, Any

def calculate_precision_at_k(retrieved_sources: List[str], expected_source: str, k: int) -> float:
    """
    Calculates Precision@K: (Relevant Retrieved Documents) / K
    """
    if k <= 0:
        return 0.0
        
    top_k_sources = retrieved_sources[:k]
    relevant_count = sum(1 for src in top_k_sources if src == expected_source)
    return relevant_count / k

def calculate_recall_at_k(retrieved_sources: List[str], expected_source: str, k: int) -> float:
    """
    Calculates Recall@K: (Relevant Retrieved Documents) / (Total Relevant Documents)
    Since this is a simplified metric for the project, we assume there is at least 1 relevant doc.
    Usually Recall denominator is total relevant in DB. We will cap at 1.0.
    """
    if k <= 0:
        return 0.0
        
    top_k_sources = retrieved_sources[:k]
    relevant_count = sum(1 for src in top_k_sources if src == expected_source)
    # Simplified assumption: assume 1 expected relevant document source
    return min(1.0, float(relevant_count))

def calculate_hit_rate_at_k(retrieved_sources: List[str], expected_source: str, k: int) -> int:
    """
    Calculates Hit Rate@K: 1 if at least one relevant document is in top K, else 0
    """
    if k <= 0:
        return 0
        
    top_k_sources = retrieved_sources[:k]
    return 1 if expected_source in top_k_sources else 0

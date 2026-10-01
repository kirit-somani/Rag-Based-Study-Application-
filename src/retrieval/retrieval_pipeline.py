import time
from typing import Dict, Any, Optional
from src.retrieval.query_processor import process_query
from src.retrieval.retriever import Retriever
from src.retrieval.score_filter import filter_by_threshold
from src.vectorstore.vector_store_manager import VectorStoreManager
from src.utils.logger import logger

def retrieve_context(
    query: str,
    vector_store: VectorStoreManager,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
    source_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the complete retrieval pipeline.
    
    Args:
        query: Raw user query.
        vector_store: Instantiated VectorStoreManager.
        top_k: Number of documents to retrieve.
        score_threshold: Minimum relevance score (0.0 - 1.0).
        source_filter: Specific document to search in.
        
    Returns:
        Structured dictionary with results and metadata.
    """
    total_start = time.time()
    
    # 1. Query Processing
    try:
        clean_query = process_query(query)
    except ValueError as e:
        return {
            "query": query,
            "found": False,
            "results": [],
            "message": str(e),
            "retrieval_time": 0.0
        }
        
    logger.info(f"Retrieval started for query: '{clean_query[:50]}...'")
    
    # 2. Vector Search
    search_start = time.time()
    retriever = Retriever(vector_store, top_k=top_k)
    raw_results = retriever.retrieve_with_scores(clean_query, source_filter)
    search_time = time.time() - search_start
    
    # 3. Score Normalization and Filtering
    filter_start = time.time()
    filtered_results = filter_by_threshold(
        results=raw_results, 
        provider=vector_store.provider,
        threshold=score_threshold
    )
    filter_time = time.time() - filter_start
    
    total_time = time.time() - total_start
    
    logger.info(f"Retrieved {len(raw_results)} results. Kept {len(filtered_results)} after threshold. Time: {total_time:.3f}s")
    
    if not filtered_results:
        return {
            "query": clean_query,
            "found": False,
            "results": [],
            "message": "No sufficiently relevant information was found in the uploaded documents.",
            "metrics": {
                "search_time": search_time,
                "filter_time": filter_time,
                "total_time": total_time
            }
        }
        
    return {
        "query": clean_query,
        "found": True,
        "results": filtered_results,
        "message": f"Found {len(filtered_results)} relevant results.",
        "metrics": {
            "search_time": search_time,
            "filter_time": filter_time,
            "total_time": total_time
        }
    }

import pytest
from langchain_core.documents import Document
from src.retrieval.score_filter import normalize_score, filter_by_threshold
from src.evaluation.retrieval_metrics import calculate_precision_at_k, calculate_recall_at_k, calculate_hit_rate_at_k

def test_normalize_score():
    # FAISS uses L2 distance
    assert normalize_score(0.0, "faiss") == 1.0
    assert normalize_score(1.0, "faiss") == 0.5
    
    # Chroma uses cosine distance
    assert normalize_score(0.0, "chroma") == 1.0
    assert normalize_score(0.2, "chroma") == 0.8

def test_filter_by_threshold():
    results = [
        (Document(page_content="A", metadata={"source": "doc1"}), 0.1), # Normalized to 0.9 (faiss)
        (Document(page_content="B", metadata={"source": "doc2"}), 0.5), # Normalized to ~0.66 (faiss)
        (Document(page_content="C", metadata={"source": "doc3"}), 1.5)  # Normalized to 0.4 (faiss)
    ]
    
    # Threshold 0.5 should keep A and B
    filtered = filter_by_threshold(results, "faiss", threshold=0.5)
    assert len(filtered) == 2
    assert filtered[0]["text"] == "A"
    assert filtered[1]["text"] == "B"
    
    # Threshold 0.95 should keep nothing
    filtered2 = filter_by_threshold(results, "faiss", threshold=0.95)
    assert len(filtered2) == 0

def test_metrics():
    retrieved = ["doc1", "doc2", "doc3", "doc1"]
    
    assert calculate_precision_at_k(retrieved, "doc1", 3) == 1/3 # doc1 appears once in top 3
    assert calculate_hit_rate_at_k(retrieved, "doc1", 3) == 1
    assert calculate_hit_rate_at_k(retrieved, "doc4", 3) == 0
    
    assert calculate_recall_at_k(retrieved, "doc1", 1) == 1.0

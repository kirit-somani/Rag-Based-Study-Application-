import pytest
import numpy as np
from src.embeddings.embedding_manager import EmbeddingManager

@pytest.fixture
def embedding_manager():
    # Use MiniLM as it's smaller and faster for tests
    return EmbeddingManager(model_key="minilm")

def test_embedding_initialization(embedding_manager):
    assert embedding_manager.get_model_name() == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding_manager.get_embedding_dimension() == 384

def test_semantic_similarity(embedding_manager):
    embeddings = embedding_manager.get_embeddings()
    
    # Generate embeddings
    query_emb = embeddings.embed_query("Binary search works on sorted data.")
    doc1_emb = embeddings.embed_query("A sorted array can be searched efficiently using binary search.")
    doc2_emb = embeddings.embed_query("The weather is pleasant today.")
    
    # Calculate cosine similarities
    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
    sim_doc1 = cosine_sim(query_emb, doc1_emb)
    sim_doc2 = cosine_sim(query_emb, doc2_emb)
    
    # The related sentence should have a significantly higher similarity score
    assert sim_doc1 > sim_doc2
    assert sim_doc1 > 0.5 # Usually very high for these sentences
    assert sim_doc2 < 0.3 # Usually low for unrelated

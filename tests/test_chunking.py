import pytest
from src.ingestion.chunker import create_chunks

def test_create_chunks():
    docs = [
        {"text": "This is a simple sentence that we will split into chunks. It has a few words.", "source": "test.pdf", "page": 1}
    ]
    
    chunks = create_chunks(docs, chunk_size=20, chunk_overlap=5)
    
    assert len(chunks) > 1
    assert "text" in chunks[0]
    assert "source" in chunks[0]
    assert chunks[0]["source"] == "test.pdf"
    assert chunks[0]["page"] == 1
    assert "chunk_id" in chunks[0]
    
    # Check unique chunk IDs
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)) # All unique

def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        create_chunks([], chunk_size=0, chunk_overlap=0)

def test_invalid_overlap():
    with pytest.raises(ValueError):
        create_chunks([], chunk_size=10, chunk_overlap=-1)
        
    with pytest.raises(ValueError):
        create_chunks([], chunk_size=10, chunk_overlap=10) # >= size

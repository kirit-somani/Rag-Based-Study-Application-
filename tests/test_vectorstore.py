import pytest
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.vector_store_manager import VectorStoreManager

@pytest.fixture(scope="module")
def embeddings():
    return EmbeddingManager("minilm").get_embeddings()

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Binary search works on sorted data.",
            metadata={"source": "DSA.pdf", "page": 10, "chunk_id": 0}
        ),
        Document(
            page_content="A stack follows the LIFO principle.",
            metadata={"source": "DSA.pdf", "page": 20, "chunk_id": 1}
        ),
        Document(
            page_content="A queue follows the FIFO principle.",
            metadata={"source": "DSA.pdf", "page": 25, "chunk_id": 2}
        )
    ]

def test_faiss_vector_store(embeddings, sample_documents):
    with tempfile.TemporaryDirectory() as tmpdir:

        vsm = VectorStoreManager("faiss", embeddings, persist_directory_base=tmpdir)
        vsm.add_documents(sample_documents)
        vsm.save()

        results = vsm.similarity_search("What data structure follows LIFO?", k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert "stack" in doc.page_content.lower()
        assert doc.metadata["source"] == "DSA.pdf"

        vsm2 = VectorStoreManager("faiss", embeddings, persist_directory_base=tmpdir)
        assert vsm2.load() is True
        results2 = vsm2.similarity_search("FIFO", k=1)
        assert "queue" in results2[0][0].page_content.lower()

def test_chroma_vector_store(embeddings, sample_documents):
    with tempfile.TemporaryDirectory() as tmpdir:

        vsm = VectorStoreManager("chroma", embeddings, persist_directory_base=tmpdir)
        vsm.add_documents(sample_documents)

        vsm.save()

        results = vsm.similarity_search("What data structure follows LIFO?", k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert "stack" in doc.page_content.lower()

        vsm2 = VectorStoreManager("chroma", embeddings, persist_directory_base=tmpdir)
        assert vsm2.load() is True
        results2 = vsm2.similarity_search("FIFO", k=1)
        assert "queue" in results2[0][0].page_content.lower()

def test_raw_chunks_conversion(embeddings):
    vsm = VectorStoreManager("faiss", embeddings)
    raw = [
        {"text": "Hello", "source": "file.pdf", "page": 1, "chunk_id": 12}
    ]
    docs = vsm.raw_chunks_to_documents(raw)
    assert len(docs) == 1
    assert docs[0].page_content == "Hello"
    assert docs[0].metadata["source"] == "file.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["chunk_id"] == 12

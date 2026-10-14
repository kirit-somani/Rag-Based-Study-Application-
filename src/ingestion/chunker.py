from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

def create_chunks(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Splits documents into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
    Preserves source and page metadata and generates a unique chunk_id.

    Args:
        documents: List of cleaned document dictionaries.
        chunk_size: Maximum size of chunks (characters).
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunk dictionaries with text, source, page, and chunk_id.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    global_chunk_counter = 0

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        page = doc["page"]

        split_texts = text_splitter.split_text(text)

        for chunk_text in split_texts:
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "chunk_id": global_chunk_counter
                })
                global_chunk_counter += 1

    return chunks

import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Cleans raw extracted text by normalizing whitespace while preserving paragraphs.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    text = re.sub(r'[ \t]+', ' ', text)

    text = re.sub(r'\n{3,}', '\n\n', text)

    text = text.strip()

    return text

def clean_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies clean_text to a list of page documents and removes empty pages.

    Args:
        documents: List of dictionaries with 'text' and metadata.

    Returns:
        List of cleaned documents, excluding those with no meaningful text.
    """
    cleaned_docs = []

    for doc in documents:
        cleaned_text = clean_text(doc.get("text", ""))

        if cleaned_text:
            cleaned_docs.append({
                "text": cleaned_text,
                "source": doc["source"],
                "page": doc["page"]
            })

    return cleaned_docs

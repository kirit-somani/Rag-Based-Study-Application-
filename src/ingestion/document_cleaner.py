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
        
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace 3 or more newlines with exactly 2 newlines (paragraph boundary)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
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
        
        if cleaned_text:  # Only keep pages that still have text after cleaning
            cleaned_docs.append({
                "text": cleaned_text,
                "source": doc["source"],
                "page": doc["page"]
            })
            
    return cleaned_docs

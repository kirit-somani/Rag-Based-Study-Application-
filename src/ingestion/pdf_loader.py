try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from pathlib import Path
from typing import List, Dict, Any
from src.utils.logger import logger

def load_pdf(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dictionaries containing text, source, and page metadata.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"File must be a PDF: {path}")

    try:
        doc = pymupdf.open(path)
    except Exception as e:
        logger.error(f"Failed to open PDF {path.name}: {e}")
        raise ValueError(f"Unable to open PDF {path.name}. It may be corrupted or encrypted.")

    if len(doc) == 0:
        logger.warning(f"PDF {path.name} is empty (0 pages).")
        return []

    documents = []
    text_found = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if text and text.strip():
            text_found = True

        documents.append({
            "text": text,
            "source": path.name,
            "page": page_num + 1
        })

    if not text_found:
        logger.warning(f"No extractable text found in {path.name}. Might be a scanned image.")

    logger.info(f"Loaded {len(documents)} pages from {path.name}")
    return documents

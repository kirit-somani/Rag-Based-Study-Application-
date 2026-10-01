from pathlib import Path
from typing import List, Dict, Any
from src.utils.logger import logger
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.document_cleaner import clean_documents
from src.ingestion.chunker import create_chunks
from src.utils.file_utils import calculate_file_hash

def process_pdf(pdf_path: str | Path, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Complete pipeline to ingest a PDF: validate, extract, clean, and chunk.
    
    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum chunk size.
        chunk_overlap: Chunk overlap.
        
    Returns:
        List of processed chunk dictionaries.
    """
    path = Path(pdf_path)
    logger.info(f"Starting ingestion pipeline for {path.name}")
    
    try:
        # Step 1 & 2: Extract text page-by-page
        raw_documents = load_pdf(path)
        if not raw_documents:
            logger.warning(f"No pages could be extracted from {path.name}")
            return []
            
        # Step 3: Clean text
        cleaned_documents = clean_documents(raw_documents)
        if not cleaned_documents:
            logger.warning(f"No meaningful text remained after cleaning {path.name}")
            return []
            
        # Step 4: Create chunks
        chunks = create_chunks(cleaned_documents, chunk_size, chunk_overlap)
        logger.info(f"Generated {len(chunks)} chunks from {path.name}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing {path.name}: {e}")
        raise e

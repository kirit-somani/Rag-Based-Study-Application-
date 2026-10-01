import pytest
from pathlib import Path
from src.ingestion.pdf_loader import load_pdf

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("nonexistent_file.pdf")

def test_invalid_extension():
    # Create a dummy txt file
    dummy_path = Path("test_dummy.txt")
    dummy_path.write_text("Hello")
    try:
        with pytest.raises(ValueError, match="File must be a PDF"):
            load_pdf(dummy_path)
    finally:
        dummy_path.unlink()

from src.ingestion.document_cleaner import clean_text, clean_documents

def test_clean_extra_spaces():
    text = "This   has    way  too many spaces."
    assert clean_text(text) == "This has way too many spaces."

def test_clean_newlines():
    text = "Paragraph 1\n\n\n\nParagraph 2"
    assert clean_text(text) == "Paragraph 1\n\nParagraph 2"

def test_empty_text():
    assert clean_text("") == ""
    assert clean_text("   \n  ") == ""

def test_clean_documents():
    docs = [
        {"text": "Valid page text", "source": "test.pdf", "page": 1},
        {"text": "   \n  ", "source": "test.pdf", "page": 2},
        {"text": "Another page\n\n\n", "source": "test.pdf", "page": 3}
    ]

    cleaned = clean_documents(docs)
    assert len(cleaned) == 2
    assert cleaned[0]["page"] == 1
    assert cleaned[1]["page"] == 3
    assert cleaned[1]["text"] == "Another page"

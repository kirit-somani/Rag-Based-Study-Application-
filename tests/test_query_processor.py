import pytest
from src.retrieval.query_processor import process_query

def test_valid_query():
    assert process_query("What is binary search?") == "What is binary search?"

    assert process_query("   What   is   binary   search???   ") == "What is binary search???"

def test_empty_query():
    with pytest.raises(ValueError, match="Please enter a question."):
        process_query("")

    with pytest.raises(ValueError, match="Please enter a question."):
        process_query("     \n   ")

def test_oversized_query():
    long_query = "A" * 1001
    with pytest.raises(ValueError, match="Your question is too long. Please shorten it."):
        process_query(long_query, max_query_length=1000)

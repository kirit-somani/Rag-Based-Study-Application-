import pytest
from unittest.mock import patch, MagicMock
from src.rag.rag_pipeline import generate_rag_response

@patch("src.rag.rag_pipeline.retrieve_context")
def test_generate_rag_response_no_context(mock_retrieve):

    mock_retrieve.return_value = {
        "found": False,
        "results": [],
        "metrics": {"search_time": 0.1}
    }

    mock_vs = MagicMock()
    mock_llm = MagicMock()

    response = generate_rag_response("What is X?", mock_vs, mock_llm)

    assert "I could not find enough information" in response["answer"]
    assert len(response["sources"]) == 0
    assert mock_llm.get_llm.called is False

@patch("src.rag.rag_pipeline.retrieve_context")
def test_generate_rag_response_with_context(mock_retrieve):

    mock_retrieve.return_value = {
        "found": True,
        "query": "What is X?",
        "results": [
            {"text": "X is an unknown variable.", "source": "math.pdf", "page": 1}
        ],
        "metrics": {"search_time": 0.1}
    }

    mock_vs = MagicMock()
    mock_llm = MagicMock()

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "X is an unknown variable based on math.pdf."

    with patch("src.rag.rag_pipeline.get_rag_prompt_template") as mock_prompt:
        mock_prompt.return_value.__or__.return_value.__or__.return_value.invoke.return_value = "X is an unknown variable."

        response = generate_rag_response("What is X?", mock_vs, mock_llm)

        assert response["answer"] == "X is an unknown variable."
        assert len(response["sources"]) == 1
        assert response["sources"][0]["source"] == "math.pdf"
        assert "generation_time" in response["metrics"]

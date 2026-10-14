import pytest
import os
from unittest.mock import patch
from src.generation.llm_manager import LLMManager

def test_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider: fake"):
        LLMManager(provider="fake")

@patch("src.generation.llm_manager.ChatOllama")
def test_ollama_initialization(mock_ollama):
    manager = LLMManager(provider="ollama", model_name="llama3", temperature=0.5)
    mock_ollama.assert_called_once_with(
        model="llama3", 
        base_url="http://localhost:11434", 
        temperature=0.5
    )
    assert manager.provider == "ollama"

@patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"})
@patch("src.generation.llm_manager.ChatOpenAI")
def test_openai_initialization(mock_openai):
    manager = LLMManager(provider="openai", model_name="gpt-4o", temperature=0.1)
    mock_openai.assert_called_once_with(
        model="gpt-4o", 
        api_key="fake_key", 
        temperature=0.1
    )
    assert manager.provider == "openai"

@patch.dict(os.environ, {}, clear=True)
def test_groq_missing_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY environment variable is not set."):
        LLMManager(provider="groq", model_name="llama-3.3-70b-versatile")

@patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq_key"})
@patch("src.generation.llm_manager.ChatOpenAI")
def test_groq_initialization(mock_openai):
    manager = LLMManager(provider="groq", model_name="llama-3.3-70b-versatile", temperature=0.2)
    mock_openai.assert_called_once_with(
        model="llama-3.3-70b-versatile",
        api_key="fake_groq_key",
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2
    )
    assert manager.provider == "groq"

@patch.dict(os.environ, clear=True)
def test_openai_missing_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is not set."):
        LLMManager(provider="openai")

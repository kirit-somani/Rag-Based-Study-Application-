import os
from typing import Any, Optional
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from src.utils.logger import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMManager:
    """
    Manages the initialization of LLM models (Ollama, OpenAI).
    """
    def __init__(self, provider: str = "ollama", model_name: str = "llama3", temperature: float = 0.0):
        """
        Initializes the LLM.
        
        Args:
            provider: 'ollama' or 'openai'
            model_name: Name of the model to use (e.g. llama3, mistral, gpt-4o)
            temperature: LLM creativity/randomness (0.0 for deterministic factual answers)
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> Any:
        if self.provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            logger.info(f"Initializing Ollama LLM: {self.model_name} at {base_url}")
            return ChatOllama(
                model=self.model_name,
                base_url=base_url,
                temperature=self.temperature
            )
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            logger.info(f"Initializing OpenAI LLM: {self.model_name}")
            return ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                temperature=self.temperature
            )
        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set.")
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = api_key
            
            # Map decommissioned or generic Groq model names to active models
            groq_model_map = {
                "llama3": "openai/gpt-oss-20b",
                "llama3-8b-8192": "openai/gpt-oss-20b",
                "llama3-70b-8192": "openai/gpt-oss-120b",
                "llama-3.1-8b-instant": "openai/gpt-oss-20b",
                "mixtral-8x7b-32768": "openai/gpt-oss-20b"
            }
            target_model = groq_model_map.get(self.model_name, self.model_name)
            logger.info(f"Initializing Groq LLM: {target_model}")
            
            # Groq exposes an OpenAI-compatible endpoint, so we reuse ChatOpenAI
            # and simply point base_url at Groq's API instead of OpenAI's.
            return ChatOpenAI(
                model=target_model,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
    def get_llm(self) -> Any:
        return self.llm

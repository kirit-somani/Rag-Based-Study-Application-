import json
import re
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.generation.llm_manager import LLMManager
from src.generation.prompt_templates import get_quiz_prompt_template, get_flashcard_prompt_template, get_summary_prompt_template
from src.vectorstore.vector_store_manager import VectorStoreManager
from src.retrieval.retrieval_pipeline import retrieve_context
from src.utils.logger import logger

def _clean_json_output(text: str) -> str:
    """Helper to strip markdown blocks if the LLM wraps json in ```json ... ```"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def generate_quiz(topic: str, vector_store: VectorStoreManager, llm_manager: LLMManager, num_questions: int = 5, source_filter: str = "All Documents") -> List[Dict]:
    """
    Generates MCQs based on the topic and context retrieved from the vector store.
    Returns a list of dicts: [{"question": "...", "options": ["A","B","C","D"], "answer": "A"}]
    """
    # Retrieve context for the topic
    retrieval_result = retrieve_context(
        query=topic,
        vector_store=vector_store,
        top_k=5,
        score_threshold=0.1,
        source_filter=source_filter
    )
    
    if not retrieval_result["results"]:
        raise ValueError(f"Could not find enough information about '{topic}' in the selected documents to generate a quiz.")
        
    context_str = ""
    for res in retrieval_result["results"]:
        context_str += res['text'] + "\n\n"
        
    prompt = get_quiz_prompt_template()
    chain = prompt | llm_manager.get_llm()
    
    response = chain.invoke({
        "topic": topic,
        "num_questions": str(num_questions),
        "context": context_str
    })
    
    try:
        json_str = _clean_json_output(response.content)
        quiz_data = json.loads(json_str)
        if isinstance(quiz_data, dict) and "questions" in quiz_data:
            return quiz_data["questions"]
        elif isinstance(quiz_data, list):
            return quiz_data
        else:
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM quiz output as JSON: {response.content}")
        raise ValueError("The LLM did not return a valid JSON format for the quiz.")


def generate_flashcards(topic: str, vector_store: VectorStoreManager, llm_manager: LLMManager, num_cards: int = 5, source_filter: str = "All Documents") -> List[Dict]:
    """
    Generates flashcards based on the topic and context retrieved from the vector store.
    Returns a list of dicts: [{"front": "...", "back": "..."}]
    """
    retrieval_result = retrieve_context(
        query=topic,
        vector_store=vector_store,
        top_k=5,
        score_threshold=0.1,
        source_filter=source_filter
    )
    
    if not retrieval_result["results"]:
        raise ValueError(f"Could not find enough information about '{topic}' in the selected documents to generate flashcards.")
        
    context_str = ""
    for res in retrieval_result["results"]:
        context_str += res['text'] + "\n\n"
        
    prompt = get_flashcard_prompt_template()
    chain = prompt | llm_manager.get_llm()
    
    response = chain.invoke({
        "topic": topic,
        "num_cards": str(num_cards),
        "context": context_str
    })
    
    try:
        json_str = _clean_json_output(response.content)
        flashcard_data = json.loads(json_str)
        if isinstance(flashcard_data, dict) and "flashcards" in flashcard_data:
            return flashcard_data["flashcards"]
        elif isinstance(flashcard_data, list):
            return flashcard_data
        else:
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM flashcard output as JSON: {response.content}")
        raise ValueError("The LLM did not return a valid JSON format for the flashcards.")

def generate_summary(topic: str, vector_store: VectorStoreManager, llm_manager: LLMManager, source_filter: str = "All Documents") -> str:
    """
    Generates a summary and key points for a topic based on the retrieved context.
    """
    retrieval_result = retrieve_context(
        query=topic,
        vector_store=vector_store,
        top_k=8,
        score_threshold=0.1,
        source_filter=source_filter
    )
    
    if not retrieval_result["results"]:
        raise ValueError(f"Could not find enough information about '{topic}' in the selected documents to generate a summary.")
        
    context_str = ""
    for res in retrieval_result["results"]:
        context_str += res['text'] + "\n\n"
        
    prompt = get_summary_prompt_template()
    chain = prompt | llm_manager.get_llm()
    
    response = chain.invoke({
        "topic": topic,
        "context": context_str
    })
    
    return response.content

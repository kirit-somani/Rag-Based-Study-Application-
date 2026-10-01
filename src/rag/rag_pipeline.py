import time
from typing import Dict, Any, Optional
from langchain_core.output_parsers import StrOutputParser
from src.retrieval.retrieval_pipeline import retrieve_context
from src.generation.prompt_templates import get_rag_prompt_template, format_context_for_prompt
from src.utils.logger import logger

def generate_rag_response(
    query: str,
    vector_store,
    llm_manager,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
    source_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline: Retrieval -> Prompt Formatting -> LLM Generation.
    
    Args:
        query: The user's question.
        vector_store: Instantiated VectorStoreManager.
        llm_manager: Instantiated LLMManager.
        top_k: Number of documents to retrieve.
        score_threshold: Minimum relevance score to keep a document.
        source_filter: Specific document to search in.
        
    Returns:
        A dictionary containing the answer, the retrieved context, and metrics.
    """
    # 1. Retrieve Context
    retrieval_response = retrieve_context(
        query=query,
        vector_store=vector_store,
        top_k=top_k,
        score_threshold=score_threshold,
        source_filter=source_filter
    )
    
    # Anti-hallucination fallback: Do not call LLM if no context is found
    if not retrieval_response["found"]:
        return {
            "answer": "I could not find enough information in the uploaded documents to answer this question.",
            "sources": [],
            "metrics": retrieval_response.get("metrics", {})
        }
        
    # 2. Format Context
    context_str = format_context_for_prompt(retrieval_response["results"])
    
    # 3. LLM Generation
    gen_start = time.time()
    prompt = get_rag_prompt_template()
    llm = llm_manager.get_llm()
    
    # LangChain LCEL (LangChain Expression Language) pipeline
    chain = prompt | llm | StrOutputParser()
    
    try:
        answer = chain.invoke({
            "context": context_str,
            "question": retrieval_response["query"]
        })
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        answer = "I'm sorry, I encountered an error while trying to generate the answer. Please check your LLM configuration."
        
    gen_time = time.time() - gen_start
    
    metrics = retrieval_response.get("metrics", {})
    metrics["generation_time"] = gen_time
    metrics["total_rag_time"] = metrics.get("total_time", 0) + gen_time
    
    return {
        "answer": answer,
        "sources": retrieval_response["results"],
        "metrics": metrics
    }

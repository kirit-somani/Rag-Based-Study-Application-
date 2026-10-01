from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = """You are an AI study assistant.

Answer the student's question using ONLY the information provided in the retrieved context below.

If the answer cannot be found in the context, clearly say:
"I could not find enough information in the uploaded documents to answer this question."

Do not invent facts.
Do not assume information that is not present in the retrieved context.
When possible, cite the source document and page number based on the metadata in the context.

Retrieved Context:
{context}
"""

def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Returns the ChatPromptTemplate for the RAG generation.
    """
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}")
    ])

def format_context_for_prompt(retrieved_results: list) -> str:
    """
    Formats the retrieved dictionary results into a string for the prompt.
    """
    context_str = ""
    for idx, res in enumerate(retrieved_results, 1):
        context_str += f"\n--- Source {idx} (File: {res['source']}, Page: {res['page']}) ---\n"
        context_str += res['text'] + "\n"
    return context_str


def get_quiz_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI tutor that creates multiple choice practice quizzes based strictly on provided context."),
        ("human", "Generate {num_questions} Multiple Choice Questions (MCQs) about \"{topic}\" based ONLY on the provided context below.\n\nReturn the response STRICTLY as a JSON array of objects in this format:\n[\n  {{\n    \"question\": \"Question text\",\n    \"options\": [\"A. ...\", \"B. ...\", \"C. ...\", \"D. ...\"],\n    \"answer\": \"A. ...\"\n  }}\n]\n\nDo not include any other text or markdown wrappers. Only return valid JSON.\n\nContext:\n{context}")
    ])

def get_flashcard_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI tutor that creates study flashcards based strictly on provided context."),
        ("human", "Generate {num_cards} flashcards about \"{topic}\" based ONLY on the provided context below.\n\nReturn the response STRICTLY as a JSON array of objects in this format:\n[\n  {{\n    \"front\": \"Question or term\",\n    \"back\": \"Answer or definition\"\n  }}\n]\n\nDo not include any other text or markdown wrappers. Only return valid JSON.\n\nContext:\n{context}")
    ])

def get_summary_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI tutor that creates chapter summaries and key points based strictly on provided context."),
        ("human", "Write a concise summary and extract key points about \"{topic}\" based ONLY on the provided context.\n\nFormat your response like this:\n\n### Summary\n(Summary paragraph)\n\n### Key Points\n- Key point 1\n- Key point 2\n\nContext:\n{context}")
    ])


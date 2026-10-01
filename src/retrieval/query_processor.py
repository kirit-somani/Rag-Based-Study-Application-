import re

def process_query(query: str, max_query_length: int = 1000) -> str:
    """
    Validates and cleans the user's natural language query.
    
    Args:
        query: The raw input string from the user.
        max_query_length: Maximum allowed length for a query.
        
    Returns:
        Cleaned query string.
        
    Raises:
        ValueError: If the query is empty, whitespace-only, or too long.
    """
    if not query:
        raise ValueError("Please enter a question.")
        
    # Remove excess whitespace but preserve intent
    cleaned = re.sub(r'\s+', ' ', query).strip()
    
    if not cleaned:
        raise ValueError("Please enter a question.")
        
    if len(cleaned) > max_query_length:
        raise ValueError("Your question is too long. Please shorten it.")
        
    return cleaned

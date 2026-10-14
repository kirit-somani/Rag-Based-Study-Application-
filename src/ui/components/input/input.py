import streamlit as st
from typing import List, Any

def text_input(label: str, placeholder: str = "", key: str = None) -> str:
    """Wrapper for styled text input."""
    return st.text_input(label, placeholder=placeholder, key=key)

import streamlit as st

def action_button(label: str, key: str = None, use_container_width: bool = False) -> bool:
    """Helper for rendering primary action button."""
    return st.button(label, key=key, use_container_width=use_container_width)

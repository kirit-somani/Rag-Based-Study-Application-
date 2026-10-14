import streamlit as st

def emit_html(html_str: str, container=st):
    """Safely emits HTML without blank lines to prevent Streamlit CommonMark parsing leaks."""
    compact_html = "".join(line.strip() for line in html_str.splitlines() if line.strip())
    container.markdown(compact_html, unsafe_allow_html=True)

def render_spacer(height_class: str = "spacer-28"):
    """Renders layout spacer without inline styling."""
    emit_html(f'<div class="{height_class}"></div>')

import streamlit as st
from typing import List
from ..base.base import emit_html

def render_system_pill(title: str, value: str, is_active: bool = False, dot_color: str = "dot-green"):
    """Renders a system status indicator pill in the sidebar."""
    active_cls = " system-pill-active" if is_active else ""
    val_cls = "system-pill-text-highlight" if is_active else "system-pill-text-normal"
    html = f"""
    <div class="system-pill{active_cls}">
        <span><span class="status-dot {dot_color}"></span> {title}</span>
        <span class="{val_cls}">{value}</span>
    </div>
    """
    emit_html(html, container=st.sidebar)

def render_prompt_chips(title: str, prompts: List[str]):
    """Renders interactive starter prompt chips."""
    chips_html = "".join([f'<span class="prompt-chip">{p}</span>' for p in prompts])
    html = f'<div class="prompts-container"><div class="prompts-title">{title}</div>{chips_html}</div>'
    emit_html(html)

def render_quiz_badge(index: int, total: int):
    """Renders assessment question badge."""
    emit_html(f'<span class="quiz-badge">Question {index} of {total}</span>')

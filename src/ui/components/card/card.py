import streamlit as st
from typing import Any
from ..base.base import emit_html

def render_doc_item(doc_name: str, description: str):
    """Renders active document card using external CSS classes."""
    html = f"""
    <div class="glass-card doc-item-card">
        <div class="doc-item-left">
            <span class="doc-item-icon">📄</span>
            <div>
                <div class="doc-item-title">{doc_name}</div>
                <div class="doc-item-desc">{description}</div>
            </div>
        </div>
    </div>
    """
    emit_html(html)

def render_citation_box(index: int, source: str, page: Any, quote: str):
    """Renders cited passage box using external CSS classes."""
    html = f"""
    <div class="citation-card">
        <div class="citation-meta">Passage {index} • {source} (Page {page})</div>
        <div class="citation-quote">"{quote}"</div>
    </div>
    """
    emit_html(html)

def render_quiz_card(index: int, total: int, question: str):
    """Renders quiz question header card using external CSS classes."""
    html = f"""
    <div class="quiz-card">
        <span class="quiz-badge">Question {index} of {total}</span>
        <div class="quiz-question-title">{question}</div>
    </div>
    """
    emit_html(html)

def render_score_card(score: int, total: int, percentage: int, label: str):
    """Renders quiz score report using external CSS classes."""
    html = f"""
    <div class="glass-card score-card-box">
        <div class="score-card-number">{score} / {total} ({percentage}%)</div>
        <div class="score-card-label">{label}</div>
    </div>
    """
    emit_html(html)

def render_flashcard_box(index: int, total: int, front: str, back: str):
    """Renders active-recall flashcard card using external CSS classes."""
    html = f"""
    <div class="flashcard-box">
        <div class="flashcard-tag">Card {index} of {total} • Concept Probe</div>
        <div class="flashcard-prompt">{front}</div>
        <div class="flashcard-answer">💡 <b>Answer:</b> {back}</div>
    </div>
    """
    emit_html(html)

def render_glass_card_open():
    """Opens a generic glass container card."""
    emit_html('<div class="glass-card summary-content-card">')

def render_glass_card_close():
    """Closes a generic glass container card."""
    emit_html('</div>')

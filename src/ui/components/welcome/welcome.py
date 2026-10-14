import streamlit as st
from typing import List, Dict
from ..base.base import emit_html
from ...text_provider import get_ui_text

def render_welcome_hero():
    """Renders the high-impact welcome hero banner."""
    tag = get_ui_text("welcome", "hero_badge")
    title = get_ui_text("welcome", "hero_title")
    subtitle = get_ui_text("welcome", "hero_subtitle")
    meta1 = get_ui_text("welcome", "stats_badge_1")
    meta2 = get_ui_text("welcome", "stats_badge_2")

    html = f"""
    <div class="welcome-hero">
        <div class="hero-tag">{tag}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle" style="margin: 0 auto 1.5rem auto;">{subtitle}</div>
        <div class="welcome-badge-strip">
            <span class="welcome-pill-meta">✨ {meta1}</span>
            <span class="welcome-pill-meta">⚡ {meta2}</span>
        </div>
    </div>
    """
    emit_html(html)

def render_feature_card_html(icon: str, title: str, desc: str):
    """Renders single feature card top visual."""
    html = f"""
    <div class="welcome-feature-card">
        <div class="welcome-feature-icon">{icon}</div>
        <div class="welcome-feature-title">{title}</div>
        <div class="welcome-feature-desc">{desc}</div>
    </div>
    """
    emit_html(html)

def render_steps_grid(steps: List[Dict[str, str]]):
    """Renders the 3-step workflow roadmap."""
    steps_html = "".join([
        f"""
        <div class="welcome-step-card">
            <div class="welcome-step-num">{s.get('step', '01')}</div>
            <div class="welcome-step-title">{s.get('title', '')}</div>
            <div class="welcome-step-desc">{s.get('desc', '')}</div>
        </div>
        """
        for s in steps
    ])
    html = f'<div class="welcome-steps-grid">{steps_html}</div>'
    emit_html(html)

def render_section_title(title: str):
    """Renders section header with proper typography."""
    html = f'<div class="welcome-section-header">{title}</div>'
    emit_html(html)

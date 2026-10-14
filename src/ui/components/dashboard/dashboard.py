import streamlit as st
from typing import List, Dict
from ..base.base import emit_html

def render_brand_header(brand_name: str, subtitle: str, logo_glyph: str = "⚡"):
    """Renders the top branding header with logo glyph."""
    html = f"""
    <div class="brand-container">
        <div class="brand-logo">{logo_glyph}</div>
        <div>
            <div class="brand-title">{brand_name}</div>
            <div class="brand-subtitle">{subtitle}</div>
        </div>
    </div>
    """
    emit_html(html, container=st.sidebar)

def render_hero_banner(tag: str, title: str, subtitle: str):
    """Renders page section hero banner."""
    html = f"""
    <div class="hero-banner">
        <div class="hero-tag">{tag}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """
    emit_html(html)

def render_stat_grid(stats: List[Dict[str, str]]):
    """Renders multi-column metrics grid."""
    cards_html = "".join([
        f'<div class="stat-card"><div class="stat-value">{s.get("value", "")}</div><div class="stat-label">{s.get("label", "")}</div></div>'
        for s in stats
    ])
    html = f'<div class="stat-grid">{cards_html}</div>'
    emit_html(html)

def render_footer(text: str):
    """Renders copyright / version tag in sidebar footer."""
    html = f'<div class="footer-tag">{text}</div>'
    emit_html(html, container=st.sidebar)

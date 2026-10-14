import re
from pathlib import Path
import streamlit as st

COMPONENTS_DIR = Path(__file__).resolve().parent / "components"
STATIC_CSS_PATH = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "style.css"

DARK_TOKENS = """
:root {
    --bg-base: #06070d;
    --bg-surface: #0e111a;
    --bg-elevated: #151926;
    --bg-card: rgba(18, 22, 36, 0.75);
    --border-glass: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(139, 92, 246, 0.35);
    --accent-primary: #a78bfa;
    --accent-secondary: #06b6d4;
    --accent-gradient: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
    --title-gradient: linear-gradient(135deg, #ffffff 20%, #cbd5e1 100%);
    --glow-purple: 0 0 25px rgba(139, 92, 246, 0.25);
    --text-high: #f8fafc;
    --text-med: #cbd5e1;
    --text-dim: #94a3b8;
    --mesh-bg-1: rgba(139, 92, 246, 0.08);
    --mesh-bg-2: rgba(6, 182, 212, 0.06);
    --header-bg: rgba(6, 7, 13, 0.85);
    --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    --hover-bg: rgba(255, 255, 255, 0.06);
    --btn-bg: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
    --btn-color: #ffffff;
    --btn-border: rgba(139, 92, 246, 0.5);
    --btn-shadow: 0 4px 18px rgba(124, 58, 237, 0.4);
    --btn-hover-bg: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
}
"""

LIGHT_TOKENS = """
:root {
    --bg-base: #f8fafc;
    --bg-surface: #ffffff;
    --bg-elevated: #f1f5f9;
    --bg-card: #ffffff;
    --border-glass: #e2e8f0;
    --border-accent: rgba(124, 58, 237, 0.35);
    --accent-primary: #7c3aed;
    --accent-secondary: #0284c7;
    --accent-gradient: linear-gradient(135deg, #7c3aed 0%, #2563eb 50%, #0891b2 100%);
    --title-gradient: linear-gradient(135deg, #0f172a 20%, #334155 100%);
    --glow-purple: 0 0 20px rgba(124, 58, 237, 0.15);
    --text-high: #0f172a;
    --text-med: #334155;
    --text-dim: #64748b;
    --mesh-bg-1: rgba(124, 58, 237, 0.04);
    --mesh-bg-2: rgba(8, 145, 178, 0.04);
    --header-bg: rgba(255, 255, 255, 0.95);
    --card-shadow: 0 4px 20px rgba(15, 23, 42, 0.07);
    --hover-bg: rgba(15, 23, 42, 0.05);
    --btn-bg: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
    --btn-color: #ffffff;
    --btn-border: #6d28d9;
    --btn-shadow: 0 4px 16px rgba(124, 58, 237, 0.35);
    --btn-hover-bg: linear-gradient(135deg, #6d28d9 0%, #4338ca 100%);
}
"""

def collect_component_styles(theme_mode: str = "dark") -> str:
    """
    Collects CSS from all modular component folders and prepends the active theme tokens.
    """
    theme_tokens = LIGHT_TOKENS if theme_mode.lower() == "light" else DARK_TOKENS
    css_parts = [theme_tokens]

    base_var_file = COMPONENTS_DIR / "base" / "variables.css"
    if base_var_file.exists():
        with open(base_var_file, "r", encoding="utf-8") as f:
            css_parts.append(f.read())

    for css_file in sorted(COMPONENTS_DIR.rglob("*.css")):
        if css_file != base_var_file:
            with open(css_file, "r", encoding="utf-8") as f:
                css_parts.append(f.read())

    if len(css_parts) == 1 and STATIC_CSS_PATH.exists():
        with open(STATIC_CSS_PATH, "r", encoding="utf-8") as f:
            css_parts.append(f.read())

    raw_css = "\n".join(css_parts)

    css_no_comments = re.sub(r"/\*.*?\*/", "", raw_css, flags=re.DOTALL)

    return " ".join(css_no_comments.split())

def inject_app_styles(theme_mode: str = "dark"):
    """Injects consolidated and theme-adapted component styles into the Streamlit DOM."""
    css_minified = collect_component_styles(theme_mode=theme_mode)
    font_link = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap">'
    st.markdown(f"{font_link}<style>{css_minified}</style>", unsafe_allow_html=True)

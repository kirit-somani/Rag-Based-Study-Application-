from ..base.base import emit_html

def render_loading_state(message: str):
    """Renders sleek loading card with animated pulse."""
    html = f"""
    <div class="loading-container">
        <div class="spinner-pulse"></div>
        <span>{message}</span>
    </div>
    """
    emit_html(html)

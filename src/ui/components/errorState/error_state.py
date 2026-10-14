from ..base.base import emit_html

def render_error_state(title: str, description: str):
    """Renders customized error alert card."""
    html = f"""
    <div class="error-state-card">
        <div class="error-state-title">{title}</div>
        <div class="error-state-desc">{description}</div>
    </div>
    """
    emit_html(html)

from ..base.base import emit_html

def render_notification(message: str, icon: str = "✨"):
    """Renders non-intrusive alert toast banner."""
    html = f"""
    <div class="notification-toast">
        <span>{icon}</span>
        <span>{message}</span>
    </div>
    """
    emit_html(html)

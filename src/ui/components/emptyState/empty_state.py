from ..base.base import emit_html

def render_empty_state(title: str, subtitle: str, icon: str = "📖"):
    """Renders formatted empty state placeholder box."""
    html = f"""
    <div class="glass-card empty-state-box">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-subtitle">{subtitle}</div>
    </div>
    """
    emit_html(html)

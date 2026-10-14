from ..base.base import emit_html

def render_task_item(title: str, status: str = "Active"):
    """Renders background task card."""
    html = f"""
    <div class="task-item-card">
        <span class="task-item-title">{title}</span>
        <span class="task-badge-active">{status}</span>
    </div>
    """
    emit_html(html)

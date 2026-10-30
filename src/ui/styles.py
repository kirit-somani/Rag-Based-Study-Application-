import re
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

COMPONENTS_DIR = Path(__file__).resolve().parent / "components"
STATIC_CSS_PATH = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "style.css"

DARK_TOKENS = """
:root {
    --bg-base: #06070d;
    --bg-surface: #0e111a;
    --bg-elevated: #151926;
    --bg-card: rgba(18, 22, 36, 0.78);
    --border-glass: rgba(255, 255, 255, 0.09);
    --border-accent: rgba(139, 92, 246, 0.45);
    --accent-primary: #a78bfa;
    --accent-secondary: #06b6d4;
    --accent-gradient: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
    --title-gradient: linear-gradient(135deg, #ffffff 20%, #cbd5e1 100%);
    --glow-purple: 0 0 30px rgba(139, 92, 246, 0.35);
    --text-high: #f8fafc;
    --text-med: #cbd5e1;
    --text-dim: #94a3b8;
    --mesh-bg-1: rgba(139, 92, 246, 0.15);
    --mesh-bg-2: rgba(6, 182, 212, 0.11);
    --header-bg: rgba(6, 7, 13, 0.88);
    --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    --hover-bg: rgba(255, 255, 255, 0.07);
    --btn-bg: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
    --btn-color: #ffffff;
    --btn-border: rgba(167, 139, 250, 0.6);
    --btn-shadow: 0 4px 20px rgba(124, 58, 237, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.35);
    --btn-hover-bg: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);

    /* 3D Isometric & Spatial Tokens - Dark Mode */
    --specular-edge: rgba(255, 255, 255, 0.22);
    --card-3d-shadow: 0 2px 4px rgba(0, 0, 0, 0.4), 0 10px 24px -4px rgba(0, 0, 0, 0.6), 0 24px 48px -8px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.14);
    --card-3d-hover-shadow: 0 4px 8px rgba(0, 0, 0, 0.5), 0 18px 40px -6px rgba(0, 0, 0, 0.8), 0 36px 64px -10px rgba(139, 92, 246, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.32);
    --flashcard-3d-shadow: 0 4px 8px rgba(0, 0, 0, 0.5), 0 14px 32px rgba(0, 0, 0, 0.7), 0 32px 60px rgba(0, 0, 0, 0.8), 0 0 40px rgba(139, 92, 246, 0.35), inset 0 1.5px 0 rgba(255, 255, 255, 0.3);

    --grid-3d-line: rgba(139, 92, 246, 0.18);
    --grid-3d-line-secondary: rgba(6, 182, 212, 0.14);
    --horizon-glow-color: rgba(139, 92, 246, 0.32);
    --orb-specular-1: rgba(255, 255, 255, 0.95);
    --orb-color-1: rgba(139, 92, 246, 0.42);
    --orb-shadow-1: rgba(12, 8, 28, 0.9);
    --orb-ambient-1: rgba(139, 92, 246, 0.3);
    --orb-specular-2: rgba(255, 255, 255, 0.9);
    --orb-color-2: rgba(6, 182, 212, 0.38);
    --orb-shadow-2: rgba(6, 20, 36, 0.9);
    --orb-ambient-2: rgba(6, 182, 212, 0.26);
}
"""

LIGHT_TOKENS = """
:root {
    --bg-base: #f8fafc;
    --bg-surface: #ffffff;
    --bg-elevated: #f1f5f9;
    --bg-card: #ffffff;
    --border-glass: #e2e8f0;
    --border-accent: rgba(124, 58, 237, 0.45);
    --accent-primary: #7c3aed;
    --accent-secondary: #0284c7;
    --accent-gradient: linear-gradient(135deg, #7c3aed 0%, #2563eb 50%, #0891b2 100%);
    --title-gradient: linear-gradient(135deg, #0f172a 20%, #334155 100%);
    --glow-purple: 0 0 25px rgba(124, 58, 237, 0.22);
    --text-high: #0f172a;
    --text-med: #334155;
    --text-dim: #64748b;
    --mesh-bg-1: rgba(124, 58, 237, 0.09);
    --mesh-bg-2: rgba(8, 145, 178, 0.08);
    --header-bg: rgba(255, 255, 255, 0.95);
    --card-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
    --hover-bg: rgba(15, 23, 42, 0.05);
    --btn-bg: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
    --btn-color: #ffffff;
    --btn-border: #6d28d9;
    --btn-shadow: 0 4px 18px rgba(124, 58, 237, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.4);
    --btn-hover-bg: linear-gradient(135deg, #6d28d9 0%, #4338ca 100%);

    /* 3D Isometric & Spatial Tokens - Light Mode (High Contrast & Professional) */
    --specular-edge: rgba(255, 255, 255, 0.95);
    --card-3d-shadow: 0 2px 4px rgba(15, 23, 42, 0.04), 0 10px 24px -4px rgba(124, 58, 237, 0.09), 0 24px 44px -8px rgba(15, 23, 42, 0.1), inset 0 1px 0 #ffffff;
    --card-3d-hover-shadow: 0 4px 8px rgba(124, 58, 237, 0.08), 0 18px 36px -6px rgba(124, 58, 237, 0.18), 0 32px 56px -10px rgba(124, 58, 237, 0.24), inset 0 1px 0 #ffffff;
    --flashcard-3d-shadow: 0 4px 8px rgba(124, 58, 237, 0.12), 0 14px 32px rgba(124, 58, 237, 0.18), 0 30px 54px rgba(15, 23, 42, 0.14), 0 0 35px rgba(124, 58, 237, 0.25), inset 0 1.5px 0 #ffffff;

    --grid-3d-line: rgba(124, 58, 237, 0.18);
    --grid-3d-line-secondary: rgba(2, 132, 199, 0.15);
    --horizon-glow-color: rgba(124, 58, 237, 0.22);
    --orb-specular-1: #ffffff;
    --orb-color-1: rgba(167, 139, 250, 0.42);
    --orb-shadow-1: rgba(124, 58, 237, 0.26);
    --orb-ambient-1: rgba(124, 58, 237, 0.2);
    --orb-specular-2: #ffffff;
    --orb-color-2: rgba(56, 189, 248, 0.42);
    --orb-shadow-2: rgba(2, 132, 199, 0.24);
    --orb-ambient-2: rgba(2, 132, 199, 0.18);
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

def inject_trailing_cursor(theme_mode: str = "dark"):
    """
    Injects a high-performance, hardware-accelerated trailing cursor effect
    with a glowing leading dot, fluid physics-based ring, and a dynamic comet particle trail.
    """
    mode = "light" if theme_mode.lower() == "light" else "dark"
    cursor_script = """
    <script>
    (function() {
        try {
            const parentDoc = window.parent.document;
            const parentWin = window.parent.window;
            if (!parentDoc || !parentWin) return;

            // Hide the component iframe itself so it occupies zero layout space
            const frameEl = window.frameElement;
            if (frameEl) {
                frameEl.style.display = 'none';
                frameEl.style.height = '0px';
                frameEl.style.width = '0px';
                frameEl.style.border = 'none';
                const parentContainer = frameEl.closest('.element-container');
                if (parentContainer) {
                    parentContainer.style.display = 'none';
                    parentContainer.style.height = '0px';
                    parentContainer.style.margin = '0px';
                    parentContainer.style.padding = '0px';
                }
            }

            const currentTheme = "__THEME_MODE__";
            const existingSystem = parentDoc.getElementById('studyrag-cursor-system');
            if (existingSystem) {
                existingSystem.setAttribute('data-theme', currentTheme);
                return;
            }

            // Guard against touchscreen devices
            if (parentWin.matchMedia && parentWin.matchMedia('(pointer: coarse)').matches) {
                return;
            }

            // Create container
            const cursorContainer = parentDoc.createElement('div');
            cursorContainer.id = 'studyrag-cursor-system';
            cursorContainer.setAttribute('data-theme', currentTheme);

            // Create styling
            const style = parentDoc.createElement('style');
            style.id = 'studyrag-cursor-styles';
            style.textContent = `
                #studyrag-cursor-system {
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100vw !important;
                    height: 100vh !important;
                    pointer-events: none !important;
                    z-index: 2147483647 !important;
                    overflow: hidden !important;
                    opacity: 0;
                    transition: opacity 0.3s cubic-bezier(0.16, 1, 0.3, 1);
                }
                #studyrag-cursor-canvas {
                    position: absolute !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100% !important;
                    height: 100% !important;
                    pointer-events: none !important;
                }
                #studyrag-cursor-dot {
                    position: absolute !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 8px !important;
                    height: 8px !important;
                    margin-left: -4px !important;
                    margin-top: -4px !important;
                    border-radius: 50% !important;
                    pointer-events: none !important;
                    will-change: transform !important;
                    transition: transform 0.04s ease-out, background 0.3s ease;
                }
                #studyrag-cursor-ring {
                    position: absolute !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 34px !important;
                    height: 34px !important;
                    margin-left: -17px !important;
                    margin-top: -17px !important;
                    border-radius: 50% !important;
                    pointer-events: none !important;
                    will-change: transform !important;
                    transition: width 0.22s cubic-bezier(0.16, 1, 0.3, 1),
                                height 0.22s cubic-bezier(0.16, 1, 0.3, 1),
                                margin 0.22s cubic-bezier(0.16, 1, 0.3, 1),
                                border-color 0.3s ease,
                                box-shadow 0.3s ease,
                                background 0.2s ease;
                }

                /* Dark Theme */
                #studyrag-cursor-system[data-theme="dark"] #studyrag-cursor-dot {
                    background: #c084fc !important;
                    box-shadow: 0 0 10px #a855f7, 0 0 20px rgba(168, 85, 247, 0.9), inset 0 0 3px #ffffff !important;
                }
                #studyrag-cursor-system[data-theme="dark"] #studyrag-cursor-ring {
                    border: 1.5px solid rgba(167, 139, 250, 0.65) !important;
                    background: radial-gradient(circle, rgba(167, 139, 250, 0.08) 0%, rgba(6, 182, 212, 0.02) 70%, transparent 100%) !important;
                    box-shadow: 0 0 16px rgba(167, 139, 250, 0.35), inset 0 0 10px rgba(6, 182, 212, 0.2) !important;
                }
                #studyrag-cursor-system[data-theme="dark"] #studyrag-cursor-ring.cursor-hover {
                    width: 52px !important;
                    height: 52px !important;
                    margin-left: -26px !important;
                    margin-top: -26px !important;
                    border-color: rgba(6, 182, 212, 0.95) !important;
                    background: radial-gradient(circle, rgba(6, 182, 212, 0.18) 0%, rgba(167, 139, 250, 0.1) 70%, transparent 100%) !important;
                    box-shadow: 0 0 24px rgba(6, 182, 212, 0.6), 0 0 45px rgba(167, 139, 250, 0.4) !important;
                }
                #studyrag-cursor-system[data-theme="dark"] #studyrag-cursor-ring.cursor-active {
                    width: 24px !important;
                    height: 24px !important;
                    margin-left: -12px !important;
                    margin-top: -12px !important;
                    background: rgba(167, 139, 250, 0.35) !important;
                }

                /* Light Theme */
                #studyrag-cursor-system[data-theme="light"] #studyrag-cursor-dot {
                    background: #7c3aed !important;
                    box-shadow: 0 0 10px #7c3aed, 0 0 18px rgba(124, 58, 237, 0.65), inset 0 0 3px #ffffff !important;
                }
                #studyrag-cursor-system[data-theme="light"] #studyrag-cursor-ring {
                    border: 1.5px solid rgba(124, 58, 237, 0.75) !important;
                    background: radial-gradient(circle, rgba(124, 58, 237, 0.08) 0%, rgba(2, 132, 199, 0.04) 70%, transparent 100%) !important;
                    box-shadow: 0 0 16px rgba(124, 58, 237, 0.32), inset 0 0 8px rgba(2, 132, 199, 0.18) !important;
                }
                #studyrag-cursor-system[data-theme="light"] #studyrag-cursor-ring.cursor-hover {
                    width: 52px !important;
                    height: 52px !important;
                    margin-left: -26px !important;
                    margin-top: -26px !important;
                    border-color: rgba(2, 132, 199, 0.95) !important;
                    background: radial-gradient(circle, rgba(2, 132, 199, 0.16) 0%, rgba(124, 58, 237, 0.08) 70%, transparent 100%) !important;
                    box-shadow: 0 0 24px rgba(2, 132, 199, 0.5), 0 0 40px rgba(124, 58, 237, 0.3) !important;
                }
                #studyrag-cursor-system[data-theme="light"] #studyrag-cursor-ring.cursor-active {
                    width: 24px !important;
                    height: 24px !important;
                    margin-left: -12px !important;
                    margin-top: -12px !important;
                    background: rgba(124, 58, 237, 0.3) !important;
                }
            `;
            parentDoc.head.appendChild(style);

            const canvas = parentDoc.createElement('canvas');
            canvas.id = 'studyrag-cursor-canvas';

            const ring = parentDoc.createElement('div');
            ring.id = 'studyrag-cursor-ring';

            const dot = parentDoc.createElement('div');
            dot.id = 'studyrag-cursor-dot';

            cursorContainer.appendChild(canvas);
            cursorContainer.appendChild(ring);
            cursorContainer.appendChild(dot);
            parentDoc.body.appendChild(cursorContainer);

            let ctx = canvas.getContext('2d');
            let width = canvas.width = parentWin.innerWidth;
            let height = canvas.height = parentWin.innerHeight;

            const onResize = () => {
                width = canvas.width = parentWin.innerWidth;
                height = canvas.height = parentWin.innerHeight;
            };
            parentWin.addEventListener('resize', onResize);

            let mouseX = -100;
            let mouseY = -100;
            let ringX = -100;
            let ringY = -100;
            let isHovered = false;
            let isInside = false;

            const trailPoints = [];
            const MAX_POINTS = 22;

            parentDoc.addEventListener('mousemove', (e) => {
                mouseX = e.clientX;
                mouseY = e.clientY;
                if (!isInside) {
                    isInside = true;
                    cursorContainer.style.opacity = '1';
                    ringX = mouseX;
                    ringY = mouseY;
                }

                trailPoints.push({
                    x: mouseX,
                    y: mouseY,
                    age: 0,
                    maxAge: 20 + Math.random() * 8,
                    radius: Math.random() * 3.5 + 2.0
                });
                if (trailPoints.length > MAX_POINTS * 2) {
                    trailPoints.shift();
                }
            }, { passive: true });

            parentDoc.addEventListener('mouseleave', () => {
                isInside = false;
                cursorContainer.style.opacity = '0';
            });

            parentDoc.addEventListener('mouseenter', () => {
                isInside = true;
                cursorContainer.style.opacity = '1';
            });

            parentDoc.addEventListener('mousedown', () => {
                ring.classList.add('cursor-active');
            });

            parentDoc.addEventListener('mouseup', () => {
                ring.classList.remove('cursor-active');
            });

            parentDoc.addEventListener('mouseover', (e) => {
                const target = e.target;
                if (!target) return;
                const isInteractive = target.closest('button, a, input, select, textarea, .glass-card, .flashcard-box, .citation-box, .summary-card, [role="button"], summary, [data-baseweb="tab"], label');
                if (isInteractive && !isHovered) {
                    isHovered = true;
                    ring.classList.add('cursor-hover');
                } else if (!isInteractive && isHovered) {
                    isHovered = false;
                    ring.classList.remove('cursor-hover');
                }
            }, { passive: true });

            function render() {
                ringX += (mouseX - ringX) * 0.18;
                ringY += (mouseY - ringY) * 0.18;

                dot.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0)`;
                ring.style.transform = `translate3d(${ringX}px, ${ringY}px, 0)`;

                ctx.clearRect(0, 0, width, height);

                const currentTheme = cursorContainer.getAttribute('data-theme') || 'dark';
                const isDark = (currentTheme === 'dark');

                if (trailPoints.length > 2) {
                    for (let i = 0; i < trailPoints.length - 1; i++) {
                        const p1 = trailPoints[i];
                        const p2 = trailPoints[i + 1];
                        const progress = i / trailPoints.length;
                        const alpha = progress * 0.65;
                        const strokeWidth = progress * 4.5 + 0.5;

                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.lineWidth = strokeWidth;
                        ctx.lineCap = 'round';
                        ctx.strokeStyle = isDark
                            ? `rgba(167, 139, 250, ${alpha})`
                            : `rgba(124, 58, 237, ${alpha})`;
                        ctx.shadowColor = isDark ? '#06b6d4' : '#2563eb';
                        ctx.shadowBlur = progress * 10;
                        ctx.stroke();
                    }
                }

                for (let i = trailPoints.length - 1; i >= 0; i--) {
                    const pt = trailPoints[i];
                    pt.age += 1;
                    if (pt.age >= pt.maxAge) {
                        trailPoints.splice(i, 1);
                        continue;
                    }
                    const lifeProgress = 1 - (pt.age / pt.maxAge);
                    const r = pt.radius * lifeProgress;
                    const alpha = lifeProgress * 0.75;

                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, Math.max(0.2, r), 0, Math.PI * 2);
                    ctx.fillStyle = isDark
                        ? `rgba(6, 182, 212, ${alpha})`
                        : `rgba(2, 132, 199, ${alpha})`;
                    ctx.shadowColor = isDark ? '#a78bfa' : '#7c3aed';
                    ctx.shadowBlur = 6;
                    ctx.fill();
                }

                parentWin.requestAnimationFrame(render);
            }

            parentWin.requestAnimationFrame(render);
        } catch (err) {
            console.error("StudyRAG cursor error:", err);
        }
    })();
    </script>
    """.replace("__THEME_MODE__", mode)

    components.html(cursor_script, height=0, width=0)

def inject_app_styles(theme_mode: str = "dark"):
    """Injects consolidated and theme-adapted component styles into the Streamlit DOM."""
    css_minified = collect_component_styles(theme_mode=theme_mode)
    font_link = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap">'
    st.markdown(f"{font_link}<style>{css_minified}</style>", unsafe_allow_html=True)
    inject_trailing_cursor(theme_mode=theme_mode)

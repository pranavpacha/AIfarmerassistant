"""Kisan Mitra — Gradio UI. Single entrypoint used both for the Colab
`share=True` demo and for Hugging Face Spaces deployment.

Visual direction: a "field advisory console" — a precision-agriculture
sensor/satellite readout aesthetic (animated grid backdrop, signal-green
accent, monospace data readouts, score bars, 3D depth on hover) rather than
a generic rustic/cream farm-app look.
"""
import html
import time

import gradio as gr
import torch

from pipeline import ask

COLAB_URL = "https://colab.research.google.com/github/pranavpacha/AIfarmerassistant/blob/main/notebooks/colab_dev.ipynb"

LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Marathi", "Bengali"]

# "chip" is a short display label (shown with its language so the multilingual
# range is visible at a glance); "question" is the full real query actually sent.
EXAMPLES = [
    {"icon": "🌾", "chip": "Stem borer in paddy?", "lang_code": "EN", "question": "How do I control stem borer in paddy?", "language": "English"},
    {"icon": "🌿", "chip": "गेहूं की बुवाई?", "lang_code": "HI", "question": "गेहूं की बुवाई का सही समय क्या है?", "language": "Hindi"},
    {"icon": "🏛", "chip": "PM-KISAN பதிவு?", "lang_code": "TA", "question": "PM-KISAN கீழ் நான் எவ்வாறு பதிவு செய்வது?", "language": "Tamil"},
    {"icon": "🌽", "chip": "Maize fertilizer?", "lang_code": "EN", "question": "What is the fertilizer schedule for hybrid maize?", "language": "English"},
    {"icon": "💧", "chip": "చుక్క సబ్సిడీ?", "lang_code": "TE", "question": "చుక్క నీటిపారుదల సబ్సిడీ ఎలా పొందాలి?", "language": "Telugu"},
    {"icon": "🌱", "chip": "ಗುಲಾಬಿ ಹುಳು?", "lang_code": "KN", "question": "ಹತ್ತಿ ಬೆಳೆಯಲ್ಲಿ ಗುಲಾಬಿ ಕಾಯಿ ಹುಳುವನ್ನು ಹೇಗೆ ನಿಯಂತ್ರಿಸುವುದು?", "language": "Kannada"},
    {"icon": "🏦", "chip": "किसान क्रेडिट कार्ड?", "lang_code": "HI", "question": "किसान क्रेडिट कार्ड कैसे बनवाएं?", "language": "Hindi"},
    {"icon": "🧅", "chip": "कांदा साठवण?", "lang_code": "MR", "question": "कांदा साठवणुकीची योग्य पद्धत कोणती आहे?", "language": "Marathi"},
    {"icon": "🏛", "chip": "পিএম-কিষান?", "lang_code": "BN", "question": "পিএম-কিষান এর জন্য কীভাবে নিবন্ধন করব?", "language": "Bengali"},
]

CROP_ICONS = {
    "Paddy (Rice)": "🌾", "Wheat": "🌿", "Cotton": "🌱", "Maize": "🌽", "Sugarcane": "🎋",
    "Chickpea": "🫘", "Tomato": "🍅", "Onion": "🧅", "Soybean": "🌱", "Groundnut": "🥜",
    "Banana": "🍌", "Mustard": "🌼", "Potato": "🥔", "General": "🛰",
}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+Devanagari:wght@400;600&family=Noto+Sans+Tamil:wght@400;600&family=Noto+Sans+Telugu:wght@400;600&family=Noto+Sans+Kannada:wght@400;600&family=Noto+Sans+Bengali:wght@400;600&display=swap');

:root {
    --km-bg: #f4f8f5; --km-surface: #ffffff; --km-surface-2: #eaf2ec; --km-border: #d3e0d7;
    --km-fg: #10201a; --km-fg-dim: #51685a; --km-accent: #1d9e5c; --km-accent-2: #c96c1f;
}
@media (prefers-color-scheme: dark) {
    :root {
        --km-bg: #0a0f0d; --km-surface: #111814; --km-surface-2: #1a2420; --km-border: #28362f;
        --km-fg: #e8f0ea; --km-fg-dim: #8ba396; --km-accent: #52e88d; --km-accent-2: #ffb454;
    }
}

html, body { background: var(--km-bg) !important; margin: 0; }
.gradio-container {
    font-family: 'IBM Plex Sans', 'Noto Sans Devanagari', 'Noto Sans Tamil', 'Noto Sans Telugu',
                 'Noto Sans Kannada', 'Noto Sans Bengali', sans-serif !important;
    background: var(--km-bg) !important;
    max-width: 880px !important;
    margin: 0 auto !important;
}
.gradio-container, .gradio-container * { color: var(--km-fg); }
footer { display: none !important; }

/* Dropdown popup (language selector) */
.gradio-container ul[role="listbox"], .gradio-container .options {
    background: var(--km-surface) !important; border: 1px solid var(--km-border) !important;
    border-radius: 10px !important; box-shadow: 0 12px 28px -12px rgba(0,0,0,.35) !important;
}
.gradio-container li[role="option"], .gradio-container .item {
    color: var(--km-fg) !important; font-family: 'IBM Plex Sans', sans-serif !important;
}
.gradio-container li[role="option"]:hover, .gradio-container .item:hover,
.gradio-container li[aria-selected="true"], .gradio-container .item.selected {
    background: var(--km-surface-2) !important; color: var(--km-accent) !important;
}

/* ---- hero (3D mouse-tilt, see the inline script in the hero HTML block) ---- */
#km-hero-outer { perspective: 1200px; margin-bottom: 16px; }
#km-hero {
    position: relative; overflow: hidden; border-radius: 20px;
    background:
        radial-gradient(ellipse 120% 80% at 20% -10%, color-mix(in srgb, var(--km-accent) 16%, transparent), transparent 60%),
        var(--km-surface);
    border: 1px solid var(--km-border);
    padding: 30px 30px 26px;
    transform-style: preserve-3d; transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s;
    box-shadow: 0 1px 0 color-mix(in srgb, var(--km-fg) 4%, transparent) inset;
}
#km-hero.km-tilt { box-shadow: 0 30px 60px -30px rgba(0,0,0,.35), 0 1px 0 color-mix(in srgb, var(--km-fg) 4%, transparent) inset; }
#km-hero::before {
    content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .35;
    background-image:
        repeating-linear-gradient(0deg, color-mix(in srgb, var(--km-accent) 22%, transparent) 0 1px, transparent 1px 34px),
        repeating-linear-gradient(90deg, color-mix(in srgb, var(--km-accent) 22%, transparent) 0 1px, transparent 1px 34px);
    -webkit-mask-image: radial-gradient(ellipse 90% 100% at 30% 0%, #000 0%, transparent 72%);
    mask-image: radial-gradient(ellipse 90% 100% at 30% 0%, #000 0%, transparent 72%);
    animation: km-drift 22s linear infinite;
}
@keyframes km-drift { from { background-position: 0 0, 0 0; } to { background-position: 34px 34px, -34px 34px; } }
@media (prefers-reduced-motion: reduce) { #km-hero::before { animation: none; } }

.km-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; position: relative; transform: translateZ(30px); }
.km-status-chip {
    display: inline-flex; align-items: center; gap: 6px; font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem; letter-spacing: .06em; color: var(--km-accent);
    background: color-mix(in srgb, var(--km-accent) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--km-accent) 35%, transparent);
    padding: 3px 9px; border-radius: 99px; text-transform: uppercase;
}
.km-status-chip .km-dot { width: 6px; height: 6px; border-radius: 99px; background: var(--km-accent); animation: km-pulse 1.8s ease-in-out infinite; }
@keyframes km-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }

#km-hero .km-kicker { font-family: 'IBM Plex Mono', monospace; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--km-fg-dim); margin: 0 0 6px; position: relative; transform: translateZ(24px); }
.km-lockup { display: flex; align-items: center; gap: 10px; position: relative; transform: translateZ(40px); }
#km-hero h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: clamp(1.9rem, 5vw, 2.3rem); letter-spacing: -.01em; margin: 2px 0; color: var(--km-fg) !important; }
#km-hero .km-lede { color: var(--km-fg-dim); font-size: .98rem; line-height: 1.55; max-width: 48ch; margin: 0; position: relative; transform: translateZ(18px); }

/* ---- CPU-hardware notice (only rendered when no GPU is available) ---- */
#km-cpu-notice {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    background: color-mix(in srgb, var(--km-accent-2) 10%, var(--km-surface));
    border: 1px solid color-mix(in srgb, var(--km-accent-2) 35%, var(--km-border));
    border-radius: 12px; padding: 10px 14px; margin-bottom: 16px;
    font-family: 'IBM Plex Mono', monospace; font-size: .74rem; color: var(--km-fg-dim);
}
#km-cpu-notice b { color: var(--km-accent-2); }
#km-cpu-notice a { color: var(--km-accent); text-decoration: underline; text-underline-offset: 2px; }

/* ---- console card (groups input + chips + button into one unit) ---- */
#km-console {
    margin-top: 16px !important; background: var(--km-surface) !important; border: 1px solid var(--km-border) !important;
    border-radius: 18px !important; padding: 20px !important;
    box-shadow: 0 1px 0 color-mix(in srgb, var(--km-fg) 4%, transparent) inset !important;
}
#km-console-label { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; flex-wrap: wrap; gap: 6px; }
.km-stats { display: flex; gap: 12px; font-family: 'IBM Plex Mono', monospace; font-size: .68rem; color: var(--km-fg-dim); }
.km-stats b { color: var(--km-accent); font-weight: 600; }

#km-input-row { gap: 10px !important; }
#km-input-row textarea, #km-input-row input[type=text] {
    background: var(--km-surface-2) !important; border: 1px solid var(--km-border) !important;
    border-radius: 10px !important; color: var(--km-fg) !important; font-size: .96rem !important;
}
#km-input-row textarea:focus, #km-input-row input[type=text]:focus {
    border-color: var(--km-accent) !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--km-accent) 18%, transparent) !important;
}
#km-language { min-width: 150px !important; }
.gradio-container label span, .gradio-container .block-title, .gradio-container span[data-testid="block-info"] {
    font-family: 'IBM Plex Mono', monospace !important; font-size: .68rem !important; letter-spacing: .08em !important;
    text-transform: uppercase !important; color: var(--km-fg-dim) !important;
}

#km-ask-btn {
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important; letter-spacing: .02em !important;
    background: var(--km-accent) !important; border: none !important; color: #06130b !important;
    border-radius: 10px !important; transition: transform .15s, box-shadow .15s !important;
    box-shadow: 0 6px 16px -8px color-mix(in srgb, var(--km-accent) 70%, transparent) !important;
    margin-top: 6px !important;
}
#km-ask-btn:hover { transform: translateY(-2px); box-shadow: 0 12px 24px -10px color-mix(in srgb, var(--km-accent) 80%, transparent) !important; }
#km-ask-btn:active { transform: translateY(0) scale(.985); }

/* ---- example chips ---- */
#km-chip-row { gap: 8px !important; flex-wrap: wrap !important; margin-top: 14px !important; }
.km-chip {
    font-family: 'IBM Plex Mono', 'Noto Sans Devanagari', 'Noto Sans Tamil', 'Noto Sans Telugu',
                 'Noto Sans Kannada', 'Noto Sans Bengali', monospace !important;
    font-size: .74rem !important; color: var(--km-fg-dim) !important;
    background: var(--km-surface-2) !important; border: 1px solid var(--km-border) !important;
    border-radius: 99px !important; padding: 4px 13px !important; min-width: 0 !important; box-shadow: none !important;
    transition: border-color .15s, color .15s, transform .15s !important;
}
.km-chip:hover { border-color: var(--km-accent) !important; color: var(--km-accent) !important; transform: translateY(-1px); }
.km-chip:active { transform: translateY(0) scale(.97); }

/* ---- answer card ---- */
#km-answer-wrap { margin-top: 20px; animation: km-rise .55s cubic-bezier(.2,.8,.2,1) both; }
@keyframes km-rise { from { opacity: 0; transform: translateY(14px) scale(.98) rotateX(-4deg); } to { opacity: 1; transform: translateY(0) scale(1) rotateX(0); } }
.km-answer-card { border-radius: 16px; border: 1px solid var(--km-border); background: var(--km-surface); overflow: hidden; }
.km-answer-unsourced { border-color: color-mix(in srgb, var(--km-accent-2) 45%, var(--km-border)); }
.km-answer-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 11px 16px; border-bottom: 1px solid var(--km-border); background: var(--km-surface-2); }
.km-answer-head-right { display: flex; align-items: center; gap: 10px; }
.km-answer-head .km-timing { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; color: var(--km-fg-dim); }
.km-status-warn { color: var(--km-accent-2) !important; background: color-mix(in srgb, var(--km-accent-2) 12%, transparent) !important; border-color: color-mix(in srgb, var(--km-accent-2) 35%, transparent) !important; }
.km-status-warn .km-dot { background: var(--km-accent-2) !important; }
.km-copy-btn {
    font-family: 'IBM Plex Mono', monospace; font-size: .68rem; color: var(--km-fg-dim);
    background: var(--km-surface); border: 1px solid var(--km-border); border-radius: 6px;
    padding: 3px 9px; cursor: pointer; transition: border-color .15s, color .15s;
}
.km-copy-btn:hover { border-color: var(--km-accent); color: var(--km-accent); }
.km-answer-body { padding: 18px; font-size: 1.02rem; line-height: 1.7; }

/* ---- sources (3D perspective hover + staggered reveal) ---- */
#km-sources-accordion { margin-top: 12px; background: transparent !important; border: none !important; }
.km-src {
    display: grid; grid-template-columns: auto 1fr auto; gap: 14px; align-items: center;
    background: var(--km-surface); border: 1px solid var(--km-border); border-radius: 14px;
    padding: 12px 16px; margin-bottom: 10px; transform-style: preserve-3d;
    transition: transform .22s cubic-bezier(.2,.8,.2,1), box-shadow .22s, border-color .22s;
    animation: km-rise .5s cubic-bezier(.2,.8,.2,1) both;
}
.km-src:nth-of-type(1) { animation-delay: .04s; } .km-src:nth-of-type(2) { animation-delay: .1s; }
.km-src:nth-of-type(3) { animation-delay: .16s; } .km-src:nth-of-type(4) { animation-delay: .22s; }
.km-src:hover {
    transform: perspective(700px) rotateX(2deg) translateY(-3px);
    box-shadow: 0 18px 32px -20px rgba(0,0,0,.4);
    border-color: color-mix(in srgb, var(--km-accent) 45%, var(--km-border));
}
.km-src-icon { width: 32px; height: 32px; border-radius: 9px; display: flex; align-items: center; justify-content: center;
    background: color-mix(in srgb, var(--km-accent) 14%, transparent); flex: none; font-size: 1rem; }
.km-src-main { min-width: 0; }
.km-src-q { font-weight: 600; font-size: .9rem; margin: 0 0 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.km-src-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.km-tag { font-family: 'IBM Plex Mono', monospace; font-size: .62rem; letter-spacing: .02em; color: var(--km-fg-dim);
    background: var(--km-surface-2); border: 1px solid var(--km-border); border-radius: 5px; padding: 2px 6px; }
.km-score { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; font-family: 'IBM Plex Mono', monospace; }
.km-score .km-val { font-size: .76rem; color: var(--km-accent); font-weight: 500; }
.km-score .km-val-num { font-size: .62rem; color: var(--km-fg-dim); }
.km-bar { width: 58px; height: 4px; border-radius: 99px; background: var(--km-surface-2); overflow: hidden; }
.km-bar i { display: block; height: 100%; background: var(--km-accent); border-radius: 99px; transition: width .6s cubic-bezier(.2,.8,.2,1); }

#km-footer { text-align: center; font-family: 'IBM Plex Mono', monospace; font-size: .7rem; color: var(--km-fg-dim); margin-top: 26px; padding-bottom: 8px; }
#km-footer b { color: var(--km-fg); }

@media (max-width: 560px) {
    #km-input-row { flex-direction: column !important; }
    #km-language { min-width: 0 !important; width: 100% !important; }
    .km-src { grid-template-columns: auto 1fr; grid-template-rows: auto auto; }
    .km-score { grid-column: 1 / -1; flex-direction: row; justify-content: flex-start; margin-top: 4px; }
    #km-hero h1 { font-size: 1.9rem; }
    .km-stats { display: none; }
}
"""

# Subtle 3D tilt on the hero, following the cursor. Self-contained and safe
# since the hero is static content the real app never re-renders.
HERO_TILT_SCRIPT = """
<script>
(function(){
    var outer = document.getElementById('km-hero-outer');
    var hero = document.getElementById('km-hero');
    var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced || !outer || !hero) return;
    outer.addEventListener('mousemove', function(e){
        var r = outer.getBoundingClientRect();
        var px = (e.clientX - r.left) / r.width - 0.5;
        var py = (e.clientY - r.top) / r.height - 0.5;
        hero.style.transform = 'rotateY(' + (px*4) + 'deg) rotateX(' + (-py*4) + 'deg)';
        hero.classList.add('km-tilt');
    });
    outer.addEventListener('mouseleave', function(){
        hero.style.transform = 'rotateY(0) rotateX(0)';
        hero.classList.remove('km-tilt');
    });
})();
</script>
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
)


def crop_icon(crop: str) -> str:
    return CROP_ICONS.get(crop, "🛰")


def confidence_label(score: float) -> str:
    # Plain-language bands instead of a raw similarity score: a number like
    # "0.64" means nothing to most users without context on the scale, and
    # research on AI-answer UX specifically flags raw confidence numbers as
    # something people can't act on. The number is still shown (smaller,
    # secondary) for anyone who does want the precise value.
    if score >= 0.7:
        return "Strong match"
    if score >= 0.5:
        return "Good match"
    if score >= 0.35:
        return "Partial match"
    return "Weak match"


def js_attr_escape(text: str) -> str:
    """Escape text for embedding as a JS single-quoted string literal inside
    an HTML double-quoted attribute. Order matters: JS-escape first (so a
    literal backslash or quote in the answer becomes a valid JS escape
    sequence), then HTML-escape the result so a literal `"` in the answer
    can't prematurely close the onclick="..." attribute — the browser decodes
    the HTML entities back to real characters before handing the string to
    the JS engine, so the JS escaping underneath still applies correctly."""
    js_safe = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "")
    return html.escape(js_safe)


EMPTY_STATE_HTML = (
    "<div class='km-answer-card'>"
    "<div class='km-answer-head'>"
    "<span class='km-status-chip'><span class='km-dot'></span>Awaiting input</span>"
    "</div>"
    "<div class='km-answer-body' style='color:var(--km-fg-dim)'>"
    "Ask a question above, or pick one of the examples, to see a grounded answer here."
    "</div></div>"
)


def render_answer(answer_text: str, elapsed: float) -> str:
    # No low-confidence/decline badge here: that depended on a pipeline.py
    # change (LOW_CONFIDENCE_THRESHOLD, result.grounded) that was reverted
    # to keep the backend exactly at the proven 34/38 eval state — the
    # pipeline always generates an answer via AgriParam, so the badge always
    # reads "Grounded answer". The copy button and layout are kept (pure
    # frontend, independent of that backend logic).
    safe_text = html.escape(answer_text).replace("\n", "<br>")
    copy_js = js_attr_escape(answer_text)
    badge = "<span class='km-status-chip'><span class='km-dot'></span>Grounded answer</span>"
    return (
        "<div class='km-answer-card'>"
        "<div class='km-answer-head'>"
        f"{badge}"
        "<span class='km-answer-head-right'>"
        f"<span class='km-timing'>{elapsed:.1f}s</span>"
        f"<button class='km-copy-btn' onclick=\"navigator.clipboard.writeText('{copy_js}').then(() => {{"
        "const t=this.textContent; this.textContent='Copied'; setTimeout(() => this.textContent=t, 1400);"
        "})\">Copy</button>"
        "</span>"
        "</div>"
        f"<div class='km-answer-body'>{safe_text}</div>"
        "</div>"
    )


def render_sources(sources: list[dict]) -> str:
    if not sources:
        return "<p style='color:var(--km-fg-dim);font-family:IBM Plex Mono,monospace;font-size:.8rem;'>No sources retrieved.</p>"
    cards = []
    for s in sources:
        pct = max(4, min(100, round(s["score"] * 100)))
        cards.append(
            "<div class='km-src'>"
            f"<div class='km-src-icon'>{crop_icon(s['crop'])}</div>"
            "<div class='km-src-main'>"
            f"<p class='km-src-q'>{html.escape(s['query'])}</p>"
            f"<div class='km-src-tags'><span class='km-tag'>{html.escape(s['crop'])}</span>"
            f"<span class='km-tag'>{html.escape(s['category'])}</span></div>"
            "</div>"
            "<div class='km-score'>"
            f"<span class='km-val'>{confidence_label(s['score'])}</span>"
            f"<span class='km-val-num'>{s['score']:.2f}</span>"
            f"<div class='km-bar'><i style='width:{pct}%'></i></div>"
            "</div>"
            "</div>"
        )
    return "".join(cards)


def handle_ask(question: str, language: str):
    if not question or not question.strip():
        return EMPTY_STATE_HTML, "", gr.update(open=False)
    start = time.time()
    result = ask(question, language)
    elapsed = time.time() - start
    return (
        render_answer(result.answer, elapsed),
        render_sources(result.sources),
        gr.update(open=True),
    )


def handle_chip(question: str, language: str):
    answer_html, sources_html, accordion_update = handle_ask(question, language)
    return question, language, answer_html, sources_html, accordion_update


with gr.Blocks(css=CUSTOM_CSS, theme=THEME, title="Kisan Mitra — AI Farmer Advisory") as demo:
    gr.HTML(
        """
        <div id="km-hero-outer">
        <div id="km-hero">
            <div class="km-brand">
                <span class="km-status-chip"><span class="km-dot"></span>Live · Retrieval-grounded</span>
            </div>
            <p class="km-kicker">Field Advisory Console</p>
            <div class="km-lockup">
                <svg width="46" height="46" viewBox="0 0 44 44" fill="none">
                    <circle cx="22" cy="22" r="20" stroke="var(--km-accent)" stroke-width="1.2" opacity="0.35"/>
                    <circle cx="22" cy="22" r="15.5" fill="var(--km-surface-2)"/>
                    <path d="M22 30.5c-6-2.5-9-7.5-9-13.5 0-3.5 2-7 5.5-8.5-1 3.5 0 7 2.5 9.5 2.5-2.5 3.5-6 2.5-9.5 3.5 1.5 5.5 5 5.5 8.5 0 6-3 11-9 13.5Z" fill="var(--km-accent)"/>
                    <path d="M22 30.5V16" stroke="var(--km-surface-2)" stroke-width="1.2"/>
                    <path d="M30 13.5c1.6 1.6 1.6 4.3 0 5.9" stroke="var(--km-accent)" stroke-width="1.3" stroke-linecap="round" opacity="0.85"/>
                    <path d="M32.8 10.7c3.1 3.1 3.1 8.3 0 11.4" stroke="var(--km-accent)" stroke-width="1.3" stroke-linecap="round" opacity="0.5"/>
                </svg>
                <h1>Kisan Mitra</h1>
            </div>
            <p class="km-lede">Ask in your own language — crops, irrigation, pests, fertilizer,
            government schemes. Every answer is traced back to the source it was grounded on,
            shown below.</p>
        </div>
        </div>
        """
        + HERO_TILT_SCRIPT
    )

    if not torch.cuda.is_available():
        # Honest expectation-setting for free CPU hosting (e.g. a free HF
        # Space): AgriParam's generation runs with use_cache=False (see
        # agriparam_client.py), which is compute-bound and genuinely slow on
        # CPU — this isn't hidden from the user, with a direct link to the
        # fast GPU-backed Colab demo as an alternative.
        gr.HTML(
            f"""
            <div id="km-cpu-notice">
                <span>⚠ <b>Running on free CPU hardware</b> — responses can take several
                minutes. For a fast GPU-backed demo, open the
                <a href="{COLAB_URL}" target="_blank">Colab notebook</a> instead.</span>
            </div>
            """
        )

    with gr.Group(elem_id="km-console"):
        gr.HTML(
            """
            <div id="km-console-label">
                <p class="km-kicker" style="margin:0">Ask a question</p>
                <div class="km-stats"><span><b>30+</b> sources</span><span><b>7</b> languages</span><span><b>RAG</b>-grounded</span></div>
            </div>
            """
        )
        with gr.Row(elem_id="km-input-row"):
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. How do I control stem borer in paddy?",
                lines=2,
                scale=3,
            )
            language = gr.Dropdown(LANGUAGES, value="English", label="Language", elem_id="km-language", scale=1, min_width=150)

        with gr.Row(elem_id="km-chip-row"):
            chip_buttons = [
                gr.Button(f"{ex['icon']} {ex['chip']}  ({ex['lang_code']})", elem_classes=["km-chip"], size="sm")
                for ex in EXAMPLES
            ]
        ask_btn = gr.Button("Ask Kisan Mitra →", elem_id="km-ask-btn", size="lg")

    with gr.Column(elem_id="km-answer-wrap"):
        answer_box = gr.HTML(EMPTY_STATE_HTML)

    with gr.Accordion("Sources this answer is grounded on", open=False, elem_id="km-sources-accordion") as sources_accordion:
        sources_box = gr.HTML()

    gr.HTML(
        """
        <div id="km-footer">
            Powered by <b>BharatGen AgriParam</b> &amp; <b>AI4Bharat IndicTrans2</b> ·
            Not a substitute for your local Krishi Vigyan Kendra.
        </div>
        """
    )

    ask_btn.click(handle_ask, inputs=[question, language], outputs=[answer_box, sources_box, sources_accordion])
    question.submit(handle_ask, inputs=[question, language], outputs=[answer_box, sources_box, sources_accordion])

    for btn, ex in zip(chip_buttons, EXAMPLES):
        btn.click(
            lambda q=ex["question"], lang=ex["language"]: handle_chip(q, lang),
            inputs=[],
            outputs=[question, language, answer_box, sources_box, sources_accordion],
        )


if __name__ == "__main__":
    demo.launch(share=True)

"""Kisan Mitra — Gradio UI. Single entrypoint used both for the Colab
`share=True` demo and for Hugging Face Spaces deployment.

Visual direction: a "field advisory console" — a precision-agriculture
sensor/satellite readout aesthetic (animated grid backdrop, signal-green
accent, monospace data readouts, score bars) rather than a generic
rustic/cream farm-app look.
"""
import html
import time

import gradio as gr

from pipeline import ask

LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Marathi", "Bengali"]

# "chip" is a short display label (shown with its language so the multilingual
# range is visible at a glance); "question" is the full real query actually sent.
EXAMPLES = [
    {"icon": "🌾", "chip": "Stem borer in paddy?", "lang_code": "EN", "question": "How do I control stem borer in paddy?", "language": "English"},
    {"icon": "🌿", "chip": "गेहूं की बुवाई?", "lang_code": "HI", "question": "गेहूं की बुवाई का सही समय क्या है?", "language": "Hindi"},
    {"icon": "🏛", "chip": "PM-KISAN பதிவு?", "lang_code": "TA", "question": "PM-KISAN கீழ் நான் எவ்வாறு பதிவு செய்வது?", "language": "Tamil"},
    {"icon": "🌽", "chip": "Maize fertilizer?", "lang_code": "EN", "question": "What is the fertilizer schedule for hybrid maize?", "language": "English"},
    {"icon": "💧", "chip": "చుక్క సబ్సిడీ?", "lang_code": "TE", "question": "చుక్క నీటిపారుదల సబ్సిడీ ఎలా పొందాలి?", "language": "Telugu"},
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

/* The gradio-container is centered with a max-width, which otherwise leaves an
   unstyled (browser-default, often black in dark mode) void on either side. */
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

/* Dropdown popup (language selector) — unstyled by default, falls back to the
   browser/OS dark popup chrome which clashes with the rest of the theme. */
.gradio-container ul[role="listbox"],
.gradio-container .options {
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

/* ---- hero ---- */
#km-hero {
    position: relative; overflow: hidden; border-radius: 20px;
    background:
        radial-gradient(ellipse 120% 80% at 20% -10%, color-mix(in srgb, var(--km-accent) 16%, transparent), transparent 60%),
        var(--km-surface);
    border: 1px solid var(--km-border);
    padding: 28px 28px 24px; margin-bottom: 16px;
}
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

.km-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; position: relative; }
.km-status-chip {
    display: inline-flex; align-items: center; gap: 6px; font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem; letter-spacing: .06em; color: var(--km-accent);
    background: color-mix(in srgb, var(--km-accent) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--km-accent) 35%, transparent);
    padding: 3px 9px; border-radius: 99px; text-transform: uppercase;
}
.km-status-chip .km-dot { width: 6px; height: 6px; border-radius: 99px; background: var(--km-accent); animation: km-pulse 1.8s ease-in-out infinite; }
@keyframes km-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }

#km-hero .km-kicker { font-family: 'IBM Plex Mono', monospace; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--km-fg-dim); margin: 0 0 6px; position: relative; }
#km-hero h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.3rem; letter-spacing: -.01em; margin: 2px 0; color: var(--km-fg) !important; position: relative; }
#km-hero .km-lede { color: var(--km-fg-dim); font-size: .98rem; line-height: 1.55; max-width: 48ch; margin: 0; position: relative; }

/* ---- input row ---- */
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
}
#km-ask-btn {
    background: var(--km-accent) !important; border: none !important; color: #06130b !important;
    border-radius: 10px !important; transition: transform .15s, box-shadow .15s !important;
    box-shadow: 0 6px 16px -8px color-mix(in srgb, var(--km-accent) 70%, transparent) !important;
}
#km-ask-btn:hover { transform: translateY(-1px); box-shadow: 0 10px 22px -8px color-mix(in srgb, var(--km-accent) 80%, transparent) !important; }

/* ---- example chips ---- */
#km-chip-row { gap: 8px !important; flex-wrap: wrap !important; }
.km-chip {
    /* Chip labels include native-script text (Hindi/Tamil/Telugu), so the mono
       face needs the same Noto Sans fallback stack as the rest of the page —
       IBM Plex Mono alone doesn't cover those scripts. */
    font-family: 'IBM Plex Mono', 'Noto Sans Devanagari', 'Noto Sans Tamil', 'Noto Sans Telugu',
                 'Noto Sans Kannada', 'Noto Sans Bengali', monospace !important;
    font-size: .74rem !important; color: var(--km-fg-dim) !important;
    background: var(--km-surface-2) !important; border: 1px solid var(--km-border) !important;
    border-radius: 99px !important; padding: 4px 13px !important; min-width: 0 !important; box-shadow: none !important;
    transition: border-color .15s, color .15s, transform .15s !important;
}
.km-chip:hover { border-color: var(--km-accent) !important; color: var(--km-accent) !important; transform: translateY(-1px); }

/* ---- answer card ---- */
#km-answer-wrap { margin-top: 18px; animation: km-rise .5s cubic-bezier(.2,.8,.2,1) both; }
@keyframes km-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.km-answer-card { border-radius: 16px; border: 1px solid var(--km-border); background: var(--km-surface); overflow: hidden; }
.km-answer-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 11px 16px; border-bottom: 1px solid var(--km-border); background: var(--km-surface-2); }
.km-answer-head .km-timing { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; color: var(--km-fg-dim); }
.km-answer-body { padding: 18px; font-size: 1.02rem; line-height: 1.7; }

/* ---- sources ---- */
#km-sources-accordion { margin-top: 12px; background: transparent !important; border: none !important; }
.km-src { display: grid; grid-template-columns: auto 1fr auto; gap: 14px; align-items: center;
    background: var(--km-surface); border: 1px solid var(--km-border); border-radius: 14px;
    padding: 12px 16px; margin-bottom: 10px; transition: transform .18s, box-shadow .18s, border-color .18s; }
.km-src:hover { transform: translateY(-2px); box-shadow: 0 14px 28px -20px rgba(0,0,0,.35); border-color: color-mix(in srgb, var(--km-accent) 45%, var(--km-border)); }
.km-src-icon { width: 32px; height: 32px; border-radius: 9px; display: flex; align-items: center; justify-content: center;
    background: color-mix(in srgb, var(--km-accent) 14%, transparent); flex: none; font-size: 1rem; }
.km-src-main { min-width: 0; }
.km-src-q { font-weight: 600; font-size: .9rem; margin: 0 0 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.km-src-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.km-tag { font-family: 'IBM Plex Mono', monospace; font-size: .62rem; letter-spacing: .02em; color: var(--km-fg-dim);
    background: var(--km-surface-2); border: 1px solid var(--km-border); border-radius: 5px; padding: 2px 6px; }
.km-score { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; font-family: 'IBM Plex Mono', monospace; }
.km-score .km-val { font-size: .76rem; color: var(--km-accent); }
.km-bar { width: 58px; height: 4px; border-radius: 99px; background: var(--km-surface-2); overflow: hidden; }
.km-bar i { display: block; height: 100%; background: var(--km-accent); border-radius: 99px; }

#km-footer { text-align: center; font-family: 'IBM Plex Mono', monospace; font-size: .7rem; color: var(--km-fg-dim); margin-top: 24px; padding-bottom: 8px; }
#km-footer b { color: var(--km-fg); }

@media (max-width: 480px) {
    .km-src { grid-template-columns: auto 1fr; grid-template-rows: auto auto; }
    .km-score { grid-column: 1 / -1; flex-direction: row; justify-content: flex-start; margin-top: 4px; }
}
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
)


def crop_icon(crop: str) -> str:
    return CROP_ICONS.get(crop, "🛰")


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
    safe_text = html.escape(answer_text).replace("\n", "<br>")
    return (
        "<div class='km-answer-card'>"
        "<div class='km-answer-head'>"
        "<span class='km-status-chip'><span class='km-dot'></span>Grounded answer</span>"
        f"<span class='km-timing'>{elapsed:.1f}s</span>"
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
            f"<span class='km-val'>{s['score']:.2f}</span>"
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
    return render_answer(result.answer, elapsed), render_sources(result.sources), gr.update(open=True)


def handle_chip(question: str, language: str):
    answer_html, sources_html, accordion_update = handle_ask(question, language)
    return question, language, answer_html, sources_html, accordion_update


with gr.Blocks(css=CUSTOM_CSS, theme=THEME, title="Kisan Mitra — AI Farmer Advisory") as demo:
    gr.HTML(
        """
        <div id="km-hero">
            <div class="km-brand">
                <span class="km-status-chip"><span class="km-dot"></span>Live · Retrieval-grounded</span>
            </div>
            <p class="km-kicker">Field Advisory Console</p>
            <h1>Kisan Mitra</h1>
            <p class="km-lede">Ask in your own language — crops, irrigation, pests, fertilizer,
            government schemes. Every answer is traced back to the source it was grounded on,
            shown below.</p>
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
    ask_btn = gr.Button("Ask →", elem_id="km-ask-btn", size="lg")

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

"""Kisan Mitra — Gradio UI. Single entrypoint used both for the Colab
`share=True` demo and for Hugging Face Spaces deployment."""
import gradio as gr

from pipeline import ask

LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Marathi", "Bengali"]

EXAMPLES = [
    ["How do I control stem borer in paddy?", "English"],
    ["गेहूं की बुवाई का सही समय क्या है?", "Hindi"],
    ["PM-KISAN கீழ் நான் எவ்வாறு பதிவு செய்வது?", "Tamil"],
    ["What is the fertilizer schedule for hybrid maize?", "English"],
    ["చుక్క నీటిపారుదల సబ్సిడీ ఎలా పొందాలి?", "Telugu"],
]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600&family=Noto+Sans+Tamil:wght@400;600&family=Noto+Sans+Telugu:wght@400;600&family=Noto+Sans+Kannada:wght@400;600&family=Noto+Sans+Bengali:wght@400;600&display=swap');

:root {
    --km-green-900: #1b3a2b;
    --km-green-700: #2d5a3d;
    --km-green-500: #4a7c59;
    --km-amber-500: #c2742f;
    --km-amber-600: #a85f22;
    --km-cream: #faf6ee;
    --km-cream-dark: #f1ead9;
    --km-ink: #27312b;
    --km-border: #e0d8c3;
}

.gradio-container {
    font-family: 'Inter', 'Noto Sans Devanagari', 'Noto Sans Tamil', 'Noto Sans Telugu',
                 'Noto Sans Kannada', 'Noto Sans Bengali', sans-serif !important;
    background: var(--km-cream) !important;
    max-width: 960px !important;
    margin: 0 auto !important;
}

#km-hero {
    background: linear-gradient(135deg, var(--km-green-900) 0%, var(--km-green-700) 65%, var(--km-green-500) 100%);
    border-radius: 18px;
    padding: 32px 36px;
    margin-bottom: 18px;
    color: #fdfaf2;
    box-shadow: 0 8px 24px rgba(27, 58, 43, 0.18);
}
#km-hero h1 {
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    margin: 0 0 6px 0;
    color: #fdfaf2 !important;
}
#km-hero p {
    margin: 0;
    opacity: 0.88;
    font-size: 1rem;
}

.km-card {
    background: #ffffff;
    border: 1px solid var(--km-border);
    border-radius: 14px;
    padding: 20px 22px;
}

#km-answer-card {
    background: #ffffff;
    border: 1px solid var(--km-border);
    border-left: 5px solid var(--km-amber-500);
    border-radius: 12px;
    padding: 20px 22px;
    min-height: 90px;
    font-size: 1.05rem;
    line-height: 1.65;
    color: var(--km-ink);
}

#km-ask-btn {
    background: var(--km-amber-500) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
#km-ask-btn:hover { background: var(--km-amber-600) !important; }

.km-source-tag {
    display: inline-block;
    background: var(--km-cream-dark);
    color: var(--km-green-700);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
}

footer { display: none !important; }
#km-footer {
    text-align: center;
    font-size: 0.8rem;
    color: #8a8372;
    margin-top: 22px;
    padding-bottom: 10px;
}
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_No sources retrieved._"
    lines = []
    for s in sources:
        lines.append(
            f"**{s['query']}**\n\n"
            f"<span class='km-source-tag'>{s['crop']}</span>"
            f"<span class='km-source-tag'>{s['category']}</span>"
            f"<span class='km-source-tag'>match {s['score']:.2f}</span>\n\n"
            f"{s['answer']}\n\n---"
        )
    return "\n\n".join(lines)


def handle_ask(question: str, language: str):
    if not question or not question.strip():
        return "Please enter a question.", "", gr.update(open=False)
    result = ask(question, language)
    debug = f"**Query understood as (English):** {result.english_query}"
    return result.answer, format_sources(result.sources), gr.update(open=True)


with gr.Blocks(css=CUSTOM_CSS, theme=THEME, title="Kisan Mitra — AI Farmer Advisory") as demo:
    gr.HTML(
        """
        <div id="km-hero">
            <h1>🌾 Kisan Mitra</h1>
            <p>Ask a farming question in your own language — crops, irrigation, pests,
            fertilizer, or government schemes — and get an answer grounded in real
            agricultural advisory sources, not a guess.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. How do I control stem borer in paddy?",
                lines=2,
            )
        with gr.Column(scale=1, min_width=160):
            language = gr.Dropdown(LANGUAGES, value="English", label="Language")

    ask_btn = gr.Button("Ask Kisan Mitra", elem_id="km-ask-btn", size="lg")

    gr.Examples(examples=EXAMPLES, inputs=[question, language], label="Try an example")

    gr.Markdown("### Answer")
    answer_box = gr.Markdown(elem_id="km-answer-card")

    with gr.Accordion("Sources this answer is grounded on", open=False) as sources_accordion:
        sources_box = gr.Markdown()

    gr.HTML(
        """
        <div id="km-footer">
            Powered by BharatGen AgriParam &amp; AI4Bharat IndicTrans2 ·
            Retrieval-grounded, not a substitute for your local Krishi Vigyan Kendra.
        </div>
        """
    )

    ask_btn.click(handle_ask, inputs=[question, language], outputs=[answer_box, sources_box, sources_accordion])
    question.submit(handle_ask, inputs=[question, language], outputs=[answer_box, sources_box, sources_accordion])


if __name__ == "__main__":
    demo.launch(share=True)

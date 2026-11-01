# Kisan Mitra — AI Farmer Advisory Assistant

A multilingual, retrieval-grounded advisory assistant for Indian farmers, built for CS3235 (AI4Bharat & BharatGen Mini-Project, Project 2).

Ask a question about crops, irrigation, fertilizers, pests, or government schemes in **English, Hindi, Tamil, Telugu, Kannada, Marathi, or Bengali**, and get a grounded answer backed by real agricultural advisory sources — not a hallucinated guess.

## How it works

```
User question (any supported language)
        │
        ▼
IndicTrans2 (indic→en)        — translate query to English if needed
        │
        ▼
Dense retrieval over the advisory corpus (FAISS + multilingual sentence embeddings)
        │
        ▼
BharatGen AgriParam            — generates a grounded answer using retrieved context
        │
        ▼
IndicTrans2 (en→indic)         — translate answer back to the user's language
        │
        ▼
Gradio UI — shows the answer + the retrieved sources it was grounded on
```

## Models & data (per project brief)

| Component | Model / Source |
|---|---|
| Generation | [`bharatgenai/AgriParam`](https://huggingface.co/bharatgenai/AgriParam) (3B, Apache-2.0, not gated) |
| Translation | [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) / `indictrans2-indic-en-dist-200M` |
| Retrieval embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Corpus | Kisan Call Centre (KCC) transcripts — [data.gov.in](https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers), Govt. of India, Open Government License + a curated agronomic seed set (see `data/`) |

## Why Colab + Hugging Face Spaces, not local

AgriParam is a 3B-parameter model — it needs ~6GB RAM in fp16 (or ~2-3GB 4-bit quantized) and runs far too slowly on CPU-only laptops. All model inference is designed to run on free GPU compute:

- **Development/testing**: Google Colab (free T4 GPU) — see `notebooks/colab_dev.ipynb`
- **Final deployed demo**: Hugging Face Spaces (free, persistent URL) — see `app.py`

Nothing here is meant to be run on a laptop without a dedicated GPU.

## Project structure

```
data/           corpus fetching + preparation (KCC API + seed set)
rag/            embedding + FAISS index building, retrieval
model/          AgriParam generation client, IndicTrans2 translation wrapper
app.py          Gradio UI — single entrypoint for both Colab demo and HF Spaces deployment
notebooks/      Colab development notebook
eval/           30-50 test cases, evaluation harness, failure analysis
```

## Setup

See [SETUP.md](SETUP.md) for the two free one-time steps required (Hugging Face token, data.gov.in API key) and exact run instructions.

## Evaluation

See `eval/` — groundedness, factual correctness, and domain QA accuracy scored across curated test cases, with a written failure analysis, per the course rubric.

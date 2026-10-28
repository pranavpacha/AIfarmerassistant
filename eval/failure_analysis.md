# Failure Analysis

**Pass rate:** 34/38 (89.5%)
**Average keyword coverage:** 0.586

This run passed with real GPU-backed generation (AgriParam 4-bit + IndicTrans2) over all 38 cases, no crashes. Scoring requires both (a) retrieval returning the expected advisory category and (b) at least 30% of the hand-picked technical keywords for that answer appearing literally in the generated text — a deliberately strict proxy for groundedness, since a correct answer phrased differently can still miss it.

## What actually failed, and why

**t14 — onion storage, t38 — maize fertilizer (typo'd query): topic drift from co-retrieved context.**
Both of these retrieved the *correct* top source (Post-Harvest / Nutrient Management respectively, high similarity scores), but the generated answer actually describes a *different* retrieved document — t14's answer is about managing a sugarcane ratoon crop, not onion storage; t38's answer blends in drip-irrigation/fertigation details that belong to a different corpus entry. This is a known RAG failure mode: when the top-k retrieved documents are concatenated into one context block, generation can pull details from the wrong one instead of staying anchored to the best match. It's a real limitation, not a scoring artifact — the answers are factually about the wrong crop.

**t30 — cyclone/heavy-rain crop protection: truncated generation with a fabricated detail.**
The answer invents a specific, ungrounded figure ("80mm/20-minute heavy rainfall warning") not present in any source, then cuts off mid-sentence before giving any actual advice. This looks like the model latching onto a plausible-sounding but invented numeric detail — a real groundedness failure worth flagging, not just an incomplete generation.

**t33 — Kannada pink bollworm (cotton): weak retrieval match + topic drift.**
Retrieval score here is notably lower (0.559 vs. 0.6-0.85 for most other cases), and the generated English answer actually describes fall-armyworm treatment (a maize pest), not pink bollworm (a cotton pest) — i.e., the corpus's cotton pink-bollworm entry wasn't the strongest match returned, and generation defaulted to a different pest-management entry entirely. Both translation directions (Kannada→English→Kannada) stayed faithful to that drifted English content, so the end-to-end translation itself is not at fault here — the miss happened at retrieval/generation, before translation ever ran.

## A more important finding the pass/fail count doesn't show

Two cases that **technically passed** are arguably the most concerning result in this whole evaluation: **t36** ("stock price of Reliance Industries") and **t37** ("file my income tax return") are deliberate out-of-domain trap questions. They passed because retrieval correctly returned low-confidence scores (0.225, 0.297 — both below the 0.35 threshold this harness treats as "no good match found"). But the *generated answers* don't reflect that uncertainty at all: t36 fabricates a specific stock price with a fake date and source; t37 invents a nonsensical description of an "Income Tax Return (IYR)" electronic system that blends unrelated agricultural program names into a confident-sounding but meaningless answer.

**The system has no explicit refusal behavior.** AgriParam generates a confident answer regardless of how irrelevant the retrieved context is — this evaluation harness only checks retrieval confidence, not whether generation actually respected it. That gap is the most actionable finding here: a production version of this assistant should check retrieval score *before* generation and return an explicit "I don't have grounded information for this" response below some confidence threshold, rather than relying on the LLM to decline on its own (it doesn't).

## Translation quality (Hindi, Tamil, Telugu, Kannada)

Comparing `english_answer` to `final_answer` across all non-English cases (t04, t17, t31, t32, t33), IndicTrans2's output stayed faithful to the English source in every case — no content was dropped or altered in translation. Where keyword-coverage scores were low for these cases (e.g., t04 Hindi: 0.5, t32 Telugu: 0.5), it's because the scoring keywords are English technical terms ("november") that don't appear literally in a correctly-translated non-English sentence — a limitation of this evaluation *method* for multilingual output, not of the translation itself.

## What this run demonstrates

- Retrieval was reliable: 38/38 cases returned the expected category (including both out-of-domain traps correctly flagging low confidence).
- Generation quality was strong on cases where retrieval returned one clearly dominant source; weaker when multiple similar-scoring sources were retrieved together.
- Translation (both directions, 4 languages tested) preserved meaning faithfully in every case checked.
- The system's main real gap is the lack of an explicit low-confidence refusal path — addressable with a simple threshold check in `pipeline.py` rather than a model change.

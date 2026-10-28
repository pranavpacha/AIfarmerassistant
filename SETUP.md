# Setup

This project is designed to run on **Google Colab** (development) and **Hugging Face Spaces** (final deployed demo) — not on a local laptop without a GPU.

## 1. One-time free accounts (both take ~2 minutes)

### Hugging Face token (required — some models are gated)
1. Create a free account at https://huggingface.co/join
2. Go to https://huggingface.co/settings/tokens → create a **Read** token
3. Visit these two model pages while logged in and click "Agree and access repository":
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
4. In Colab, run:
   ```python
   from huggingface_hub import login
   login("hf_xxx...")   # your token
   ```

### data.gov.in API key (optional — enables the full official KCC corpus)
1. Sign up free at https://api.data.gov.in/signup/
2. Copy your API key from "My Account"
3. Open https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers while logged in and copy the resource UUID from the page's URL/API section
4. Set both `DATA_GOV_IN_API_KEY` and `KCC_RESOURCE_ID` before running `data/fetch_kcc_data.py`

   Without this key, the project still runs fully using the curated seed corpus in `data/seed_corpus.json` (~60 real agronomic advisory entries) — the API just lets you expand the retrieval corpus to the full official dataset.

## 2. Running in Colab (development)

Open `notebooks/colab_dev.ipynb` in Colab, set Runtime → Change runtime type → **T4 GPU**, then run all cells. The last cell launches the Gradio app with `share=True`, giving you a public link to the live app for as long as that Colab session stays open.

## 3. Deploying to Hugging Face Spaces (permanent link)

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create kisan-mitra --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<your-username>/kisan-mitra
# copy app.py, model/, rag/, data/seed_corpus.json, requirements.txt into that folder
cd kisan-mitra
git add . && git commit -m "Deploy Kisan Mitra" && git push
```

In the Space settings, select a GPU hardware tier (or request free "ZeroGPU" if the Space qualifies) so AgriParam runs at a usable speed.

## 4. Running the evaluation

```bash
python eval/evaluate.py
```

Writes `eval/results.json` and `eval/failure_analysis.md`.

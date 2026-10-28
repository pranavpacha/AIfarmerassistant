"""Wraps the AI4Bharat IndicTrans2 distilled 200M checkpoints for indic->english
(query understanding) and english->indic (answer delivery) translation."""
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

EN_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"

# FLORES-200 language codes for the languages exposed in the UI
LANG_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Marathi": "mar_Deva",
    "Bengali": "ben_Beng",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Direction:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(DEVICE)
        self.model.eval()
        self.processor = IndicProcessor(inference=True)

    @torch.inference_mode()
    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        batch = self.processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(
            batch, truncation=True, padding="longest", max_length=256, return_tensors="pt"
        ).to(DEVICE)
        output_tokens = self.model.generate(
            **inputs, num_beams=5, num_return_sequences=1, max_length=256
        )
        decoded = self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.processor.postprocess_batch(decoded, lang=tgt_lang)


@lru_cache(maxsize=1)
def _en_indic() -> _Direction:
    return _Direction(EN_INDIC_MODEL)


@lru_cache(maxsize=1)
def _indic_en() -> _Direction:
    return _Direction(INDIC_EN_MODEL)


def to_english(text: str, src_language: str) -> str:
    if src_language == "English":
        return text
    src_code = LANG_CODES[src_language]
    return _indic_en().translate([text], src_lang=src_code, tgt_lang="eng_Latn")[0]


def from_english(text: str, tgt_language: str) -> str:
    if tgt_language == "English":
        return text
    tgt_code = LANG_CODES[tgt_language]
    return _en_indic().translate([text], src_lang="eng_Latn", tgt_lang=tgt_code)[0]

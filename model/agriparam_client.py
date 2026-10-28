"""Load BharatGen AgriParam and generate grounded advisory answers from
retrieved context, using the model's documented <user>/<context>/<assistant>
prompt format."""
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "bharatgenai/AgriParam"


class AgriParamClient:
    # 4-bit quantization is skipped by default: with use_cache=False below, every
    # generation step recomputes full attention from scratch, and bnb's per-op
    # dequantization overhead compounds badly under that workload. FP16 is both
    # faster here and well within a T4's 16GB VRAM for a 3B model (~6GB).
    def __init__(self, load_in_4bit: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
        kwargs = dict(trust_remote_code=True, device_map="auto")
        if torch.cuda.is_available():
            if load_in_4bit:
                kwargs["load_in_4bit"] = True
            else:
                # bfloat16 is NOT well-accelerated on T4 (Turing lacks native BF16
                # tensor cores); float16 is what T4 is actually fast at.
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
        self.model.eval()

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            f"<context>\n{context}\n</context>\n"
            f"<user>{question}</user>\n<assistant>"
        )

    @torch.inference_mode()
    def answer(self, question: str, context: str, max_new_tokens: int = 256) -> str:
        prompt = self._build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id,
            # AgriParam's custom modeling code (modeling_parambharatgen.py) indexes
            # past_key_values with the old tuple-of-tensors layout; current transformers
            # passes a Cache object whose per-layer entries start as None, which breaks
            # that code's `.shape` access. Disabling the cache avoids that broken path
            # entirely (slower generation, but correct).
            use_cache=False,
        )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.split("</assistant>")[0].strip()


@lru_cache(maxsize=1)
def get_agriparam() -> AgriParamClient:
    return AgriParamClient()

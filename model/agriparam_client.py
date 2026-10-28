"""Load BharatGen AgriParam and generate grounded advisory answers from
retrieved context, using the model's documented <user>/<context>/<assistant>
prompt format.

Generation uses a custom decode loop instead of transformers' generate(),
because AgriParam's custom modeling code (modeling_parambharatgen.py, loaded
via trust_remote_code) is a pre-Cache-class LLaMA-style fork: it builds and
consumes past_key_values as plain per-layer (key, value) tuples throughout
(confirmed by reading the actual source), with no knowledge of the Cache/
DynamicCache classes. Current transformers' generate() initializes caching
with a DynamicCache instance by default, and the model's own
`past_key_values[0][0].shape[2]` crashes on that object's empty first-layer
slot. Calling the model's forward() directly, passing back exactly the
tuple-of-tuples it returns as its own `past_key_values`, sidesteps that
mismatch entirely instead of fighting generate()'s cache initialization —
and restores real KV-caching, which is dramatically faster than the
use_cache=False recompute-every-step workaround this replaces.
"""
import gc
from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "bharatgenai/AgriParam"


def _sample_next_token(logits: torch.Tensor, generated_ids: torch.Tensor,
                        temperature: float, top_p: float, repetition_penalty: float) -> torch.Tensor:
    """Standard temperature + repetition-penalty + nucleus (top-p) sampling
    over a single-step logits vector, shape (vocab_size,)."""
    logits = logits.clone()

    # Repetition penalty: discourage tokens already generated (HF's standard
    # formula — divide positive logits, multiply negative ones).
    for token_id in set(generated_ids.tolist()):
        if logits[token_id] > 0:
            logits[token_id] /= repetition_penalty
        else:
            logits[token_id] *= repetition_penalty

    logits = logits / max(temperature, 1e-5)
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep the smallest set of tokens whose cumulative probability >= top_p;
    # always keep at least the single most likely token.
    cutoff = (cumulative > top_p).nonzero()
    keep = cutoff[0].item() + 1 if len(cutoff) > 0 else len(sorted_probs)
    keep = max(keep, 1)
    sorted_probs = sorted_probs[:keep]
    sorted_idx = sorted_idx[:keep]
    sorted_probs = sorted_probs / sorted_probs.sum()

    choice = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx[choice]


class AgriParamClient:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
        kwargs = dict(trust_remote_code=True, device_map="auto")
        if torch.cuda.is_available():
            # bfloat16 is NOT well-accelerated on T4 (Turing lacks native BF16
            # tensor cores); float16 is what T4 is actually fast at. With real
            # caching restored, 4-bit's dequantization overhead no longer
            # compounds across a full recompute every step, but FP16 is still
            # simpler and comfortably fits a T4 for a 3B model (~5.4GB).
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
        self.model.eval()

        if not torch.cuda.is_available():
            # CPU deployment (e.g. a free Hugging Face Space has no GPU):
            # PyTorch's built-in dynamic INT8 quantization is CPU-native (no
            # extra dependencies, unlike bitsandbytes which is CUDA-first) and
            # specifically speeds up the CPU matrix multiplications that
            # dominate generation cost.
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            f"<context>\n{context}\n</context>\n"
            f"<user>{question}</user>\n<assistant>"
        )

    @torch.inference_mode()
    def answer(self, question: str, context: str, max_new_tokens: int = 200) -> str:
        prompt = self._build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        eos_id = self.tokenizer.eos_token_id

        past_key_values = None
        current_input = input_ids
        # Tracked explicitly and grown by one each step (standard HF
        # generate() behaviour, mirrored here since the model's own
        # prepare_inputs_for_generation accepts attention_mask, implying its
        # forward() expects one rather than gracefully handling None).
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        generated = input_ids[0]
        outputs = None

        try:
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = _sample_next_token(
                    outputs.logits[0, -1, :], generated,
                    temperature=0.3, top_p=0.9, repetition_penalty=1.15,
                )
                generated = torch.cat([generated, next_token])
                if next_token.item() == eos_id:
                    break
                # After the first step, only the newly generated token is fed
                # in — the cache already holds everything before it.
                current_input = next_token.view(1, 1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                    dim=-1,
                )

            text = self.tokenizer.decode(
                generated[input_ids.shape[1]:], skip_special_tokens=True
            )
            return text.split("</assistant>")[0].strip()
        finally:
            # outputs (this iteration's full logits + the cache it produced)
            # must be released explicitly here, same reasoning as past_key_values:
            # it's still a live local when this block runs, so empty_cache()
            # below wouldn't free its memory otherwise.
            del inputs, past_key_values, current_input, generated, attention_mask, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@lru_cache(maxsize=1)
def get_agriparam() -> AgriParamClient:
    return AgriParamClient()

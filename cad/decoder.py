from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import torch


@dataclass
class CADConfig:
    alpha: float = 1.0
    top_p: float = 1.0
    temperature: float = 0.0
    max_new_tokens: int = 256


class ContextAwareDecoder:
    """
    Context-aware decoding for causal LMs using a context and a prior prompt.

    Combined logits:
        (1 + alpha) * logits(context) - alpha * logits(prior)
    """

    def __init__(self, model, tokenizer, device: str, use_chat_template: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_chat_template = use_chat_template

    def generate(
        self,
        context_prompt: Union[str, Sequence[str]],
        prior_prompt: Union[str, Sequence[str]],
        config: CADConfig,
    ) -> Union[str, List[str]]:
        single = isinstance(context_prompt, str)
        contexts = [context_prompt] if single else list(context_prompt)
        priors = [prior_prompt] if isinstance(prior_prompt, str) else list(prior_prompt)

        if len(contexts) != len(priors):
            raise ValueError("Context and prior prompt lists must be the same length.")

        outputs = []
        for ctx, pri in zip(contexts, priors):
            outputs.append(self._generate_one(ctx, pri, config))
        return outputs[0] if single else outputs

    def _generate_one(self, context_prompt: str, prior_prompt: str, config: CADConfig) -> str:
        ctx_inputs = self._encode(context_prompt)
        pri_inputs = self._encode(prior_prompt)

        ctx_ids = ctx_inputs["input_ids"]
        pri_ids = pri_inputs["input_ids"]

        past_ctx = None
        past_pri = None
        generated = []

        for _ in range(config.max_new_tokens):
            ctx_out = self.model(
                input_ids=ctx_ids[:, -1:] if past_ctx is not None else ctx_ids,
                past_key_values=past_ctx,
                use_cache=True,
            )
            pri_out = self.model(
                input_ids=pri_ids[:, -1:] if past_pri is not None else pri_ids,
                past_key_values=past_pri,
                use_cache=True,
            )

            logits_ctx = ctx_out.logits[:, -1, :]
            logits_pri = pri_out.logits[:, -1, :]

            combined = (1.0 + config.alpha) * logits_ctx - config.alpha * logits_pri
            combined = apply_temperature(combined, config.temperature)
            combined = top_p_filtering(combined, config.top_p)

            next_id = sample_next_token(combined, config.temperature)
            generated.append(next_id)

            ctx_ids = torch.cat([ctx_ids, next_id], dim=1)
            pri_ids = torch.cat([pri_ids, next_id], dim=1)
            past_ctx = ctx_out.past_key_values
            past_pri = pri_out.past_key_values

            if self.tokenizer.eos_token_id is not None and next_id.item() == self.tokenizer.eos_token_id:
                break

        gen_ids = torch.cat(generated, dim=1) if generated else ctx_ids[:, 0:0]
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    def _encode(self, prompt: str):
        if self.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt", padding=False)
        return {k: v.to(self.device) for k, v in encoded.items()}


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return logits
    return logits / temperature


def sample_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        next_token = torch.argmax(logits, dim=-1)
    else:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_token.unsqueeze(0)


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    mask = torch.zeros_like(sorted_mask).scatter(1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

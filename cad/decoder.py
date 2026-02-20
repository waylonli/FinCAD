from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch


@dataclass
class CADConfig:
    alpha: float = 1.0
    top_p: float = 1.0
    temperature: float = 0.0
    max_new_tokens: int = 256
    stop_token_ids: Optional[List[int]] = None


class ContextAwareDecoder:
    """
    Context-aware decoding for causal LMs using a context and a prior prompt.

    Combined logits:
        (1 + alpha) * logits(context) - alpha * logits(prior)

    When alpha=0, falls back to standard single-pass generation.
    When alpha>0, batches context and prior into a single forward pass
    for ~2x speedup over the naive sequential approach.
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

        N = len(contexts)

        if config.alpha == 0:
            # No CAD — standard generation per sample
            outputs = []
            for ctx in contexts:
                ctx_ids = self._encode(ctx)["input_ids"]
                outputs.append(self._generate_single(ctx_ids, config))
            return outputs[0] if single else outputs

        if N == 1:
            # Single-sample CAD — use existing pair-batched path
            ctx_ids = self._encode(contexts[0])["input_ids"]
            pri_ids = self._encode(priors[0])["input_ids"]
            result = self._generate_batched(ctx_ids, pri_ids, config)
            return result if single else [result]

        # Multi-sample CAD — batch all N context+prior pairs together
        ctx_ids_list = [self._encode(ctx)["input_ids"] for ctx in contexts]
        pri_ids_list = [self._encode(pri)["input_ids"] for pri in priors]
        outputs = self._generate_batched_multi(ctx_ids_list, pri_ids_list, config)
        return outputs[0] if single else outputs

    # ------------------------------------------------------------------
    # Baseline: single forward pass per token (alpha=0)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _generate_single(self, input_ids: torch.Tensor, config: CADConfig) -> str:
        past = None
        generated: List[torch.Tensor] = []
        ids = input_ids

        for _ in range(config.max_new_tokens):
            out = self.model(
                input_ids=ids[:, -1:] if past is not None else ids,
                past_key_values=past,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            logits = apply_temperature(logits, config.temperature)
            logits = top_p_filtering(logits, config.top_p)

            next_id = sample_next_token(logits, config.temperature)
            generated.append(next_id)
            ids = torch.cat([ids, next_id], dim=1)
            past = out.past_key_values

            if self._should_stop(next_id, config):
                break

        gen_ids = torch.cat(generated, dim=1) if generated else ids[:, 0:0]
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # CAD: batched context + prior in one forward pass (alpha>0)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _generate_batched(
        self, ctx_ids: torch.Tensor, pri_ids: torch.Tensor, config: CADConfig,
    ) -> str:
        device = ctx_ids.device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        ctx_len = ctx_ids.size(1)
        pri_len = pri_ids.size(1)
        max_len = max(ctx_len, pri_len)

        # Left-pad to equal length so we can batch as (2, max_len)
        ctx_padded = torch.nn.functional.pad(ctx_ids, (max_len - ctx_len, 0), value=pad_id)
        pri_padded = torch.nn.functional.pad(pri_ids, (max_len - pri_len, 0), value=pad_id)
        batched_ids = torch.cat([ctx_padded, pri_padded], dim=0)

        ctx_mask = torch.nn.functional.pad(
            torch.ones(1, ctx_len, device=device, dtype=torch.long),
            (max_len - ctx_len, 0), value=0,
        )
        pri_mask = torch.nn.functional.pad(
            torch.ones(1, pri_len, device=device, dtype=torch.long),
            (max_len - pri_len, 0), value=0,
        )
        attn_mask = torch.cat([ctx_mask, pri_mask], dim=0)

        past = None
        generated: List[torch.Tensor] = []

        for _ in range(config.max_new_tokens):
            if past is None:
                # Prefill: full padded prompts
                out = self.model(
                    input_ids=batched_ids,
                    attention_mask=attn_mask,
                    use_cache=True,
                )
            else:
                # Decode: same generated token fed to both branches
                step_ids = generated[-1].expand(2, -1)
                attn_mask = torch.cat(
                    [attn_mask, torch.ones(2, 1, device=device, dtype=torch.long)],
                    dim=1,
                )
                out = self.model(
                    input_ids=step_ids,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=True,
                )

            past = out.past_key_values
            logits_ctx = out.logits[0:1, -1, :]
            logits_pri = out.logits[1:2, -1, :]

            combined = (1.0 + config.alpha) * logits_ctx - config.alpha * logits_pri
            combined = apply_temperature(combined, config.temperature)
            combined = top_p_filtering(combined, config.top_p)

            next_id = sample_next_token(combined, config.temperature)
            generated.append(next_id)

            if self._should_stop(next_id, config):
                break

        gen_ids = torch.cat(generated, dim=1) if generated else ctx_ids[:, 0:0]
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # CAD: vectorized batch across N samples (alpha>0, N>1)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _generate_batched_multi(
        self,
        ctx_ids_list: List[torch.Tensor],
        pri_ids_list: List[torch.Tensor],
        config: CADConfig,
    ) -> List[str]:
        """Batch N context+prior pairs into a single (2N, max_len) forward pass.

        Optimised decode loop: logit combination, temperature, top-p, and
        sampling are fully vectorised across all N samples per step.
        Token state stays on GPU (no .item() syncs during generation).
        Attention mask is pre-allocated to avoid per-step torch.cat.
        """
        N = len(ctx_ids_list)
        device = ctx_ids_list[0].device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        alpha = config.alpha
        two_n = 2 * N

        # Build stop-token tensor for vectorised checking
        stop_ids: List[int] = []
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
        if config.stop_token_ids:
            stop_ids.extend(config.stop_token_ids)
        stop_tensor = (
            torch.tensor(stop_ids, device=device, dtype=torch.long)
            if stop_ids else None
        )

        # Collect all 2N sequences and find max prompt length
        all_ids = list(ctx_ids_list) + list(pri_ids_list)
        all_lens = [ids.size(1) for ids in all_ids]
        max_prompt_len = max(all_lens)

        # Pre-allocate attention mask: (2N, max_prompt_len + max_new_tokens)
        total_len = max_prompt_len + config.max_new_tokens
        attn_mask = torch.zeros(two_n, total_len, device=device, dtype=torch.long)

        # Left-pad input ids and fill attention mask for prompt positions
        padded = []
        for row, (ids, seq_len) in enumerate(zip(all_ids, all_lens)):
            padded.append(
                torch.nn.functional.pad(ids, (max_prompt_len - seq_len, 0), value=pad_id)
            )
            attn_mask[row, max_prompt_len - seq_len : max_prompt_len] = 1

        batched_ids = torch.cat(padded, dim=0)  # (2N, max_prompt_len)

        # Generation state — all on GPU, no Python lists of tensors
        past = None
        generated = torch.full(
            (N, config.max_new_tokens), pad_id, device=device, dtype=torch.long,
        )
        done = torch.zeros(N, dtype=torch.bool, device=device)
        last_tokens = torch.full((N,), pad_id, device=device, dtype=torch.long)
        gen_lengths = torch.zeros(N, dtype=torch.long, device=device)
        cur_len = max_prompt_len  # tracks total KV-cache length

        for step in range(config.max_new_tokens):
            if past is None:
                # Prefill
                out = self.model(
                    input_ids=batched_ids,
                    attention_mask=attn_mask[:, :cur_len],
                    use_cache=True,
                )
            else:
                # Build step_ids on GPU — no .item() round-trips
                tokens_for_step = torch.where(done, pad_id, last_tokens)  # (N,)
                step_ids = tokens_for_step.repeat(2).unsqueeze(1)  # (2N, 1)

                # Extend attention mask for the new token position
                attn_mask[:, cur_len] = 1
                cur_len += 1

                out = self.model(
                    input_ids=step_ids,
                    attention_mask=attn_mask[:, :cur_len],
                    past_key_values=past,
                    use_cache=True,
                )

            past = out.past_key_values

            # Vectorised logit combination across all N samples at once
            logits_ctx = out.logits[:N, -1, :]   # (N, vocab)
            logits_pri = out.logits[N:, -1, :]   # (N, vocab)
            combined = (1.0 + alpha) * logits_ctx - alpha * logits_pri

            combined = apply_temperature(combined, config.temperature)
            combined = top_p_filtering(combined, config.top_p)

            # Vectorised sampling — no per-sample Python loop
            if config.temperature is None or config.temperature <= 0:
                next_tokens = torch.argmax(combined, dim=-1)  # (N,)
            else:
                probs = torch.nn.functional.softmax(combined, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Mask finished samples, store, and update lengths
            next_tokens = torch.where(done, pad_id, next_tokens)
            generated[:, step] = next_tokens
            last_tokens = next_tokens
            gen_lengths += (~done).long()

            # Vectorised stop check — single GPU op instead of N .item() calls
            if stop_tensor is not None:
                hits = (next_tokens.unsqueeze(1) == stop_tensor.unsqueeze(0)).any(dim=1)
                done = done | hits

            if done.all():
                break

        # Decode results — one GPU→CPU sync for lengths, then slice
        lengths = gen_lengths.tolist()
        results = []
        for i in range(N):
            gen_ids = generated[i, :lengths[i]]
            results.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_stop(self, next_id: torch.Tensor, config: CADConfig) -> bool:
        token_id = next_id.item()
        if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
            return True
        if config.stop_token_ids and token_id in config.stop_token_ids:
            return True
        return False

    def _encode(self, prompt: str):
        if self.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            kwargs = dict(
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            try:
                sig = inspect.signature(self.tokenizer.apply_chat_template)
                if "enable_thinking" in sig.parameters:
                    kwargs["enable_thinking"] = False
            except (ValueError, TypeError):
                pass
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
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

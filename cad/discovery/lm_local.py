"""In-process HuggingFace Transformers LM for DSPy (no server required)."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import dspy
    from dspy.clients.base_lm import BaseLM
except ImportError:
    BaseLM = object  # type: ignore[assignment,misc]

try:
    from litellm import ModelResponse
    from litellm.types.utils import Choices, Message, Usage
except ImportError:
    ModelResponse = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class TransformersLM(BaseLM):
    """DSPy-compatible LM that runs a HuggingFace model in-process.

    This avoids the need for a separate vLLM / TGI / SGLang server.
    The model is loaded once on construction and kept in GPU memory.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> None:
        # BaseLM expects (model, model_type, ...) but we handle init ourselves
        super().__init__(
            model=f"transformers/{model_name}",
            model_type="chat",
            temperature=0.0,
            max_tokens=max_new_tokens,
            cache=kwargs.pop("cache", True),
            callbacks=kwargs.pop("callbacks", None),
            num_retries=0,
        )
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens

        logger.info("Loading model %s ...", model_name)
        cuda_available = torch.cuda.is_available()
        dtype = torch.bfloat16 if cuda_available else torch.float32
        device_map = device if device != "auto" else ("auto" if cuda_available else None)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._hf_model.eval()
        self._device = next(self._hf_model.parameters()).device
        logger.info("Model loaded on %s (dtype=%s)", self._device, dtype)

        # Ensure pad token exists
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # DSPy BaseLM interface
    # ------------------------------------------------------------------

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a completion and return a litellm-compatible response."""
        text = self._build_prompt(prompt, messages)
        max_tokens = kwargs.get("max_tokens", self._max_new_tokens)
        temperature = kwargs.get("temperature", 0.0)

        inputs = self._tokenizer(text, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"].to(self._device)
        input_len = input_ids.shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._hf_model.generate(input_ids, **gen_kwargs)

        new_tokens = output_ids[0, input_len:]
        completion = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._make_response(completion, input_len, len(new_tokens))

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Async variant (just calls sync forward; transformers has no async)."""
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> str:
        """Convert DSPy's prompt/messages into a single text string."""
        if messages:
            # Try chat template first
            if (
                hasattr(self._tokenizer, "chat_template")
                and self._tokenizer.chat_template is not None
            ):
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            # Fallback: concatenate messages
            parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(content)
                elif role == "user":
                    parts.append(content)
                elif role == "assistant":
                    parts.append(content)
            return "\n\n".join(parts)
        if prompt:
            return prompt
        return ""

    @staticmethod
    def _make_response(
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Any:
        """Build a litellm-compatible ModelResponse."""
        if ModelResponse is not None:
            return ModelResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model="transformers/local",
                choices=[
                    Choices(
                        index=0,
                        message=Message(role="assistant", content=completion),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        # Fallback dict if litellm not available
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AdapterInitConfig:
    model_name: str
    use_chat_template: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: Optional[str] = None


class BaseLMAdapter:
    """Abstract interface so steering logic can work with Transformers or vLLM."""

    def __init__(self, config: AdapterInitConfig):
        self.config = config

    @property
    def device(self) -> str:
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int:
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    @property
    def supports_hooks(self) -> bool:
        """True when the backend can expose per-layer forward hooks."""
        raise NotImplementedError

    @property
    def tokenizer(self):
        raise NotImplementedError

    def tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        raise NotImplementedError

    def generate(self, inputs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_layer(self, layer_id: int):
        raise NotImplementedError


class TransformersAdapter(BaseLMAdapter):
    """Thin wrapper around a HF causal LM so we can reuse steering code."""

    def __init__(self, config: AdapterInitConfig):
        super().__init__(config)
        dtype = torch.bfloat16 if config.device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=dtype,
            device_map="auto" if config.device == "cuda" else None,
            cache_dir=config.cache_dir,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Detect number of layers across architectures
        num_layers = getattr(self.model.config, "num_hidden_layers", None)
        if num_layers is None:
            if hasattr(self.model, "model"):
                num_layers = len(self.model.model.layers)
            else:
                num_layers = len(self.model.transformer.h)
        self._num_layers = num_layers

    @property
    def device(self) -> str:
        return self.config.device

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def supports_hooks(self) -> bool:
        return True

    def _apply_chat_template(self, prompt) -> Dict[str, torch.Tensor]:
        if isinstance(prompt, str):
            conversations = [[{"role": "user", "content": prompt}]]
        else:
            conversations = [[{"role": "user", "content": p}] for p in prompt]
        return self._tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
        )

    def tokenize(self, prompt) -> Dict[str, torch.Tensor]:
        """Prompt can be a string or list of strings."""
        if self.config.use_chat_template:
            encoded = self._apply_chat_template(prompt if isinstance(prompt, str) else list(prompt))
        else:
            encoded = self._tokenizer(prompt, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in encoded.items()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        with torch.no_grad():
            return self.model(**inputs)

    def generate(self, inputs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        gen_kwargs = dict(
            max_new_tokens=kwargs.pop("max_new_tokens", 256),
            do_sample=kwargs.pop("do_sample", False),
            temperature=kwargs.pop("temperature", 0.0),
            pad_token_id=self.eos_token_id,
            repetition_penalty=kwargs.pop("repetition_penalty", 1.1),
        )
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            return self.model.generate(**inputs, **gen_kwargs)

    def get_layer(self, layer_id: int):
        if hasattr(self.model, "model"):
            return self.model.model.layers[layer_id]
        return self.model.transformer.h[layer_id]


class VLLMAdapter(BaseLMAdapter):
    """
    Placeholder adapter for vLLM. The surface mirrors `TransformersAdapter`, but
    per-layer hooks are not yet implemented in this project. This keeps the
    steering controller compatible once vLLM exposes an activation API.
    """

    def __init__(self, config: AdapterInitConfig):
        super().__init__(config)
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "vLLM is not installed. Install it to use the VLLMAdapter."
            ) from exc

        self._SamplingParams = SamplingParams
        # Load lazily; steering code may still run in Transformers-only mode.
        self.model = LLM(model=config.model_name, tensor_parallel_size=1)

        # vLLM does not expose per-layer modules today; store a best-effort count.
        self._num_layers = None

    @property
    def device(self) -> str:
        # vLLM abstracts device placement
        return "vllm"

    @property
    def tokenizer(self):
        raise NotImplementedError("vLLM manages tokenization internally.")

    @property
    def eos_token_id(self) -> int:
        # vLLM handles eos internally; placeholder for API parity
        return 0

    @property
    def num_layers(self) -> int:
        return self._num_layers or 0

    @property
    def supports_hooks(self) -> bool:
        return False

    def tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Tokenization is handled inside vLLM; steering currently relies on "
            "Transformers for activation extraction."
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        raise NotImplementedError("Per-layer forward hooks are not supported in vLLM yet.")

    def generate(self, inputs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        # vLLM inference happens via text API; provide a minimal shim.
        prompts: List[str] = kwargs.pop("prompts", None) or []
        sampling = self._SamplingParams(
            max_tokens=kwargs.pop("max_new_tokens", 256),
            temperature=kwargs.pop("temperature", 0.0),
        )
        outputs = self.model.generate(prompts, sampling)
        # Return a list-like object of strings to match transformers decode flow.
        texts = [o.outputs[0].text for o in outputs]
        return texts

    def get_layer(self, layer_id: int):
        raise NotImplementedError("Layer hooks are not implemented for vLLM yet.")

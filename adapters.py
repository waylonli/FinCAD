from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import inspect

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def _supports_enable_thinking(tokenizer) -> bool:
    """Check if a tokenizer's chat template accepts enable_thinking."""
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
        return "enable_thinking" in sig.parameters
    except (ValueError, TypeError):
        return False


@dataclass
class AdapterInitConfig:
    model_name: str
    use_chat_template: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: Optional[str] = None
    attn_implementation: Optional[str] = None  # e.g. "flash_attention_2"
    compile_model: bool = False


class TransformersAdapter:
    """Thin wrapper around a HF causal LM for baseline and CAD evaluation."""

    def __init__(self, config: AdapterInitConfig):
        self.config = config
        dtype = torch.bfloat16 if config.device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )
        # Auto-detect best attention implementation on CUDA
        attn_impl = config.attn_implementation
        if attn_impl is None and config.device == "cuda":
            try:
                from transformers.utils import is_flash_attn_2_available
                if is_flash_attn_2_available():
                    attn_impl = "flash_attention_2"
            except ImportError:
                pass
            if attn_impl is None:
                attn_impl = "sdpa"
            print(f"[Adapter] Auto-selected attn_implementation={attn_impl!r}")

        load_kwargs: Dict[str, Any] = dict(
            dtype=dtype,
            device_map="auto" if config.device == "cuda" else None,
            cache_dir=config.cache_dir,
        )
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_kwargs,
        )

        if config.compile_model:
            print("[Adapter] Compiling model with torch.compile (mode='reduce-overhead')...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def device(self) -> str:
        return self.config.device

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    def _apply_chat_template(self, prompt) -> Dict[str, torch.Tensor]:
        if isinstance(prompt, str):
            conversations = [[{"role": "user", "content": prompt}]]
        else:
            conversations = [[{"role": "user", "content": p}] for p in prompt]
        kwargs: Dict[str, Any] = dict(
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
        )
        if _supports_enable_thinking(self._tokenizer):
            kwargs["enable_thinking"] = False
        return self._tokenizer.apply_chat_template(conversations, **kwargs)

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
        temperature = kwargs.pop("temperature", 0.0)
        gen_kwargs = dict(
            max_new_tokens=kwargs.pop("max_new_tokens", 256),
            do_sample=kwargs.pop("do_sample", temperature is not None and temperature > 0),
            temperature=temperature,
            pad_token_id=self.eos_token_id,
        )
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            return self.model.generate(**inputs, **gen_kwargs)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable evaluation runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

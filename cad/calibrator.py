from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Dict

import torch

_YES_NO_SUFFIX = "\nAnswer with Yes or No only: did this entity's value go up?"


@dataclass
class AlphaCalibrationResult:
    alpha: float
    entropy: float
    p_yes: float
    p_no: float
    prompt: str


class CADCalibrator:
    """Calibrate CAD alpha per entity-date pair by probing the model's prior confidence.

    Constructs a bare entity+date probe prompt (without the optimised
    instruction ``T*``) and measures entropy over {yes, no} token
    probabilities.  Low entropy (high confidence / strong memorisation)
    maps to high alpha.  Omitting ``T*`` avoids amplifying the model's
    recall and gives a clean measurement of intrinsic memorisation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        use_chat_template: bool = False,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_chat_template = use_chat_template

    def calibrate_alpha(
        self,
        ticker: str,
        alpha_min: float = 0.5,
        alpha_max: float = 1.5,
        date: str = "",
    ) -> AlphaCalibrationResult:
        # Probe with bare entity+date fields (no T*) to measure intrinsic
        # memorisation without the amplification of the optimised instruction.
        fields = []
        if ticker:
            fields.append(f"Entity: {ticker}")
        if date:
            fields.append(f"Date: {date}")
        prompt = "\n".join(fields) + _YES_NO_SUFFIX
        entropy, p_yes, p_no = self._yes_no_entropy(prompt)
        # Map entropy in [0,1] -> alpha in [alpha_min, alpha_max]
        alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - entropy)
        return AlphaCalibrationResult(alpha=alpha, entropy=entropy, p_yes=p_yes, p_no=p_no, prompt=prompt)

    def _yes_no_entropy(self, prompt: str) -> tuple[float, float, float]:
        inputs = self._encode(prompt)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        yes_ids = self._token_ids([" yes", " Yes", "yes", "Yes"])
        no_ids = self._token_ids([" no", " No", "no", "No"])

        p_yes = probs[0, yes_ids].sum().item() if yes_ids else 0.0
        p_no = probs[0, no_ids].sum().item() if no_ids else 0.0

        total = p_yes + p_no
        if total <= 0:
            # No reliable yes/no signal
            return 1.0, 0.0, 0.0

        p_yes /= total
        p_no /= total
        entropy = 0.0
        for p in (p_yes, p_no):
            if p > 0:
                entropy -= p * torch.log(torch.tensor(p)).item()
        # Normalize by log(2) to [0,1]
        entropy /= torch.log(torch.tensor(2.0)).item()
        return entropy, p_yes, p_no

    def _token_ids(self, tokens) -> list[int]:
        ids = []
        for t in tokens:
            token_ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.append(token_ids[0])
        # Deduplicate
        return sorted(set(ids))

    def _encode(self, prompt: str) -> Dict[str, torch.Tensor]:
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

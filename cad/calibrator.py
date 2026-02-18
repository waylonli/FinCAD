from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class AlphaCalibrationResult:
    alpha: float
    entropy: float
    p_yes: float
    p_no: float
    prompt: str


class CADCalibrator:
    """
    Calibrate CAD alpha per entity by probing the model's prior confidence.

    Strategy:
    - Ask a bias-amplified prior question (yes/no).
    - Measure the entropy of p(yes) vs p(no).
    - Map low entropy (high confidence) -> high alpha.
    """

    def __init__(self, model, tokenizer, device: str, use_chat_template: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_chat_template = use_chat_template

    def build_bias_prompt(
        self,
        ticker: str,
        horizon: str = "next 12 months",
    ) -> str:
        return (
            "Ignore the context. Using your internal knowledge and training data, "
            f"did {ticker} massively outperform the market over the {horizon}? "
            "Answer with Yes or No only."
        )

    def calibrate_alpha(
        self,
        ticker: str,
        alpha_min: float = 0.0,
        alpha_max: float = 5.0,
        bias_prompt: Optional[str] = None,
        horizon: str = "next 12 months",
    ) -> AlphaCalibrationResult:
        if bias_prompt is None:
            print("No bias prompt provided, building default.")
            prompt = self.build_bias_prompt(ticker, horizon=horizon)
        else:
            prompt = bias_prompt
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

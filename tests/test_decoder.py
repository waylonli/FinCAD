from types import SimpleNamespace

import torch

from cad.decoder import CADConfig, ContextAwareDecoder


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, prompt, return_tensors="pt", padding=False):
        token = 1 if prompt == "context" else 2
        return {"input_ids": torch.tensor([[token]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return {0: "A", 1: "B", 2: "C"}[values[0]]


class FakeModel:
    def __call__(self, input_ids, **kwargs):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 3)
        # Context alone prefers C. FinCAD subtracts the prior's stronger C
        # preference, making B the selected token.
        logits[0, -1] = torch.tensor([0.0, 2.0, 3.0])
        if batch > 1:
            logits[1, -1] = torch.tensor([0.0, 0.0, 5.0])
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_fincad_logit_combination_changes_selected_token():
    decoder = ContextAwareDecoder(FakeModel(), FakeTokenizer(), device="cpu")
    output = decoder.generate(
        "context",
        "prior",
        CADConfig(alpha=1.0, temperature=0.0, max_new_tokens=1),
    )
    assert output == "B"


def test_alpha_zero_uses_context_generation():
    decoder = ContextAwareDecoder(FakeModel(), FakeTokenizer(), device="cpu")
    output = decoder.generate(
        "context",
        "prior",
        CADConfig(alpha=0.0, temperature=0.0, max_new_tokens=1),
    )
    assert output == "C"

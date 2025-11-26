from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .adapters import BaseLMAdapter


@dataclass
class ScanSettings:
    """Configuration for layer scanning and vector extraction."""

    start_ratio: float = 0.45  # start scanning from 45% depth
    end_ratio: float = 0.85  # end at 85% depth
    layer_step: int = 1  # skip every n layers during scan
    max_scan_samples: int = 24  # samples to keep per class during scan
    val_split: float = 0.3
    accuracy_threshold: float = 0.03  # tolerance below best layer


class SteeringController:
    """
    End-to-end pipeline to:
    1) Collect activations,
    2) Train linear probes,
    3) Extract normalized steering vectors,
    4) Apply vectors at inference via forward hooks.
    """

    def __init__(self, adapter: BaseLMAdapter):
        self.adapter = adapter
        self.layer_ids: List[int] = []
        self.steering_vectors: Dict[int, torch.Tensor] = {}
        self.last_similarities: Dict[int, float] = {}

    # --- Activation utilities -------------------------------------------------
    def _capture_activation(self, prompt: str, layer_id: int) -> torch.Tensor:
        """Capture the final-token activation from a specific layer."""
        inputs = self.adapter.tokenize(prompt)
        activations: List[torch.Tensor] = []

        def hook_fn(_, __, output):
            hidden = output[0] if isinstance(output, tuple) else output
            last_token = hidden[:, -1, :].detach()
            activations.append(last_token)

        layer_module = self.adapter.get_layer(layer_id)
        handle = layer_module.register_forward_hook(hook_fn)
        try:
            self.adapter.forward(inputs)
        finally:
            handle.remove()

        return activations[0].to(torch.float32).view(-1)

    # --- Layer search and vector extraction ----------------------------------
    def scan_layers(
        self,
        memory_prompts: Sequence[str],
        generic_prompts: Sequence[str],
        settings: Optional[ScanSettings] = None,
    ) -> List[int]:
        """
        Finds layers whose activations separate Memory vs Logic prompts.
        Returns layers within `accuracy_threshold` of the best probe accuracy.
        """
        if settings is None:
            settings = ScanSettings()

        start_layer = int(self.adapter.num_layers * settings.start_ratio)
        end_layer = int(self.adapter.num_layers * settings.end_ratio)
        check_layers = list(range(start_layer, end_layer, settings.layer_step))

        print(f"Scanning layers: {check_layers}")

        max_acc = -1.0
        layer_scores: Dict[int, float] = {}

        scan_mem = list(memory_prompts)[: settings.max_scan_samples]
        scan_gen = list(generic_prompts)[: settings.max_scan_samples]
        y = np.array([1] * len(scan_mem) + [0] * len(scan_gen))

        for layer in check_layers:
            try:
                mem_acts = [self._capture_activation(p, layer).cpu().numpy() for p in scan_mem]
                gen_acts = [self._capture_activation(p, layer).cpu().numpy() for p in scan_gen]
                X = np.concatenate([mem_acts, gen_acts])

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=settings.val_split, stratify=y, random_state=42
                )
                clf = LogisticRegression(random_state=42, solver="liblinear", max_iter=500)
                clf.fit(X_train, y_train)

                acc = clf.score(X_val, y_val)
                layer_scores[layer] = acc
                print(f"Layer {layer:02d}: probe val acc = {acc:.2%}")
                max_acc = max(max_acc, acc)
            except Exception as exc:
                print(f"Skipping layer {layer}: {exc}")

        winners = [l for l, score in layer_scores.items() if score >= max_acc - settings.accuracy_threshold]
        winners.sort()
        print(f"Selected layers: {winners} (best={max_acc:.2%}, tol={settings.accuracy_threshold:.2%})")

        self.layer_ids = winners
        return winners

    def fit_steering_vectors(
        self,
        memory_prompts: Sequence[str],
        generic_prompts: Sequence[str],
        layers: Optional[Iterable[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Train a probe per layer and store normalized steering vectors."""
        if layers is None:
            layers = self.layer_ids
        layers = list(layers)
        if not layers:
            raise ValueError("No layers provided for steering vector extraction.")

        print(f"Fitting steering vectors for layers: {layers}")
        y = np.concatenate([np.ones(len(memory_prompts)), np.zeros(len(generic_prompts))])

        for layer in layers:
            print(f"Layer {layer}: collecting activations...")
            mem_acts = [self._capture_activation(p, layer) for p in memory_prompts]
            gen_acts = [self._capture_activation(p, layer) for p in generic_prompts]

            X_mem = torch.stack(mem_acts).cpu().numpy()
            X_gen = torch.stack(gen_acts).cpu().numpy()
            X = np.concatenate([X_mem, X_gen])

            probe = LogisticRegression(
                random_state=42,
                solver="liblinear",
                class_weight="balanced",
                max_iter=500,
            )
            probe.fit(X, y)
            print(f"  Train accuracy: {probe.score(X, y):.2%}")

            vector = torch.tensor(probe.coef_[0], dtype=torch.float32, device=self.adapter.device)
            vector = vector / torch.norm(vector)
            self.steering_vectors[layer] = vector

        return self.steering_vectors

    # --- Persistence ---------------------------------------------------------
    def save_vectors(self, path: str) -> None:
        """
        Save layer_ids and steering_vectors to a file.
        """
        payload = {
            "layer_ids": self.layer_ids,
            "vectors": {str(k): v.cpu() for k, v in self.steering_vectors.items()},
        }
        torch.save(payload, path)

    def load_vectors(self, path: str) -> None:
        """
        Load layer_ids and steering_vectors from a file.
        """
        payload = torch.load(path, map_location=self.adapter.device)
        self.layer_ids = [int(x) for x in payload.get("layer_ids", [])]
        vectors = payload.get("vectors", {})
        self.steering_vectors = {int(k): v.to(self.adapter.device) for k, v in vectors.items()}

    # --- Inference with steering ---------------------------------------------
    def generate(
        self,
        prompt: Union[str, List[str]],
        strength: float = 0.0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        """
        Generate text with optional steering applied uniformly across stored vectors.
        Supports single prompt (str) or batched prompts (List[str]).
        """
        if strength != 0.0 and not self.adapter.supports_hooks:
            raise NotImplementedError("The current adapter does not expose layer hooks for steering.")

        self.last_similarities = {}
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        inputs = self.adapter.tokenize(prompts)  # type: ignore[arg-type]
        input_lengths = compute_input_lengths(inputs)

        # Baseline
        if strength == 0.0 or not self.steering_vectors:
            outputs = self.adapter.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            return decode_outputs(self.adapter, outputs, inputs, input_lengths, single=isinstance(prompt, str))

        handles = []

        def make_hook(layer_id: int, vector: torch.Tensor):
            def hook_fn(_, __, output):
                hidden = output[0] if isinstance(output, tuple) else output
                perturb = vector.view(1, 1, -1) * strength
                perturbed = hidden + perturb

                if layer_id not in self.last_similarities:
                    original_vec = hidden[:, -1, :].detach().to(torch.float32)
                    perturbed_vec = perturbed[:, -1, :].detach().to(torch.float32)
                    sims = torch.nn.functional.cosine_similarity(
                        original_vec, perturbed_vec, dim=1
                    ).mean().item()
                    self.last_similarities[layer_id] = sims

                if isinstance(output, tuple):
                    return (perturbed,) + output[1:]
                return perturbed

            return hook_fn

        try:
            for layer_id, vector in self.steering_vectors.items():
                module = self.adapter.get_layer(layer_id)
                vec = vector.to(
                    device=module_device(module),
                    dtype=hidden_dtype(module),
                )
                handles.append(module.register_forward_hook(make_hook(layer_id, vec)))

            outputs = self.adapter.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        finally:
            for handle in handles:
                handle.remove()

        return decode_outputs(self.adapter, outputs, inputs, input_lengths, single=isinstance(prompt, str))


def hidden_dtype(module: torch.nn.Module) -> torch.dtype:
    """Utility: try to infer the module's parameter dtype for casting vectors."""
    for param in module.parameters():
        return param.dtype
    return torch.float32


def module_device(module: torch.nn.Module) -> torch.device:
    """Utility: try to infer the module's device placement."""
    for param in module.parameters():
        return param.device
    return torch.device("cpu")


def decode_outputs(
    adapter: BaseLMAdapter,
    outputs,
    inputs: Dict[str, torch.Tensor],
    input_lengths: List[int],
    single: bool = True,
) -> Union[str, List[str]]:
    """
    Unify decoding across adapters.
    - Transformers: decode token ids per sample, dropping prompt tokens.
    - vLLM (text list): return the generated text directly.
    """
    if isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], str):
        return outputs[0] if single else list(outputs)

    decoded = []
    for i, input_len in enumerate(input_lengths):
        decoded.append(
            adapter.tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)  # type: ignore
        )
    return decoded[0] if single else decoded


def compute_input_lengths(inputs: Dict[str, torch.Tensor]) -> List[int]:
    """Compute per-sample input lengths for decoding offsets."""
    if "attention_mask" in inputs:
        mask = inputs["attention_mask"]
        return mask.sum(dim=1).tolist()
    input_ids = inputs["input_ids"]
    return [input_ids.shape[1]] * input_ids.shape[0]

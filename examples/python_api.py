"""Minimal FinCAD Python API example using a released discovery profile."""

from adapters import AdapterInitConfig, TransformersAdapter
from cad import CADConfig, ContextAwareDecoder
from cad.discovery import NegativePromptBuilder

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DISCOVERY_FILE = "results/discovery/qwen2.5-7b-instruct.json"

context = "Context available at the historical decision date."
task = "Predict up or down. Return one word."

adapter = TransformersAdapter(
    AdapterInitConfig(model_name=MODEL_ID, use_chat_template=True)
)
builder = NegativePromptBuilder.from_file(DISCOVERY_FILE)
decoder = ContextAwareDecoder(
    adapter.model,
    adapter.tokenizer,
    device=adapter.device,
    use_chat_template=True,
)

output = decoder.generate(
    context_prompt=f"{context}\n\n{task}",
    prior_prompt=builder.build(entity="NVDA", date="2018-06-29", task_prompt=task),
    config=CADConfig(alpha=1.0, temperature=0.0, max_new_tokens=32),
)
print(output)

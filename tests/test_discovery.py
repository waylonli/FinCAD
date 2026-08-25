import json

from cad.discovery.builder import NegativePromptBuilder
from cad.discovery.config import OptimizedInstruction
from cad.discovery.registry import load_instruction, save_instruction


def test_negative_prompt_preserves_task_format():
    builder = NegativePromptBuilder(
        OptimizedInstruction(
            instruction="Recall training knowledge about {entity} after {date}.",
            model_name="example/model",
            score=0.75,
        )
    )
    prompt = builder.build(
        entity="NVDA",
        date="2018-06-29",
        task_prompt="Return exactly: up or down.",
    )

    assert "NVDA" in prompt
    assert "2018-06-29" in prompt
    assert "Return exactly: up or down." in prompt


def test_discovery_registry_round_trip(tmp_path):
    instruction = OptimizedInstruction(
        instruction="Recall what you know.",
        model_name="org/Example-7B",
        score=0.5,
        metadata={"optimizer": "MIPROv2"},
    )
    path = save_instruction(instruction, output_dir=str(tmp_path))
    loaded = load_instruction(path=str(path))

    assert loaded == instruction
    assert json.loads(path.read_text())["model_name"] == "org/Example-7B"

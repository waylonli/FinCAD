import json
from pathlib import Path

from cad.cli import _read_text, build_parser

ROOT = Path(__file__).resolve().parents[1]


def test_all_discovery_jsons_are_valid_and_use_public_model_ids():
    paths = sorted((ROOT / "results" / "discovery").glob("*.json"))
    assert paths
    for path in paths:
        data = json.loads(path.read_text())
        assert data["instruction"].strip()
        assert data["model_name"].strip()
        assert not data["model_name"].startswith(("/", "~")), path
        assert "/models/" not in data["model_name"], path


def test_paper_config_references_released_profiles():
    models = json.loads((ROOT / "configs" / "paper" / "models.json").read_text())
    assert len(models) == 11
    for model in models.values():
        assert (ROOT / model["discovery_file"]).exists()


def test_cli_parser_and_file_input(tmp_path):
    context = tmp_path / "context.txt"
    context.write_text("historical evidence")
    assert _read_text(None, str(context), "context") == "historical evidence"
    args = build_parser().parse_args(
        [
            "--model-name",
            "org/model",
            "--discovery-file",
            "profile.json",
            "--context",
            "evidence",
            "--task",
            "answer",
            "--alpha",
            "1",
        ]
    )
    assert args.use_chat_template is True
    assert args.alpha == 1.0

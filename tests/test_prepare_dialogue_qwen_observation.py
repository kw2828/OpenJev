"""Preparation regressions without loading tokenizers or model parameters."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_partial_cache_lookup_requests_only_pinned_model_inputs(monkeypatch, tmp_path):
    import sys

    calls = []
    def snapshot(model, **kwargs):
        calls.append((model, kwargs))
        if "allow_patterns" not in kwargs:
            raise RuntimeError("README.md and .gitattributes deliberately absent")
        assert kwargs["local_files_only"] is True
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_dialogue_qwen_observation.py"
    spec = importlib.util.spec_from_file_location("qwen_preparation_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.cached_model_path() == tmp_path
    model, options = calls[0]
    assert model == module.MODEL_ID
    assert options["revision"] == module.MODEL_REVISION
    assert options["allow_patterns"] == ["*.json", "*.safetensors", "*.jinja", "*.txt"]

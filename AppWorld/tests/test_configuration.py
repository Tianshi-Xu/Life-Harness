from pathlib import Path
import tempfile

import pytest

from life_harness_appworld.configuration import local_agent_config


def install_prompts(root):
    for name in ("api_predictor.txt", "function_calling_agent/instructions.txt",
                 "function_calling_agent/demos.json"):
        path = root / "experiments/prompts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")


def test_default_configuration_needs_no_custom_upstream_experiment():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install_prompts(root)
        config = local_agent_config(root, "http://127.0.0.1:8000/")
        assert config["model_config"]["base_url"] == "http://127.0.0.1:8000/v1"
        assert config["api_predictor_config"]["model_config"]["base_url"] == "http://127.0.0.1:8000/v1"
        assert Path(config["prompt_file_path"]).is_file()
        assert config["log_lm_calls"] is True
        config["model_config"]["name"] = "changed"
        assert local_agent_config(root, "http://127.0.0.1:8000/v1")["model_config"]["name"] == "Qwen3-4B"


def test_missing_upstream_prompts_fail_before_model_calls():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="experiment prompts"):
            local_agent_config(Path(tmp), "http://127.0.0.1:8000")


def test_installed_prompt_package_can_live_outside_appworld_root():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        install_prompts(root / "installed_agent")
        config = local_agent_config(root / "benchmark", "http://localhost:8000/v1",
            prompts_directory=root / "installed_agent/experiments/prompts")
        assert Path(config["api_predictor_config"]["prompt_file_path"]).is_file()


def test_invalid_endpoint_is_rejected():
    with pytest.raises(ValueError, match="HTTP"):
        local_agent_config(Path("."), "not-an-endpoint")

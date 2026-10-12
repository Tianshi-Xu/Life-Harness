"""Load the shipped local-server config without machine-local experiments."""
import json
from importlib.resources import files
from pathlib import Path


def local_agent_config(appworld_root: Path, model_server_url: str, *, prompts_directory: Path | None = None) -> dict:
    endpoint = model_server_url.rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("MODEL_SERVER_URL must be an HTTP(S) endpoint")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    prompts = (prompts_directory if prompts_directory is not None
               else appworld_root.resolve() / "experiments" / "prompts")
    config = json.loads(files("life_harness_appworld").joinpath("configs/local_function_calling.json").read_text())

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str):
            return value.replace("{prompts}", str(prompts)).replace("{endpoint}", endpoint)
        return value

    config = resolve(config)
    required = [config["prompt_file_path"], config["demo_messages_file_path"],
                config["api_predictor_config"]["prompt_file_path"]]
    if any(not Path(path).is_file() for path in required):
        raise FileNotFoundError("Install the AppWorld agents package and its experiment prompts")
    return config

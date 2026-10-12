"""Resolve API references at consumption time, never in saved experiment config."""
import os
import re


def credentials():
    key = os.environ.get("MR_API_KEY")
    host = os.environ.get("MR_HOST")
    if not key or not host:
        raise RuntimeError("Set MR_API_KEY and MR_HOST in the process environment")
    return key, host.rstrip("/") + "/protocol/openai/v1"


def bind_connections(module):
    original = module.get_llm_config

    def get(config, role="agent"):
        value = original(config, role)
        for key, item in value.items():
            if isinstance(item, str):
                match = re.fullmatch(r"\$\{env\.([A-Z0-9_]+)\}", item)
                if match:
                    value[key] = os.environ[match.group(1)]
        return value
    get.__wrapped__ = original
    module.get_llm_config = get


def configure_environment():
    key, url = credentials()
    os.environ.update(AHE_EVOLVE_API_KEY=key, AHE_EVOLVE_BASE_URL=url,
        AHE_USER_API_KEY=key, AHE_USER_API_BASE=url, QA_API_KEY=key,
        QA_BASE_URL=url, QA_MODEL_NAME="deepseek-v4-flash")
    return url

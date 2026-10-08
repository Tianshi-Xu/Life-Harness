"""Actual SDK transport regression; no benchmark tasks or grading data."""
import os
import sys
from pathlib import Path
from nexau import Agent
from nexau.archs.main_sub.config.config import AgentConfig
from nexau.core.messages import ReasoningBlock
from nexau.archs.main_sub.execution.hooks import Middleware, HookResult
from .api_broker import Broker
from .connection import credentials
from .records import write_json


class StripReasoningProbe(Middleware):
    def before_model(self, hook_input):
        messages = [m.model_copy(update={"content": [b for b in m.content
                    if not isinstance(b, ReasoningBlock)]}) for m in hook_input.messages]
        return HookResult.with_modifications(messages=messages)


def main():
    key, url = credentials()
    broker = Broker(key, url)
    os.environ["LLM_BASE_URL"] = broker.url
    workspace = Path(__file__).resolve().parents[1] / "smokes/api_stream_probe/workspace"
    sys.path.insert(0, str(workspace))
    config = AgentConfig.from_yaml(workspace / "code_agent.yaml")
    # Diagnostic-only fixture, never used in optimizer or benchmark execution.
    config.middlewares = [*(config.middlewares or []), StripReasoningProbe()]
    agent = Agent(config=config)
    result = None
    try:
        result = agent.run(message="Run the public ping transport test.")
    finally:
        broker.close()
        proof = {"completed": result is not None, "hits": broker.reasoning.hits,
            "misses": broker.reasoning.misses, "diagnostics": broker.reasoning.miss_details,
            "observed": [{"ids": list(ids), **value} for ids, value in broker.reasoning.observed_ids.items()]}
        write_json(Path(__file__).resolve().parents[1] / "smokes/api_stream_probe/restoration_proof.json", proof)
        print(proof)


if __name__ == "__main__":
    main()

"""Use the tracked OS Task and Docker controller without benchmark edits."""
import argparse
import asyncio
import copy
import glob
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
AB = ROOT / "AgentBench"


def parameters(split, network="agentbench-fc_default", life=False):
    import yaml
    if split not in {"train", "test"}:
        raise ValueError("explicit native train/test split required")
    document = yaml.safe_load((AB / "configs/tasks/os.yaml").read_text())
    config = copy.deepcopy(document["default"]["parameters"])
    config.update(copy.deepcopy(document["os-env_train" if split == "train" else "os-std"]["parameters"]))
    config.update(name=f"os-comparison-{split}", concurrency=1, sample_size=None,
                  task_ids=None, enabled=life, **{f"h{i}": life for i in range(2, 6)})
    # Match the tracked Docker/Redis stack, connecting from a host process.
    config["env_options"] = {"network_name": network, "state_driver": "redis",
                             "state_options": {"connection": {"host": "127.0.0.1"}}}
    # Pin glob order before integer IDs are assigned by the unchanged Task.
    files = []
    for item in config["data_config"]["files"]:
        paths = sorted(glob.glob(str(AB / item["problem_file"])))
        if not paths:
            raise ValueError("missing native OS task data")
        files.extend({**item, "problem_file": path,
                      "script_dir": str(AB / item["script_dir"])} for path in paths)
    config["data_config"]["files"] = files
    return config


def task(split, network="agentbench-fc_default", manifest=None):
    os.chdir(AB)
    if str(AB) not in sys.path:
        sys.path.insert(0, str(AB))
    from src.server.tasks.os_interaction import task as module
    receipt = None
    if manifest is not None:
        from evolution.harness_hooks import install_agentbench
        receipt = install_agentbench(module, "os", manifest)
    instance = module.OSInteraction(**parameters(split, network, manifest is not None))
    from .transport import install_shell_transport
    install_shell_transport(instance.env_controller)
    return instance, receipt


async def close_controller(instance):
    background = instance.env_controller_background_task
    if background is not None:
        background.cancel()
        await asyncio.gather(background, return_exceptions=True)
        instance.env_controller_background_task = None
    client = instance.env_controller._client
    if client is not None:
        await client.close()
        instance.env_controller._client = None


class ScriptedSession:
    """No model calls; public action path and native scoring/cleanup smoke only."""
    def __init__(self):
        self.history, self.rewards, self.turn = [], [], 0
        self.controller = SimpleNamespace(env_output=SimpleNamespace(history=[]))

    def inject(self, item):
        if isinstance(item, list):
            for value in item:
                self.inject(value)
        elif isinstance(item, dict) and "role" in item:
            self.history.append(copy.deepcopy(item))
        elif hasattr(item, "reward"):
            self.rewards.append(item.model_dump(mode="json"))

    def set_tools(self, tools):
        self.tools = copy.deepcopy(tools)

    def set_full_history(self, value):
        pass

    def cover(self, messages):
        self.history = copy.deepcopy(messages)

    async def action(self):
        self.turn += 1
        name, arguments = (("bash_action", {"script": "pwd"}) if self.turn == 1
                           else ("finish_action", {"thought": "Infrastructure smoke completed."}))
        message = {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"smoke_{self.turn}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}]}
        self.history.append(copy.deepcopy(message))
        return SimpleNamespace(messages=[message])


async def smoke(network="agentbench-fc_default"):
    instance, _ = task("train", network)
    from src.server.tasks.os_interaction.task import Container
    cycles = []
    try:
        for image in ("default", "packages", "ubuntu"):
            async with Container(instance.env_controller, image) as container:
                result = await container.execute("printf 'LIFE_OS_SMOKE_OK\\n'")
                if result.exit_code != 0 or b"LIFE_OS_SMOKE_OK" not in result.output:
                    raise RuntimeError("Docker environment action smoke failed: " + image)
                cycles.append({"image": image, "reset_action_close": True})
        ids = instance.get_indices()
        if not ids:
            raise RuntimeError("native OS TRAIN is empty")
        session = ScriptedSession()
        session.set_tools(instance.tools)
        result = await instance.start_sample(ids[0], session)
        if result.status.value.replace(" ", "_") in {"unknown", "task_error", "cancelled", "running"}:
            raise RuntimeError("native OS TRAIN smoke failed: " + str(result.result))
        if not any(m.get("role") == "tool" and m.get("tool_call_id") == "smoke_1" for m in session.history):
            raise RuntimeError("native OS action did not produce public tool feedback")
        return {"domain": "os", "split": "train", "paid_calls": 0,
                "cycles": cycles, "native_episode_status": result.status.value,
                "native_episode_turns": session.turn, "train_pool_size": len(ids),
                "released_harness_enabled": instance.harness_config.enabled}
    finally:
        await close_controller(instance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--network", default="agentbench-fc_default")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    result = asyncio.run(smoke(args.network))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Run genuine NexAU in its own interpreter with an episode-local RPC channel."""
import argparse
import json
import os
import sys
from pathlib import Path

from . import environment_rpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--agent-config", default="code_agent.yaml")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    sys.path.insert(0, str(workspace))
    # Imports are process-local; no candidate modules leak between episodes.
    channel = environment_rpc.Channel(int(os.environ["AHE_RPC_READ_FD"]),
                                      int(os.environ["AHE_RPC_WRITE_FD"]))
    environment_rpc.CHANNEL = channel
    agent = None
    tracer = None
    try:
        from nexau import Agent
        from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
        seed = channel.receive()
        if seed.get("kind") != "init":
            raise RuntimeError("first RPC message must be init")
        policy = seed.get("capability_policy")
        if policy:
            from .capabilities import validate_workspace
            validate_workspace(workspace, policy, args.agent_config)
        agent = Agent.from_yaml(workspace / args.agent_config)
        # A genuine tracer, including evolved middleware/tools/subagent calls.
        tracer = next((t for t in agent.config.tracers if isinstance(t, InMemoryTracer)), None)
        if tracer is None:
            raise RuntimeError("workspace must retain its native InMemoryTracer")
        if policy:
            from .capabilities import install_runtime_guard
            install_runtime_guard(agent, channel, policy)
        messages = seed["messages"]
        native_instructions = "\n\n".join(m.get("content", "") for m in messages
                                             if m["role"] == "system")
        context = {"native_instructions": native_instructions, "working_directory": str(workspace)}
        if policy:
            context.update(frozen_memory=(workspace / "LongTermMEMORY.md").read_text(),
                frozen_skills="\n\n".join(p.read_text() for p in sorted((workspace / "skills").rglob("*.md"))))
        first = True
        while True:
            history = [m for m in messages[:-1] if m["role"] != "system"] if first else None
            result = agent.run(message=messages[-1].get("content", ""),
                               history=history,
                               context=context)
            first = False
            if isinstance(result, tuple):
                result = result[0]
            channel.send({"kind": "final", "content": result or ""})
            followup = channel.receive()
            if followup.get("kind") != "continue":
                raise RuntimeError("expected continuation after a text reply")
            messages = followup["messages"]
            if not messages:
                messages = [{"role": "user", "content": "Continue."}]
            # Keep native user/environment messages intact, not just the last one.
            if len(messages) > 1:
                messages = [{"role": "user", "content": json.dumps(messages, ensure_ascii=False)}]
    except environment_rpc.EpisodeFinished:
        pass
    except BaseException as exc:
        channel.send({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if tracer is not None:
            from .records import write_json
            sys.path.insert(0, "/upstream")
            from trace_converter import extract_trace_data_from_inmemory_dump
            raw = tracer.dump_traces()
            write_json(args.trace_dir / "nexau_in_memory_tracer.json", raw)
            cleaned = extract_trace_data_from_inmemory_dump(raw,
                include_system_prompt_message=True, include_user_message=True,
                capture_errors=True, jsonable_output=True)
            write_json(args.trace_dir / "nexau_in_memory_tracer.cleaned.json", cleaned)
            channel.send({"kind": "closed", "nexau_spans": len(raw)})


if __name__ == "__main__":
    main()

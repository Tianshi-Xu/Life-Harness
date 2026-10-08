"""Mechanical, policy-free NexAU workspace generation from public tool schemas."""
import argparse
import json
from pathlib import Path

import yaml

from .backend import HERE, describe
from .records import write_json


def prepare(domain, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("never overwrite a baseline workspace")
    destination.mkdir(parents=True)
    schema = describe(domain, destination / "public_tool_schema.json")
    (destination / "tools").mkdir()
    (destination / "tool_descriptions").mkdir()
    for folder in ("middleware", "skills", "sub_agents"):
        (destination / folder).mkdir()
    (destination / "tools/__init__.py").write_text("")
    functions = ["from ahe_native.environment_rpc import invoke\n"]
    tools = []
    for item in schema["tools"]:
        fn = item["function"]
        name = fn["name"]
        if not name.isidentifier():
            raise ValueError("tool name is not a Python identifier")
        functions.append(f"def {name}(**arguments):\n    return invoke({name!r}, arguments)\n")
        tools.append({"name": name, "yaml_path": f"./tool_descriptions/{name}.tool.yaml",
                      "binding": f"tools.native:{name}"})
        (destination / f"tool_descriptions/{name}.tool.yaml").write_text(yaml.safe_dump({
            "type": "tool", "name": name, "description": fn.get("description", ""),
            "input_schema": fn["parameters"], "disable_parallel": True}, sort_keys=False))
    (destination / "tools/native.py").write_text("\n".join(functions))
    # Original task instructions remain public runtime input, not evolved policy.
    # The scaffold contributes no additional domain strategies.
    (destination / "systemprompt.md").write_text("{{ native_instructions }}\n")
    for name in ("LongTermMEMORY.md", "ShortTermMEMORY.md"):
        (destination / name).write_text("")
    config = {"type": "agent", "name": f"ahe_{domain}",
        "system_prompt": "./systemprompt.md", "system_prompt_type": "jinja",
        "tool_call_mode": "openai", "max_iterations": 300, "max_context_tokens": 40000,
        "llm_config": {"model": "${env.LLM_MODEL}", "base_url": "${env.LLM_BASE_URL}",
            "api_key": "${env.LLM_API_KEY}", "max_tokens": 4096, "temperature": 0.0,
            "api_type": "openai_chat_completion", "stream": False,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "tools": tools, "tracers": [{"import": "nexau.archs.tracer.adapters.in_memory:InMemoryTracer"}]}
    (destination / "code_agent.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (destination / "native_api.md").write_text(
        "# Native benchmark transport\n\n"
        "The public task instructions are provided via native_instructions at runtime.\n"
        "Task environments and evaluation are outside this workspace and immutable.\n"
        "tools/native.py forwards ordinary named actions/arguments to the native environment.\n"
        "You may evolve these Python wrappers, add internal tools, middleware, skills, "
        "subagents or memory, and revise the system prompt while keeping native instructions.\n"
        "The RPC bridge and grader are infrastructure and must not be changed.\n"
        "Do not change LLM configuration/tracing or read hidden grader data.\n"
        "There are no released Life rules, H2-H5 ownership requirements or single-mechanism limits.\n")
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(prepare(args.domain, args.destination))


if __name__ == "__main__":
    main()

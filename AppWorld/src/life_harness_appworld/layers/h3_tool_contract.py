from __future__ import annotations

from copy import deepcopy
from typing import Any


class H3ToolContract:
    """Embed stable AppWorld calling conventions into tool docs exactly once.

    H3 receives and returns tool descriptions only.  It cannot add/remove tools,
    inspect the trajectory, or prescribe a task-solving workflow.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        contracts = config.get("parameter_contracts", {})
        self.parameter_contracts = contracts if isinstance(contracts, dict) else {}
        scoped_contracts = config.get("tool_parameter_contracts", {})
        self.tool_parameter_contracts = (
            scoped_contracts if isinstance(scoped_contracts, dict) else {}
        )

    def apply(self, function_docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        tools = deepcopy(function_docs)
        augmented = 0
        for tool in tools:
            function = tool.get("function", {})
            parameters = function.get("parameters", {})
            properties = parameters.get("properties", {})
            if not isinstance(properties, dict):
                continue
            hints = [
                str(self.parameter_contracts[name]).strip()
                for name in properties
                if name in self.parameter_contracts
                and str(self.parameter_contracts[name]).strip()
            ]
            function_name = str(function.get("name", ""))
            for tool_prefix, contracts in self.tool_parameter_contracts.items():
                if not function_name.startswith(str(tool_prefix)) or not isinstance(
                    contracts, dict
                ):
                    continue
                hints.extend(
                    str(contracts[name]).strip()
                    for name in properties
                    if name in contracts and str(contracts[name]).strip()
                )
            if not hints:
                continue
            base = str(function.get("description", "")).rstrip()
            function["description"] = base + "\n\n[H3 environment contract] " + " ".join(hints)
            augmented += 1
        return tools, augmented

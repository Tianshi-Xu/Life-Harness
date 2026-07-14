from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


SENSITIVE_ARGUMENTS = {"password", "access_token"}


@dataclass(frozen=True)
class ToolCall:
    """One model-proposed action after its arguments have been parsed."""

    call_id: str
    name: str
    arguments: dict[str, Any]

    def signature(self) -> str:
        safe_arguments = {
            key: "<REDACTED>" if key in SENSITIVE_ARGUMENTS else value
            for key, value in self.arguments.items()
        }
        return self.name + "|" + json.dumps(
            safe_arguments, sort_keys=True, separators=(",", ":"), default=str
        )


@dataclass(frozen=True)
class GateDecision:
    """H2's complete output contract.

    H2 may pass the same action, perform an unambiguous representation repair,
    or block it.  It cannot synthesize a different semantic action.
    """

    call: ToolCall
    blocked: bool = False
    feedback: str = ""
    repairs: list[str] = field(default_factory=list)

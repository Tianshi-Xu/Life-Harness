from __future__ import annotations

import json
import re
from typing import Any

from life_harness_appworld.runtime_types import GateDecision, ToolCall


INTEGER_PATTERN = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
TOKEN_PLACEHOLDER_PATTERN = re.compile(
    r"^(?:YOUR_[A-Z0-9_]*TOKEN|<[^>]*TOKEN[^>]*>|[a-zA-Z0-9_]+__get_access_token\(\))$",
    re.I,
)

# Known values from the AppWorld tutorial examples that the model
# sometimes copies verbatim instead of retrieving real credentials.
TUTORIAL_VALUES = frozenset({
    "p8!Wq^3T", "Y2#kL!7z", "n4@H9!xP", "R7^d$1mB",
    "J5!qX^0c", "Z1*v9#hQ", "F8#r2@kD",
    "AbcDefGhIjKlMnOpQrStUvWxYz123456",
    "XyZ01234AbCdEfGhIjKlMnOpQrStUvWx",
    "LmNoPqRsTuVwXyZaBcDeFgHiJkMnOpQr",
    "default_token", "test_token", "access_token",
    "your_token_here", "placeholder",
})


class H2ActionGate:
    """Validate one proposed action immediately before environment execution.

    This layer deliberately has no trajectory or task state.  The only repair
    enabled by default is a lossless JSON-representation repair (for example,
    ``"7"`` to integer ``7`` when the schema requires an integer).  H2 never
    selects another tool, invents missing values, or changes semantic values.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.coerce_integer_strings = bool(config.get("coerce_integer_strings", True))
        self.reject_unknown_arguments = bool(config.get("reject_unknown_arguments", True))
        self.drop_undeclared_access_token = bool(
            config.get("drop_undeclared_access_token", True)
        )
        self.min_steps_before_complete = max(
            0, int(config.get("min_steps_before_complete", 0))
        )
        self.min_steps_before_fail = max(
            self.min_steps_before_complete,
            int(config.get("min_steps_before_fail", self.min_steps_before_complete)),
        )

    def realize(
        self,
        call_id: str,
        name: str,
        raw_arguments: Any,
        tools: dict[str, dict[str, Any]],
    ) -> GateDecision:
        """Parse and validate a raw function call as one H2 lifecycle hook."""
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except (json.JSONDecodeError, TypeError):
            return self._block(
                ToolCall(call_id, name, {}),
                "arguments are not valid JSON; emit one JSON object matching the tool schema",
            )
        if not isinstance(arguments, dict):
            return self._block(
                ToolCall(call_id, name, {}),
                "arguments must be a JSON object",
            )
        return self.apply(ToolCall(call_id, name, arguments), tools)

    def apply(
        self,
        call: ToolCall,
        tools: dict[str, dict[str, Any]],
    ) -> GateDecision:
        if call.name not in tools:
            return self._block(call, f"unknown or unavailable tool `{call.name}`")

        schema = tools[call.name].get("parameters", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict):
            properties = {}
        if not isinstance(required, list):
            required = []

        arguments = dict(call.arguments)
        repairs: list[str] = []
        if (
            self.drop_undeclared_access_token
            and "access_token" in arguments
            and "access_token" not in properties
        ):
            arguments.pop("access_token")
            repairs.append("removed undeclared `access_token`")

        unknown = sorted(set(arguments) - set(properties))
        if unknown and self.reject_unknown_arguments:
            accepted = sorted(set(properties))
            return self._block(
                call,
                "undeclared argument(s): " + ", ".join(f"`{name}`" for name in unknown)
                + ". Accepted arguments: " + ", ".join(f"`{name}`" for name in accepted),
            )

        missing = [name for name in required if name not in arguments]
        if missing:
            return self._block(
                call,
                "missing required argument(s): " + ", ".join(f"`{name}`" for name in missing),
            )

        for argument_name, value in arguments.items():
            property_schema = properties.get(argument_name, {})
            expected_type = property_schema.get("type") if isinstance(property_schema, dict) else None
            if (
                argument_name == "access_token"
                and isinstance(value, str)
                and (
                    TOKEN_PLACEHOLDER_PATTERN.fullmatch(value.strip())
                    or value.strip() in TUTORIAL_VALUES
                    or value.strip().startswith(("AbcDef", "XyZ0", "LmNoP"))
                )
            ):
                app_name = call.name.split("__", 1)[0] if "__" in call.name else "the app"
                return self._block(
                    call,
                    f"`access_token` is a placeholder, tutorial example value, or "
                    f"unevaluated expression. You must call {app_name}__login first "
                    f"and use the returned access_token (a JWT starting with 'eyJ').",
                )
            if (
                argument_name == "password"
                and isinstance(value, str)
                and value.strip() in TUTORIAL_VALUES
                and call.name.endswith("__login")
            ):
                app_name = call.name.split("__", 1)[0]
                return self._block(
                    call,
                    f"`password` is a tutorial example value, not the real password. "
                    f"Call supervisor__show_account_passwords and use the exact password "
                    f"listed for `{app_name}`.",
                )
            if (
                expected_type == "integer"
                and self.coerce_integer_strings
                and isinstance(value, str)
                and INTEGER_PATTERN.fullmatch(value)
            ):
                arguments[argument_name] = int(value)
                repairs.append(f"coerced `{argument_name}` from an integer string")
                value = arguments[argument_name]
            if not self._matches_type(value, expected_type):
                return self._block(
                    call,
                    f"argument `{argument_name}` must have JSON type `{expected_type}`",
                )

        realized = ToolCall(call.call_id, call.name, arguments)
        return GateDecision(call=realized, repairs=repairs)

    def apply_batch(
        self,
        decisions: list[GateDecision],
        step: int = 0,
    ) -> list[GateDecision]:
        """Delay terminal submission until parallel action results are observed
        and prevent premature submission before sufficient exploration."""
        decisions = [
            self._block_premature_submission(decision, step)
            for decision in decisions
        ]
        has_other_executable_action = any(
            not decision.blocked
            and decision.call.name != "supervisor__complete_task"
            for decision in decisions
        )
        if not has_other_executable_action:
            return decisions
        output: list[GateDecision] = []
        for decision in decisions:
            if decision.blocked or decision.call.name != "supervisor__complete_task":
                output.append(decision)
                continue
            output.append(
                GateDecision(
                    call=decision.call,
                    blocked=True,
                    feedback=(
                        "[H2 action gate] Terminal submission was not executed in "
                        "parallel with other actions. Observe their results first, then "
                        "submit in a later turn."
                    ),
                    repairs=decision.repairs,
                )
            )
        return output

    def _block_premature_submission(
        self, decision: GateDecision, step: int
    ) -> GateDecision:
        if decision.blocked or decision.call.name != "supervisor__complete_task":
            return decision
        status = decision.call.arguments.get("status", "success")
        if status == "fail":
            threshold = self.min_steps_before_fail
            message = (
                f"[H2 action gate] Failure submission at step {step} is premature. "
                "The task is doable. Continue interacting with the environment: "
                "gather information, perform the required actions, and only "
                "submit failure if you have exhausted all approaches near the step budget."
            )
        else:
            threshold = self.min_steps_before_complete
            message = (
                f"[H2 action gate] Terminal submission at step {step} is premature. "
                "Continue interacting with the environment: gather information, "
                "perform the required actions, and submit only after verifying completion."
            )
        if step >= threshold:
            return decision
        return GateDecision(
            call=decision.call,
            blocked=True,
            feedback=message,
            repairs=decision.repairs,
        )

    @staticmethod
    def _matches_type(value: Any, expected_type: Any) -> bool:
        if expected_type is None:
            return True
        if isinstance(expected_type, list):
            return any(H2ActionGate._matches_type(value, item) for item in expected_type)
        checks = {
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "null": lambda item: item is None,
        }
        check = checks.get(expected_type)
        return True if check is None else bool(check(value))

    @staticmethod
    def _block(call: ToolCall, reason: str) -> GateDecision:
        return GateDecision(
            call=call,
            blocked=True,
            feedback=f"[H2 action gate] Action not executed: {reason}.",
        )

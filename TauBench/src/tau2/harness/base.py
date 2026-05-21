"""Base types for the harness system."""

import json
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from tau2.environment.db import DB


@runtime_checkable
class HarnessRule(Protocol):
    """Protocol for a single pre-execution validation rule.

    Implementations check DB state (and tool kwargs) and raise ValueError
    when the call violates a policy rule.  The environment catches the
    ValueError and returns ToolMessage(error=True, content="Error: …"),
    giving the agent a chance to self-correct.
    """

    tool_name: str

    def check(self, db: DB, **kwargs: Any) -> None:
        """Validate the tool call.

        Args:
            db: Current domain database snapshot.
            **kwargs: The exact keyword arguments that will be passed to the
                tool (mirrors the tool's own parameter list).

        Raises:
            ValueError: If the rule is violated.  The message should be clear
                enough for the LLM to understand what went wrong and why.
        """
        ...


@runtime_checkable
class HarnessAnnotator(Protocol):
    """Protocol for post-execution tool response annotation (H4 / H6).

    Implementations receive the DB snapshot and the successful tool result,
    and may return a short annotation string to append to the tool response
    visible to the agent.  Returning ``None`` means no annotation.

    Unlike HarnessRule this fires on the SUCCESS path — it never blocks the
    call.  Its purpose is to inject state-aware context (order status tables,
    eligibility flags, completion reminders) at the exact moment in the
    conversation where it is most useful.

    Replay safety
    -------------
    Annotators run AFTER the tool executes, so they see the updated DB state.
    Like HarnessRule, implementations MUST only read from DB/kwargs; they must
    NOT have side-effects or maintain cross-call state.
    """

    tool_name: str

    def annotate(self, db: DB, result: Any, **kwargs: Any) -> str | None:
        """Produce an annotation to append to the tool response.

        Args:
            db: Current domain database snapshot (post-execution).
            result: The raw return value of the tool (a Pydantic model, str,
                dict, etc.).
            **kwargs: The keyword arguments that were passed to the tool.

        Returns:
            A short annotation string, or ``None`` to add nothing.
        """
        ...


def _serialize_tool_result(result: Any) -> str:
    """Replicate Environment.to_json_str serialisation for annotation attachment.

    ``Environment.to_json_str`` already returns the string as-is, so if we
    pre-serialise here the environment will pass it through without modification.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return json.dumps(result.model_dump(), default=str)
    if isinstance(result, list):
        return json.dumps(
            [r.model_dump() if isinstance(r, BaseModel) else r for r in result],
            default=str,
        )
    return json.dumps(result, default=str)


class HarnessedToolKitMixin:
    """Mixin that adds pre-execution harness validation and post-execution
    annotation to a ToolKitBase subclass.

    Subclasses define class-level dicts:

    ``harness_rules`` – maps tool names to lists of HarnessRule instances
    (H2 pre-execution validation, existing behaviour)::

        harness_rules = {
            "cancel_reservation": [CancelFlightRule()],
        }

    ``harness_annotators`` – maps tool names to lists of HarnessAnnotator
    instances (H4/H6 post-execution annotation, new behaviour)::

        harness_annotators = {
            "get_user_details": [OrderStatusAnnotator()],
            "exchange_delivered_order_items": [ExchangeCompletionAnnotator()],
        }

    Replay safety
    -------------
    Both rules and annotators must only inspect DB state (reconstructed from
    WRITE tools during set_state replay).  Rules that depend on READ-tool
    history or conversation text must NOT be implemented here.
    """

    harness_rules: dict[str, list] = {}
    harness_annotators: dict[str, list] = {}

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Run H2 checks, execute the tool, then apply H4/H6 annotations."""
        for rule in self.harness_rules.get(tool_name, []):
            # Pass toolkit=self so rules can inspect in-memory toolkit state
            # (e.g. _agent_discoverable_tools_state).  Existing rules ignore it
            # via **kwargs — fully backward-compatible.
            rule.check(self.db, toolkit=self, **kwargs)  # type: ignore[attr-defined]

        result = super().use_tool(tool_name, **kwargs)  # type: ignore[misc]
        calls = getattr(self, "_harness_successful_calls", None)
        if calls is None:
            calls = []
            setattr(self, "_harness_successful_calls", calls)
        calls.append({"tool_name": tool_name, "kwargs": dict(kwargs)})

        annotations = [
            ann
            for annotator in self.harness_annotators.get(tool_name, [])
            for ann in [
                annotator.annotate(self.db, result, toolkit=self, **kwargs)  # type: ignore[attr-defined]
            ]
            if ann
        ]
        if annotations:
            return _serialize_tool_result(result) + "\n\n" + "\n".join(annotations)
        return result

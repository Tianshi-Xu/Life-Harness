"""Harness module: pluggable pre-execution validation for tau2 domain tools.

A harness intercepts tool calls before they modify the database, checks policy
rules derived from the domain's policy.md, and raises ValueError (which the
environment converts to ToolMessage(error=True)) when a rule is violated.

This lets the LLM agent self-correct after receiving the error message, rather
than silently making a wrong DB change that the evaluator would penalise.

Usage
-----
Enable via env_kwargs when running:

    tau2 run --domain airline --env-kwargs harness_enabled=True ...

Design notes
------------
- H2 harnesses only inspect DB state (not conversation history), so they are
  safe for set_state replay: the DB state is identical during simulation and
  replay, so the same error is produced both times.
- H4 (stateful conversation-level gates) is deliberately **not** implemented
  as a WRITE gate; it is instead realised as H5 (system-prompt injection).
"""

from tau2.harness.base import HarnessedToolKitMixin, HarnessRule
from tau2.harness.h3_tools import (
    H3AirlineToolDescriptionMixin,
    H3RetailToolDescriptionMixin,
    H3TelecomToolDescriptionMixin,
    H3ToolDescriptionMixin,
)

__all__ = [
    "HarnessedToolKitMixin",
    "HarnessRule",
    "H3ToolDescriptionMixin",
    "H3AirlineToolDescriptionMixin",
    "H3RetailToolDescriptionMixin",
    "H3TelecomToolDescriptionMixin",
]

from __future__ import annotations

import hashlib
import re
from typing import Any

from life_harness_appworld.runtime_types import ToolCall


ENVIRONMENT_ERROR_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:execution failed|execution error|traceback|error\b)|response status code (?:is )?\d{3}",
    re.I,
)
CREATE_CONFLICT_PATTERN = re.compile(
    r"response status code (?:is )?409|already exists", re.I
)
AUTHENTICATION_FAILURE_PATTERN = re.compile(
    r"response status code (?:is )?401|invalid credentials|access token is missing, invalid or expired",
    re.I,
)


class H4TrajectoryMonitor:
    """Observe executed actions and emit bounded next-turn trajectory feedback.

    H4 is post-execution only.  It never changes an observation, rewrites model
    history, blocks a future call, or passes state to H2/H3/H5.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.duplicate_threshold = max(2, int(config.get("duplicate_threshold", 3)))
        self.error_streak_threshold = max(2, int(config.get("error_streak_threshold", 2)))
        self.budget_fraction = float(config.get("budget_fraction", 0.8))
        self.max_no_action_history_characters = int(
            config.get("max_no_action_history_characters", 12000)
        )
        self.max_auth_emissions = max(1, int(config.get("max_auth_emissions", 3)))
        self.reset()

    def reset(self) -> None:
        self.action_observations: list[tuple[str, str]] = []
        self.error_streak = 0
        self.no_tool_streak = 0
        self.emitted: set[str] = set()
        self.emit_counts: dict[str, int] = {}

    @staticmethod
    def is_environment_error(output: str) -> bool:
        return bool(ENVIRONMENT_ERROR_PATTERN.search(output))

    def compact_no_action_message(self, message: dict[str, Any]) -> int:
        """Drop only oversized private text from a turn that issued no action."""
        if message.get("tool_calls"):
            return 0
        reasoning = str(message.get("reasoning_content", "") or "")
        content = str(message.get("content", "") or "")
        removed = len(reasoning) + len(content)
        if removed <= self.max_no_action_history_characters:
            return 0
        message["reasoning_content"] = ""
        message["content"] = ""
        return removed

    def observe(
        self,
        step: int,
        max_steps: int,
        calls: list[ToolCall],
        outputs: list[str],
        no_tool: bool = False,
    ) -> list[tuple[str, str]]:
        interventions: list[tuple[str, str]] = []

        if no_tool:
            self.no_tool_streak += 1
            if self.no_tool_streak >= 2:
                interventions.extend(
                    self._emit_once(
                        "no_tool_streak",
                        "[H4 trajectory monitor] Two consecutive turns produced no environment action. Choose one available tool call or submit only if the task is complete.",
                    )
                )
        else:
            self.no_tool_streak = 0

        for call, output in zip(calls, outputs, strict=False):
            observation_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()[:16]
            self.action_observations.append((call.signature(), observation_hash))
            self.error_streak = self.error_streak + 1 if self.is_environment_error(output) else 0
            if CREATE_CONFLICT_PATTERN.search(output):
                interventions.extend(
                    self._emit_once(
                        "create_conflict",
                        "[H4 trajectory monitor] A create action was rejected because the record already exists. Do not retry creation for that item; inspect the existing record with an available read/list tool, then use an available update action only if its observed state differs from the requested state.",
                    )
                )
            if AUTHENTICATION_FAILURE_PATTERN.search(output):
                app_name = call.name.split("__", 1)[0] if "__" in call.name else "target"
                interventions.extend(
                    self._emit_auth(app_name)
                )

        if self.error_streak >= self.error_streak_threshold:
            interventions.extend(
                self._emit_once(
                    "error_streak",
                    f"[H4 trajectory monitor] {self.error_streak} consecutive executed actions returned environment errors. Do not repeat the same failing pattern; re-check the available tool schema and use identifiers observed from successful reads.",
                )
            )

        tail = self.action_observations[-self.duplicate_threshold :]
        if len(tail) == self.duplicate_threshold and len(set(tail)) == 1:
            interventions.extend(
                self._emit_once(
                    "exact_duplicate:" + tail[-1][0] + ":" + tail[-1][1],
                    f"[H4 trajectory monitor] The same action produced the same observation {self.duplicate_threshold} times consecutively. Change the tool or arguments before continuing.",
                )
            )

        if len(self.action_observations) >= 4:
            a, b, c, d = self.action_observations[-4:]
            if a == c and b == d and a != b:
                interventions.extend(
                    self._emit_once(
                        "abab:" + a[0] + ":" + b[0],
                        "[H4 trajectory monitor] The last four executed steps form an A-B-A-B loop with unchanged observations. Choose an action outside this two-step cycle.",
                    )
                )

        budget_step = max(1, int(max_steps * self.budget_fraction))
        if step >= budget_step:
            interventions.extend(
                self._emit_once(
                    "budget",
                    f"[H4 trajectory monitor] {step} of {max_steps} steps have been used. Do not submit merely because the budget is low. Prioritize unresolved requirements, batch independent same-schema actions when valid, and submit only after verifying completion.",
                )
            )
        return interventions

    def _emit_once(self, key: str, message: str) -> list[tuple[str, str]]:
        if key in self.emitted:
            return []
        self.emitted.add(key)
        return [(key.split(":", 1)[0], message)]

    def _emit_auth(self, app_name: str) -> list[tuple[str, str]]:
        key = f"authentication_failure:{app_name}"
        count = self.emit_counts.get(key, 0)
        if count >= self.max_auth_emissions:
            return []
        self.emit_counts[key] = count + 1
        count = count + 1
        if count == 1:
            message = (
                f"[H4 trajectory monitor] Authentication for `{app_name}` was rejected. "
                f"Do not retry the rejected value or use a password as `access_token`; "
                f"read the exact supervisor profile/account password. If that app's login "
                f"tool is available, call it with its documented username convention and "
                f"use only the returned access token."
            )
        elif count == 2:
            message = (
                f"[H4 trajectory monitor] Authentication for `{app_name}` still rejected. "
                f"You must: 1) call supervisor__show_account_passwords to get the correct "
                f"password for {app_name}, 2) call {app_name}__login with the supervisor "
                f"email and that exact password, 3) use only the returned access_token."
            )
        else:
            message = (
                f"[H4 trajectory monitor] Repeated authentication failure for `{app_name}`. "
                f"Stop guessing. Call supervisor__show_profile for the email, "
                f"supervisor__show_account_passwords for the password, then "
                f"{app_name}__login(username=email, password=password). "
                f"Use the returned access_token for all subsequent {app_name} calls."
            )
        return [("authentication_failure", message)]

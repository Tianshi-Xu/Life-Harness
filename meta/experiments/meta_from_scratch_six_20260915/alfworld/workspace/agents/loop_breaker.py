"""Repetition-triggered recovery reminder.

Statelessly reconstructs recent (action, observation) pairs from the request
messages. When the trailing actions repeat (two identical actions in a row, or
a genuine period-2 / period-3 action-observation cycle), a compact generic
reminder is appended to the content of the last message so the next model turn
is nudged toward a different offered action. With no repetition detected the
request is returned unchanged.
"""

import json
import re

_ACTIONS_MARKER = re.compile(r"AVAILABLE ACTIONS", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_GOAL_MARKER = "Your task is to:"
_MAX_PAIRS = 12


def register():
    return None


class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------
    def transform_request(self, request):
        try:
            return self._with_recovery_reminder(request)
        except Exception:
            return request

    def transform_response(self, response, public_history):
        return response

    # ------------------------------------------------------------------
    # Reminder construction
    # ------------------------------------------------------------------
    def _with_recovery_reminder(self, request):
        if not isinstance(request, dict):
            return request
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            return request
        pairs = self._recent_pairs(messages)
        if len(pairs) < 2:
            return request
        description = self._repetition_description(pairs)
        if description is None:
            return request
        last = messages[-1]
        if not isinstance(last, dict):
            return request
        content = last.get("content")
        if not isinstance(content, str):
            return request
        reminder = self._build_reminder(description, self._goal_text(messages))
        updated_request = dict(request)
        updated_messages = list(messages)
        updated_last = dict(last)
        updated_last["content"] = content + "\n\n" + reminder
        updated_messages[-1] = updated_last
        updated_request["messages"] = updated_messages
        return updated_request

    def _build_reminder(self, description, goal):
        parts = ["Reminder: " + description + " with no progress."]
        if goal:
            parts.append("Goal: " + goal + ".")
        parts.append(
            "Choose a DIFFERENT action strictly from the currently offered "
            "action list; never repeat the same action twice in a row; try an "
            "unvisited location or an unexplored container; re-read the goal "
            "and check the exact object name and its required state."
        )
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(text):
        return _WS_RE.sub(" ", text).strip().casefold()

    @staticmethod
    def _observation_head(content):
        match = _ACTIONS_MARKER.search(content)
        return content[: match.start()] if match else content

    @staticmethod
    def _extract_action(message):
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return None
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            parsed = None
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except Exception:
                    parsed = None
            elif isinstance(arguments, dict):
                parsed = arguments
            if isinstance(parsed, dict):
                action = parsed.get("action")
                if isinstance(action, str) and action.strip():
                    return action
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                return name
        return None

    def _recent_pairs(self, messages):
        pairs = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            action = self._extract_action(message)
            if action is None:
                continue
            if index + 1 >= len(messages):
                continue
            following = messages[index + 1]
            if not isinstance(following, dict):
                continue
            if following.get("role") not in ("tool", "user"):
                continue
            content = following.get("content")
            if not isinstance(content, str):
                continue
            pairs.append((action, content))
        return pairs[-_MAX_PAIRS:]

    def _goal_text(self, messages):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            start = content.find(_GOAL_MARKER)
            if start != -1:
                start += len(_GOAL_MARKER)
                match = _ACTIONS_MARKER.search(content, start)
                end = match.start() if match else len(content)
                return content[start:end].strip()
            return content[:200].strip()
        return ""

    # ------------------------------------------------------------------
    # Repetition detection
    # ------------------------------------------------------------------
    def _repetition_description(self, pairs):
        keys = [
            (self._normalize(action), self._normalize(self._observation_head(obs)))
            for action, obs in pairs
        ]
        actions = [key[0] for key in keys]
        if len(actions) >= 2 and actions[-1] == actions[-2]:
            count = 1
            cursor = len(actions) - 1
            while cursor > 0 and actions[cursor] == actions[cursor - 1]:
                count += 1
                cursor -= 1
            return (
                'the action "'
                + pairs[-1][0]
                + '" was repeated '
                + str(count)
                + " times in a row"
            )
        for period in (2, 3):
            if len(pairs) < 3 * period:
                continue
            if all(
                keys[-1 - offset] == keys[-1 - offset - period]
                for offset in range(2 * period)
            ):
                cycle = " then ".join(
                    '"' + pairs[-period + step][0] + '"' for step in range(period)
                )
                return (
                    "the cycle of actions "
                    + cycle
                    + " repeated 3 times"
                )
        return None

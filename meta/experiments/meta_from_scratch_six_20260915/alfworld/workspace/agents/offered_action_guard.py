"""Stateless request reminders layered on repetition-triggered recovery.

Statelessly reconstructs recent (action, observation) pairs from the request
messages. When the trailing actions repeat (two identical actions in a row, or
a genuine period-2 / period-3 action-observation cycle), a compact generic
reminder is appended to the content of the last message. Additionally, when
the goal requires an object to be cleaned, heated or cooled and no offered
transform action on that object has been chosen yet, a short goal-derived
precondition reminder is appended in the same place. When the goal
asks to look at or examine an object under a lamp-like device, an earlier
chosen action already used that device, and the goal object has not been
picked up from an offered action, a short hold reminder is appended in the
same place. Finally, when the goal asks to put an object in or on a
destination, the goal object has never been picked up, the trailing actions
stall (one action repeats, or a take/put/clean/heat/cool/use action produced
an observation without its success phrase while offered go-to/open actions
remain unused), a short search redirect is appended in the same place.
Finally, when the most recently chosen action was not among the actions
offered just before it, a short offered-action guard reminder quoting that
action is appended in the same place, telling the agent not to repeat the
unoffered action, to choose the next action verbatim from the currently
offered list, and to first choose an offered action that makes the intended
one possible. Malformed input, a goal with no applicable precondition, a
lamp goal, a goal object already picked up, no stall, no recent chosen
action, an absent or unparsable offered-action block, or a most recent
action that is still offered leaves the request unchanged.
"""

import json
import re

_ACTIONS_MARKER = re.compile(r"AVAILABLE ACTIONS", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_GOAL_MARKER = "Your task is to:"
_MAX_PAIRS = 12

_PUT_STATE_GOAL_RE = re.compile(
    r"^put\s+(?:a\s+|an\s+|the\s+)?(?P<state>clean|hot|cool)\s+"
    r"(?P<object>.+?)\s+(?P<prep>in(?:to)?|on(?:to)?)\s+"
    r"(?P<dest>.+?)\s*[.?!]?\s*$"
)
_TRANSFORM_GOAL_RE = re.compile(
    r"^(?P<verb>clean|heat|cool)\s+(?:some\s+|a\s+|an\s+|the\s+)?"
    r"(?P<object>.+?)\s*"
    r"(?:and\s+put\s+(?:it|them)\s+(?P<prep>in(?:to)?|on(?:to)?)\s+"
    r"(?P<dest>.+?))?\s*[.?!]?\s*$"
)
_OBJECT_INDEX_RE = re.compile(r"\s+\d+$")
_STATE_TO_VERB = {"clean": "clean", "hot": "heat", "cool": "cool"}
_VERB_TO_STATE = {"clean": "clean", "heat": "hot", "cool": "cool"}
_LIGHT_GOAL_RE = re.compile(
    r"^(?:look at|examine)\s+(?:the\s+|a\s+|an\s+)?(?P<object>.+?)\s+"
    r"(?:under|with)\s+(?:the\s+)?(?P<device>[a-z]*(?:lamp|light)[a-z]*)\s*[.?!]?\s*$"
)
_PLAIN_PUT_GOAL_RE = re.compile(
    r"^put\s+(?:some\s+|a\s+|an\s+|the\s+|two\s+)?(?P<object>.+?)\s+"
    r"(?P<prep>in(?:to)?|on(?:to)?)\s+(?P<dest>.+?)\s*[.?!]?\s*$"
)
_FIND_PUT_GOAL_RE = re.compile(
    r"^find\s+\S+\s+(?P<object>.+?)\s+and\s+put\s+them\s+"
    r"(?P<prep>in(?:to)?|on(?:to)?)\s+(?P<dest>.+?)\s*[.?!]?\s*$"
)
_PICKUP_PHRASE = "you pick up"
_MANIPULATION_PHRASES = {
    "take": "you pick up",
    "put": "you put",
    "clean": "you clean",
    "heat": "you heat",
    "cool": "you cool",
    "use": "you turn",
}
_MOVE_PREFIXES = ("go to ", "open ")
_MAX_STALL_ACTIONS = 12
_STALL_REPEATS = 4
_MAX_EFFECT_WINDOW = 4
_MAX_LOCATIONS = 3
_ACTION_VERBS = (
    "go",
    "open",
    "close",
    "take",
    "put",
    "clean",
    "heat",
    "cool",
    "use",
    "move",
    "examine",
    "look",
    "inventory",
    "toggle",
)


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
            updated = self._with_state_reminder(self._with_recovery_reminder(request))
            updated = self._with_light_hold_reminder(updated)
            updated = self._with_search_redirect(updated)
            return self._with_offered_action_guard(updated)
        except Exception:
            return request

    def transform_response(self, response, public_history):
        return response

    # ------------------------------------------------------------------
    # Repetition reminder (unchanged behavior)
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
    # State-precondition reminder
    # ------------------------------------------------------------------
    def _with_state_reminder(self, request):
        if not isinstance(request, dict):
            return request
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            return request
        last = messages[-1]
        if not isinstance(last, dict):
            return request
        content = last.get("content")
        if not isinstance(content, str):
            return request
        parsed = self._parse_state_goal(self._goal_text(messages))
        if parsed is None:
            return request
        verb, state, obj, prep, dest = parsed
        if self._transform_already_chosen(messages, verb, obj):
            return request
        reminder = self._build_state_reminder(verb, state, obj, prep, dest)
        updated_request = dict(request)
        updated_messages = list(messages)
        updated_last = dict(last)
        updated_last["content"] = content + "\n\n" + reminder
        updated_messages[-1] = updated_last
        updated_request["messages"] = updated_messages
        return updated_request

    def _parse_state_goal(self, goal):
        if not isinstance(goal, str) or not goal.strip():
            return None
        text = self._normalize(goal)
        match = _PUT_STATE_GOAL_RE.match(text)
        if match:
            state = match.group("state")
            return (
                _STATE_TO_VERB.get(state, ""),
                state,
                self._object_token(match.group("object")),
                self._preposition(match.group("prep")),
                self._clean_phrase(match.group("dest")),
            )
        match = _TRANSFORM_GOAL_RE.match(text)
        if match:
            verb = match.group("verb")
            dest = match.group("dest")
            return (
                verb,
                _VERB_TO_STATE.get(verb, ""),
                self._object_token(match.group("object")),
                self._preposition(match.group("prep")),
                self._clean_phrase(dest) if dest else "",
            )
        return None

    @staticmethod
    def _preposition(prep):
        if isinstance(prep, str) and prep.startswith("on"):
            return "on"
        return "in"

    def _clean_phrase(self, phrase):
        return self._normalize(phrase).strip("\"'")

    def _object_token(self, phrase):
        token = self._clean_phrase(phrase)
        return _OBJECT_INDEX_RE.sub("", token).strip()

    def _transform_already_chosen(self, messages, verb, obj):
        if not verb or not obj:
            return False
        pattern = re.compile(r"\b" + re.escape(obj) + r"\b")
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            action = self._extract_action(message)
            if action is None:
                continue
            action_norm = self._normalize(action)
            if not action_norm.startswith(verb):
                continue
            if not pattern.search(action_norm):
                continue
            offered = self._offered_before(messages, index)
            if action_norm in offered:
                return True
        return False

    def _offered_before(self, messages, index):
        for cursor in range(index - 1, -1, -1):
            message = messages[cursor]
            if not isinstance(message, dict):
                continue
            if message.get("role") not in ("tool", "user"):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                return set()
            match = _ACTIONS_MARKER.search(content)
            if match is None:
                return set()
            offered = set()
            for line in content[match.end():].splitlines():
                line = line.strip().lstrip(":").strip()
                if line:
                    offered.add(self._normalize(line))
            return offered
        return set()

    def _build_state_reminder(self, verb, state, obj, prep, dest):
        if dest:
            where = prep + " " + dest
            lead = (
                'State check: "'
                + obj
                + '" must be '
                + state
                + " before going "
                + where
                + "."
            )
            tail = (
                " Find and take "
                + obj
                + ", carry it there, choose the offered "
                + verb
                + " action, then deliver it "
                + where
                + "."
            )
        else:
            lead = 'State check: "' + obj + '" must be ' + state + "."
            tail = (
                " Find and take "
                + obj
                + ", carry it there, choose the offered "
                + verb
                + " action, then deliver it."
            )
        body = (
            ' A "'
            + verb
            + '" action is offered only while you hold '
            + obj
            + " at the right appliance/location; actions not in the "
            "offered list do nothing."
        )
        return lead + body + tail

    # ------------------------------------------------------------------
    # Lamp-hold reminder
    # ------------------------------------------------------------------
    def _with_light_hold_reminder(self, request):
        if not isinstance(request, dict):
            return request
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            return request
        last = messages[-1]
        if not isinstance(last, dict):
            return request
        content = last.get("content")
        if not isinstance(content, str):
            return request
        parsed = self._parse_light_goal(self._goal_text(messages))
        if parsed is None:
            return request
        obj, device = parsed
        if not self._light_device_action_seen(messages, device):
            return request
        if self._light_hold_already_chosen(messages, obj):
            return request
        reminder = self._build_light_hold_reminder(obj, device)
        updated_request = dict(request)
        updated_messages = list(messages)
        updated_last = dict(last)
        updated_last["content"] = content + "\n\n" + reminder
        updated_messages[-1] = updated_last
        updated_request["messages"] = updated_messages
        return updated_request

    def _parse_light_goal(self, goal):
        if not isinstance(goal, str) or not goal.strip():
            return None
        match = _LIGHT_GOAL_RE.match(self._normalize(goal))
        if match is None:
            return None
        obj = self._object_token(match.group("object"))
        device = self._clean_phrase(match.group("device"))
        if not obj or not device:
            return None
        return obj, device

    def _light_device_action_seen(self, messages, device):
        if not device:
            return False
        pattern = re.compile(r"\b" + re.escape(device) + r"\b")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            action = self._extract_action(message)
            if action is None:
                continue
            action_norm = self._normalize(action)
            if action_norm.startswith("take"):
                continue
            if pattern.search(action_norm):
                return True
        return False

    def _light_hold_already_chosen(self, messages, obj):
        if not obj:
            return False
        pattern = re.compile(r"\b" + re.escape(obj) + r"\b")
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            action = self._extract_action(message)
            if action is None:
                continue
            action_norm = self._normalize(action)
            if not action_norm.startswith("take"):
                continue
            if not pattern.search(action_norm):
                continue
            if action_norm in self._offered_before(messages, index):
                return True
        return False

    @staticmethod
    def _build_light_hold_reminder(obj, device):
        return (
            'Hold check: to look at "'
            + obj
            + '" under the "'
            + device
            + '" — the task completes only while HOLDING '
            + obj
            + " and the "
            + device
            + " is on. Switching the "
            + device
            + " on or off again does not finish it. Find "
            + obj
            + '; if "take" is not offered, empty your hands first, then take '
            + obj
            + " and use the "
            + device
            + " while still holding it."
        )

    # ------------------------------------------------------------------
    # Search redirect
    # ------------------------------------------------------------------
    def _with_search_redirect(self, request):
        if not isinstance(request, dict):
            return request
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            return request
        last = messages[-1]
        if not isinstance(last, dict):
            return request
        content = last.get("content")
        if not isinstance(content, str):
            return request
        parsed = self._parse_put_goal(self._goal_text(messages))
        if parsed is None:
            return request
        obj, prep, dest = parsed
        if not obj:
            return request
        if self._goal_object_acquired(messages, obj):
            return request
        if not self._search_stalled(messages):
            return request
        offered = self._offered_move_actions(messages)
        if not offered:
            return request
        reminder = self._build_search_redirect(obj, prep, dest, offered)
        updated_request = dict(request)
        updated_messages = list(messages)
        updated_last = dict(last)
        updated_last["content"] = content + "\n\n" + reminder
        updated_messages[-1] = updated_last
        updated_request["messages"] = updated_messages
        return updated_request

    def _parse_put_goal(self, goal):
        if not isinstance(goal, str) or not goal.strip():
            return None
        text = self._normalize(goal)
        if self._parse_light_goal(text) is not None:
            return None
        parsed = self._parse_state_goal(text)
        if parsed is not None:
            verb, state, obj, prep, dest = parsed
            if not obj:
                return None
            return obj, prep, dest
        match = _FIND_PUT_GOAL_RE.match(text)
        if match is None:
            match = _PLAIN_PUT_GOAL_RE.match(text)
        if match is None:
            return None
        obj = self._object_token(match.group("object"))
        if not obj:
            return None
        return (
            obj,
            self._preposition(match.group("prep")),
            self._clean_phrase(match.group("dest")),
        )

    def _goal_object_acquired(self, messages, obj):
        pattern = re.compile(r"\b" + re.escape(obj) + r"\b")
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            action = self._extract_action(message)
            if action is None:
                continue
            action_norm = self._normalize(action)
            if not action_norm.startswith("take"):
                continue
            if not pattern.search(action_norm):
                continue
            observation = self._following_observation(messages, index)
            if observation is None:
                continue
            head = self._normalize(self._observation_head(observation))
            if _PICKUP_PHRASE in head and pattern.search(head):
                return True
        return False

    def _search_stalled(self, messages):
        if self._repeated_action(messages):
            return True
        pairs = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            action = self._extract_action(message)
            if action is None:
                continue
            observation = self._following_observation(messages, index)
            if observation is None:
                continue
            pairs.append((action, observation))
        for action, observation in pairs[-_MAX_EFFECT_WINDOW:]:
            action_norm = self._normalize(action)
            for prefix, phrase in _MANIPULATION_PHRASES.items():
                if not action_norm.startswith(prefix):
                    continue
                head = self._normalize(self._observation_head(observation))
                if phrase not in head:
                    return True
                break
        return False

    def _repeated_action(self, messages):
        actions = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            action = self._extract_action(message)
            if action is not None:
                actions.append(self._normalize(action))
        recent = actions[-_MAX_STALL_ACTIONS:]
        if not recent:
            return False
        counts = {}
        for action in recent:
            counts[action] = counts.get(action, 0) + 1
        return max(counts.values(), default=0) in range(
            _STALL_REPEATS, len(recent) + 1
        )

    def _offered_move_actions(self, messages):
        chosen = set()
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            action = self._extract_action(message)
            if action is not None:
                chosen.add(self._normalize(action))
        offered = []
        seen = set()
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") not in ("tool", "user"):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            match = _ACTIONS_MARKER.search(content)
            if match is None:
                continue
            for line in content[match.end():].splitlines():
                line = self._normalize(line.strip().lstrip(":").strip())
                if not line.startswith(_MOVE_PREFIXES):
                    continue
                if line in seen or line in chosen:
                    continue
                seen.add(line)
                offered.append(line)
        return offered

    def _build_search_redirect(self, obj, prep, dest, offered):
        if dest:
            head = (
                "Search check: "
                + obj
                + " has not been picked up yet; only "
                + obj
                + " "
                + prep
                + " "
                + dest
                + " completes the task, other objects do not count."
            )
        else:
            head = (
                "Search check: "
                + obj
                + " has not been picked up yet; the task requires "
                + obj
                + " itself, other objects do not count."
            )
        listed = ", ".join(offered[:_MAX_LOCATIONS])
        tail = (
            " Not yet tried: "
            + listed
            + ". Closed containers hide their contents, so choose the offered "
            "open action. An action on an object you are not holding has no "
            "effect; check your inventory first."
        )
        return head + tail

    @staticmethod
    def _following_observation(messages, index):
        if not 0 <= index + 1 < len(messages):
            return None
        following = messages[index + 1]
        if not isinstance(following, dict):
            return None
        if following.get("role") not in ("tool", "user"):
            return None
        content = following.get("content")
        if not isinstance(content, str):
            return None
        return content

    # ------------------------------------------------------------------
    # Offered-action guard
    # ------------------------------------------------------------------
    def _with_offered_action_guard(self, request):
        if not isinstance(request, dict):
            return request
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            return request
        last = messages[-1]
        if not isinstance(last, dict):
            return request
        content = last.get("content")
        if not isinstance(content, str):
            return request
        chosen = self._latest_chosen_action(messages)
        if chosen is None:
            return request
        index, action = chosen
        offered = self._offered_before(messages, index)
        if not offered:
            return request
        if self._normalize(action) in offered:
            return request
        reminder = self._build_offered_action_guard(action)
        updated_request = dict(request)
        updated_messages = list(messages)
        updated_last = dict(last)
        updated_last["content"] = content + "\n\n" + reminder
        updated_messages[-1] = updated_last
        updated_request["messages"] = updated_messages
        return updated_request

    def _latest_chosen_action(self, messages):
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            action = self._extract_action(message)
            if action is None:
                continue
            action_norm = self._normalize(action)
            if self._is_action_verb(action_norm):
                return index, action
        return None

    @staticmethod
    def _is_action_verb(action_norm):
        for verb in _ACTION_VERBS:
            if action_norm == verb or action_norm.startswith(verb + " "):
                return True
        return False

    @staticmethod
    def _build_offered_action_guard(action):
        return (
            'Offered-action check: your last choice "'
            + action
            + '" was not among the actions offered then, so it did not take '
            "effect - the environment either repeated the same view or "
            "silently performed a different offered action. Do not repeat "
            "it; choose the next action verbatim from the offered list. An "
            "action is offered only while it is possible: take requires "
            "being at the object with empty hands (check your inventory; put "
            "a held item down first), put requires the object in your hands, "
            "open requires a closed container at this location, and go to "
            "requires a different destination. If the intended action is not "
            "offered, first choose an offered action that enables it."
        )

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
                return "the cycle of actions " + cycle + " repeated 3 times"
        return None

"""Goal-directed planner + action guard for the ALFWorld text transport.

Evidence (iteration-3 released traces, 10 failing trajectories):

* ``_parse_goal`` only looked at the words *immediately before* the first
  ``put ... in/on ...`` clause, so for the very common
  ``find two cloth and put them in shelf`` phrasing the object was read as the
  pronoun ``them`` -> the goal object type came out empty and the whole planner
  went blind (no "object is here" hint, no decisive overrides).  Same for
  ``examine the pillow with the desklamp`` (a whole goal family the planner did
  not know about at all).
* every real placement is phrased ``You put the <obj> in/on the <place>.`` but
  the parser looked for ``in the`` / ``on the``; the ``in/on`` form never
  matched, so ``carried`` was never cleared and ``placed`` never advanced.
  The planner therefore kept recommending ``take <already placed object>`` and
  the agent looped ``take kettle 1 from cabinet 1`` / ``put kettle 1 in/on
  cabinet 1`` until the episode budget ran out (multi-object goals can never
  finish that way).
* a ``put`` of the goal object at the *wrong* destination (``put cloth 1 in/on
  sinkbasin 1``, ``put pencil 1 in/on drawer 1``) was accepted because only
  wrong-*object* puts were guarded.
* the fallback plan ``examine <current place>`` was itself a repeat, and the
  loop-breaker could not replace an action that equalled the plan, producing
  ``examine handtowelholder 1`` x4 loops.
* when the model emitted the call as plain text on the very first turn the
  salvage backstop failed because ``available`` was still empty (it is only
  populated from observations), so the agent repeated the same text for the
  whole episode without a single action.
* the third goal family ``look at <obj> under the <light>`` was not recognised
  at all: ``_parse_goal`` only knew ``examine X with Y``, so ``look at pen under
  the desklamp`` came out as a *placement* goal with an empty destination.  The
  planner then reported ``goal complete`` the moment the object had been dropped
  anywhere, and while the agent stood next to the ``desklamp`` the offered
  ``use desklamp 1`` action (the only thing that can satisfy a light goal) was
  never recommended.  Episodes of this family therefore burned their whole
  50-action budget on a fake "done" state.

The environment already tells the truth: every observation ends with an
``AVAILABLE ACTIONS:`` block listing the exact ``take <obj> n from <place>`` /
``put <obj> n in/on <place>`` / ``clean|heat|cool <obj> n with <appliance>``
actions that are legal *right now*.  This middleware holds a small per-episode
state (goal, mode, carried object, placed object ids, visited/examined places,
closed containers, no-progress actions) and:

* normalises every emitted tool call to ``take_action(action=...)``;
* completes an abbreviated action that is a unique prefix of an offered one;
* recovers a textual ``take_action(action="...")`` / quoted action from a
  plain-text reply, even before the first observation is available;
* computes a deterministic next step (take the goal object when offered, apply
  the required clean/heat/cool transform, carry it to the goal destination and
  put it there, otherwise expand the search over not-yet-visited receptacles);
* overrides the model at decisive moments: illegal actions, wrong-type takes,
  misplaced goal objects, ignoring an offered goal action, and repeats;
* writes a short ``[PLAN]`` block (goal, carried object, progress, unchecked
  places, recommended action) into every observation.

All state is per agent instance, i.e. per episode.  Nothing here executes an
action of its own: it only rewrites the arguments of the single tool call the
model already produced.
"""
from __future__ import annotations

import json
import re

from nexau.archs.main_sub.execution.hooks import HookResult, Middleware
from nexau.archs.main_sub.execution.parse_structures import ToolCall
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock

TOOL_NAME = "take_action"
ACTION_MARKER = "AVAILABLE ACTIONS:"
MAX_FORCED_RETRIES = 4
MAX_RECENT = 12
MAX_NOTE_PLACES = 6
MAX_SEARCH_DEVIATIONS = 2

# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


def _flatten_text(value, decode=False):
    """Best-effort extraction of the human readable text of a tool payload."""
    if isinstance(value, str):
        if decode:
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                payload = None
                try:
                    payload = json.loads(stripped)
                except Exception:
                    payload = None
                if isinstance(payload, (str, list, dict)):
                    return _flatten_text(payload, decode)
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _flatten_text(item, decode)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("result", "text", "content", "observation", "action"):
            if key in value:
                text = _flatten_text(value[key], decode)
                if text:
                    return text
    return ""


def _available_actions(text):
    """Read the action list that follows the AVAILABLE ACTIONS marker."""
    if not text or ACTION_MARKER not in text:
        return []
    tail = text.split(ACTION_MARKER, 1)[1]
    if "\n" not in tail and "\\n" in tail:
        tail = tail.replace("\\n", "\n")
    actions = []
    for line in tail.split("\n"):
        candidate = line.strip().strip('"').strip(",")
        if not candidate or candidate == "}":
            continue
        if candidate.startswith("[") or candidate.startswith("*"):
            break
        actions.append(candidate)
    return actions


def _message_text(message, decode=False):
    content = message.content
    if not isinstance(content, list):
        return _flatten_text(content, decode)
    parts = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolResultBlock):
            parts.append(_flatten_text(block.content, decode))
        elif isinstance(block, ToolUseBlock):
            parts.append(_flatten_text(block.input, decode))
    return "\n".join(part for part in parts if part)


def _action_of(call):
    """Read the intended action string out of a (possibly malformed) call."""
    parameters = call.parameters
    if isinstance(parameters, dict):
        value = parameters.get("action")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if call.tool_name == TOOL_NAME:
            return ""
    elif isinstance(parameters, str) and parameters.strip():
        return parameters.strip()
    if call.tool_name == TOOL_NAME:
        return ""
    name = call.tool_name
    if isinstance(name, str):
        return name.strip()
    return ""


def _obj_type(phrase):
    """``statue 1`` -> ``statue``; ``the soapbar`` -> ``soapbar``."""
    if not phrase:
        return ""
    phrase = phrase.strip().rstrip(".!?,").lower()
    phrase = phrase.replace("the ", " ", 1) if phrase.startswith("the ") else phrase
    parts = [p for p in phrase.split() if p]
    if not parts:
        return ""
    return parts[0].strip()


def _obj_id(phrase):
    """Normalised object identity used to remember what has been delivered."""
    return re.sub(r"\s+", " ", (phrase or "").strip().lower()).strip(" .")


def _same_type(a, b):
    """Object-type equality that tolerates a trailing plural 's'."""
    if not a or not b:
        return False
    if a == b:
        return True
    return a.rstrip("s") == b.rstrip("s")


def _insert_note(output, note):
    """Put ``note`` just before the AVAILABLE ACTIONS block (or append)."""
    if isinstance(output, str):
        if ACTION_MARKER in output:
            return output.replace(ACTION_MARKER, note.lstrip("\n") + "\n" + ACTION_MARKER, 1)
        return output + note
    if isinstance(output, dict):
        for key in ("result", "text", "content", "observation"):
            if key in output:
                updated = dict(output)
                updated[key] = _insert_note(output[key], note)
                return updated
    return output


# ---------------------------------------------------------------------------
# goal parsing
# ---------------------------------------------------------------------------

_ARTICLES = {
    "a", "an", "the", "some", "one", "two", "three", "four", "five", "six",
    "2", "3", "4", "5", "6", "couple", "and", "then",
}
_PRONOUNS = {
    "it", "them", "they", "this", "that", "these", "those", "its", "both", "all",
}
_VERBS = {
    "put", "place", "find", "get", "take", "grab", "pick", "bring", "move",
    "go", "look", "examine", "use", "turn", "switch", "clean", "heat", "cool",
    "wash", "make", "is", "are", "was", "were", "be", "there",
}
_PREPS = {
    "in", "on", "into", "with", "to", "from", "at", "under", "of", "for", "by",
    "near", "next", "inside", "onto",
}
_STATES = {
    "hot", "heated", "warm", "cold", "chilled", "refrigerated", "frozen",
    "clean", "cleaned", "dirty",
}

_RECEPTACLE = {
    "drawer", "cabinet", "shelf", "shelves", "sidetable", "dresser", "desk",
    "coffeetable", "countertop", "counter", "diningtable", "table", "sofa",
    "couch", "armchair", "bed", "ottoman", "tvstand", "fridge", "microwave",
    "garbagecan", "trashcan", "toilet", "bathtubbasin", "sinkbasin",
    "handtowelholder", "towelholder", "toiletpaperhanger", "coffeemachine",
    "stoveburner", "toaster", "safe", "cart", "desklamp", "laundryhamper",
    "bench", "chair", "nightstand", "floor", "peehole",
}

_BEDDING = {"pillow", "blanket", "sheet", "duvet", "comforter"}
_COOKWARE = {
    "pot", "pan", "tray", "bowl", "plate", "dish", "casserole", "wok",
    "skillet", "saucepan", "cuttingboard",
}
_LINEN = {"cloth", "rag", "sponge", "dishsponge", "towel", "handtowel", "curtain"}


def _sing_place(word):
    if not word:
        return word
    if word in _RECEPTACLE:
        return word
    if word.endswith("s") and word[:-1] in _RECEPTACLE:
        return word[:-1]
    if word.endswith("es") and word[:-2] in _RECEPTACLE:
        return word[:-2]
    return word


def _parse_goal(goal_text):
    """Return ``(target, dest, transform, count, mode)`` for the task line.

    ``mode`` is ``"examine"`` for the ``examine X with Y`` /
    ``look at X under the Y`` goal family (the object has to be held while the
    tool - a lamp - is switched on) and ``"put"`` for every other
    (put / find / clean / heat / cool) goal.
    """
    text = (goal_text or "").lower().strip().rstrip(".")
    text = re.sub(r"\s+", " ", text)

    # --- examine family -------------------------------------------------
    # "look at the pen under the desklamp": hold the object while the lamp
    # (the tool) is on.
    la = re.search(r"\blook\s+at\s+(?:the\s+|a\s+|an\s+|some\s+)?([a-z]+)", text)
    if la:
        tool = ""
        wt = re.search(r"\b(?:under|with|in|near|by)\s+(?:the\s+|a\s+|an\s+|some\s+)?([a-z]+)", text)
        if wt:
            tool = _sing_place(wt.group(1))
        return la.group(1), tool, "", 1, "examine"

    ex = re.search(r"\bexamine\s+(?:the\s+|a\s+|an\s+|some\s+)?([a-z]+)", text)
    if ex:
        tool = ""
        wt = re.search(r"\bwith\s+(?:the\s+|a\s+|an\s+|some\s+)?([a-z]+)", text)
        if wt:
            tool = _sing_place(wt.group(1))
        return ex.group(1), tool, "", 1, "examine"

    # --- required state transform --------------------------------------
    transform = ""
    tv = re.search(r"\b(clean|cleaned|heat|heated|cool|cooled|wash|washed)\b", text)
    if tv:
        word = tv.group(1)
        if word.startswith("clean") or word.startswith("wash"):
            transform = "clean"
        elif word.startswith("heat"):
            transform = "heat"
        else:
            transform = "cool"
    elif re.search(r"\b(hot|warm)\b", text):
        transform = "heat"
    elif re.search(r"\b(cold|chilled|refrigerated|frozen)\b", text):
        transform = "cool"

    # --- destination ----------------------------------------------------
    dest = ""
    dm = re.search(r"\b(?:in|on|into)\s+(?:the\s+|a\s+|an\s+|some\s+)?([a-z]+)", text)
    if dm:
        dest = _sing_place(dm.group(1))

    # --- goal object ----------------------------------------------------
    target = ""
    for word in re.findall(r"[a-z]+", text):
        if word in _VERBS or word in _PREPS or word in _ARTICLES or word in _PRONOUNS:
            continue
        if word in _STATES or word in _RECEPTACLE:
            continue
        if dest and word == dest:
            continue
        target = word
        break

    count = 1
    if re.search(r"\b(two|2|couple|both)\b", text):
        count = 2
    elif re.search(r"\b(three|3)\b", text):
        count = 3
    return target, dest, transform, count, "put"


# ---------------------------------------------------------------------------
# ALFWorld object -> receptacle priors (domain knowledge, not task specific)
# ---------------------------------------------------------------------------

_FOOD = {
    "apple", "banana", "bread", "tomato", "lettuce", "potato", "egg", "eggs",
    "pepper", "carrot", "corn", "peach", "orange", "lemon", "chicken", "fish",
    "meat", "cake", "pie", "cheese", "butter", "yogurt", "sugar", "flour",
    "onion", "garlic", "broccoli", "spinach", "cucumber", "watermelon",
    "strawberry", "grape", "pear", "plum", "melon", "juice", "milk", "water",
    "wine", "beer", "soda", "drink", "cream", "soup", "cereal", "pasta",
    "rice", "bean", "nuts", "cookie", "candy", "chocolate",
}
_DRINK = {
    "mug", "cup", "glassbottle", "teapot", "kettle", "glass", "jar",
    "pitcher", "bottle", "winebottle", "vase",
}
_UTENSIL = {
    "spoon", "fork", "knife", "butterknife", "ladle", "spatula", "whisk",
    "tongs", "chopsticks", "knifes",
}
_WRITING = {
    "pen", "pencil", "marker", "notepad", "paper", "envelope", "book",
    "newspaper", "magazine", "folder", "card", "creditcard", "postit",
    "letter", "document", "map",
}
_HYGIENE = {
    "soapbar", "soapbottle", "spraybottle", "detergent",
    "toiletpaper", "tissuebox", "brush", "toothbrush",
    "razor", "shampoo", "conditioner", "lotion", "cream",
}
_SMALL = {
    "watch", "clock", "alarmclock", "keychain", "key", "coin", "cellphone",
    "cd", "dvd", "remotecontrol", "lighter", "matches", "candle",
    "flashlight", "battery", "padlock", "ring", "necklace", "earring",
    "wallet", "purse", "sunglasses", "glasses", "headphones", "speaker",
    "camera", "mouse", "charger", "adapter", "multimeter", "box", "basket",
}
_DECOR = {
    "statue", "houseplant", "plant", "flower", "painting", "picture",
    "frame", "art", "sculpture", "ornament", "decoration", "trophy",
    "television", "tv", "laptop", "computer", "keyboard", "monitor",
}
_FABRIC = {
    "cushion", "rug", "clothes", "shirt", "jacket", "scarf", "hat", "glove",
}

_DEFAULT_ORDER = [
    "drawer", "cabinet", "shelf", "sidetable", "dresser", "desk",
    "coffeetable", "sofa", "armchair", "tvstand", "bed", "ottoman",
    "countertop", "diningtable", "fridge", "microwave", "garbagecan",
    "toilet", "bathtubbasin", "sinkbasin", "handtowelholder", "towelholder",
    "toiletpaperhanger", "coffeemachine", "stoveburner", "toaster", "safe",
    "cart", "desklamp", "laundryhamper",
]

APPLIANCES = {
    "heat": ["microwave", "stoveburner"],
    "cool": ["fridge"],
    "clean": ["sinkbasin", "bathtubbasin"],
}

# A closed receptacle costs two actions to inspect ("go to" + "open") while an
# open surface reveals its contents on arrival, so when two receptacles have a
# comparable prior the surface is worth checking first.  These are exactly the
# receptacles the environment reports as "The <name> is closed".
CLOSED_KINDS = {"cabinet", "drawer", "fridge", "microwave", "safe"}
CLOSED_PENALTY = 3

# A scene can hold a *bank* of identical closed containers (a large kitchen in
# the released traces had 20/27 cabinets and up to 12 drawers).  Every member of
# such a bank costs two actions to rule out and, per the released evidence, the
# goal object was never inside one of them: of the nine objects actually picked
# up in the iteration-6 traces, eight lay on an open surface and the ninth was
# in the fridge.  Walking a bank first therefore burned 36-54 of the 50 allowed
# actions before the useful part of the scene was even reached (near-miss
# `native_027ffe04...`, loss `native_580811fa...`).  A closed *kind* with at
# least this many instances is therefore treated as generic storage and searched
# after everything else; a lone closed container (one fridge, one microwave,
# a single drawer) keeps its ordinary prior, so object-specific priors such as
# "food lives in the fridge" are untouched.
CONTAINER_BANK_MIN = 3

# Receptacles that hold food / cookware / crockery in a kitchen scene.
KITCHEN_SURFACES = ["countertop", "stoveburner", "sinkbasin", "diningtable",
                    "coffeemachine", "toaster", "shelf", "coffeetable"]

# Goal family `look at <object> under the <light>` / `examine <object> with the
# <light>`: the object has to be held while the lamp is switched on, and the
# switch is the `use <lamp>` action, which the environment only offers while the
# agent stands at the surface that carries the lamp.
LIGHT_KINDS = {"desklamp", "floorlamp", "lamp", "tablelamp", "ceilinglamp",
               "light", "lightbulb", "lightswitch"}
# Surfaces a lamp usually sits on, used to order the search for it.
LIGHT_PRIOR = ["sidetable", "desk", "coffeetable", "dresser", "shelf",
               "countertop", "table", "diningtable", "nightstand", "tvstand",
               "bed", "sofa", "armchair", "cabinet", "drawer", "cart"]


def _is_light_kind(kind):
    """True for lamp-like object kinds (``desklamp``, ``floorlamp``, ...)."""
    if not kind:
        return False
    if kind in LIGHT_KINDS:
        return True
    return "lamp" in kind or "light" in kind


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _prior_for(target, transform):
    order = []
    if transform == "cool":
        order.append("fridge")
    if (transform and target not in _LINEN and target not in _BEDDING
            and target not in _FABRIC):
        # A clean / heat / cool goal can only name a transformable object, and
        # those are handled in the kitchen: the object is found on a kitchen
        # surface (observed: clean kettle -> stoveburner, clean tomato ->
        # diningtable, clean fork -> countertop, hot egg -> countertop, cool
        # pot -> stoveburner), not buried in the first of thirteen cabinets.
        order += KITCHEN_SURFACES
    if target in _COOKWARE:
        order += ["countertop", "stoveburner", "sinkbasin", "microwave",
                  "diningtable", "cabinet", "shelf", "fridge", "drawer",
                  "coffeemachine", "toaster"]
    elif target in _LINEN:
        order += ["bathtubbasin", "sinkbasin", "countertop", "drawer",
                  "cabinet", "towelholder", "handtowelholder", "shelf",
                  "toilet", "garbagecan", "diningtable"]
    elif target in _BEDDING:
        order += ["bed", "dresser", "sofa", "armchair", "shelf", "cabinet",
                  "coffeetable", "ottoman"]
    elif target in _FOOD:
        order += ["fridge", "countertop", "diningtable", "microwave",
                  "sinkbasin", "cabinet", "garbagecan"]
    elif target in _DRINK:
        order += ["cabinet", "countertop", "shelf", "coffeetable",
                  "diningtable", "coffeemachine", "microwave", "sinkbasin"]
    elif target in _UTENSIL:
        order += ["drawer", "countertop", "diningtable", "sinkbasin",
                  "cabinet", "shelf"]
    elif target in _WRITING:
        order += ["drawer", "desk", "sidetable", "coffeetable", "shelf",
                  "countertop", "dresser", "sofa"]
    elif target in _HYGIENE:
        order += ["cabinet", "countertop", "drawer", "sinkbasin",
                  "bathtubbasin", "toilet", "towelholder",
                  "handtowelholder", "toiletpaperhanger", "shelf",
                  "garbagecan"]
    elif target in _DECOR:
        order += ["shelf", "cabinet", "coffeetable", "dresser", "desk",
                  "sidetable", "tvstand", "countertop"]
    elif target in _FABRIC:
        order += ["towelholder", "handtowelholder", "sofa", "armchair",
                  "bed", "shelf", "dresser", "cabinet", "coffeetable"]
    elif target in _SMALL:
        order += ["drawer", "sidetable", "dresser", "desk", "coffeetable",
                  "shelf", "sofa", "armchair", "tvstand", "cabinet",
                  "countertop", "bed"]
    else:
        order += ["drawer", "cabinet", "shelf", "sidetable", "dresser",
                  "desk", "coffeetable", "sofa", "armchair", "tvstand",
                  "countertop", "bed", "fridge", "microwave"]
    return _dedupe(order + _DEFAULT_ORDER)


# ---------------------------------------------------------------------------
# middleware
# ---------------------------------------------------------------------------

_PUT_RE = re.compile(r"You put the (.+?) (?:in/on|into|in|on) (?:the )?(.+?)\.", re.I)
_PICK_RE = re.compile(r"You pick up the (.+?) from the ", re.I)
_SEEN_RE = re.compile(r"you see\s+([^.\n]*)", re.I)
_SEEN_PLACE_RE = re.compile(r"(?:On|In) the ([a-z]+ \d+),? you see", re.I)
_OBJ_ID_RE = re.compile(r"\b([a-z][a-z]*)\s+(\d+)\b")


class GoalPlannerMiddleware(Middleware):
    """External memory + decisive-action planner for the ALFWorld episode."""

    def __init__(self, max_recent=MAX_RECENT):
        self.max_recent = int(max_recent)
        self.mode = "put"
        self.target = ""
        self.dest = ""
        self.transform = ""
        self.count = 1
        self.placed = 0
        self.transformed = False
        self.light_on = False
        self.light_put = False
        self.examined_lit = False
        self.light_places = {}
        self.carried = ""
        self.visited = []
        self.visited_set = set()
        self.examined = set()
        self.closed = set()
        self.place = ""
        self.seen_at = {}
        self.taken = set()
        self.noop = set()
        self.recent = []
        self.last_action = ""
        self.last_obs = ""
        self.repeat_count = 0
        self.available = []
        self.prior = _prior_for("", "")
        self.goal_text = ""
        self.forced = 0
        self.done = set()
        self.deviations = 0

    # ------------------------------------------------------------------
    # state updates
    # ------------------------------------------------------------------
    def _ensure_goal(self, messages):
        if self.goal_text:
            return
        for message in messages:
            try:
                text = _message_text(message)
            except Exception:
                continue
            if "Your task is to:" in text:
                raw = text.split("Your task is to:", 1)[1]
                raw = raw.split("AVAILABLE ACTIONS")[0].strip()
                self.goal_text = raw
                (self.target, self.dest, self.transform,
                 self.count, self.mode) = _parse_goal(raw)
                self.prior = _prior_for(self.target, self.transform)
                avail = _available_actions(text)
                if avail:
                    self.available = avail
                return

    def _observe(self, action, observation):
        action = (action or "").strip()
        fresh = _available_actions(observation)
        if fresh:
            self.available = fresh

        if action.startswith("go to "):
            place = action[6:].strip()
            self.place = place
            if place and place not in self.visited_set:
                self.visited_set.add(place)
                self.visited.append(place)
        elif action.startswith("open "):
            self.place = action[5:].strip()
        elif action.startswith("examine "):
            what = action[8:].strip()
            self.examined.add(what)
            if self.light_on and self.carried and what == self.carried:
                # examined in the light of the (already switched on) lamp
                self.examined_lit = True

        for match in re.finditer(r"The ([a-z]+ \d+) is closed", observation):
            self.closed.add(match.group(1))
        for match in re.finditer(r"The ([a-z]+ \d+) is open", observation):
            self.closed.discard(match.group(1))

        picked = _PICK_RE.search(observation)
        if picked:
            self.carried = picked.group(1).strip()
            obj_id = _obj_id(self.carried)
            self.taken.add(obj_id)
            for where, ids in list(self.seen_at.items()):
                if obj_id in ids:
                    self.seen_at[where] = [i for i in ids if i != obj_id]

        self._record_seen(observation)

        put = _PUT_RE.search(observation)
        if put:
            obj = put.group(1).strip()
            place = put.group(2).strip()
            self.carried = ""
            if (self.mode == "examine" and self.target and self.place
                    and place == self.place
                    and _same_type(_obj_type(obj), self.target)):
                self.light_put = True
            if self.target and _same_type(_obj_type(obj), self.target):
                if (not self.dest) or _same_type(_obj_type(place), self.dest):
                    obj_id = _obj_id(obj)
                    if obj_id not in self.done:
                        self.done.add(obj_id)
                        self.placed += 1

        if self.transform and action.startswith(self.transform + " "):
            if not observation.startswith("Nothing happens"):
                self.transformed = True

        if action.startswith("use ") and not observation.startswith("Nothing happens"):
            # The tool/lamp of a `look at X under the Y` goal is on now.  Never
            # issue the same `use` again - it is a toggle and would switch the
            # light back off.
            self.light_on = True

        if self.place:
            for candidate in self.available:
                if candidate.startswith("use ") and _is_light_kind(_obj_type(candidate[4:])):
                    # `use <lamp>` is only offered next to the lamp, so this is
                    # the most reliable memory of where the light stands.
                    self.light_places[_obj_type(candidate[4:])] = self.place

        if observation == self.last_obs and action:
            self.noop.add(action)
        self.last_obs = observation

        if action and action == self.last_action:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.last_action = action
        self.recent.append(action)
        while len(self.recent) > self.max_recent:
            self.recent.pop(0)

    def _record_seen(self, observation):
        """Remember which objects are visible at the receptacle we are at."""
        if not observation or "you see" not in observation:
            return
        text = observation.split("[PLAN]", 1)[0]
        if "Looking quickly around you" in text or text.lstrip().startswith(
                "You are in the middle of a room"):
            # whole-room description (`look`): it lists containers, not contents
            return
        matches = list(_SEEN_RE.finditer(text))
        if not matches:
            return
        clause = matches[-1].group(1)
        place = self.place
        pm = _SEEN_PLACE_RE.search(text)
        if pm:
            place = pm.group(1).strip()
        if not place:
            return
        ids = []
        for om in _OBJ_ID_RE.finditer(clause):
            kind = om.group(1).lower()
            if kind in _RECEPTACLE:
                continue
            ids.append("%s %s" % (kind, om.group(2)))
        self.seen_at[place] = _dedupe(ids)

    def _recall_place(self):
        """A visited receptacle that was seen to hold an unclaimed goal object."""
        if not self.target or self.placed >= self.count:
            return ""
        rank = {name: index for index, name in enumerate(self.prior)}
        best = None
        for place, ids in self.seen_at.items():
            hit = False
            for obj_id in ids:
                if (obj_id not in self.taken and obj_id not in self.done
                        and _same_type(_obj_type(obj_id), self.target)):
                    hit = True
                    break
            if not hit:
                continue
            kind = _obj_type(place)
            order = rank.get(kind, len(rank))
            penalty = CLOSED_PENALTY if kind in CLOSED_KINDS else 0
            digits = re.findall(r"\d+", place)
            number = int(digits[0]) if digits else 0
            key = (order + penalty, order, number, place)
            if best is None or key < best[0]:
                best = (key, place)
        return best[1] if best else ""

    def _recall_step(self):
        """Legal action that returns to a place where the goal object was seen."""
        place = self._recall_place()
        if not place:
            return ""
        if place == self.place:
            for candidate in self.available:
                if candidate == "open " + place:
                    return candidate
            return ""
        for candidate in self.available:
            if candidate == "go to " + place:
                return candidate
        return ""

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------
    def _take_target(self):
        for candidate in self.available:
            if not candidate.startswith("take "):
                continue
            body = candidate[5:]
            if " from " not in body:
                continue
            obj = body.split(" from ", 1)[0].strip()
            if not _same_type(_obj_type(obj), self.target):
                continue
            if _obj_id(obj) in self.done:
                continue
            return candidate
        return ""

    def _place_target(self):
        carried = self.carried
        if not carried:
            return ""
        prefix = "put " + carried + " in/on "
        for candidate in self.available:
            if candidate.startswith(prefix):
                place = candidate[len(prefix):]
                if not self.dest or _same_type(_obj_type(place), self.dest):
                    return candidate
        return ""

    def _put_down(self, carried):
        if not carried:
            return ""
        prefix = "put " + carried + " in/on "
        for candidate in self.available:
            if candidate.startswith(prefix):
                return candidate
        return ""

    def _put_on(self, carried, place):
        """The `put <object> in/on <place>` action for one specific receptacle."""
        if not carried or not place:
            return ""
        candidate = "put %s in/on %s" % (carried, place)
        return candidate if candidate in self.available else ""

    def _transform_target(self):
        carried = self.carried
        if not carried or not self.transform:
            return ""
        prefix = self.transform + " " + carried
        for candidate in self.available:
            if candidate.startswith(prefix):
                return candidate
        kinds = APPLIANCES.get(self.transform, [])
        for kind in kinds:
            for candidate in self.available:
                if candidate.startswith("go to ") and _obj_type(candidate[6:]) == kind:
                    return candidate
        for candidate in self.available:
            if candidate.startswith("open ") and _obj_type(candidate[5:]) in kinds:
                return candidate
        return ""

    def _go_to_dest(self):
        if not self.dest:
            return ""
        for candidate in self.available:
            if candidate.startswith("go to ") and _same_type(_obj_type(candidate[6:]), self.dest):
                return candidate
        return ""

    def _go_to_kind(self, kind):
        if not kind:
            return ""
        for candidate in self.available:
            if candidate.startswith("go to ") and _same_type(_obj_type(candidate[6:]), kind):
                return candidate
        return ""

    def _open_kind(self, kind):
        if not kind:
            return ""
        for candidate in self.available:
            if candidate.startswith("open ") and _same_type(_obj_type(candidate[5:]), kind):
                return candidate
        return ""

    def _open_dest(self):
        return self._open_kind(self.dest)

    def _open_closed(self):
        for candidate in self.available:
            if candidate.startswith("open ") and candidate[5:].strip() in self.closed:
                return candidate
        return ""

    def _use_any(self):
        if self.dest:
            for candidate in self.available:
                if candidate.startswith("use ") and _same_type(_obj_type(candidate[4:]), self.dest):
                    return candidate
        for candidate in self.available:
            if candidate.startswith("use "):
                return candidate
        return ""

    def _kind_counts(self):
        """How many offered ``go to <kind> N`` places share each kind.

        The environment lists every receptacle of the room in each observation,
        so this is the multiplicity of the receptacle *kind* in the scene (27
        cabinets, 12 drawers, one fridge, ...).
        """
        counts = {}
        for candidate in self.available:
            if not candidate.startswith("go to "):
                continue
            kind = _obj_type(candidate[6:].strip())
            if kind:
                counts[kind] = counts.get(kind, 0) + 1
        return counts

    def _place_key(self, place, rank, counts=None):
        """Sort key for an unvisited receptacle (lower is investigated sooner).

        Ordinary receptacles are ranked by the object prior with the usual
        closed-container penalty.  A closed *kind* that the scene offers
        ``CONTAINER_BANK_MIN`` times or more is a bank of generic storage: no
        member of it is investigated until every other receptacle has been, so
        that a 20-27 container kitchen cannot swallow the whole action budget.
        """
        name = place[6:] if place.startswith("go to ") else place
        kind = _obj_type(name)
        order = rank.get(kind, len(rank))
        digits = re.findall(r"\d+", name)
        number = int(digits[0]) if digits else 0
        if counts is None:
            counts = self._kind_counts()
        bank = (kind in CLOSED_KINDS
                and counts.get(kind, 0) >= CONTAINER_BANK_MIN)
        if bank:
            return (1, order, number, name)
        penalty = CLOSED_PENALTY if kind in CLOSED_KINDS else 0
        return (0, order + penalty, order, number, name)

    def _unvisited_places(self):
        places = [
            candidate for candidate in self.available
            if candidate.startswith("go to ") and candidate[6:].strip() not in self.visited_set
        ]
        rank = {name: index for index, name in enumerate(self.prior)}
        counts = self._kind_counts()

        def key(place):
            return self._place_key(place, rank, counts)

        places.sort(key=key)
        return places

    def _fresh(self, candidate):
        if not candidate:
            return False
        if candidate in self.noop:
            return False
        if candidate == self.last_action and self.repeat_count >= 1:
            return False
        return True

    def _novel_action(self):
        pool = []
        for candidate in self.available:
            if candidate == self.last_action:
                continue
            if self.recent.count(candidate) >= 2:
                continue
            if candidate in self.noop and self.recent.count(candidate) >= 1:
                continue
            pool.append(candidate)
        if not pool:
            pool = [c for c in self.available if c != self.last_action]
        if not pool:
            return ""

        def pref(action):
            if action.startswith("go to "):
                return 0
            if action.startswith("open "):
                return 1
            if action.startswith("examine "):
                return 2
            if action.startswith("take "):
                return 3
            if action.startswith("put "):
                return 4
            if action == "look":
                return 5
            if action == "inventory":
                return 7
            return 6

        pool.sort(key=pref)
        return pool[0]

    def _go_to_place(self, place):
        """The `go to <place>` action, when that place is somewhere else."""
        if not place or place == self.place:
            return ""
        for candidate in self.available:
            if candidate == "go to " + place:
                return candidate
        return ""

    def _kind_in_seen(self, ids, kind):
        """True when one of the remembered object ids has the requested type."""
        for obj_id in ids:
            obj_kind = _obj_type(obj_id)
            if _same_type(obj_kind, kind):
                return True
            if _is_light_kind(kind) and _is_light_kind(obj_kind):
                return True
        return False

    def _place_with_kind(self, kind, prior=None):
        """A remembered receptacle that was seen to hold an object of ``kind``."""
        if not kind:
            return ""
        rank = {name: index for index, name in enumerate(prior or self.prior)}
        best = None
        for place, ids in self.seen_at.items():
            if not self._kind_in_seen(ids, kind):
                continue
            digits = re.findall(r"\d+", place)
            number = int(digits[0]) if digits else 0
            order = rank.get(_obj_type(place), len(rank))
            key = (order, number, place)
            if best is None or key < best[0]:
                best = (key, place)
        return best[1] if best else ""

    def _light_place(self):
        """Where the lamp of a light goal was last seen ('' when unseen)."""
        if not self.dest:
            return ""
        place = self.light_places.get(self.dest)
        if place:
            return place
        for kind, place in self.light_places.items():
            if _is_light_kind(kind) and _is_light_kind(self.dest):
                return place
        if _is_light_kind(self.dest):
            return self._place_with_kind(self.dest, LIGHT_PRIOR)
        return ""

    def _examine_target(self):
        """The legal `examine <goal object>` action, if the environment offers it."""
        for candidate in self.available:
            if (candidate.startswith("examine ")
                    and _same_type(_obj_type(candidate[8:]), self.target)):
                return candidate
        return ""

    def _stay_action(self):
        """A harmless action that keeps the agent at the place that matters.

        Used after the lamp is on: whatever the environment expects next, it is
        observed from here, so wandering off to another receptacle can only
        lose the goal.
        """
        for candidate in self.available:
            if (candidate.startswith("examine ")
                    and _same_type(_obj_type(candidate[8:]), self.dest)):
                return candidate
        for name in ("look", "inventory"):
            if name in self.available:
                return name
        return ""

    def _search_places(self):
        """Unvisited receptacles, ordered for the current sub-goal.

        While the goal object is in hand the search is for the lamp, so
        lamp-carrying surfaces are visited first; otherwise the ordinary
        object prior applies.
        """
        places = self._unvisited_places()
        carrying = bool(self.carried) and _same_type(_obj_type(self.carried), self.target)
        if not places or not (self.mode == "examine" and carrying):
            return places
        rank = {name: index for index, name in enumerate(LIGHT_PRIOR)}

        def key(place):
            name = place[6:]
            digits = re.findall(r"\d+", name)
            number = int(digits[0]) if digits else 0
            return (rank.get(_obj_type(name), len(rank)), number, name)

        return sorted(places, key=key)

    def _plan_examine(self):
        """Plan the `examine X with Y` / `look at X under Y` goal family.

        The goal object has to be in hand while the tool (a lamp) is switched
        on, and the switch is the `use <lamp>` action, which the environment
        only offers while the agent stands at the surface that carries the lamp:
          1. take the goal object (or return to the place it was seen at);
          2. walk to the lamp and `use` it - a lamp is a toggle, so never twice;
          3. `examine` the object, and only if the episode has not ended by
             then leave the object on the lamp's surface, which is the literal
             reading of "the object under the lamp".
        The environment terminates the episode the moment the goal is satisfied,
        so the later steps are only ever reached when the earlier ones did not
        already win.
        """
        carrying = bool(self.carried) and _same_type(_obj_type(self.carried), self.target)
        tool = self.dest
        light_place = self._light_place()
        if not carrying:
            if not self.light_put:
                step = self._take_target()
                if self._fresh(step):
                    return step
                step = self._go_to_place(self._place_with_kind(self.target))
                if self._fresh(step):
                    return step
            if not tool:
                step = self._examine_target()
                if self._fresh(step):
                    return step
        else:
            at_light = bool(light_place) and light_place == self.place
            if not self.light_on:
                step = self._use_any()          # only offered at the lamp
                if self._fresh(step):
                    return step
            if (self.light_on or not tool or at_light) and not self.examined_lit:
                step = self._examine_target()
                if self._fresh(step):
                    return step
            if self.light_on and at_light:
                if not self.light_put:
                    step = self._put_on(self.carried, self.place)
                    if self._fresh(step):
                        return step
                step = self._stay_action()
                if self._fresh(step):
                    return step
            if light_place and not at_light:
                step = self._go_to_place(light_place)
                if self._fresh(step):
                    return step
            if not self.light_on:
                step = self._open_kind(tool)
                if self._fresh(step):
                    return step

        if self.carried and not carrying:
            step = self._put_down(self.carried)
            if self._fresh(step):
                return step

        step = self._go_to_kind(tool)
        if self._fresh(step):
            return step

        if not carrying and not self.light_put:
            step = self._recall_step()
            if self._fresh(step):
                return step

        step = self._open_closed()
        if self._fresh(step):
            return step

        for place in self._search_places():
            if self._fresh(place):
                return place

        return self._novel_action()

    def _plan(self):
        """The single most useful legal action right now ('' if unknown)."""
        if not self.available:
            return ""
        if self.mode == "examine":
            return self._plan_examine()

        carrying_target = bool(self.carried) and _same_type(_obj_type(self.carried), self.target)
        if carrying_target:
            if self.transform and not self.transformed:
                step = self._transform_target()
                if self._fresh(step):
                    return step
            step = self._place_target()
            if self._fresh(step):
                return step
            step = self._open_dest()
            if self._fresh(step):
                return step
            step = self._go_to_dest()
            if self._fresh(step):
                return step
        else:
            step = self._take_target()
            if self._fresh(step):
                return step

        if self.carried and not carrying_target:
            step = self._put_down(self.carried)
            if self._fresh(step):
                return step

        step = self._recall_step()
        if self._fresh(step):
            return step

        step = self._open_closed()
        if self._fresh(step):
            return step

        for place in self._unvisited_places():
            if self._fresh(place):
                return place

        step = self._take_target()
        if self._fresh(step):
            return step

        return self._novel_action()

    # ------------------------------------------------------------------
    # override policy
    # ------------------------------------------------------------------
    def _is_decisive(self, plan):
        if not plan:
            return False
        if plan == self._take_target():
            return True
        if plan in (self._place_target(), self._transform_target(),
                    self._open_dest(), self._open_closed(),
                    self._recall_step()):
            return True
        carrying_target = bool(self.carried) and _same_type(_obj_type(self.carried), self.target)
        if carrying_target and plan == self._go_to_dest():
            return True
        return False

    def _is_search_plan(self, plan):
        """True when the plan is a plain receptacle-search move."""
        return bool(plan) and plan.startswith("go to ")

    def _is_exploratory(self, action):
        return bool(action) and (
            action.startswith("go to ")
            or action.startswith("examine ")
            or action in ("look", "inventory")
        )

    def _override(self, action):
        if not action or not self.available:
            return ""
        plan = self._plan()
        if action not in self.available:
            return plan if plan and plan != action else ""
        if self.mode == "examine":
            plan = self._plan()
            if not plan or plan == action:
                return ""
            if plan == self._take_target():
                return plan
            if action.startswith("take "):
                body = action[5:]
                obj = _obj_type(body.split(" from ", 1)[0] if " from " in body else body)
                if obj and not _same_type(obj, self.target):
                    # only the goal object belongs in the hand of this goal
                    return plan
            if not self.dest:
                return plan if plan == self._examine_target() else ""
            if not self.light_on:
                if plan == self._use_any():
                    return plan
                if plan == self._go_to_place(self._light_place()):
                    return plan
                if (action.startswith("examine ") and self.carried
                        and action[8:].strip() == self.carried
                        and plan.startswith("go to ")):
                    # an examine before the lamp is on cannot satisfy the goal
                    return plan
                # dropping the goal object before the light is on loses the goal
                if action.startswith("put ") and _same_type(
                        _obj_type(action[4:].split(" in/on ", 1)[0].strip()),
                        self.target):
                    return plan
                return ""
            step = self._examine_target()
            if step and plan == step:
                return plan
            if plan.startswith("put ") and self.place and plan.endswith(self.place):
                return plan
            return ""
        carrying_target = bool(self.carried) and _same_type(_obj_type(self.carried), self.target)
        if self.target:
            if action.startswith("take "):
                body = action[5:]
                obj_phrase = body.split(" from ", 1)[0] if " from " in body else body
                obj = _obj_type(obj_phrase)
                if obj and not _same_type(obj, self.target):
                    return plan if plan and plan != action else ""
                if carrying_target:
                    return plan if plan and plan != action else ""
                if _obj_id(obj_phrase) in self.done:
                    return plan if plan and plan != action else ""
            elif action.startswith("put "):
                body = action[4:]
                if " in/on " in body:
                    obj = _obj_type(body.split(" in/on ", 1)[0])
                    place = _obj_type(body.split(" in/on ", 1)[1])
                else:
                    obj = _obj_type(body)
                    place = ""
                if obj and not _same_type(obj, self.target):
                    return plan if plan and plan != action else ""
                if _same_type(obj, self.target) and self.dest and place \
                        and not _same_type(place, self.dest):
                    return plan if plan and plan != action else ""
        if carrying_target and plan and plan != action:
            return plan
        if plan and plan != action and self._is_decisive(plan):
            return plan
        return ""

    # ------------------------------------------------------------------
    # repair / salvage
    # ------------------------------------------------------------------
    def _repair(self, action, available):
        if not action or not available:
            return action
        if action in available:
            return action
        lowered = action.lower()
        for item in available:
            if item.lower() == lowered:
                return item
        matches = [item for item in available if item.lower().startswith(lowered)]
        if len(matches) == 1:
            return matches[0]
        return action

    def _should_replace(self, action):
        if not action:
            return False
        if action == self.last_action and self.repeat_count >= 1:
            return True
        if len(self.recent) >= 4:
            window = self.recent[-4:]
            if window[0] == window[2] and window[1] == window[3] and window[0] != window[1]:
                if action == self.recent[-4] or action == self.recent[-3]:
                    return True
        if action in self.noop and self.recent.count(action) >= 2:
            return True
        return False

    def _align_history(self, messages, calls):
        target = None
        for message in reversed(messages):
            if message.role == Role.ASSISTANT:
                target = message
                break
        if target is None or not isinstance(target.content, list):
            return
        index = {}
        for call in calls:
            if call.tool_call_id:
                index[call.tool_call_id] = call
        for block in target.content:
            if not isinstance(block, ToolUseBlock):
                continue
            call = index.get(block.id)
            if call is None:
                continue
            block.name = call.tool_name
            if isinstance(call.parameters, dict):
                block.input = dict(call.parameters)

    def _salvage(self, messages, available):
        text = ""
        for message in reversed(messages):
            if message.role == Role.ASSISTANT:
                text = _message_text(message)
                break
        text = (text or "").strip()
        if not text:
            return ""
        text = text.strip("`").strip()
        if "take_action" in text or "action" in text:
            match = re.search(r'action\s*[=:]\s*["\']([^"\']+)["\']', text)
            if match:
                text = match.group(1).strip()
            elif re.match(r"(?i)take_action\s*\(", text):
                inner = text[text.find("(") + 1: text.rfind(")")].strip()
                inner = re.sub(r"(?i)^action\s*[=:]\s*", "", inner).strip()
                inner = inner.strip().strip("\"'`").strip()
                if inner:
                    text = inner
        text = text.strip().strip("\"'`").strip()
        if not text:
            return ""
        if available:
            if text in available:
                return text
            lowered = text.lower()
            for action in available:
                if action.lower() == lowered:
                    return action
            if len(lowered) <= 200:
                for action in available:
                    if len(action) > 3 and action.lower() in lowered:
                        return action
            matches = [a for a in available if a.lower().startswith(lowered)]
            if len(matches) == 1:
                return matches[0]
            return ""
        if re.match(r"(?i)^(go to|open|close|take|put|examine|look|inventory|"
                    r"clean|heat|cool|use|move|turn)\b", text) and len(text) <= 80:
            return text
        return ""

    # ------------------------------------------------------------------
    # the note the model sees
    # ------------------------------------------------------------------
    def _note(self):
        if not self.goal_text:
            return ""
        lines = []
        carry = self.carried or "nothing"
        if self.mode == "examine" and self.dest:
            lines.append(
                "[PLAN] goal=%s | carrying=%s | %s=%s"
                % (self.goal_text.rstrip("."), carry, self.dest,
                   "on" if self.light_on else "off")
            )
        else:
            lines.append(
                "[PLAN] goal=%s | carrying=%s | placed=%d/%d"
                % (self.goal_text.rstrip("."), carry, self.placed, self.count)
            )
        if self.mode == "examine":
            object_name = self.target or "the object"
            if self.light_put:
                lines.append("[PLAN] %s is lying under the %s - the goal should "
                             "be met now" % (object_name, self.dest or "light"))
            elif self.light_on:
                lines.append("[PLAN] the %s is on - now examine %s"
                             % (self.dest or "tool", object_name))
            elif self.dest:
                lines.append("[PLAN] goal: hold %s while using the %s"
                             % (object_name, self.dest))
            else:
                lines.append("[PLAN] goal: examine %s" % object_name)
            if not self.light_on:
                use_step = self._use_any()
                if use_step:
                    lines.append("[PLAN] *** THE %s IS HERE -> %s ***"
                                 % ((self.dest or "tool").upper(), use_step))
                else:
                    place = self._light_place()
                    if place and place != self.place:
                        lines.append("[PLAN] the %s was seen at %s -> go back there"
                                     % (self.dest or "tool", place))
            elif not self.carried:
                lines.append("[PLAN] take %s first, then examine it in the light"
                             % object_name)
        elif self.placed >= self.count > 0:
            lines.append("[PLAN] goal complete - the episode ends by itself now")
        else:
            target_take = self._take_target()
            if target_take:
                lines.append("[PLAN] *** YOUR OBJECT IS HERE -> %s ***" % target_take)
            elif self.carried and _same_type(_obj_type(self.carried), self.target):
                extra = ""
                if self.transform and not self.transformed:
                    extra = " (apply '%s' first)" % self.transform
                lines.append("[PLAN] deliver %s to %s%s"
                             % (self.carried, self.dest or "the goal place", extra))
            else:
                recall = self._recall_place()
                if recall:
                    lines.append(
                        "[PLAN] you already saw the %s at %s -> go back there first"
                        % (self.target or "goal object", recall))
                places = self._unvisited_places()
                if places:
                    shown = ", ".join(place[6:] for place in places[:MAX_NOTE_PLACES])
                    extra = len(places) - MAX_NOTE_PLACES
                    if extra > 0:
                        shown += " (+%d more)" % extra
                    lines.append("[PLAN] still unchecked: %s" % shown)
                elif self.target:
                    lines.append("[PLAN] every listed place checked - open the closed containers")
        plan = self._plan()
        if plan:
            lines.append("[PLAN] next action -> %s" % plan)
        return "\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------
    def after_model(self, hook_input):
        parsed = hook_input.parsed_response
        if parsed is None:
            return HookResult.no_changes()
        messages = hook_input.messages
        try:
            self._ensure_goal(messages)
        except Exception:
            pass
        available = self.available

        calls = parsed.tool_calls
        if calls:
            changed = False
            plan_fallback = self._plan()
            if not plan_fallback and "look" in available:
                plan_fallback = "look"
            for call in calls:
                action = _action_of(call)
                if not action:
                    if plan_fallback:
                        action = plan_fallback
                        call.tool_name = TOOL_NAME
                        call.parameters = {"action": action}
                        changed = True
                    continue
                action = self._repair(action, available)
                override = self._override(action)
                if not override and self._should_replace(action):
                    plan = self._plan()
                    if plan and plan != action:
                        override = plan
                    else:
                        alt = self._novel_action()
                        if alt and alt != action:
                            override = alt
                if override:
                    self.deviations = 0
                else:
                    plan = self._plan()
                    if (plan and plan != action and self._is_search_plan(plan)
                            and self._is_exploratory(action)):
                        self.deviations += 1
                        if self.deviations >= MAX_SEARCH_DEVIATIONS:
                            override = plan
                            self.deviations = 0
                    else:
                        self.deviations = 0
                if override:
                    action = override
                if (call.tool_name != TOOL_NAME
                        or not isinstance(call.parameters, dict)
                        or call.parameters.get("action") != action):
                    call.tool_name = TOOL_NAME
                    call.parameters = {"action": action}
                    changed = True
            if changed:
                self._align_history(messages, calls)
                return HookResult.with_modifications(parsed_response=parsed)
            return HookResult.no_changes()

        if parsed.sub_agent_calls:
            return HookResult.no_changes()

        salvage = self._salvage(messages, available)
        if salvage:
            parsed.tool_calls = [
                ToolCall(tool_name=TOOL_NAME, parameters={"action": salvage}, source="xml")
            ]
            return HookResult.with_modifications(parsed_response=parsed)

        if self.forced < MAX_FORCED_RETRIES:
            self.forced += 1
            note = Message(
                role=Role.USER,
                content=[TextBlock(text=(
                    "[FORMAT ERROR] Your last reply contained no tool call, so no action was "
                    "taken. Reply with exactly ONE call to the tool `take_action` and put the "
                    "action string in its `action` argument, for example: "
                    "take_action(action=\"look\"). Never write plain text and never use the "
                    "action text as the tool name."
                ))],
            )
            return HookResult.with_modifications(messages=list(messages) + [note], force_continue=True)
        return HookResult.no_changes()

    def after_tool(self, hook_input):
        if hook_input.tool_name != TOOL_NAME:
            return HookResult.no_changes()
        if not self.goal_text:
            try:
                self._ensure_goal(getattr(hook_input, "messages", None) or [])
            except Exception:
                pass
        action = ""
        tool_input = hook_input.tool_input
        if isinstance(tool_input, dict):
            value = tool_input.get("action")
            if isinstance(value, str):
                action = value.strip()
        if not action:
            return HookResult.no_changes()

        observation = _flatten_text(hook_input.tool_output)
        if observation:
            try:
                self._observe(action, observation)
            except Exception:
                pass

        self.forced = 0
        note = self._note()
        if not note:
            return HookResult.no_changes()
        output = _insert_note(hook_input.tool_output, note)
        if output is hook_input.tool_output:
            return HookResult.no_changes()
        return HookResult.with_modifications(tool_output=output)

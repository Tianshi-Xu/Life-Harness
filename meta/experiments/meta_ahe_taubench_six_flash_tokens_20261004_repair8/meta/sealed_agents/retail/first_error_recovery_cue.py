"""First-error recovery cue middleware for the retail LLM agent.

ONE mechanism: when the LATEST observation(s) are tool errors, embed a single,
short, concrete corrective cue directly INSIDE the error observation text (the
first matching category wins). The cue names the specific fix for that class of
error -- do not repeat the identical call, use a real identifier from the data
already seen, use the tool that matches the order's present status, or realign
the caller's argument lists -- so the solver's next action changes course.

The middleware is a pure pass-through when no trailing tool error qualifies. It
never calls a model, never touches the filesystem/network, keeps no cross-episode
mutable state (only the single overwritten ``_last_history``), and degrades to
identity behaviour on any unexpected structure.
"""

import json

MARKER = "[GUIDANCE"

# Ordered (needle-tuple, cue) categories; the FIRST matching category wins.
_CUE_SHOULD_DIFFER = (
    "[GUIDANCE: the new item id equals the old id. Do NOT repeat this call. "
    "Look up the product variants and choose a DIFFERENT available item id of "
    "the same product as the replacement.]"
)

_CUE_PAYMENT_METHOD = (
    "[GUIDANCE: wrong payment method. Do NOT repeat this call. Reuse the "
    "payment method id that appears in the order's own details (e.g. "
    "gift_card_... / credit_card_...); never invent one or reuse an id from "
    "another order.]"
)

_CUE_STATUS = (
    "[GUIDANCE: this action is not allowed for the order's current status. Do "
    "NOT repeat it. Tell the user it cannot be done in this state and use the "
    "tool that matches the present status (pending: cancel_pending_order / "
    "modify_pending_order_*; delivered: return_delivered_order_items / "
    "exchange_delivered_order_items).]"
)

_CUE_SHOULD_MATCH = (
    "[GUIDANCE: the lists must align - item_ids and new_item_ids must be the "
    "same length and each new id must be the same product as the id in the same "
    "position. Rebuild both lists from the order details.]"
)

_CUE_NOT_FOUND = (
    "[GUIDANCE: that id was not found. Do NOT repeat this exact call. Copy "
    "every id EXACTLY as given by the user or by a prior tool result (order ids "
    "keep their leading '#', e.g. #W1234567) and never use an id that appears "
    "only as an example inside a tool description. If it still fails, call "
    "get_user_details and reuse an id from its returned list.]"
)


# --------------------------------------------------------------------------- #
# Pure helpers (no side effects)
# --------------------------------------------------------------------------- #
def _text_of(content):
    """Render a message/tool content value as text (str-or-JSON)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, sort_keys=True)
    except Exception:
        return str(content)


def _is_error(text):
    """True when the rendered text looks like an error observation."""
    low = text.lower()
    return low.startswith("error") or "error:" in low


def classify_cue(error_text):
    """Return ONE short cue string for the FIRST matching category, else None.

    Categories are checked in a fixed order; the first match wins. The input is
    lowercased once. Never raises: any unexpected input yields ``None``.
    """
    try:
        if error_text is None:
            return None
        low = (error_text if isinstance(error_text, str) else str(error_text)).lower()

        if "should be different from the old item id" in low:
            return _CUE_SHOULD_DIFFER
        if "payment method not found" in low or "original payment method" in low:
            return _CUE_PAYMENT_METHOD
        if (
            "non-pending" in low
            or "non-delivered" in low
            or "cannot be cancelled" in low
            or "cannot be modified" in low
            or "cannot be returned" in low
            or "cannot be exchanged" in low
        ):
            return _CUE_STATUS
        if "should match" in low:
            return _CUE_SHOULD_MATCH
        if "not found" in low:
            return _CUE_NOT_FOUND
        return None
    except Exception:
        return None


def _is_message_list(value):
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, dict) for x in value)
    )


def _append_cue(message, cue):
    """Append the cue to a message's content in place. Returns True on change."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = content + "\n" + cue
        return True
    if isinstance(content, list):
        content.append({"type": "text", "text": cue})
        return True
    return False


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context
        self._last_history = None

    # -- extraction -------------------------------------------------------- #
    def _extract_messages(self, request):
        """Return (messages, attach) where ``attach`` re-binds msgs to request.

        ``attach`` is a tuple ``("dict", key)`` or ``("attr", name)`` when the
        list lives on the request, or ``None`` when it came from the recorded
        history fallback (in which case a copy is used and nothing is bound, so
        nothing accumulates across calls).
        """
        if isinstance(request, dict):
            for key in ("messages", "input", "history", "chat"):
                if _is_message_list(request.get(key)):
                    return request[key], ("dict", key)
        else:
            msgs = getattr(request, "messages", None)
            if _is_message_list(msgs):
                return msgs, ("attr", "messages")

        if _is_message_list(self._last_history):
            # fallback source: work on a copy so no cross-call accumulation.
            return list(self._last_history), None

        return None, None

    # -- public API -------------------------------------------------------- #
    def transform_request(self, request):
        try:
            msgs, _attach = self._extract_messages(request)
            if msgs is None or not msgs:
                return request

            last = msgs[-1]
            if not (isinstance(last, dict) and last.get("role") == "tool"):
                return request

            # Trailing run of tool-role messages ending at the list tail.
            i = len(msgs) - 1
            while (
                i >= 0
                and isinstance(msgs[i], dict)
                and msgs[i].get("role") == "tool"
            ):
                i -= 1
            trailing = msgs[i + 1:]

            for msg in trailing:
                text = _text_of(msg.get("content"))
                if MARKER in text:
                    continue
                if not _is_error(text):
                    continue
                cue = classify_cue(text)
                if cue is None:
                    continue
                _append_cue(msg, cue)

            return request
        except Exception:
            return request

    def transform_response(self, response, public_history):
        try:
            if isinstance(public_history, list):
                self._last_history = public_history
        except Exception:
            pass
        return response


def register():
    return None

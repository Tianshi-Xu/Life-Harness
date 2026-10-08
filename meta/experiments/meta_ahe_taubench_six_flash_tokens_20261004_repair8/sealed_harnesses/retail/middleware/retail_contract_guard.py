"""Deterministic contract guard for the native retail tool set.

Every rule below comes from a failure that is visible in a recorded episode of
this environment:

* the agent authenticates with an invented placeholder (``user@example.com``,
  first/last ``user``/``example``, zip ``12345``) instead of asking the customer,
  burning the error budget on a call that can never succeed;
* the agent hands ``new_item_ids`` values that it never read from any tool
  result (the ids printed inside the tool documentation), or sends the very
  same item id as its own replacement - and this environment allows an order to
  be exchanged only once, so one bad call destroys the whole task;
* the agent transfers the customer to a human agent before it has authenticated
  or looked at a single order, which abandons requests the tools can handle.

The middleware never executes a tool itself. It only

1. repairs outgoing arguments (``before_tool``) - an order id that is missing
   its ``#`` prefix or that arrives as a bare number is normalised to the
   ``#W1234567`` form the environment requires, and id lists are coerced to
   strings; and
2. removes a call that is provably wrong (``after_model``) and appends a short
   framework note explaining what to do instead, so the model continues the
   turn with better information instead of wasting an environment error.

All state is re-derived from the episode's own message history, so the
middleware keeps no state between episodes.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from nexau.archs.main_sub.execution.hooks import (
    AfterModelHookInput,
    AfterToolHookInput,
    BeforeToolHookInput,
    HookResult,
    Middleware,
)
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock

_NOTE_MARK = "[CONTRACT NOTE]"

# A guard may only fire this many times in one episode; a call whose note key is
# already exhausted is allowed through so that the episode can never be stuck.
# The budget has to be well above the per-key limits because every note key is
# spent at most a couple of times: recorded episodes spent the old budget of 6
# within the first few turns and then ran the rest of the conversation with no
# protection at all, which is exactly when the expensive mistakes happened.
_MAX_NOTES = 14
_DEFAULT_PER_KEY = 2
_PER_KEY = {
    "unseen-new-item": 1,
    "unseen-payment": 1,
    "transfer-early": 3,
    "auth-repeat": 2,
    "read-profile": 1,
    "wrong-product": 2,
    "unavailable-variant": 2,
}

# --------------------------------------------------------------------- shapes

_ORDER_W = re.compile(r"^#?[Ww](\d{5,})$")
_ORDER_NUMBER = re.compile(r"^#?(\d{5,8})$")
_ITEM_ID = re.compile(r"^\d{7,}$")
_PAYMENT_ID = re.compile(r"^(?:gift_card|credit_card|paypal|debit_card|bank_account)_\d+$")
_EMAIL = re.compile(r"^([a-z0-9._%+-]+)@([a-z0-9.-]+)$")

_ID_TOKEN = re.compile(
    r"#?W\d{5,}"
    r"|\b\d{7,}\b"
    r"|(?:gift_card|credit_card|paypal)_\d+"
    r"|[a-z][a-z_]*_\d{3,}"
)

# A customer id such as "amelia_silva_7726", as returned by the auth tools.
_USER_ID = re.compile(r"[A-Za-z][A-Za-z_]*_\d{3,}")

_PLACEHOLDER_EMAIL_LOCAL = frozenset(
    {"user", "name", "email", "test", "example", "your", "you", "customer", "someone", "placeholder", "foo", "bar"}
)
_PLACEHOLDER_NAME_FIRST = frozenset({"user", "example", "test", "name", "placeholder", "someone"})
_PLACEHOLDER_NAME_LAST = frozenset({"example", "test", "name", "placeholder", "unknown"})
_PLACEHOLDER_ZIPS = frozenset({"12345", "00000", "1234", "11111", "zip"})

_ITEM_TOOLS = frozenset(
    {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
    }
)
_AUTH_TOOLS = frozenset(
    {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
    }
)
_PAYMENT_TOOLS = frozenset(
    {
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        "modify_pending_order_items",
        "modify_pending_order_payment",
    }
)

_ERROR_TEXT = re.compile(r"(?:^|\n)\s*Error\b|Error: ")


# --------------------------------------------------------------------- helpers


def _bare_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    name = value.strip()
    if name.startswith("Tool: "):
        name = name[len("Tool: ") :].strip()
    return name


def _text_of(block: Any) -> str:
    if isinstance(block, TextBlock):
        return block.text or ""
    if isinstance(block, str):
        return block
    return ""


def _result_text(block: ToolResultBlock) -> str:
    content = block.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text_of(inner) for inner in content)
    if isinstance(content, (dict, tuple)):
        return _output_text(content)
    return ""


def _is_error(text: str) -> bool:
    return bool(_ERROR_TEXT.search(text or ""))


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for entry in value:
        if isinstance(entry, bool):
            continue
        if isinstance(entry, (str, int)):
            out.append(str(entry).strip())
    return out


def _note_keys(text: str) -> list[str]:
    keys = []
    for chunk in (text or "").split(_NOTE_MARK)[1:]:
        key = chunk.split("\n", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def _strip_calls(messages: Any, dropped: set) -> list:
    """Remove the dropped tool-use blocks so no call is left unanswered."""

    updated = []
    for message in messages or []:
        if getattr(message, "role", None) is not Role.ASSISTANT:
            updated.append(message)
            continue
        kept = []
        removed = False
        for block in message.content or []:
            if isinstance(block, ToolUseBlock) and str(block.id) in dropped:
                removed = True
                continue
            kept.append(block)
        if not removed:
            updated.append(message)
            continue
        if not kept:
            kept = [TextBlock(text="[invalid tool call removed by contract guard]")]
        updated.append(message.model_copy(update={"content": kept}))
    return updated


def _framework_note(text: str) -> Message:
    return Message(role=Role.FRAMEWORK, content=[TextBlock(text=text)])


def _repair_order_id(value: Any) -> Any:
    """Return the ``#W1234567`` form of an order id, else the original value."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    match = _ORDER_W.match(text)
    if match:
        fixed = "#W" + match.group(1)
        return fixed if fixed != text else value
    match = _ORDER_NUMBER.match(text)
    if match:
        return "#W" + match.group(1)
    return value


# ------------------------------------------------------------------- scanning


class _Scan:
    """A read-only view of the episode rebuilt from the message history."""

    def __init__(self) -> None:
        self.notes: dict[str, int] = {}
        self.seen_ids: set[str] = set()
        self.resolved: bool = False
        self.auth_user_id: str = ""
        # item id -> product id / product name / options, from get_order_details
        self.item_product: dict[str, str] = {}
        self.item_name: dict[str, str] = {}
        self.item_options: dict[str, dict[str, str]] = {}
        # variant item id -> product id / availability / short label, and
        # product id -> {variant item id: available}, from get_product_details
        self.variant_owner: dict[str, str] = {}
        self.variant_available: dict[str, bool] = {}
        self.variant_label: dict[str, str] = {}
        self.product_variants: dict[str, dict[str, bool]] = {}
        self.product_name: dict[str, str] = {}


def _add_ids(text: str, scan: _Scan) -> None:
    if not isinstance(text, str) or not text:
        return
    for token in _ID_TOKEN.findall(text):
        scan.seen_ids.add(token)
        if token.startswith("#"):
            scan.seen_ids.add(token[1:])
        elif _ORDER_W.match(token):
            scan.seen_ids.add("#" + token)


def _payload(text: Any) -> Any:
    """Return the object a read tool answered with, from a stored tool result.

    The framework stores a tool result as a JSON document wrapping the value the
    tool returned (``{"result": "<json string>"}``); older records keep the
    Python ``repr`` of that wrapper. Both shapes are unwrapped here, and a value
    that cannot be decoded is reported as ``None`` so that a rule can only fire
    on data that was really read.
    """

    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    value: Any = None
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(raw)
            break
        except Exception:
            value = None
    for _ in range(4):
        if isinstance(value, str):
            inner: Any = None
            for loader in (json.loads, ast.literal_eval):
                try:
                    inner = loader(value.strip())
                    break
                except Exception:
                    inner = None
            if inner is None:
                return None
            value = inner
        elif isinstance(value, dict) and "result" in value:
            value = value["result"]
        else:
            break
    return value


def _brief(options: Any) -> str:
    if not isinstance(options, dict) or not options:
        return ""
    parts = []
    for key in sorted(options):
        parts.append("%s=%s" % (key, options[key]))
    return ", ".join(parts)


def _index_order(text: str, scan: _Scan) -> None:
    data = _payload(text)
    if not isinstance(data, dict):
        return
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id") or "").strip()
        if not item_id:
            continue
        product_id = str(item.get("product_id") or "").strip()
        if product_id:
            scan.item_product[item_id] = product_id
        name = str(item.get("name") or "").strip()
        if name:
            scan.item_name[item_id] = name
        options = item.get("options")
        if isinstance(options, dict):
            scan.item_options[item_id] = {
                str(key).lower(): str(value).lower() for key, value in options.items()
            }


def _index_product(text: str, scan: _Scan) -> None:
    data = _payload(text)
    if not isinstance(data, dict):
        return
    product_id = str(data.get("product_id") or "").strip()
    variants = data.get("variants")
    if not product_id or not isinstance(variants, dict):
        return
    name = str(data.get("name") or "").strip()
    if name:
        scan.product_name[product_id] = name
    table = scan.product_variants.setdefault(product_id, {})
    for variant_id, info in variants.items():
        vid = str(variant_id).strip()
        if not vid:
            continue
        available: bool | None = None
        options: Any = None
        if isinstance(info, dict):
            raw_available = info.get("available")
            if isinstance(raw_available, bool):
                available = raw_available
            options = info.get("options")
        scan.variant_owner[vid] = product_id
        table[vid] = available
        if available is not None:
            scan.variant_available[vid] = available
        label = _brief(options)
        if label:
            scan.variant_label[vid] = label


def _scan(messages: Any) -> _Scan:
    scan = _Scan()
    pending: dict[str, str] = {}
    for message in messages or []:
        blocks = getattr(message, "content", None) or []
        role = getattr(message, "role", None)
        for block in blocks:
            if isinstance(block, ToolUseBlock):
                pending[str(block.id)] = _bare_name(block.name)
            elif isinstance(block, ToolResultBlock):
                name = pending.pop(str(block.tool_use_id), None)
                text = _result_text(block)
                _add_ids(text, scan)
                if name == "get_user_details" and text.strip() and not _is_error(text):
                    scan.resolved = True
                if not _is_error(text):
                    if name == "get_order_details":
                        _index_order(text, scan)
                    elif name == "get_product_details":
                        _index_product(text, scan)
                if (
                    name in ("find_user_id_by_email", "find_user_id_by_name_zip")
                    and text.strip()
                    and not _is_error(text)
                ):
                    found = _USER_ID.search(text)
                    if found:
                        scan.auth_user_id = found.group(0)
            elif isinstance(block, TextBlock):
                text = block.text or ""
                if role is Role.USER and text.strip():
                    _add_ids(text, scan)
                if role in (Role.ASSISTANT, Role.FRAMEWORK, Role.USER):
                    for key in _note_keys(text):
                        scan.notes[key] = scan.notes.get(key, 0) + 1
    return scan


# --------------------------------------------------------------------- rules


def _placeholder_identity(name: str, arguments: dict) -> str:
    """Return a short reason when an authentication call uses a dummy value."""

    if name == "find_user_id_by_email":
        email = str(arguments.get("email") or "").strip().lower()
        match = _EMAIL.match(email)
        if match and match.group(1) in _PLACEHOLDER_EMAIL_LOCAL and not any(ch.isdigit() for ch in match.group(1)):
            return email
        return ""
    if name == "find_user_id_by_name_zip":
        first = str(arguments.get("first_name") or "").strip().lower()
        last = str(arguments.get("last_name") or "").strip().lower()
        zipcode = str(arguments.get("zip") or "").strip().lower()
        if zipcode in _PLACEHOLDER_ZIPS:
            return "zip " + zipcode
        if first in _PLACEHOLDER_NAME_FIRST or last in _PLACEHOLDER_NAME_LAST:
            return "%s %s" % (first, last)
        return ""
    return ""


def _output_text(value: Any) -> str:
    """The readable text of a tool result, whatever wrapper it arrived in."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_output_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_output_text(item) for item in value)
    return "" if value is None else str(value)


def _with_hint(value: Any, hint: str, only_errors: bool) -> Any:
    """Return ``value`` with ``hint`` appended to its readable leaves."""

    if isinstance(value, str):
        if value.strip() and (not only_errors or _is_error(value)):
            return value + "\n" + hint
        return value
    if isinstance(value, dict):
        return {key: _with_hint(item, hint, only_errors) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_hint(item, hint, only_errors) for item in value]
    return value


def _product_label(product_id: str, scan: _Scan) -> str:
    name = scan.product_name.get(product_id) or ""
    return "%s (%s)" % (name, product_id) if name else product_id


def _available_lines(product_id: str, scan: _Scan) -> str:
    """A few available variants of ``product_id``, as printed by the listing."""

    table = scan.product_variants.get(product_id) or {}
    lines = []
    for variant_id in sorted(table):
        if table.get(variant_id) is False:
            continue
        label = scan.variant_label.get(variant_id, "")
        lines.append(
            "  - item_id %s%s" % (variant_id, (" - " + label) if label else "")
        )
        if len(lines) >= 5:
            break
    return "\n".join(lines)


def _wrong_product_note(old: str, new: str, old_product: str, scan: _Scan) -> str:
    chosen = _product_label(scan.variant_owner.get(new, ""), scan)
    wanted = _product_label(old_product, scan)
    item_name = scan.item_name.get(old) or "the item being replaced"
    options = _brief(scan.item_options.get(old))
    detail = (" - " + options) if options else ""
    lines = _available_lines(old_product, scan)
    listed = (
        "\nThe variants of %s that the listing marks available are:\n%s\n" % (wanted, lines)
        if lines
        else ""
    )
    return (
        "new_item_ids contains %s, which is a variant of %s, but the item it is matched with (%s, a "
        "%s%s) belongs to %s. These tools can only replace an item with another variant of that same "
        "item's product - they cannot swap one product for a different one, and the environment "
        "answers \"Variant not found\" when the ids do not belong together.%sIf the customer asked "
        "for a different product, this is simply not something these tools can do: say so plainly and "
        "carry on with the rest of their request. Otherwise pick the replacement from the variants of "
        "%s."
        % (new, chosen, old, item_name, detail, wanted, listed, wanted)
    )


def _unavailable_note(new: str, product_id: str, scan: _Scan) -> str:
    label = scan.variant_label.get(new, "")
    chosen = ("-" + label) if label else ""
    wanted = _product_label(product_id, scan)
    lines = _available_lines(product_id, scan)
    listed = (
        "\nThe variants of %s that are available are:\n%s\n" % (wanted, lines)
        if lines
        else ""
    )
    return (
        "new_item_ids contains %s%s, which get_product_details lists as \"available\": false, so the "
        "environment rejects the replacement and the call is wasted (the single write this order "
        "allows is not consumed, but the turn is).%sChoose one of the available variants, or if none "
        "of them matches what the customer asked for, tell the customer that this item cannot be "
        "changed the way they want." % (new, chosen, listed)
    )


def _problem(name: str, arguments: dict, scan: _Scan) -> tuple[str, str] | None:
    """Return ``(note key, explanation)`` when a call must not be executed."""

    if not isinstance(arguments, dict):
        return None

    if name in ("find_user_id_by_email", "find_user_id_by_name_zip"):
        placeholder = _placeholder_identity(name, arguments)
        if placeholder:
            return (
                "placeholder-identity",
                "You called %s with the value %r, which is a documentation placeholder, not data "
                "the customer gave you. You do not know this customer's identity yet, so this call "
                "cannot succeed. Ask the customer for their email address, or for their first name, "
                "last name and zip code, and use exactly what they answer." % (name, placeholder),
            )

    if name in ("find_user_id_by_email", "find_user_id_by_name_zip") and scan.auth_user_id:
        return (
            "auth-repeat",
            "This customer is already identified: a previous %s call returned %s. Calling the "
            "authentication tools again cannot give you anything new, and guessing an email or a "
            "zip code will only fail. Call get_user_details with user_id=\"%s\" - it returns this "
            "customer's orders (order ids, statuses, items) and their payment methods - and take "
            "everything you need from that result."
            % (name, scan.auth_user_id, scan.auth_user_id),
        )

    if name in _PAYMENT_TOOLS and not scan.resolved:
        payment = str(arguments.get("payment_method_id") or "").strip()
        if payment:
            return (
                "read-profile",
                "You are about to settle money with payment_method_id %s, but this conversation has "
                "never read the customer's profile (no successful get_user_details). Read it first: it "
                "returns the customer's own payment methods and their orders. A return's refund settles "
                "on the method the order was paid with - copy that id out of the order's payment_history "
                "- and if the customer has not already told you which of their methods to use for a "
                "price difference, ask them. Then re-send this call with that payment method id." % payment,
            )

    if name in _ITEM_TOOLS:
        item_ids = _str_list(arguments.get("item_ids"))
        new_ids = _str_list(arguments.get("new_item_ids"))
        if item_ids and new_ids and len(item_ids) != len(new_ids):
            return (
                "ids-length",
                "item_ids and new_item_ids must have the same length: the replacement in each "
                "position belongs to the item item_ids[position]. You sent %d item id(s) and %d "
                "replacement id(s). Rebuild both lists in the same item order."
                % (len(item_ids), len(new_ids)),
            )
        for old, new in zip(item_ids, new_ids):
            if not old or not new:
                continue
            if old == new:
                return (
                    "self-exchange",
                    "new_item_ids contains %s, the item that is being replaced, so this call would "
                    "exchange an item for itself and would consume the single exchange allowed for "
                    "this order. If the order is pending the right tool is modify_pending_order_items "
                    "and if it is delivered it is exchange_delivered_order_items - check the status in "
                    "get_order_details first. Then call get_product_details with the item's product_id "
                    "and choose a DIFFERENT variant with \"available\": true that matches what the "
                    "customer asked for." % old,
                )
            owner = scan.variant_owner.get(new)
            old_product = scan.item_product.get(old)
            if owner and old_product and owner != old_product:
                return ("wrong-product", _wrong_product_note(old, new, old_product, scan))
            if owner and scan.variant_available.get(new) is False:
                return ("unavailable-variant", _unavailable_note(new, owner, scan))
        if scan.seen_ids:
            for new in new_ids:
                if _ITEM_ID.match(new) and new not in scan.seen_ids:
                    return (
                        "unseen-new-item",
                        "new_item_ids contains %s, which has never appeared in any tool result or in "
                        "the customer's messages - it is a documentation example or a guess, and the "
                        "environment will reject it. Check the order status in get_order_details "
                        "(pending orders use modify_pending_order_items, delivered orders use "
                        "exchange_delivered_order_items), then open get_product_details for the "
                        "product of the item being replaced and copy an available variant item id "
                        "from that result." % new,
                    )

    if name in _PAYMENT_TOOLS and scan.seen_ids:
        payment = str(arguments.get("payment_method_id") or "").strip()
        if _PAYMENT_ID.match(payment) and payment not in scan.seen_ids:
            return (
                "unseen-payment",
                "payment_method_id %s has never appeared in any tool result or in the customer's "
                "messages, so it is invented and the refund or charge would be booked against a "
                "method that does not belong to this customer. Call get_user_details for this "
                "customer and copy one of the payment method ids from that result." % payment,
            )

    if name == "transfer_to_human_agents" and not scan.resolved:
        return (
            "transfer-early",
            "A transfer is only for requests that these tools cannot handle, and you have not "
            "resolved this customer yet (no successful get_user_details in this conversation). "
            "Authenticate with find_user_id_by_email or find_user_id_by_name_zip, read "
            "get_user_details and the orders involved, then continue with the request. Transfer "
            "only if it is still impossible after that.",
        )

    return None


# ------------------------------------------------------------------ middleware


class RetailContractGuardMiddleware(Middleware):
    """Repairs action arguments and blocks provably wrong actions."""

    # ------------------------------------------------------------- arguments

    def before_tool(self, hook_input: BeforeToolHookInput) -> HookResult:
        try:
            arguments = hook_input.tool_input
            if not isinstance(arguments, dict) or not arguments:
                return HookResult.no_changes()
            updated = dict(arguments)
            changed = False

            if "order_id" in updated:
                fixed = _repair_order_id(updated.get("order_id"))
                if fixed != updated.get("order_id"):
                    updated["order_id"] = fixed
                    changed = True

            for key in ("item_ids", "new_item_ids"):
                value = updated.get(key)
                if isinstance(value, list) and any(isinstance(entry, int) and not isinstance(entry, bool) for entry in value):
                    updated[key] = _str_list(value)
                    changed = True

            if not changed:
                return HookResult.no_changes()
            return HookResult.with_modifications(tool_input=updated)
        except Exception:  # pragma: no cover - never break the loop
            return HookResult.no_changes()

    # ----------------------------------------------------------------- results

    def after_tool(self, hook_input: AfterToolHookInput) -> HookResult:
        try:
            return self._after_tool(hook_input)
        except Exception:  # pragma: no cover - never break the loop
            return HookResult.no_changes()

    def _after_tool(self, hook_input: AfterToolHookInput) -> HookResult:
        name = _bare_name(getattr(hook_input, "tool_name", ""))
        output = hook_input.tool_output
        text = _output_text(output)
        if not text.strip() or _NOTE_MARK in text:
            return HookResult.no_changes()

        hint = ""
        only_errors = True
        if name in _AUTH_TOOLS and not _is_error(text):
            found = _USER_ID.search(text)
            if found:
                only_errors = False
                hint = (
                    _NOTE_MARK
                    + " next-step\nYou are authenticated as "
                    + found.group(0)
                    + ". The next call is get_user_details with user_id=\""
                    + found.group(0)
                    + "\": it returns this customer's orders (order ids, statuses and items) and "
                    "their payment methods. Do not ask the customer for an order id you can read "
                    "from that result, and do not call an authentication tool again."
                )
        elif _is_error(text) and name in _ITEM_TOOLS:
            hint = self._replacement_hint(hook_input, text)
        elif _is_error(text) and re.search(r"insufficient|balance", text, re.IGNORECASE):
            hint = (
                _NOTE_MARK
                + " next-step\nThat payment method does not have enough balance. The order's "
                "payment history is not the customer's wallet: call get_user_details for this "
                "customer to see the payment methods they own, then ask the customer which one "
                "should be used for this change and send that id."
            )
        if not hint:
            return HookResult.no_changes()
        decorated = _with_hint(output, hint, only_errors)
        return HookResult.with_modifications(tool_output=decorated)

    def _replacement_hint(self, hook_input: AfterToolHookInput, text: str) -> str:
        """Point at the real variants after a rejected replacement id."""

        if not re.search(
            r"not found or available|variant not found|variant not available",
            text,
            re.IGNORECASE,
        ):
            return ""
        return (
            _NOTE_MARK
            + " next-step\nThe replacement id was rejected: an exchange or an item "
            "modification can only put ANOTHER VARIANT OF THE SAME PRODUCT in the item's place, "
            "and only one the product listing marks \"available\": true. The product_id and the "
            "current item_id are printed next to the item in get_order_details. Call "
            "get_product_details with that item's product_id and copy one of the item ids it "
            "lists as \"available\": true; never reuse an id from another product, and never "
            "repeat the rejected one - it cannot start working. If no available variant matches "
            "what the customer asked for, say so plainly instead of retrying."
        )

    # ----------------------------------------------------------------- calls

    def after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        try:
            return self._after_model(hook_input)
        except Exception:  # pragma: no cover - never break the loop
            return HookResult.no_changes()

    def _after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        parsed = hook_input.parsed_response
        calls = list(getattr(parsed, "tool_calls", None) or [])
        if not calls:
            return HookResult.no_changes()

        scan = _scan(hook_input.messages)
        if sum(scan.notes.values()) >= _MAX_NOTES:
            return HookResult.no_changes()

        kept = []
        dropped: set[str] = set()
        notes: list[str] = []
        for call in calls:
            name = _bare_name(getattr(call, "tool_name", ""))
            arguments = getattr(call, "parameters", None) or {}
            problem = _problem(name, arguments, scan)
            if problem is None:
                kept.append(call)
                continue
            key, explanation = problem
            if scan.notes.get(key, 0) >= _PER_KEY.get(key, _DEFAULT_PER_KEY):
                kept.append(call)
                continue
            dropped.add(str(getattr(call, "tool_call_id", "")))
            notes.append("%s %s\n%s" % (_NOTE_MARK, key, explanation))

        if not dropped:
            return HookResult.no_changes()

        try:
            parsed.tool_calls = kept
        except Exception:  # pragma: no cover - defensive
            return HookResult.no_changes()

        messages = _strip_calls(hook_input.messages, dropped)
        messages.append(
            _framework_note(
                "%d invalid tool call(s) were NOT executed.\n\n" % len(dropped) + "\n\n".join(notes)
            )
        )
        return HookResult.with_modifications(
            parsed_response=parsed,
            messages=messages,
            force_continue=True,
        )

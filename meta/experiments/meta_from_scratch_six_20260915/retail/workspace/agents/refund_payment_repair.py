"""Compact state-aware digests for verbose tool payloads.

The plugin wraps the plain domain toolkit's tool executor and appends a short
factual digest of the relevant database state to the result text of a small set
of read-only lookup tools. Digests are additive and machine-readable: the
original payload is preserved, tool errors propagate untouched, and any failure
while building or serializing a digest falls back to the original value.
"""

import json

_MARKER = "[STATE DIGEST]"
_TRUNCATED = "... (truncated)"
_MAX_DIGEST_LEN = 1200
_MIN_VARIANTS = 3
_MAX_UNAVAILABLE = 8
_MAX_ITEMS = 12
_MAX_USER_DIGEST_LEN = 1800
_MAP_LADDER = ((24, True), (16, True), (10, True), (10, False))
_PAYMENT_REPAIR_TRIGGER = "original payment method"

_INSTALLED = False
_ORIGINAL_USE_TOOL = None


def _value(obj, name, default=None):
    """Read a field from a mapping or attribute holder."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _mapping(obj):
    """Return ``obj`` when it behaves like a mapping, else an empty dict."""
    if callable(getattr(obj, "items", None)):
        return obj
    return {}


def _available(variant):
    """Return the availability flag of a variant."""
    return bool(_value(variant, "available", False))


def _money(value):
    """Format a monetary value with two decimals."""
    return f"{float(value):.2f}"


def _options_text(options):
    """Render option key/value pairs sorted by key."""
    pairs = _mapping(options)
    if not pairs:
        return "-"
    return ";".join(f"{key}={pairs[key]}" for key in sorted(pairs, key=str))


def _variant_line(variant_id, variant):
    """Render one product variant as a single line."""
    return (
        f"item {variant_id} available={'true' if _available(variant) else 'false'} "
        f"price={_money(_value(variant, 'price', 0.0))} "
        f"options={_options_text(_value(variant, 'options'))}"
    )


def _item_line(item):
    """Render one order item as a single line."""
    return (
        f"item {_value(item, 'item_id', '?')} "
        f"price={_money(_value(item, 'price', 0.0))} "
        f"options={_options_text(_value(item, 'options'))}"
    )


def _method_text(payment_method):
    """Render one payment method with the details that identify it."""
    method_id = _value(payment_method, "id", "?")
    source = _value(payment_method, "source") or "unknown"
    if source == "gift_card":
        balance = _money(_value(payment_method, "balance", 0.0))
        return f"{method_id} type=gift_card balance={balance}"
    if source == "credit_card":
        brand = _value(payment_method, "brand", "?")
        last_four = _value(payment_method, "last_four", "?")
        return f"{method_id} type=credit_card brand={brand} last4={last_four}"
    return f"{method_id} type={source}"


def _methods_line(user):
    """Render a user and their payment methods on one line."""
    methods = list(_mapping(_value(user, "payment_methods")).values())
    head = f"user {_value(user, 'user_id', '?')} payment_methods={len(methods)}"
    if not methods:
        return head
    return head + " " + " | ".join(_method_text(method) for method in methods)


def _fit(header, body):
    """Join digest lines, truncating the tail to respect the length cap."""
    budget = _MAX_DIGEST_LEN - len(_MARKER) - 1
    lines = [header] + list(body)
    if len("\n".join(lines)) <= budget:
        return "\n".join(lines)
    lines = lines[:-1]
    while lines and len("\n".join(lines + [_TRUNCATED])) > budget:
        lines = lines[:-1]
    if not lines:
        return ""
    return "\n".join(lines + [_TRUNCATED])


def _product_for(db, product_id=None, item_id=None):
    """Find a product by id, or the product that owns an item id."""
    products = _mapping(_value(db, "products"))
    if product_id is not None:
        return products.get(product_id)
    for product in products.values():
        if item_id in _mapping(_value(product, "variants")):
            return product
    return None


def _product_digest(toolkit, product_id=None, item_id=None):
    """Build an availability-first variant digest, or None if too small."""
    product = _product_for(_value(toolkit, "db"), product_id, item_id)
    if product is None:
        return None
    variants = _mapping(_value(product, "variants"))
    if len(variants) < _MIN_VARIANTS:
        return None
    available = [key for key, variant in variants.items() if _available(variant)]
    unavailable = [key for key, variant in variants.items() if not _available(variant)]
    header = (
        f"product {_value(product, 'product_id') or product_id or '?'} "
        f"name={_value(product, 'name') or '?'} variants={len(variants)} "
        f"available={len(available)} unavailable={len(unavailable)}"
    )
    body = [_variant_line(key, variants[key]) for key in available]
    body += [_variant_line(key, variants[key]) for key in unavailable[:_MAX_UNAVAILABLE]]
    return _fit(header, body)


def _order_digest(toolkit, order_id):
    """Build a digest of an order, its payments and the owner's methods."""
    db = _value(toolkit, "db")
    order = _mapping(_value(db, "orders")).get(order_id)
    if order is None:
        return None
    items = list(_value(order, "items") or [])
    header = (
        f"order {_value(order, 'order_id') or order_id} "
        f"status={_value(order, 'status', '?')} items={len(items)}"
    )
    body = [_item_line(item) for item in items[:_MAX_ITEMS]]
    for payment in _value(order, "payment_history") or []:
        body.append(
            f"payment {_value(payment, 'transaction_type', '?')} "
            f"amount={_money(_value(payment, 'amount', 0.0))} "
            f"method={_value(payment, 'payment_method_id', '?')}"
        )
    owner = _mapping(_value(db, "users")).get(_value(order, "user_id"))
    if owner is not None:
        body.append(_methods_line(owner))
    return _fit(header, body)


def _flat_text(value, fallback="?"):
    """Flatten one database field into a single line of printable ASCII."""
    raw = str(value if value is not None else "")
    text = "".join(char if char.isprintable() else " " for char in raw)
    text = " ".join(text.encode("ascii", "ignore").decode("ascii").split())
    return text or fallback


def _order_items(order):
    """Return an order's items as a list, tolerating malformed payloads."""
    items = _value(order, "items")
    if isinstance(items, (list, tuple)):
        return list(items)
    return list(_mapping(items).values())


def _map_item_text(item, name_len):
    """Render one item of an order as ``name item id price`` on one line."""
    name = _flat_text(_value(item, "name"))
    if len(name) > name_len:
        name = name[:name_len] + "..."
    item_id = _flat_text(_value(item, "item_id"))
    return f"{name} {item_id} {_money(_value(item, 'price', 0.0))}"


def _address_text(address, with_address2):
    """Render an order's own shipping address as one flattened line."""
    parts = [_flat_text(_value(address, "address1"))]
    second = _flat_text(_value(address, "address2"), "")
    if with_address2 and second:
        parts.append(second)
    parts.append(_flat_text(_value(address, "city")))
    parts.append(_flat_text(_value(address, "state")))
    parts.append(_flat_text(_value(address, "zip")))
    return ", ".join(parts)


def _map_order_line(order, order_id, name_len, with_address2, show_items=True):
    """Render one order of the user map as a single ASCII line."""
    items = _order_items(order)
    rendered = [_map_item_text(item, name_len) for item in items] if show_items else []
    head = (
        f"order {_flat_text(_value(order, 'order_id') or order_id)} "
        f"status={_flat_text(_value(order, 'status'))} "
        f"ship={_address_text(_value(order, 'address'), with_address2)} "
        f"items={len(rendered)}"
    )
    if rendered:
        return head + " :: " + "; ".join(rendered)
    if not show_items and items:
        return head + f" :: +{len(items)} items (order lookup)"
    return head


def _map_block(orders, order_ids, total, name_len, with_address2, show_items=True):
    """Render the map coverage line plus one line per included order."""
    lines = []
    for order_id in order_ids:
        order = orders.get(order_id)
        if order is not None:
            lines.append(
                _map_order_line(order, order_id, name_len, with_address2, show_items)
            )
    return [f"orders={total} shown={len(lines)}"] + lines


def _map_fits(header, method_lines, map_lines):
    """Return True when the composed user digest respects its length cap."""
    text = "\n".join([header] + list(method_lines) + list(map_lines))
    return len(text) <= _MAX_USER_DIGEST_LEN


def _user_map_lines(header, method_lines, orders, raw_order_ids):
    """Build the largest order map that fits, compressing names first.

    Tries the full list at decreasing name lengths, then without the secondary
    address line, then with trailing order lines dropped and ``shown`` kept
    accurate. Only when a single order still cannot fit does that order's item
    list become an explicit ``+k items (order lookup)`` marker; if even that
    does not fit, no map is returned and the caller keeps methods only.
    """
    order_ids = [order_id for order_id in raw_order_ids if orders.get(order_id) is not None]
    if not order_ids:
        return []
    total = len(raw_order_ids)
    for name_len, with_address2 in _MAP_LADDER:
        lines = _map_block(orders, order_ids, total, name_len, with_address2)
        if _map_fits(header, method_lines, lines):
            return lines
    name_len, with_address2 = _MAP_LADDER[-1]
    keep = len(order_ids)
    while keep > 1:
        keep -= 1
        lines = _map_block(orders, order_ids[:keep], total, name_len, with_address2)
        if _map_fits(header, method_lines, lines):
            return lines
    lines = _map_block(
        orders, order_ids[:1], total, name_len, with_address2, show_items=False
    )
    if _map_fits(header, method_lines, lines):
        return lines
    return []


def _user_digest(toolkit, user_id):
    """Build a digest of a user, payment methods and a bounded order map.

    The unchanged payment-method lines always come first. The map that follows
    is derived purely from the current database: one line per resolvable order
    of the user, in stored order, carrying the order id, status, the order's own
    shipping address and its item list (name, item id, price), with the total
    number of order ids and the number of shown orders on a leading line. Names
    are shortened, then the secondary address is dropped, then trailing order
    lines are dropped, only as far as needed to keep the digest inside its own
    larger cap; the item list of a lone order that still cannot fit becomes an
    explicit count marker instead of being hidden. Any failure while building
    the map falls back to the methods-only digest, so the method lines are
    always preserved.
    """
    db = _value(toolkit, "db")
    user = _mapping(_value(db, "users")).get(user_id)
    if user is None:
        return None
    methods = list(_mapping(_value(user, "payment_methods")).values())
    header = f"user {_value(user, 'user_id') or user_id} payment_methods={len(methods)}"
    method_lines = [f"method {_method_text(method)}" for method in methods]
    try:
        raw_orders = _value(user, "orders")
        orders = _mapping(_value(db, "orders"))
        if orders and isinstance(raw_orders, (list, tuple)) and len(raw_orders) > 0:
            map_lines = _user_map_lines(header, method_lines, orders, list(raw_orders))
            if map_lines:
                text = "\n".join([header] + method_lines + map_lines)
                if len(text) <= _MAX_USER_DIGEST_LEN:
                    return text
    except Exception:
        pass
    return _fit(header, method_lines)


def _argument(name, args, kwargs):
    """Return a named keyword argument, else the first positional one."""
    if name in kwargs:
        return kwargs[name]
    return args[0] if args else None


def _build_digest(toolkit, tool_name, args, kwargs):
    """Return the digest for a supported read tool, else None."""
    if tool_name == "get_product_details":
        return _product_digest(toolkit, product_id=_argument("product_id", args, kwargs))
    if tool_name == "get_item_details":
        return _product_digest(toolkit, item_id=_argument("item_id", args, kwargs))
    if tool_name == "get_order_details":
        return _order_digest(toolkit, _argument("order_id", args, kwargs))
    if tool_name == "get_user_details":
        return _user_digest(toolkit, _argument("user_id", args, kwargs))
    return None


def _serialize_result(result):
    """Serialize a tool result like the host does, or None if impossible."""
    if isinstance(result, str):
        return result
    dump = getattr(result, "model_dump", None)
    if callable(dump):
        return json.dumps(dump(), default=str)
    if isinstance(result, list):
        items = [
            item.model_dump() if callable(getattr(item, "model_dump", None)) else item
            for item in result
        ]
        return json.dumps(items, default=str)
    if isinstance(result, dict):
        return json.dumps(result, default=str)
    try:
        return json.dumps(result, default=str)
    except Exception:
        return None


def _repair_kwargs(toolkit, exc, kwargs):
    """Return kwargs with the order's own payment method, or None.

    Acts only on the domain's original-payment rejection and only when the
    call carried both an order id and a payment method. The replacement is
    the payment method stored first in the order's history; it is refused
    when missing, unreadable, not a string, already attempted, or when any
    read of the live state fails, so the call keeps its original error.
    """
    if not isinstance(exc, ValueError):
        return None
    if _PAYMENT_REPAIR_TRIGGER not in str(exc).lower():
        return None
    if "order_id" not in kwargs or "payment_method_id" not in kwargs:
        return None
    try:
        orders = _mapping(_value(_value(toolkit, "db"), "orders"))
        history = _value(orders.get(kwargs["order_id"]), "payment_history")
        if not isinstance(history, (list, tuple)) or not history:
            return None
        resolved = _value(history[0], "payment_method_id")
    except Exception:
        return None
    if not isinstance(resolved, str) or not resolved:
        return None
    if resolved == kwargs["payment_method_id"]:
        return None
    return {**kwargs, "payment_method_id": resolved}


def register():
    """Install the digest annotator on the plain domain toolkit."""
    global _INSTALLED, _ORIGINAL_USE_TOOL
    if _INSTALLED:
        return None
    try:
        from tau2.domains.retail.tools import RetailTools

        original = RetailTools.use_tool
        if getattr(original, "_state_digest_wrapped", False):
            _INSTALLED = True
            return None

        def use_tool(self, tool_name, *args, **kwargs):
            """Execute a tool, digesting results and repairing one payment rejection."""
            try:
                result = original(self, tool_name, *args, **kwargs)
            except ValueError as exc:
                repaired = _repair_kwargs(self, exc, kwargs)
                if repaired is None:
                    raise
                try:
                    result = original(self, tool_name, *args, **repaired)
                except Exception:
                    raise exc
            try:
                digest = _build_digest(self, tool_name, args, kwargs)
                if not digest:
                    return result
                serialized = _serialize_result(result)
                if serialized is None:
                    return result
                return f"{serialized}\n\n{_MARKER}\n{digest}"
            except Exception:
                return result

        use_tool._state_digest_wrapped = True
        _ORIGINAL_USE_TOOL = original
        RetailTools.use_tool = use_tool
    except Exception:
        return None
    _INSTALLED = True
    return None


class AgentHarness:
    """Identity harness; state digests are installed by register()."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        return request

    def transform_response(self, response, public_history):
        return response

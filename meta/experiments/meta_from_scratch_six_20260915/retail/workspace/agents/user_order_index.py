"""Compact state-aware digests for verbose tool payloads.

The plugin wraps the plain domain toolkit's tool executor and appends a short
factual digest of the relevant database state to the result text of a small set
of read-only lookup tools. Digests are additive and machine-readable: the
original payload is preserved, tool errors propagate untouched, and any failure
while building or serializing a digest falls back to the original value.

The digest of a user lookup additionally carries a bounded order index: every
order owned by the user is listed on one compact line with its status, the
ship-to city/state/zip and its leading items (product name, item id, price),
so the order a request refers to can be located from a single observation
instead of rescanning one large order payload at a time. The index covers a
bounded number of orders and of items per order, and it shortens long product
names just enough for the whole digest to stay inside the length cap, so every
indexed order id remains visible even for users with many orders.
"""

import json

_MARKER = "[STATE DIGEST]"
_TRUNCATED = "... (truncated)"
_MAX_DIGEST_LEN = 1200
_MIN_VARIANTS = 3
_MAX_UNAVAILABLE = 8
_MAX_ITEMS = 12
_MAX_ORDERS_SHOWN = 6
_MAX_ORDER_ITEMS_SHOWN = 3
_MAX_NAME_LEN = 40
_MIN_NAME_LEN = 6
_MAX_FIELD_LEN = 24

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


def _flat_text(value, limit):
    """Render one database field as single-line ASCII text of bounded length."""
    text = " ".join(str(value if value is not None else "?").split())
    text = text.encode("ascii", "ignore").decode("ascii") or "?"
    if limit <= 0:
        return ""
    return text[:limit]


def _order_items(order):
    """Return an order's items as a list, tolerating malformed payloads."""
    items = _value(order, "items")
    if isinstance(items, (list, tuple)):
        return list(items)
    return list(_mapping(items).values())


def _order_item_text(item, name_len):
    """Render one ordered item as ``name item_id price``."""
    name = _flat_text(_value(item, "name", "?"), name_len)
    item_id = _flat_text(_value(item, "item_id", "?"), _MAX_FIELD_LEN)
    return f"{name} {item_id} {_money(_value(item, 'price', 0.0))}"


def _order_head(order, order_id):
    """Render the identifying fields of one order as a single-line prefix."""
    address = _value(order, "address")
    return (
        f"order {_flat_text(_value(order, 'order_id') or order_id, _MAX_FIELD_LEN)} "
        f"status={_flat_text(_value(order, 'status', '?'), _MAX_FIELD_LEN)} "
        f"ship={_flat_text(_value(address, 'city', '?'), _MAX_FIELD_LEN)},"
        f"{_flat_text(_value(address, 'state', '?'), _MAX_FIELD_LEN)},"
        f"{_flat_text(_value(address, 'zip', '?'), _MAX_FIELD_LEN)} "
        f"items={len(_order_items(order))}"
    )


def _order_lines(orders, order_ids, name_len=None):
    """Render orders as compact lines and count the item slots they open.

    ``name_len`` bounds the item names; ``None`` drops the item details so the
    order ids stay visible when the digest has almost no room left.
    """
    lines = []
    slots = 0
    for order_id in order_ids:
        order = orders.get(order_id)
        if order is None:
            continue
        line = _order_head(order, order_id)
        if name_len is None:
            lines.append(line)
            continue
        items = _order_items(order)
        shown = items[:_MAX_ORDER_ITEMS_SHOWN]
        slots += len(shown)
        entries = [_order_item_text(item, name_len) for item in shown]
        if entries:
            line += ": " + "; ".join(entries)
        hidden = len(items) - len(shown)
        if hidden > 0:
            line += f" +{hidden} more"
        lines.append(line)
    return lines, slots


def _room_left(header, body):
    """Return the digest length still available for appended body lines."""
    budget = _MAX_DIGEST_LEN - len(_MARKER) - 1
    return budget - len("\n".join([header] + list(body))) - 1


def _order_index(db, user, room):
    """Render the user's orders as compact lines that fit in ``room`` chars."""
    raw_orders = _value(user, "orders")
    if not isinstance(raw_orders, (list, tuple)):
        return []
    orders = _mapping(_value(db, "orders"))
    order_ids = [order_id for order_id in raw_orders if orders.get(order_id) is not None]
    if not order_ids:
        return []
    shown_ids = order_ids[:_MAX_ORDERS_SHOWN]
    hidden_orders = len(order_ids) - len(shown_ids)
    tail = f"+{hidden_orders} more orders" if hidden_orders > 0 else None
    skeleton, slots = _order_lines(orders, shown_ids, 0)
    fixed = sum(len(line) + 1 for line in skeleton)
    if tail is not None:
        fixed += len(tail) + 1
    spare = room - fixed
    if slots and spare < slots * _MIN_NAME_LEN:
        lines = _order_lines(orders, shown_ids)[0]
    else:
        name_len = _MAX_NAME_LEN if not slots else min(_MAX_NAME_LEN, spare // slots)
        lines = _order_lines(orders, shown_ids, name_len)[0]
    if tail is not None:
        lines.append(tail)
    return lines


def _user_digest(toolkit, user_id):
    """Build a digest of a user, their payment methods and their orders."""
    db = _value(toolkit, "db")
    user = _mapping(_value(db, "users")).get(user_id)
    if user is None:
        return None
    methods = list(_mapping(_value(user, "payment_methods")).values())
    header = f"user {_value(user, 'user_id') or user_id} payment_methods={len(methods)}"
    body = [f"method {_method_text(method)}" for method in methods]
    body.extend(_order_index(db, user, _room_left(header, body)))
    return _fit(header, body)


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
            """Execute a tool, appending a state digest to read-tool results."""
            result = original(self, tool_name, *args, **kwargs)
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

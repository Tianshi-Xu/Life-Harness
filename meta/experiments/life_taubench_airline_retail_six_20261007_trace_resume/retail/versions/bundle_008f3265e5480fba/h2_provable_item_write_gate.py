"""H2: bounded action control against the current business records.

Every rule here replaces an action whose failure the CURRENT records already
prove, and each rule is per-episode bounded and deduplicated by signature:

* a verbatim repeat of a call that already came back with an error,
* a transfer to a human agent while no customer turn ever asked for one,
* an order id that is absent from the records (after a pure format recovery),
* a return/exchange/cancel/modify aimed at an order whose recorded status
  forbids that tool,
* an item-level write the catalog contradicts: an item id that is not in the
  target order, an item offered as its own replacement, a replacement list that
  does not line up positionally with the items it replaces, an unknown
  replacement id, a replacement belonging to another product, or a replacement
  that is marked unavailable.

TRAIN evidence for the last group: recorded episodes call
exchange_delivered_order_items / modify_pending_order_items with new_item_ids
equal to item_ids (the item id the model just read back from get_item_details)
and the environment accepts that write, so the customer's requested option is
never ordered and the final records never match the request. The feedback for
that case names the product's own AVAILABLE variants, closest first, with the
option values that differ, so the next action can carry a real replacement id.

No other action is touched: an allowed action is returned exactly as proposed,
and the layer returns the original message unchanged whenever it cannot prove
the failure or its per-episode budget is spent. Feedback is a single tool
message for the suppressed call, naming the recorded error, status, order items
or available variants so the next action has usable arguments.
"""
from json import dumps, loads

TARGET_LAYER = 'h2'

STATUS_RULES = {
    'cancel_pending_order': 'pending',
    'modify_pending_order_address': 'pending',
    'modify_pending_order_items': 'pending',
    'modify_pending_order_payment': 'pending',
    'return_delivered_order_items': 'delivered',
    'exchange_delivered_order_items': 'delivered',
}
ORDER_TOOLS = tuple(STATUS_RULES) + ('get_order_details',)
LOOKUP_TOOLS = ('find_user_id_by_email', 'find_user_id_by_name_zip')
ITEM_TOOLS = ('exchange_delivered_order_items', 'modify_pending_order_items')
HUMAN_WORDS = ('human', 'agent', 'representative', 'supervisor', 'manager',
               'someone', 'person', 'escalate', 'escalation')
MAX_VETOES = 4
MAX_ALTERNATIVES = 5
USE_PRODUCT_DETAILS = ("Use an item id that get_product_details returned for the same product; the"
                       " values printed in tool descriptions are format examples, not this"
                       " customer's items.")


def _text(value):
    return value if isinstance(value, str) else ''


def _short(text, limit=160):
    collapsed = ' '.join(_text(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + '...'


def _is_error(value):
    return _text(value).lstrip().lower().startswith('error')


def _json_object(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _calls(item):
    result = []
    calls = item.get('tool_calls')
    if not isinstance(calls, list):
        return result
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get('function')
        if not isinstance(function, dict):
            continue
        arguments = function.get('arguments')
        params = _json_object(arguments)
        if params is None:
            params = arguments if isinstance(arguments, dict) else {}
        result.append({'id': call.get('id'), 'name': _text(function.get('name')),
                       'params': params, 'canonical': dumps(params, sort_keys=True)})
    return result


def _tool_results(history):
    pending = {}
    results = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get('role')
        if role == 'assistant':
            for call in _calls(item):
                if isinstance(call['id'], str):
                    pending[call['id']] = call
            continue
        if role == 'tool':
            key = item.get('tool_call_id')
            if not isinstance(key, str):
                key = item.get('id')
            results.append((pending.get(key) if isinstance(key, str) else None,
                            _text(item.get('content'))))
    return results


def _tokens(text):
    words = []
    current = []
    for char in _text(text):
        if char.isalnum() or char == '_':
            current.append(char)
        elif current:
            words.append(''.join(current))
            current = []
    if current:
        words.append(''.join(current))
    return words


def _human_requested(history):
    for item in history:
        if not isinstance(item, dict) or item.get('role') != 'user':
            continue
        lowered = set(word.lower() for word in _tokens(item.get('content')))
        for word in HUMAN_WORDS:
            if word in lowered:
                return True
    return False


def _normalize(value):
    return ''.join(char for char in value.lower() if char.isalnum())


def _recover_order_id(text, orders):
    """Pure formatting recovery: the id differs only in punctuation/prefix."""
    target = _normalize(text)
    if not target:
        return None
    exact = [oid for oid in orders if _normalize(oid) == target]
    if len(exact) == 1:
        return exact[0]
    if len(target) >= 6 and target.isdigit():
        tail = [oid for oid in orders if _normalize(oid).endswith(target)]
        if len(tail) == 1:
            return tail[0]
    return None


def _account_orders(users, uid):
    if not uid:
        return []
    user = users.get(uid)
    if not isinstance(user, dict):
        return []
    listed = user.get('orders')
    if not isinstance(listed, list):
        return []
    return [oid for oid in listed if isinstance(oid, str)]


def _account_facts(users, orders, uid):
    parts = []
    for oid in _account_orders(users, uid):
        order = orders.get(oid)
        status = order.get('status') if isinstance(order, dict) else None
        parts.append('%s [%s]' % (oid, status if isinstance(status, str) else 'unknown'))
    if not parts:
        return ''
    return "The authenticated customer '%s' has orders: %s." % (uid, ', '.join(parts))


def _order_item_ids(order):
    result = []
    if not isinstance(order, dict):
        return result
    items = order.get('items')
    if not isinstance(items, list):
        return result
    for entry in items:
        if isinstance(entry, dict) and isinstance(entry.get('item_id'), str):
            result.append(entry['item_id'])
    return result


def _option_text(options):
    if not isinstance(options, dict):
        return ''
    parts = []
    for key in sorted(options):
        value = options[key]
        if isinstance(value, (str, int, float)):
            parts.append('%s=%s' % (key, value))
    return '; '.join(parts)


def _catalog(products):
    """item_id -> the variant record that proves what an item is."""
    index = {}
    if not isinstance(products, dict):
        return index
    for pid in sorted(products):
        product = products[pid]
        if not isinstance(product, dict):
            continue
        variants = product.get('variants')
        if not isinstance(variants, dict):
            continue
        name = product.get('name')
        for key in sorted(variants):
            variant = variants[key]
            if not isinstance(variant, dict):
                continue
            iid = variant.get('item_id')
            if not isinstance(iid, str) or not iid:
                iid = key if isinstance(key, str) else ''
            if not iid:
                continue
            options = variant.get('options')
            index[iid] = {
                'item_id': iid,
                'product_id': pid if isinstance(pid, str) else '',
                'product': name if isinstance(name, str) else '',
                'options': options if isinstance(options, dict) else {},
                'available': variant.get('available'),
                'price': variant.get('price'),
            }
    return index


def _alternatives(products, catalog, item_id, limit=MAX_ALTERNATIVES):
    """The same product's other AVAILABLE variants, closest option change first."""
    info = catalog.get(item_id)
    if not isinstance(info, dict):
        return ''
    product = products.get(info['product_id']) if isinstance(products, dict) else None
    variants = product.get('variants') if isinstance(product, dict) else None
    if not isinstance(variants, dict):
        return ''
    base = info['options']
    rows = []
    for key in sorted(variants):
        variant = variants[key]
        if not isinstance(variant, dict):
            continue
        iid = variant.get('item_id')
        if not isinstance(iid, str) or not iid:
            iid = key if isinstance(key, str) else ''
        if not iid or iid == item_id or variant.get('available') is not True:
            continue
        options = variant.get('options')
        if not isinstance(options, dict):
            continue
        changed = [k for k in sorted(options) if base.get(k) != options[k]]
        if not changed:
            continue
        rows.append((len(changed),
                     iid,
                     '; '.join('%s %s -> %s' % (k, base.get(k), options[k]) for k in changed),
                     variant.get('price')))
    rows.sort(key=lambda row: (row[0], row[1]))
    parts = []
    for _, iid, changed, price in rows[:limit]:
        price_text = (' $%s' % price) if isinstance(price, (int, float)) else ''
        parts.append('%s [%s]%s' % (iid, changed, price_text))
    if not parts:
        return ("No other available variant of product '%s' is on file, so no replacement id can"
                " satisfy 'same product, different option'." % info['product'])
    more = '' if len(rows) <= limit else ' (+%d more available variants)' % (len(rows) - limit)
    return ("Available variants of the same product '%s' that differ from item '%s' (%s), closest"
            " first: %s%s" % (info['product'], item_id, _option_text(base), '; '.join(parts), more))


def _item_problem(name, params, order, products, catalog):
    """A provable item-level defect in a proposed item write, or ''."""
    item_ids = params.get('item_ids')
    if not isinstance(item_ids, list) or not item_ids:
        return ''
    for iid in item_ids:
        if not isinstance(iid, str) or not iid:
            return ''
    order_items = _order_item_ids(order)
    if order_items:
        missing = [iid for iid in item_ids if iid not in order_items]
        if missing:
            return ("No action was executed. Item id(s) %s are not in order '%s', whose recorded"
                    " items are %s; the environment cannot act on a line the order does not have."
                    " Read the order again with get_order_details and use its own item ids and no"
                    " others." % (', '.join(missing), params.get('order_id'), ', '.join(order_items)))
    if name not in ITEM_TOOLS:
        return ''
    new_ids = params.get('new_item_ids')
    if not isinstance(new_ids, list):
        return ''
    if len(new_ids) != len(item_ids):
        return ("No action was executed. new_item_ids must hold one replacement id for every item"
                " id, in the same positions (%d items, %d replacements were sent), and the"
                " environment rejects a positional mismatch. Rebuild the call so each replacement"
                " lines up with the item it replaces, and confirm the pairs with the customer."
                % (len(item_ids), len(new_ids)))
    for index in range(len(item_ids)):
        old = item_ids[index]
        new = new_ids[index]
        if not isinstance(new, str) or not new:
            return ''
        alternatives = _alternatives(products, catalog, old)
        old_info = catalog.get(old)
        new_info = catalog.get(new)
        if new == old:
            return ("No action was executed. Item '%s' was offered as its own replacement, which"
                    " changes no option; the customer asked for a different variant of the same"
                    " product. %s Use one of those replacement ids (or another available id you"
                    " read back from get_product_details) and confirm the exact option with the"
                    " customer before writing."
                    % (old, alternatives or USE_PRODUCT_DETAILS))
        if not isinstance(old_info, dict) or not isinstance(new_info, dict):
            return ("No action was executed. Replacement item id '%s' is not an item in the current"
                    " catalog, so the environment returns an error instead of exchanging anything."
                    " %s" % (new, alternatives or USE_PRODUCT_DETAILS))
        if (old_info.get('product_id') and new_info.get('product_id')
                and new_info.get('product_id') != old_info.get('product_id')):
            return ("No action was executed. Replacement item '%s' belongs to product '%s', not to"
                    " product '%s' of the item it replaces, and a product type cannot change. %s"
                    % (new, new_info.get('product'), old_info.get('product'),
                       alternatives or USE_PRODUCT_DETAILS))
        if new_info.get('available') is False:
            return ("No action was executed. Replacement item '%s' is marked unavailable in the"
                    " current records, so the environment rejects it. %s"
                    % (new, alternatives or USE_PRODUCT_DETAILS))
    return ''


def _with_arguments(message, call_id, params):
    updated = dict(message)
    calls = []
    for call in message.get('tool_calls') or []:
        if isinstance(call, dict) and call.get('id') == call_id:
            call = dict(call)
            function = dict(call.get('function') or {})
            function['arguments'] = dumps(params)
            call['function'] = function
        calls.append(call)
    updated['tool_calls'] = calls
    return updated


class Harness:
    def bind_shared_state(self, state):
        self.state = state
        self.db = None

    def bind_environment_state(self, state):
        db = state.get('db') if isinstance(state, dict) else None
        self.db = db if isinstance(db, dict) else None

    def h2(self, message, history):
        try:
            return self._control(message, history)
        except Exception:
            return message

    def _control(self, message, history):
        if not isinstance(message, dict):
            return message
        calls = _calls(message)
        if len(calls) != 1:
            return message
        call = calls[0]
        name = call['name']
        call_id = call['id']
        if not name or not isinstance(call_id, str) or not call_id:
            return message
        db = self.db
        users = db.get('users') if isinstance(db, dict) else None
        orders = db.get('orders') if isinstance(db, dict) else None
        products = db.get('products') if isinstance(db, dict) else None
        if not isinstance(users, dict):
            users = {}
        if not isinstance(orders, dict):
            orders = {}
        if not isinstance(products, dict):
            products = {}
        catalog = _catalog(products)
        results = _tool_results(history)
        uid = self._session_customer(results, users)
        facts = _account_facts(users, orders, uid)
        vetoes = self.state.get('h2.vetoes')
        if not isinstance(vetoes, int):
            vetoes = 0

        # 1. Verbatim repeat of a call that already returned an error.
        for other, content in results:
            if (isinstance(other, dict) and other.get('name') == name
                    and other.get('canonical') == call['canonical']
                    and _is_error(content)):
                extra = ''
                for value in call['params'].values():
                    if isinstance(value, str) and value in users and name in LOOKUP_TOOLS:
                        extra = (" That value is a user id on file, not an email address;"
                                 " authenticate with the email on file, or ask the customer to"
                                 " confirm their name and zip.")
                        break
                return self._veto(message, call_id, 'repeat:' + name + ':' + call['canonical'], vetoes,
                                  "No action was executed. This exact call already returned '%s';"
                                  " repeating the same arguments cannot change that result.%s%s Change"
                                  " the arguments or take the next step."
                                  % (_short(content, 120), extra, (' ' + facts) if facts else ''))

        # 2. Transfer with no customer request for a human agent anywhere.
        if name == 'transfer_to_human_agents':
            if not _human_requested(history):
                return self._veto(message, call_id, 'transfer', vetoes,
                                  "No action was executed. transfer_to_human_agents is only allowed"
                                  " when the customer explicitly asks for a human agent, and no"
                                  " customer turn in this conversation asked for one. The recorded"
                                  " retail records already answer this request%s State plainly what"
                                  " the policy allows for each open request (including the exact"
                                  " refund or delivery rule) and finish the work with the retail"
                                  " tools; do not hand off."
                                  % (': ' + facts if facts else '.'))
            return message

        if name not in ORDER_TOOLS:
            return message

        # 3/4/5. Order-scoped calls: recover a malformed id, otherwise veto an id the
        # records do not contain, a status the tool cannot act on, or an item-level
        # write the catalog contradicts.
        oid = call['params'].get('order_id')
        if not isinstance(oid, str):
            return message
        effective = oid
        recovered = None
        if oid not in orders:
            recovered = _recover_order_id(oid, orders)
            if recovered:
                effective = recovered
        if effective not in orders:
            if uid:
                tail = (" Use an order id from the customer's own account (get_user_details):"
                        " %s" % facts)
            else:
                tail = (" Authenticate the customer first (find_user_id_by_email /"
                        " find_user_id_by_name_zip); no authenticated account is known yet.")
            return self._veto(message, call_id, 'unknown:' + name + ':' + _normalize(oid), vetoes,
                              "No action was executed. Order id '%s' is not in the current business"
                              " records, so this call could only return 'Error: Order not found';"
                              " the '#W0000000' value in a tool description is only a format"
                              " example.%s" % (oid, tail))
        order = orders.get(effective)
        status = order.get('status') if isinstance(order, dict) else None
        required = STATUS_RULES.get(name)
        if required and status != required:
            return self._veto(message, call_id, 'status:' + name + ':' + effective, vetoes,
                              "No action was executed. Order '%s' currently has status '%s' and"
                              " '%s' requires a %s order; the environment rejects it.%s"
                              % (effective, status if isinstance(status, str) else 'unknown',
                                 name, required, (' ' + facts) if facts else ''))
        problem = _item_problem(name, call['params'], order, products, catalog)
        if problem:
            return self._veto(message, call_id, 'item:' + name + ':' + call['canonical'], vetoes,
                              problem)
        if recovered:
            params = dict(call['params'])
            params['order_id'] = recovered
            return _with_arguments(message, call_id, params)
        return message

    def _veto(self, message, call_id, signature, vetoes, text):
        seen = self.state.get('h2.signatures')
        if not isinstance(seen, list):
            seen = []
        if vetoes >= MAX_VETOES or signature in seen:
            return message
        seen.append(signature)
        self.state['h2.signatures'] = seen
        self.state['h2.vetoes'] = vetoes + 1
        result = dict(message)
        result['_harness_control'] = {
            'action': 'suppress_tool',
            'messages': [{'role': 'tool', 'tool_call_id': call_id, 'content': text}],
        }
        return result

    def _session_customer(self, results, users):
        uid = ''
        for call, content in results:
            if _is_error(content):
                continue
            stripped = content.strip()
            name = call.get('name') if isinstance(call, dict) else ''
            if name in LOOKUP_TOOLS:
                if stripped in users:
                    uid = stripped
            elif name == 'get_user_details':
                parsed = _json_object(stripped)
                if isinstance(parsed, dict):
                    value = parsed.get('user_id')
                    if isinstance(value, str) and value in users:
                        uid = value
        return uid

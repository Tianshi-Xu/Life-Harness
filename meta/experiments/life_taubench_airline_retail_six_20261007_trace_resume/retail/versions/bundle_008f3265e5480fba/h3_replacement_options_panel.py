"""H3: record-grounded, model-only context for the retail solver.

The note this layer appends is a pure function of (public history, bound
agent-side business records) recomputed on every solver call, so it can never go
stale and needs no cross-turn state. It changes the model-visible view only:
the episode history, the recorded transcript and the user simulator never see it.

Two kinds of content are emitted:

* the account panel the previous revision established (authenticated customer,
  their own order ids and statuses, the error-specific recovery note), and
* a replacement-options panel that appears only when the latest customer turn
  asks to change an item the authenticated customer already owns. TRAIN evidence:
  recorded episodes call exchange_delivered_order_items /
  modify_pending_order_items with new_item_ids equal to item_ids - the id the
  model had just read back from get_item_details - and the environment accepts
  that no-op write, so the requested option is never ordered. The panel states
  the recorded option values of the customer's own item and, closest change
  first, the available variants of the SAME product with the option values that
  differ, so the replacement id can come from a real record instead of the id
  being replaced.

The three description additions state the tool semantics that were misread: a
real id comes from the customer's own account, modify_user_address does not
change an order's shipping address, and a replacement id must be a different
available variant of the same product read from get_product_details.
"""
from json import loads

TARGET_LAYER = 'h3'

LOOKUP_TOOLS = ('find_user_id_by_email', 'find_user_id_by_name_zip')
MARK = '[H3 record context] '
MARK_OPTIONS = '[H3 replacement options] '
INTENT_WORDS = ('exchange', 'exchanged', 'swap', 'swapped', 'replace', 'replaced',
                'replacement', 'switch', 'switched', 'different', 'another',
                'modify', 'modified', 'change', 'changed')
MAX_PRODUCTS = 2
MAX_VARIANTS = 5

DESCRIPTION_NOTES = {
    'get_user_details': (
        " Call this right after authenticating the customer: it returns that customer's own"
        " order ids, so never ask the customer for an order id or item id that this record"
        " already holds."
    ),
    'get_order_details': (
        " The order id must be one of the ids in the customer's own account"
        " (get_user_details); the value above is only a format example and a user id is"
        " never an order id."
    ),
    'get_product_details': (
        " Use this to read every variant of a product (item id, option values, availability,"
        " price); all exchange and modification replacement ids must be taken from this"
        " result."
    ),
    'modify_user_address': (
        " This changes only the user's default profile address. It never changes the shipping"
        " address of an existing order: a pending order's shipping address changes only"
        " through modify_pending_order_address on that order."
    ),
    'modify_pending_order_items': (
        " Each new item id must be an available variant item of the same product with a"
        " different option value than the item it replaces; never pass the current item id"
        " and never invent an id - take it from get_product_details."
    ),
    'exchange_delivered_order_items': (
        " Each new item id must be an available variant item of the same product with a"
        " different option value than the item it replaces; never pass the current item id"
        " and never invent an id - take it from get_product_details."
    ),
}

STATUS_RULE_TEXT = (
    'cancel and modify tools need a pending order; return and exchange tools need a'
    ' delivered order and can be used once per order'
)


def _text(value):
    return value if isinstance(value, str) else ''


def _short(text, limit=180):
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
        name = function.get('name')
        arguments = function.get('arguments')
        params = _json_object(arguments)
        if params is None:
            params = arguments if isinstance(arguments, dict) else {}
        result.append({'id': call.get('id'), 'name': _text(name), 'params': params})
    return result


def _tool_results(history):
    """Every tool result in history order, paired with the call that produced it."""
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


def _tokens(content):
    words = []
    current = []
    for char in _text(content):
        if char.isalnum() or char == '_':
            current.append(char)
        elif current:
            words.append(''.join(current))
            current = []
    if current:
        words.append(''.join(current))
    return words


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


def _order_facts(users, orders, uid):
    parts = []
    for oid in _account_orders(users, uid):
        order = orders.get(oid)
        status = order.get('status') if isinstance(order, dict) else None
        parts.append('%s [%s]' % (oid, status if isinstance(status, str) else 'unknown'))
    return ', '.join(parts)


def _option_text(options):
    if not isinstance(options, dict):
        return ''
    parts = []
    for key in sorted(options):
        value = options[key]
        if isinstance(value, (str, int, float)):
            parts.append('%s=%s' % (key, value))
    return '; '.join(parts)


def _last_customer_text(messages):
    text = ''
    for item in messages:
        if not isinstance(item, dict) or item.get('role') != 'user':
            continue
        content = item.get('content')
        if isinstance(content, str) and content.strip():
            text = content
    return text


class Harness:
    def bind_environment_state(self, state):
        db = state.get('db') if isinstance(state, dict) else None
        self.db = db if isinstance(db, dict) else None

    def h3(self, tools):
        updated = []
        for tool in tools:
            if isinstance(tool, dict):
                function = tool.get('function')
                if isinstance(function, dict):
                    note = DESCRIPTION_NOTES.get(function.get('name'))
                    if note:
                        tool = dict(tool)
                        function = dict(function)
                        function['description'] = _text(function.get('description')).rstrip() + note
                        tool['function'] = function
            updated.append(tool)
        return updated

    def h3_messages(self, messages):
        try:
            note = self._note(messages)
            if note:
                for item in messages:
                    if isinstance(item, dict) and item.get('role') == 'system':
                        item['content'] = _text(item.get('content')) + '\n\n' + note
                        break
        except Exception:
            return messages
        return messages

    def _note(self, messages):
        base = self._account_note(messages)
        options = self._options_note(messages)
        if options:
            return base + '\n\n' + options if base else options
        return base

    def _account_note(self, messages):
        db = self.db
        if not db:
            return ''
        users = db.get('users')
        orders = db.get('orders')
        if not isinstance(users, dict) or not isinstance(orders, dict):
            return ''
        results = _tool_results(messages)
        uid, _read = self._session_customer(results, users)
        claimed = uid if uid else self._claimed_customer(messages, users)
        facts = _order_facts(users, orders, claimed)
        if results and _is_error(results[-1][1]):
            note = self._error_note(results[-1], uid, claimed, users, orders, facts)
            if note:
                return note
        if uid:
            if facts:
                return (MARK + "The customer is authenticated as '%s'. Orders on file: %s."
                        " Read them with get_user_details('%s') and take every order id, item id and"
                        " payment id from these records or from get_order_details - never from the"
                        " customer's memory and never from the example printed in a tool description."
                        " Do not ask the customer for an id the records already hold, and do not"
                        " transfer to a human agent while these records can still answer the request."
                        % (uid, facts, uid))
            return (MARK + "The customer is authenticated as '%s'. Read their own record with"
                    " get_user_details('%s') before asking the customer for any order or item id,"
                    " and do not transfer to a human agent while that record can answer the request."
                    % (uid, uid))
        if claimed:
            user = users.get(claimed)
            email = _text(user.get('email')) if isinstance(user, dict) else ''
            zipcode = ''
            if isinstance(user, dict) and isinstance(user.get('address'), dict):
                zipcode = _text(user['address'].get('zip'))
            return (MARK + "The customer's own record is '%s' (email on file '%s', zip on file '%s')."
                    " Authentication is still required: ask the customer to confirm that email or"
                    " their name and zip, then call find_user_id_by_email / find_user_id_by_name_zip"
                    " and finish with get_user_details('%s') to read their orders%s."
                    % (claimed, email, zipcode, claimed,
                       (': ' + facts) if facts else ''))
        return ''

    def _options_note(self, messages):
        db = self.db
        if not db:
            return ''
        users = db.get('users')
        orders = db.get('orders')
        products = db.get('products')
        if (not isinstance(users, dict) or not isinstance(orders, dict)
                or not isinstance(products, dict)):
            return ''
        uid, _read = self._session_customer(_tool_results(messages), users)
        if not uid:
            return ''
        text = _last_customer_text(messages)
        if not text:
            return ''
        lowered = [word.lower() for word in _tokens(text)]
        if not any(word in INTENT_WORDS for word in lowered):
            return ''
        wanted = set(lowered)
        found = []
        for oid in _account_orders(users, uid):
            order = orders.get(oid)
            if not isinstance(order, dict):
                continue
            entries = order.get('items')
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                iid = entry.get('item_id')
                pid = entry.get('product_id')
                if not isinstance(iid, str) or not isinstance(pid, str):
                    continue
                if any(pair[0] == iid for pair in found):
                    continue
                product = products.get(pid)
                name = ''
                if isinstance(product, dict) and isinstance(product.get('name'), str):
                    name = product['name']
                words = [word.lower() for word in _tokens(name) if len(word) >= 3]
                if not words or not all(word in wanted for word in words):
                    continue
                found.append((iid, pid, name, entry))
            if len(found) >= MAX_PRODUCTS:
                break
        parts = []
        for iid, pid, name, entry in found[:MAX_PRODUCTS]:
            line = self._variant_lines(products, iid, pid, name, entry.get('options'))
            if line:
                parts.append(line)
        if not parts:
            return ''
        return (MARK_OPTIONS + "The customer asked to change an item they already own, so the"
                " replacement id must be a DIFFERENT, AVAILABLE variant of the SAME product,"
                " confirmed with the customer before the write. Recorded options only: "
                + ' | '.join(parts) +
                " Use one of those item ids in new_item_ids (never the current item id, never an"
                " unavailable one, never an id from outside this record).")

    def _variant_lines(self, products, item_id, product_id, name, options):
        product = products.get(product_id)
        variants = product.get('variants') if isinstance(product, dict) else None
        if not isinstance(variants, dict):
            return ''
        current = options if isinstance(options, dict) else {}
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
            opts = variant.get('options')
            if not isinstance(opts, dict):
                continue
            changed = [k for k in sorted(opts) if current.get(k) != opts[k]]
            if not changed:
                continue
            rows.append((len(changed), iid,
                         '; '.join('%s %s -> %s' % (k, current.get(k), opts[k]) for k in changed),
                         variant.get('price')))
        label = "%s (product %s): the customer's item %s [%s]" % (name, product_id, item_id,
                                                                 _option_text(current))
        if not rows:
            return label + " has no other available variant on file."
        rows.sort(key=lambda row: (row[0], row[1]))
        listed = []
        for _, iid, changed, price in rows[:MAX_VARIANTS]:
            price_text = (' $%s' % price) if isinstance(price, (int, float)) else ''
            listed.append('%s [%s]%s' % (iid, changed, price_text))
        more = '' if len(rows) <= MAX_VARIANTS else ' (+%d more)' % (len(rows) - MAX_VARIANTS)
        return (label + "; available replacements, closest option change first: " + '; '.join(listed)
                + more + '.')

    def _error_note(self, last, uid, claimed, users, orders, facts):
        call, content = last
        error = _short(content, 120)
        name = call.get('name') if isinstance(call, dict) else ''
        params = call.get('params') if isinstance(call, dict) and isinstance(call.get('params'), dict) else {}
        if name in LOOKUP_TOOLS:
            argument = ''
            for key in ('email', 'first_name', 'zip', 'last_name'):
                if isinstance(params.get(key), str):
                    argument += (' ' if argument else '') + params[key]
            if uid:
                return (MARK + "The identity lookup with '%s' returned '%s', but the customer is"
                        " already authenticated as '%s'; do not run further identity lookups. Read the"
                        " account with get_user_details('%s') and continue from it%s Do not transfer to"
                        " a human agent while these records can answer the request."
                        % (argument, error, uid, uid, ('. Orders on file: ' + facts) if facts else '.'))
            if claimed:
                user = users.get(claimed)
                email = _text(user.get('email')) if isinstance(user, dict) else ''
                return (MARK + "The identity lookup with '%s' returned '%s'. The customer's own record"
                        " is '%s' (email on file '%s'). Ask the customer to confirm that email or their"
                        " name and zip, complete authentication with find_user_id_by_email /"
                        " find_user_id_by_name_zip, then read get_user_details('%s'). Do not repeat an"
                        " identity lookup that already returned an error%s"
                        % (argument, error, claimed, email, claimed,
                           ('. Orders on file: ' + facts) if facts else '.'))
            return (MARK + "Authentication is unresolved ('%s'). Ask the customer for the email"
                    " address, or for the name and zip code on file, and retry"
                    " find_user_id_by_email / find_user_id_by_name_zip with those exact values instead"
                    " of a guessed one. Do not transfer to a human agent while the customer's own"
                    " record can still be found." % error)
        oid = params.get('order_id')
        if isinstance(oid, str) and oid not in orders:
            return (MARK + "Order id '%s' is not in the business records, so repeating it can only"
                    " return the same error.%s Use an order id taken from the customer's own account"
                    " (get_user_details) or the id the customer actually confirmed; the '#W0000000'"
                    " value in a tool description is only a format example."
                    % (oid, (' Records: ' + facts + '.') if facts else ''))
        if isinstance(oid, str) and oid in orders:
            order = orders.get(oid)
            status = order.get('status') if isinstance(order, dict) else None
            if isinstance(status, str):
                return (MARK + "Order '%s' currently has status '%s': %s. Take the customer's own"
                        " order ids and item ids from get_user_details / get_order_details instead of"
                        " repeating an action the records already reject."
                        % (oid, status, STATUS_RULE_TEXT))
        if name in ('exchange_delivered_order_items', 'modify_pending_order_items'):
            return (MARK + "The last call returned '%s'. A replacement id must be an available"
                    " variant of the same product with a different option value, read from"
                    " get_product_details, never the item id it replaces.%s"
                    % (error, (' Records: ' + facts + '.') if facts else ''))
        return (MARK + "The last call returned '%s'. Do not resend the same arguments; change them or"
                " take the next step.%s Take ids from the customer's own records (get_user_details /"
                " get_order_details) rather than from the customer's memory or a description example."
                % (error, (' Records: ' + facts + '.') if facts else ''))

    def _session_customer(self, results, users):
        uid = ''
        read = False
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
                        read = True
        return uid, read

    def _claimed_customer(self, messages, users):
        claimed = ''
        for item in messages:
            if not isinstance(item, dict) or item.get('role') != 'user':
                continue
            content = item.get('content')
            if not isinstance(content, str):
                continue
            for token in _tokens(content):
                if token in users:
                    claimed = token
        return claimed

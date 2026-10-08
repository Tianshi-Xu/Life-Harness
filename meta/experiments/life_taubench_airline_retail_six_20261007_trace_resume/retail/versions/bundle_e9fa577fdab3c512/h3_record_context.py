"""H3: record-grounded, model-only context for the retail solver.

The note this layer appends is a pure function of (public history, bound
agent-side business records) recomputed on every solver call, so it can never go
stale and needs no cross-turn state. It changes the model-visible view only:
the episode history, the recorded transcript and the user simulator never see it.

Evidence (TRAIN batch): after authenticating, the solver repeatedly asks the
customer for order/item ids that the account already holds and can copy the
format example '#W0000000' out of a description; both end in error loops or an
unusable handoff. The note names the customer's own order ids and statuses at the
moment the solver has to act, and the description additions state the two tool
semantics that were misread (id provenance and address scope).
"""
from json import loads

TARGET_LAYER = 'h3'

LOOKUP_TOOLS = ('find_user_id_by_email', 'find_user_id_by_name_zip')
MARK = '[H3 record context] '

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
    'modify_user_address': (
        " This changes only the user's default profile address. It never changes the shipping"
        " address of an existing order: a pending order's shipping address changes only"
        " through modify_pending_order_address on that order."
    ),
    'modify_pending_order_items': (
        " Each new item id must be an available variant item of the same product and must"
        " differ from the item id it replaces."
    ),
    'exchange_delivered_order_items': (
        " Each new item id must be an available variant item of the same product and must"
        " differ from the item id it replaces."
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
    for char in content:
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

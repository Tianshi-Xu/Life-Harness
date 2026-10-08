"""H2: bounded action control against the current business records.

Every rule here replaces an action whose failure the CURRENT records already
prove, and each one is per-episode bounded and deduplicated by signature:

* a verbatim repeat of a call that already came back with an error,
* an order id that is absent from the records (unchanged arguments could only
  return 'Error: Order not found'), after trying a pure format recovery,
* a return/exchange/cancel/modify aimed at an order whose recorded status
  forbids that tool.

No other action is touched: an allowed action is returned exactly as proposed,
and the layer returns the original message unchanged whenever it cannot prove
the failure or its budget is spent. Feedback is a single tool message for the
suppressed call, naming the recorded error or status plus the authenticated
customer's real order ids and statuses, so the next action has usable arguments.
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
MAX_VETOES = 3


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
        if not isinstance(users, dict) or not isinstance(orders, dict):
            users = users if isinstance(users, dict) else {}
            orders = orders if isinstance(orders, dict) else {}
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

        if name not in ORDER_TOOLS:
            return message

        # 2/3. Order-scoped calls: recover a malformed id, otherwise veto an id the
        # records do not contain or a status the tool cannot act on.
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
                tail = (' Use an order id from the customer\'s own account (get_user_details):'
                        ' %s' % facts)
            else:
                tail = (" Authenticate the customer first (find_user_id_by_email /"
                        " find_user_id_by_name_zip); no authenticated account is known yet.")
            return self._veto(message, call_id, 'unknown:' + name + ':' + _normalize(oid), vetoes,
                              "No action was executed. Order id '%s' is not in the current business"
                              " records, so this call could only return 'Error: Order not found'; the"
                              " '#W0000000' value in a tool description is only a format example.%s"
                              % (oid, tail))
        order = orders.get(effective)
        status = order.get('status') if isinstance(order, dict) else None
        required = STATUS_RULES.get(name)
        if required and status != required:
            return self._veto(message, call_id, 'status:' + name + ':' + effective, vetoes,
                              "No action was executed. Order '%s' currently has status '%s' and '%s'"
                              " requires a %s order; the environment rejects it.%s"
                              % (effective, status if isinstance(status, str) else 'unknown',
                                 name, required, (' ' + facts) if facts else ''))
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

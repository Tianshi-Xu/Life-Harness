"""H4: track the identified customer and surface account facts or the next check.

The monitor reads only public history (assistant actions, tool results) and the
read-only business DB binding. It never performs an action and never rewrites
history: its single hint per turn is model-visible guidance.
"""
from json import loads

TARGET_LAYER = 'h4'

NUDGE = (
    "No customer identified yet. Technical support must identify the customer first: ask for the "
    "phone number and pass it to get_customer_by_phone exactly as the customer states it, dashes "
    "included (for example 555-123-2002); a name lookup also needs the date of birth. The phone "
    "steps in the support manual are the customer's actions, not yours."
)

OPEN_BILLS = ('Overdue', 'Awaiting Payment', 'Disputed')


def _rows(db, name):
    value = db.get(name)
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _text(value):
    return value if isinstance(value, str) else ''


def _money(value):
    return f"{value:.2f}" if isinstance(value, (int, float)) else '0.00'


def _customer_in(value, depth):
    """Find the first customer_id inside a decoded public tool result."""
    if depth > 6:
        return None
    if isinstance(value, dict):
        direct = value.get('customer_id')
        if isinstance(direct, str) and direct:
            return direct
        for key in sorted(value):
            found = _customer_in(value[key], depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _customer_in(item, depth + 1)
            if found:
                return found
    return None


def _customer_from(messages):
    """The most recent customer_id visible in public tool results, if any."""
    if not isinstance(messages, list):
        return None
    found = None
    for item in messages:
        if not isinstance(item, dict) or item.get('role') != 'tool':
            continue
        content = item.get('content')
        if not isinstance(content, str) or 'customer_id' not in content:
            continue
        try:
            value = loads(content)
        except ValueError:
            continue
        name = _customer_in(value, 0)
        if name:
            found = name
    return found


def _requested_total(state):
    value = state.get('user_side_requests')
    if not isinstance(value, dict):
        return 0
    total = 0
    for amount in value.values():
        if isinstance(amount, int) and amount > 0:
            total += amount
    return total


class Harness:
    state = None
    env = None
    context = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def bind_environment_state(self, state):
        self.env = state

    def h4(self, messages, remaining):
        state = self.state if isinstance(self.state, dict) else None
        if state is None:
            return []
        found = _customer_from(messages)
        if found:
            state['customer_id'] = found
        customer = state.get('customer_id')
        if isinstance(customer, str) and customer:
            facts = self.account(customer)
            if facts and state.get('h4_facts') != facts:
                state['h4_facts'] = facts
                return [facts]
            return []
        requested = _requested_total(state)
        announced = state.get('h4_nudge_at')
        if requested >= 2 and (not isinstance(announced, int) or requested - announced >= 3):
            state['h4_nudge_at'] = requested
            return [NUDGE]
        return []

    def account(self, customer):
        db = self.env.get('db') if isinstance(self.env, dict) else None
        if not isinstance(db, dict):
            return None
        record = None
        for row in _rows(db, 'customers'):
            if row.get('customer_id') == customer:
                record = row
                break
        if not isinstance(record, dict):
            return None
        owned = record.get('line_ids')
        owned = owned if isinstance(owned, list) else []
        lines = []
        for row in _rows(db, 'lines'):
            if row.get('line_id') not in owned:
                continue
            label = _text(row.get('line_id')) + ' ' + _text(row.get('phone_number'))
            status = _text(row.get('status'))
            if status != 'Active':
                line = label + ' ' + status
                if _text(row.get('suspension_start_date')):
                    line += ' since ' + _text(row.get('suspension_start_date'))
                if _text(row.get('contract_end_date')):
                    line += ', contract ends ' + _text(row.get('contract_end_date'))
                lines.append(line)
            elif self.over_limit(row, db):
                lines.append(label + ' data over the plan limit')
        bills = []
        for row in _rows(db, 'bills'):
            if row.get('customer_id') != customer:
                continue
            status = _text(row.get('status'))
            if status in OPEN_BILLS:
                bills.append(_text(row.get('bill_id')) + ' ' + status + ' $'
                             + _money(row.get('total_due')) + ' due '
                             + _text(row.get('due_date')))
        head = ('Account facts for ' + customer + ' ' + _text(record.get('full_name')) + ' ('
                + _text(record.get('account_status')) + '): lines '
                + ('; '.join(lines) if lines else 'none suspended or over limit'))
        tail = ('. Open bills ' + '; '.join(bills)) if bills else '. No open bills'
        return (head + tail + '. Use your own account tools for changes; the phone steps are the '
                "customer's.")

    def over_limit(self, row, db):
        used = row.get('data_used_gb')
        if not isinstance(used, (int, float)):
            return False
        for plan in _rows(db, 'plans'):
            if plan.get('plan_id') == row.get('plan_id'):
                limit = plan.get('data_limit_gb')
                if isinstance(limit, (int, float)) and used > limit:
                    return True
        return False

"""H4: keep the identified customer's account in view and name the permitted step.

Reads only public history (user messages, assistant actions, public tool results)
and the detached read-only business-DB binding. It never performs an action, never
rewrites history, and emits at most one model-visible hint per changed signature.

The hint is derived from the public policy the solver already has: identify the
customer, map the number the customer gave to their line, and report that line's
status, contract end date, roaming flag, data usage against the plan limit and the
open bills for the customer, together with the account-side action each anomaly
permits. Reporting the state again after a change closes the progress loop: the
model sees that its write changed the account and what is still open.
"""
from json import loads
from re import IGNORECASE, compile, search

TARGET_LAYER = 'h4'

OPEN_BILLS = ('Overdue', 'Awaiting Payment', 'Disputed')

NUDGE = (
    "No customer identified yet. Technical support must identify the customer first: ask for the "
    "phone number and pass it to get_customer_by_phone exactly as the customer states it, dashes "
    "included (for example 555-123-2002); a name lookup also needs the date of birth. The phone "
    "steps in the support manual are the customer's actions, not yours."
)

TRAVEL = compile(
    r"\b(abroad|overseas|travel(?:l)?ing|travel(?:l)?er|on vacation|on holiday|international|"
    r"another country|outside (?:the|my) (?:home )?(?:network|country|coverage|area)|roaming|"
    r"in (?:france|spain|italy|germany|canada|mexico|japan|china|india|brazil|england|britain|"
    r"australia|ireland|portugal|netherlands|greece|turkey|egypt|thailand|korea))\b",
    IGNORECASE)

TODAY = compile(r"current time is\s+(\d{4}-\d{2}-\d{2})", IGNORECASE)
CUSTOMER_ID = compile(r'"customer_id"\s*:\s*"([^"]{1,64})"')


def _rows(db, name):
    value = db.get(name)
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _text(value):
    return value if isinstance(value, str) else ''


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _gb(value):
    return f"{value:g}" if _number(value) is not None else '?'


def _money(value):
    return f"{value:.2f}" if _number(value) is not None else '0.00'


def _digits(value):
    return ''.join(ch for ch in value if ch.isdigit()) if isinstance(value, str) else ''


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
        value = None
        try:
            value = loads(content)
        except ValueError:
            value = None
        name = _customer_in(value, 0) if value is not None else None
        if not name:
            hit = CUSTOMER_ID.search(content)
            name = hit.group(1) if hit else None
        if name:
            found = name
    return found


def _phone_from(messages):
    """The phone number the agent last passed to the public phone lookup."""
    if not isinstance(messages, list):
        return ''
    found = ''
    for item in messages:
        if not isinstance(item, dict) or item.get('role') != 'assistant':
            continue
        calls = item.get('tool_calls')
        if not isinstance(calls, list):
            continue
        for call in calls:
            function = call.get('function') if isinstance(call, dict) else None
            if not isinstance(function, dict) or function.get('name') != 'get_customer_by_phone':
                continue
            parsed = function.get('arguments')
            if isinstance(parsed, str):
                try:
                    parsed = loads(parsed)
                except ValueError:
                    parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get('phone_number'), str):
                found = parsed['phone_number']
    return found


def _travel(messages):
    """True when the customer said they are outside their home network."""
    if not isinstance(messages, list):
        return False
    for item in messages:
        if not isinstance(item, dict) or item.get('role') != 'user':
            continue
        content = item.get('content')
        if not isinstance(content, str) or content.lstrip()[:4] == '[H4]':
            continue
        if TRAVEL.search(content):
            return True
    return False


def _today(messages):
    """The episode date stated by the public policy, when it is visible."""
    if not isinstance(messages, list):
        return ''
    for item in messages:
        if not isinstance(item, dict) or item.get('role') != 'system':
            continue
        content = item.get('content')
        hit = TODAY.search(content) if isinstance(content, str) else None
        if hit:
            return hit.group(1)
    return ''


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
            report = self.report(customer, messages)
            if report and state.get('h4_report') != report:
                state['h4_report'] = report
                return [report]
            return []
        requested = _requested_total(state)
        announced = state.get('h4_nudge_at')
        if requested >= 2 and (not isinstance(announced, int) or requested - announced >= 3):
            state['h4_nudge_at'] = requested
            return [NUDGE]
        return []

    def report(self, customer, messages):
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
        today = _today(messages)
        phone = _phone_from(messages)
        owned = record.get('line_ids')
        owned = [item for item in owned if isinstance(item, str)] if isinstance(owned, list) else []
        lines = [row for row in _rows(db, 'lines') if row.get('line_id') in owned]
        primary = self.primary(record, lines, phone)
        pieces = [self.head(record, primary, phone), self.line_list(lines, primary, db, today),
                  self.bill_list(db, customer)]
        options = self.options(customer, primary, db, today, _travel(messages))
        if options:
            pieces.append('Account-side next step: ' + ' '.join(options))
        pieces.append("The phone steps in the support manual are the customer's actions, not tool "
                      "calls you can make.")
        return ' '.join(piece for piece in pieces if piece)

    def primary(self, record, lines, phone):
        wanted = _digits(phone)
        for row in lines:
            if wanted and _digits(row.get('phone_number')) == wanted:
                return row
        contact = _digits(record.get('phone_number'))
        for row in lines:
            if contact and _digits(row.get('phone_number')) == contact:
                return row
        return lines[0] if len(lines) == 1 else None

    def head(self, record, primary, phone):
        text = ('Customer ' + _text(record.get('customer_id')) + ' '
                + _text(record.get('full_name')) + ' ('
                + _text(record.get('account_status')) + ').')
        if primary is not None and phone:
            text += (' The number the customer gave, ' + phone + ', is line '
                     + _text(primary.get('line_id')) + '.')
        return text

    def line_list(self, lines, primary, db, today):
        if not lines:
            return 'No line records are attached to this customer.'
        entries = []
        for row in lines:
            if row is primary:
                entries.append(self.line_detail(row, db, today))
            else:
                entries.append(self.line_brief(row, db, today))
        return 'Lines: ' + '; '.join(entries) + '.'

    def line_detail(self, row, db, today):
        pieces = [_text(row.get('line_id')) + ' ' + _text(row.get('phone_number')) + ' '
                  + _text(row.get('status'))]
        if _text(row.get('suspension_start_date')):
            pieces.append('suspended since ' + _text(row.get('suspension_start_date')))
        end = _text(row.get('contract_end_date'))
        if end:
            pieces.append('contract ends ' + end
                          + (' (already past)' if today and end < today else ''))
        plan = self.plan(row, db)
        if plan is not None:
            pieces.append('plan ' + _text(plan.get('plan_id')))
            used = _number(row.get('data_used_gb'))
            limit = _number(plan.get('data_limit_gb'))
            if used is not None:
                data = 'data ' + _gb(used)
                if limit is not None:
                    data += ' of ' + _gb(limit) + ' GB' + (', over the plan limit' if used > limit else '')
                else:
                    data += ' GB'
                pieces.append(data)
        pieces.append('roaming ' + ('enabled' if row.get('roaming_enabled') else 'disabled'))
        return ', '.join(pieces)

    def line_brief(self, row, db, today):
        pieces = [_text(row.get('line_id')) + ' ' + _text(row.get('phone_number')) + ' '
                  + _text(row.get('status'))]
        if row.get('roaming_enabled') is False:
            pieces.append('roaming disabled')
        if self.over_limit(row, db):
            pieces.append('over the plan data limit')
        end = _text(row.get('contract_end_date'))
        if today and end and end < today:
            pieces.append('contract ended ' + end)
        return ', '.join(pieces)

    def bill_list(self, db, customer):
        bills = []
        for row in _rows(db, 'bills'):
            if row.get('customer_id') != customer:
                continue
            status = _text(row.get('status'))
            if status in OPEN_BILLS:
                bills.append(_text(row.get('bill_id')) + ' ' + status + ' $'
                             + _money(row.get('total_due')) + ' due ' + _text(row.get('due_date')))
        if bills:
            return 'Open bills: ' + '; '.join(bills) + '.'
        return 'No open bills for this customer.'

    def options(self, customer, primary, db, today, travel):
        overdue = []
        awaiting = []
        for row in _rows(db, 'bills'):
            if row.get('customer_id') != customer:
                continue
            status = _text(row.get('status'))
            if status == 'Overdue':
                overdue.append(row)
            elif status == 'Awaiting Payment':
                awaiting.append(row)
        out = []
        if overdue:
            bill = _text(overdue[0].get('bill_id'))
            out.append('bill ' + bill + ' is Overdue, so send_payment_request(' + customer + ', '
                       + bill + ') is permitted; the customer then pays it from their payment '
                       'requests and you re-check that the status is PAID.')
        elif awaiting:
            out.append('a bill is Awaiting Payment: the customer still has to pay it, and you '
                       're-check that the status is PAID before confirming anything.')
        if primary is not None:
            line = _text(primary.get('line_id'))
            status = _text(primary.get('status'))
            end = _text(primary.get('contract_end_date'))
            if status == 'Suspended':
                if overdue:
                    out.append('line ' + line + ' stays Suspended until every overdue bill is paid.')
                elif today and end and end < today:
                    out.append('the contract for line ' + line + ' has ended, so you cannot lift '
                               'this suspension; escalate the case to a human agent.')
                else:
                    out.append('resume_line(' + customer + ', ' + line + ') is permitted for the '
                               'suspended line, and the customer must reboot the device afterwards.')
            elif status == 'Active':
                if self.over_limit(primary, db):
                    out.append('line ' + line + ' is over its plan data limit, so the customer can '
                               'refuel up to 2 GB (refuel_data) or change plan.')
                if primary.get('roaming_enabled') is False:
                    if travel:
                        out.append('the customer is outside their home network and line ' + line
                                   + ' is not roaming enabled, so enable_roaming(' + customer + ', '
                                   + line + ') is offered at no cost.')
                    else:
                        out.append('line ' + line + ' is not roaming enabled; that can be enabled '
                                   'at no cost only while the customer is outside their home '
                                   'network.')
        return out[:3]

    def plan(self, row, db):
        for plan in _rows(db, 'plans'):
            if plan.get('plan_id') == row.get('plan_id'):
                return plan
        return None

    def over_limit(self, row, db):
        used = _number(row.get('data_used_gb'))
        if used is None:
            return False
        plan = self.plan(row, db)
        limit = _number(plan.get('data_limit_gb')) if plan is not None else None
        return limit is not None and used > limit

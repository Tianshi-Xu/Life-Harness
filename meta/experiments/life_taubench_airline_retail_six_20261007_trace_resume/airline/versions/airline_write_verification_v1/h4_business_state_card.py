"""H4: episode-local business-state card and next-step guidance (airline).

Reads the bound read-only business DB and the public history, then emits ONE short
harness message. The card carries the facts the solver cannot reliably recompute
from the transcript: membership level, booked cabin, legs with their live status,
insurance flag, payments so far, booked bags, the policy's free-bag allowance, and
whether the cancellation conditions hold. The guidance line picks up the shared
state written by the H2 gate (a write waiting for the user's explicit confirmation,
or a write that was refused before execution).

Evidence from TRAIN: episodes answer membership/bag questions from the user's own
claim instead of the records (a silver user was told a "gold" allowance), and the
solver continues to retry or restate actions the policy forbids. The card is
informational; it never executes or overrides an action.
"""
from datetime import datetime, timedelta
from json import loads
from re import IGNORECASE, compile

TARGET_LAYER = 'h4'

CURRENT_TIME = datetime(2024, 5, 15, 15, 0, 0)
HARNESS_PREFIXES = ('[h2]', '[h3]', '[h4]', '[h5]')
FLOWN_STATES = ('landed', 'flying')
BAG_ALLOWANCE = {
    'regular': {'basic_economy': 0, 'economy': 1, 'business': 2},
    'silver': {'basic_economy': 1, 'economy': 2, 'business': 3},
    'gold': {'basic_economy': 2, 'economy': 3, 'business': 4},
}
RID_TOKEN = compile(r"\b[A-Z0-9]{6}\b")
CONFIRM_START = compile(
    r"^\s*(?:(?:yes|yeah|yep|yup|absolutely|exactly|correct|confirmed)\b"
    r"|(?:go ahead|please do|proceed|do it|i confirm|i agree|"
    r"that(?:'s| is) (?:right|correct|fine))\b"
    r"|(?:ok(?:ay)?|sure|perfect|great|alright)\s*[,.!\-]?\s*"
    r"(?:please|go ahead|proceed|do it|yes|confirm)\b)", IGNORECASE)
CONFIRM_ANY = compile(
    r"\b(?:yes,?\s*please|please (?:do|proceed|go ahead)(?!\s+not)|go ahead(?!\s+not)|"
    r"you (?:can|may) proceed|i confirm|i agree|please proceed|"
    r"would like you to proceed)\b", IGNORECASE)
ACTION_WORDS = compile(
    r"\b(?:cancel|change|modify|update|upgrade|book|refund|bag|baggage|baggages|insurance|"
    r"allowance|reschedul\w*|move|transfer)\b", IGNORECASE)


def _business_db(env):
    if not isinstance(env, dict):
        return None
    db = env.get('db')
    return db if isinstance(db, dict) else None


def _is_harness_text(content):
    if not isinstance(content, str):
        return False
    lowered = content.lstrip().lower()
    for prefix in HARNESS_PREFIXES:
        if lowered.startswith(prefix):
            return True
    return False


def _public_texts(messages):
    texts = []
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, dict) or item.get('role') != 'user':
                continue
            content = item.get('content')
            if isinstance(content, str) and not _is_harness_text(content):
                texts.append(content)
    return texts


def _confirmed(text):
    if not isinstance(text, str) or not text.strip():
        return False
    if CONFIRM_START.search(text):
        return True
    return bool(CONFIRM_ANY.search(text))


def _reservation(db, rid):
    if not isinstance(db, dict) or not isinstance(rid, str):
        return None
    reservations = db.get('reservations')
    if not isinstance(reservations, dict):
        return None
    record = reservations.get(rid)
    return record if isinstance(record, dict) else None


def _reservation_in_play(db, messages):
    if not isinstance(messages, list):
        return None
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        content = item.get('content')
        if not isinstance(content, str):
            continue
        if item.get('role') == 'tool' and content.lstrip().startswith('{'):
            try:
                payload = loads(content)
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                record = _reservation(db, payload.get('reservation_id'))
                if record is not None:
                    return record
        for token in RID_TOKEN.findall(content):
            record = _reservation(db, token)
            if record is not None:
                return record
    return None


def _membership(db, user_id):
    if not isinstance(db, dict) or not isinstance(user_id, str):
        return None
    users = db.get('users')
    if not isinstance(users, dict):
        return None
    record = users.get(user_id)
    if not isinstance(record, dict):
        return None
    membership = record.get('membership')
    return membership if isinstance(membership, str) else None


def _leg_status(db, flight_number, date):
    if not isinstance(db, dict) or not isinstance(flight_number, str):
        return None
    flights = db.get('flights')
    if not isinstance(flights, dict):
        return None
    record = flights.get(flight_number)
    if not isinstance(record, dict):
        return None
    dates = record.get('dates')
    if not isinstance(dates, dict):
        return None
    day = dates.get(date)
    if not isinstance(day, dict):
        return None
    status = day.get('status')
    return status if isinstance(status, str) else None


def _created_delta(created):
    if not isinstance(created, str):
        return None
    try:
        stamp = datetime.strptime(created[:19], '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        return None
    delta = CURRENT_TIME - stamp
    return delta if delta >= timedelta(0) else None


def _legs_text(db, res):
    parts = []
    flights = res.get('flights')
    if isinstance(flights, list):
        for leg in flights:
            if not isinstance(leg, dict):
                continue
            status = _leg_status(db, leg.get('flight_number'), leg.get('date'))
            parts.append('%s %s-%s %s%s' % (leg.get('flight_number'), leg.get('origin'),
                                            leg.get('destination'), leg.get('date'),
                                            ' [%s]' % status if status else ''))
    return ', '.join(parts) if parts else 'none'


def _passenger_count(res):
    passengers = res.get('passengers')
    return len(passengers) if isinstance(passengers, list) else 0


def _paid_total(res):
    total = 0
    history = res.get('payment_history')
    if isinstance(history, list):
        for entry in history:
            if isinstance(entry, dict):
                amount = entry.get('amount')
                if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                    total += amount
    return int(total)


def _allowance(membership, cabin):
    row = BAG_ALLOWANCE.get(membership) if isinstance(membership, str) else None
    if not isinstance(row, dict):
        return None
    value = row.get(str(cabin))
    return value if isinstance(value, int) else None


def _cancel_note(db, res):
    flown = False
    cancelled = False
    flights = res.get('flights')
    if isinstance(flights, list):
        for leg in flights:
            if not isinstance(leg, dict):
                continue
            status = _leg_status(db, leg.get('flight_number'), leg.get('date'))
            if status in FLOWN_STATES:
                flown = True
            elif status == 'cancelled':
                cancelled = True
            elif (status is None and isinstance(leg.get('date'), str)
                  and leg.get('date') < '2024-05-15'):
                flown = True
    if flown:
        return ('not possible here: part of the itinerary has already flown, a human agent is '
                'needed.')
    if cancelled:
        return 'permitted: the airline cancelled a flight in this reservation.'
    if str(res.get('cabin')).lower() == 'business':
        return 'permitted: business reservation.'
    delta = _created_delta(res.get('created_at'))
    if delta is not None and delta <= timedelta(hours=24):
        return 'permitted: booked within the last 24 hours.'
    if str(res.get('insurance')).lower() == 'yes':
        return ('permitted only with travel insurance plus a health or weather reason; otherwise '
                'deny.')
    return ('not permitted: booked more than 24 hours ago, cabin not business, no travel insurance, '
            'no flight cancelled by the airline.')


def _card(db, res):
    rid = res.get('reservation_id')
    user_id = res.get('user_id')
    membership = _membership(db, user_id)
    cabin = res.get('cabin')
    count = _passenger_count(res)
    created = res.get('created_at')
    delta = _created_delta(created)
    age = ('%s days before 2024-05-15 15:00' % delta.days) if delta is not None else 'unknown'
    text = ('Business DB facts (read-only snapshot) for reservation %s: user %s, membership %s; '
            'cabin %s; flights %s; %s passenger(s); insurance "%s"; paid $%s; booked bags %s '
            '(%s non-free); created %s (%s).'
            % (rid, user_id, membership, cabin, _legs_text(db, res), count,
               res.get('insurance'), _paid_total(res), res.get('total_baggages'),
               res.get('nonfree_baggages'), str(created)[:10], age))
    per_passenger = _allowance(membership, cabin)
    if per_passenger is not None and count:
        text += (' Free-bag allowance for %s/%s: %s per passenger x %s = %s free bags; extra bags '
                 '$50 each.' % (membership, cabin, per_passenger, count, per_passenger * count))
    text += ' Cancellation: ' + _cancel_note(db, res)
    return text


def _guidance(state, messages, res):
    if not isinstance(state, dict):
        return ''
    pending = state.get('h2_pending')
    blocked = state.get('h2_block')
    texts = _public_texts(messages)
    latest = texts[-1] if texts else ''
    if isinstance(pending, dict) and pending.get('tool'):
        if _confirmed(latest):
            return ('Next action: the user just confirmed, so perform the pending %s call for %s '
                    'now, exactly as it was listed.' % (pending.get('tool'), pending.get('key')))
        return ('Next action: a %s for %s is waiting for the user to answer with an explicit yes; '
                'do not execute it and do not report it as done before that.'
                % (pending.get('tool'), pending.get('key')))
    if isinstance(blocked, dict) and blocked.get('tool'):
        return ('Next action: the earlier %s for %s was replaced before execution because of policy '
                '(%s); do not repeat it - state the policy reason and offer a human transfer if the '
                'user insists.' % (blocked.get('tool'), blocked.get('key'), blocked.get('reason')))
    if res is not None and ACTION_WORDS.search(latest):
        return ('Next action: check the DB facts above against what the user claims, list the exact '
                'action details you would apply, and ask for an explicit yes before any tool call '
                'that updates the booking database.')
    return ''


class Harness:
    state = None
    context = None
    env = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def bind_environment_state(self, state):
        self.env = state

    def h4(self, messages, remaining):
        try:
            return self._compose(messages)
        except Exception:
            return []

    def _compose(self, messages):
        db = _business_db(self.env)
        parts = []
        record = _reservation_in_play(db, messages)
        if record is not None and db is not None:
            parts.append(_card(db, record))
        guidance = _guidance(self.state, messages, record)
        if guidance:
            parts.append(guidance)
        return [' '.join(parts)] if parts else []

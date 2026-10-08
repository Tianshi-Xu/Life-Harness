"""H4: episode-local reservation identity, business-state card and next step.

TRAIN evidence (public traces only) shows two harness-side divergences that the
incumbent card created:

* The card bound the episode to a reservation id that was the first six-character
  token anywhere in recent text.  The caller's own reservation list is exactly
  such a payload, so `get_user_details` made the harness present the *first* of
  seven reservations as "the" booking; the solver then quoted those facts, asked
  the caller to confirm that booking, and in one trace executed a cancellation of
  it after the caller had confirmed a different reservation.
* The standing line "list the exact action details and ask for an explicit yes
  before any tool call that updates the booking database" fired on every turn
  whose user text contained an action word.  Even after the caller answered yes,
  the solver re-sent the identical request (or an identical summary) turn after
  turn until the caller stopped, and the requested booking-database change never
  happened.

This revision therefore (a) anchors the reservation only on evidence that really
identifies one - an id the caller typed, a reservation-scoped tool call argument,
or a reservation-scoped tool result - and emits a compact index of the caller's
reservations instead of guessing when no such evidence exists, and (b) replaces
the standing confirmation instruction with a state-derived next step: carry out a
pending action the caller already confirmed for that exact reservation, void it
when the caller answered about another reservation, and never send the same
request twice.  Confirmation enforcement stays in H2, which replaces an
unconfirmed write before it executes.

Shared state: H2 owns `h2_pending` / `h2_block`; this layer reads them and
consumes `h2_pending` when the caller's newest message points at a different
reservation (recorded as `h4_void`).  State is episode-local and reset by the
runtime.
"""
from datetime import datetime, timedelta
from json import loads
from re import IGNORECASE, compile

TARGET_LAYER = 'h4'

CURRENT_TIME = datetime(2024, 5, 15, 15, 0, 0)
HARNESS_PREFIXES = ('[h2]', '[h3]', '[h4]', '[h5]')
FLOWN_STATES = ('landed', 'flying')
WORD_BUDGET = 100
INDEX_LIMIT = 8
BAG_ALLOWANCE = {
    'regular': {'basic_economy': 0, 'economy': 1, 'business': 2},
    'silver': {'basic_economy': 1, 'economy': 2, 'business': 3},
    'gold': {'basic_economy': 2, 'economy': 3, 'business': 4},
}
RID_TOKEN = compile(r"\b[A-Za-z0-9]{6}\b")
UID_TOKEN = compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,3}\b")
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
    r"\b(?:cancel|change|modify|update|upgrade|bag|baggage|baggages|insurance|allowance|"
    r"reschedul\w*|move|refund|transfer|switch)\b", IGNORECASE)
LIST_ASK = compile(
    r"\b(?:all (?:my|of my|the)|every(?: one)?|other (?:reservations?|flights?|bookings?)|"
    r"more (?:reservations?|flights?|bookings?)|list (?:of )?(?:my|the|all)|"
    r"show me (?:all|my|the)|which reservations?|how many reservations?)\b", IGNORECASE)
CONFIRM_ASK = compile(
    r"\b(?:confirm|confirmation|proceed|would you like me to|shall i|do you want me to|"
    r"reply|yes/no|yes or no)\b", IGNORECASE)
BOOK_ASK = compile(
    r"\b(?:book|booking|reserve a|reserve me|new reservation|new flights?)\b", IGNORECASE)


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


def _tool_calls(message):
    result = []
    calls = message.get('tool_calls')
    if not isinstance(calls, list):
        return result
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get('function')
        if not isinstance(function, dict):
            continue
        name = function.get('name')
        if not isinstance(name, str):
            continue
        arguments = function.get('arguments')
        if isinstance(arguments, str):
            try:
                arguments = loads(arguments)
            except ValueError:
                arguments = None
        result.append((name, arguments if isinstance(arguments, dict) else {}))
    return result


def _user_texts(messages):
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


def _user_record(db, uid):
    if not isinstance(db, dict) or not isinstance(uid, str):
        return None
    users = db.get('users')
    if not isinstance(users, dict):
        return None
    record = users.get(uid)
    return record if isinstance(record, dict) else None


def _named_reservations(db, text):
    """Every reservation id the text mentions that exists in the records."""
    found = []
    if not isinstance(text, str) or not text:
        return found
    for token in RID_TOKEN.findall(text):
        rid = token if _reservation(db, token) is not None else None
        if rid is None:
            upper = token.upper()
            if upper != token and _reservation(db, upper) is not None:
                rid = upper
        if rid is not None and rid not in found:
            found.append(rid)
    return found


def _first_rid(db, text):
    found = _named_reservations(db, text)
    return found[0] if found else None


def _first_uid(db, text):
    if not isinstance(text, str) or not text:
        return None
    for token in UID_TOKEN.findall(text):
        if _user_record(db, token) is not None:
            return token
    return None


def _payload(content):
    if not isinstance(content, str) or not content.lstrip().startswith('{'):
        return None
    try:
        value = loads(content)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _caller_id(db, messages):
    """Caller identity only from profile evidence: a typed id or a user lookup."""
    if not isinstance(messages, list):
        return None
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        role = item.get('role')
        content = item.get('content')
        if role == 'user':
            if isinstance(content, str) and not _is_harness_text(content):
                uid = _first_uid(db, content)
                if uid:
                    return uid
        elif role == 'tool':
            payload = _payload(content)
            if isinstance(payload, dict) and isinstance(payload.get('reservations'), list):
                uid = payload.get('user_id')
                if _user_record(db, uid) is not None:
                    return uid
        elif role == 'assistant':
            for name, arguments in _tool_calls(item):
                if name != 'get_user_details':
                    continue
                uid = arguments.get('user_id')
                if _user_record(db, uid) is not None:
                    return uid
    return None


def _subject(db, messages, caller):
    """Most recent reservation-scoped evidence: caller text, call argument, result."""
    if not isinstance(messages, list):
        return None
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        role = item.get('role')
        content = item.get('content')
        rid = None
        if role == 'user':
            if isinstance(content, str) and not _is_harness_text(content):
                rid = _first_rid(db, content)
        elif role == 'tool':
            payload = _payload(content)
            if isinstance(payload, dict):
                value = payload.get('reservation_id')
                if isinstance(value, str) and _reservation(db, value) is not None:
                    rid = value
        elif role == 'assistant':
            for _name, arguments in _tool_calls(item):
                value = arguments.get('reservation_id')
                if isinstance(value, str) and _reservation(db, value) is not None:
                    rid = value
                    break
        if rid is None:
            continue
        record = _reservation(db, rid)
        owner = record.get('user_id') if isinstance(record, dict) else None
        if caller is not None and isinstance(owner, str) and owner != caller:
            return None
        return rid
    return None


def _membership(db, uid):
    record = _user_record(db, uid)
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


def _short_date(value):
    return value[5:] if isinstance(value, str) and len(value) >= 10 else str(value)


def _legs_text(db, res):
    parts = []
    flights = res.get('flights')
    if isinstance(flights, list):
        for leg in flights[:2]:
            if not isinstance(leg, dict):
                continue
            status = _leg_status(db, leg.get('flight_number'), leg.get('date'))
            parts.append('%s %s-%s %s%s' % (leg.get('flight_number'), leg.get('origin'),
                                            leg.get('destination'), _short_date(leg.get('date')),
                                            ' [%s]' % status if status else ''))
        if len(flights) > 2:
            parts.append('+%s legs' % (len(flights) - 2))
    return ', '.join(parts) if parts else 'none'


def _route(res):
    flights = res.get('flights')
    if isinstance(flights, list) and flights:
        first = flights[0] if isinstance(flights[0], dict) else {}
        last = flights[-1] if isinstance(flights[-1], dict) else {}
        origin = first.get('origin')
        destination = last.get('destination')
        if isinstance(origin, str) and isinstance(destination, str):
            return '%s-%s' % (origin, destination)
    return 'route?'


def _dates(res):
    flights = res.get('flights')
    days = []
    if isinstance(flights, list):
        for leg in flights:
            if isinstance(leg, dict) and isinstance(leg.get('date'), str):
                days.append(_short_date(leg.get('date')))
    if not days:
        return 'date?'
    if days[0] == days[-1]:
        return days[0]
    return '%s/%s' % (days[0], days[-1])


def _passenger_count(res):
    passengers = res.get('passengers')
    return len(passengers) if isinstance(passengers, list) else 0


def _payments(res):
    total = 0
    negative = False
    history = res.get('payment_history')
    if isinstance(history, list):
        for entry in history:
            if not isinstance(entry, dict):
                continue
            amount = entry.get('amount')
            if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                total += amount
                if amount < 0:
                    negative = True
    return int(total), negative


def _refunded(res):
    total, negative = _payments(res)
    return negative and total <= 0


def _allowance(membership, cabin):
    row = BAG_ALLOWANCE.get(membership) if isinstance(membership, str) else None
    if not isinstance(row, dict):
        return None
    value = row.get(str(cabin))
    return value if isinstance(value, int) else None


def _cancel_note(db, res):
    if _refunded(res):
        return 'not needed (full refund already recorded)'
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
        return 'not possible here (part flown; human agent)'
    if cancelled:
        return 'allowed (airline cancelled a leg)'
    if str(res.get('cabin')).lower() == 'business':
        return 'allowed (business cabin)'
    delta = _created_delta(res.get('created_at'))
    if delta is not None and delta <= timedelta(hours=24):
        return 'allowed (booked within 24h)'
    if str(res.get('insurance')).lower() == 'yes':
        return 'allowed only with insurance plus a health or weather reason'
    return 'not allowed (booked >24h, not business, no insurance, no airline cancellation)'


def _card(db, res):
    rid = res.get('reservation_id')
    uid = res.get('user_id')
    membership = _membership(db, uid)
    count = _passenger_count(res)
    parts = ['DB facts %s: user %s (%s); cabin %s; %s; %s pax; insurance %s; bags %s (%s non-free);'
             % (rid, uid, membership, res.get('cabin'), _legs_text(db, res), count,
                res.get('insurance'), res.get('total_baggages'), res.get('nonfree_baggages'))]
    if not _refunded(res):
        parts.append('paid $%s;' % _payments(res)[0])
    per = _allowance(membership, res.get('cabin'))
    if per is not None and count:
        parts.append('free bags %s (%s/pax, extra $50 each);' % (per * count, per))
    parts.append('created %s;' % str(res.get('created_at'))[:10])
    parts.append('cancel: %s.' % _cancel_note(db, res))
    return ' '.join(parts)


def _index_entries(db, uid):
    record = _user_record(db, uid)
    entries = []
    rids = record.get('reservations') if isinstance(record, dict) else None
    if not isinstance(rids, list):
        return entries
    for rid in rids:
        res = _reservation(db, rid)
        if res is None:
            continue
        suffix = ' cancelled' if _refunded(res) else ''
        entries.append('%s %s %s %s%s' % (rid, _route(res), _dates(res), res.get('cabin'), suffix))
    return entries


def _index_text(uid, entries, limit, tail):
    used = entries[:limit]
    if not used:
        return ''
    text = 'Caller %s reservations: %s.' % (uid, '; '.join(used))
    if tail:
        text += ' ' + tail
    return text


def _awaiting_execution(messages):
    """The solver asked for confirmation last and has not acted since."""
    asked = False
    if not isinstance(messages, list):
        return False
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get('role') != 'assistant':
            continue
        content = item.get('content')
        if isinstance(content, str) and content.strip() and not _is_harness_text(content):
            asked = bool(CONFIRM_ASK.search(content))
            break
        if item.get('tool_calls'):
            return False
    return asked


def _guidance(db, state, messages, latest, rid, uid):
    pending = state.get('h2_pending') if isinstance(state, dict) else None
    blocked = state.get('h2_block') if isinstance(state, dict) else None
    confirmed = _confirmed(latest)
    named = _named_reservations(db, latest)
    if isinstance(pending, dict) and pending.get('tool'):
        key = pending.get('key')
        if named and key not in named:
            if isinstance(state, dict):
                state['h2_pending'] = None
                state['h4_void'] = {'tool': pending.get('tool'), 'key': key, 'to': named[0]}
            return ('Next action: the caller\'s latest message is about %s, not %s, so the prepared '
                    '%s on %s is void - serve %s instead.'
                    % (named[0], key, pending.get('tool'), key, named[0]))
        if confirmed:
            return ('Next action: the caller has already given the explicit yes for %s %s, so make '
                    'exactly that call now and do not send the request again.'
                    % (pending.get('tool'), key))
        return ('Next action: %s %s is waiting for the caller\'s explicit yes; do not execute it '
                'yet, and answer whatever else the caller asked instead of repeating the request.'
                % (pending.get('tool'), key))
    if isinstance(blocked, dict) and blocked.get('tool'):
        return ('Next action: the %s on %s was replaced before execution (policy: %s); do not '
                'repeat it - state that reason once.'
                % (blocked.get('tool'), blocked.get('key'), blocked.get('reason')))
    if _awaiting_execution(messages) and confirmed and rid is not None:
        return ('Next action: the caller already answered yes to the details you listed and nothing '
                'has been done yet, so carry out that change now instead of sending the same '
                'request again, or state plainly which policy condition blocks it.')
    if LIST_ASK.search(latest) and uid is not None:
        return ('Next action: the caller is asking about all of their reservations - answer from '
                'the list above (id, route, dates, cabin) instead of one reservation only.')
    if rid is not None and ACTION_WORDS.search(latest):
        return ('Next action: state the one change to %s with its exact details once and carry it '
                'out as soon as the caller agrees; never send the same request twice.' % rid)
    if rid is None and uid is not None and ACTION_WORDS.search(latest):
        return ('Next action: which reservation is in play is not established - match the caller\'s '
                'route and dates against the list above, or ask which one, before quoting facts or '
                'acting.')
    if rid is None and uid is None and ACTION_WORDS.search(latest):
        return ('Next action: no reservation is identified yet - get the caller\'s user id or '
                'reservation id before quoting facts or acting.')
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
        if db is None:
            return []
        texts = _user_texts(messages)
        latest = texts[-1] if texts else ''
        uid = _caller_id(db, messages)
        rid = _subject(db, messages, uid)
        card = ''
        if rid is not None:
            record = _reservation(db, rid)
            if record is not None:
                card = _card(db, record)
        guide = _guidance(db, self.state, messages, latest, rid, uid)
        entries = _index_entries(db, uid) if uid is not None else []
        if rid is not None and not LIST_ASK.search(latest):
            entries = []
        if entries and rid is None and BOOK_ASK.search(latest):
            entries = []
        if entries and rid is None:
            tail = ('Which reservation is in play is not established: match the caller route and '
                    'dates to this list, or ask which one.')
        elif entries:
            tail = 'The caller asked about all of their reservations: answer from this list.'
        else:
            tail = ''
        limit = INDEX_LIMIT
        index = _index_text(uid, entries, limit, tail) if entries else ''
        while (limit > 0 and entries
               and len(' '.join([x for x in (card, index, guide) if x]).split()) > WORD_BUDGET):
            limit -= 1
            index = _index_text(uid, entries, limit, tail)
        parts = [part for part in (card, index, guide) if part]
        text = ' '.join(parts)
        if len(text.split()) > WORD_BUDGET:
            text = ' '.join(text.split()[:WORD_BUDGET])
        return [text] if text else []

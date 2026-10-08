"""H4: episode-local reservation identity, record card and state-derived next step.

TRAIN evidence (public traces only) shows four harness-side divergences:

* The next-step line was derived with a *different* confirmation reading than the
  H2 gate.  H4 matched "go ahead" inside "if Noah's portion is $650 or less, go
  ahead and upgrade just Noah" and "I agree" inside "before I agree to anything",
  so it told the solver "the caller has already given the explicit yes for
  cancel_reservation YAX4DR - make exactly that call now" while H2 was correctly
  vetoing the same call.  The solver re-sent the refused action turn after turn and
  the caller stopped.  This revision uses one reader, the same one H2 uses, and
  treats a refusal of the prepared action as a reason to void it.  That reader now
  also recognises an unconditional confirmation that is not the first sentence
  ("Oui, YES! I confirm. Please proceed..."), which is how the caller in the
  repeated-action trace actually answered while the gate kept re-asking.

* The card omitted record facts the solver must quote, and the one fact it did
  carry about bags was ambiguous: "free bags 8 (4/pax)" on a gold booking whose
  record held no bags at all was read as "this reservation allows 8 bags", and the
  solver then wrote total_baggages = 8 after the caller asked for two bags.  The
  card now separates the bags on the record from the free allowance.  The card also
  carries the passengers with their dates of birth, and for the caller the
  profile's membership, payment ids with their amounts and the per-passenger
  checked-bag allowance for every cabin.

* The profile line and the allowance table were emitted only when the newest
  message happened to contain one of a few literals, so "how many suitcases does
  my reservation allow" - the exact question in two TRAIN episodes that share one
  task - produced no allowance row at all in one of them, and the solver answered
  from memory with the wrong membership rate (1 bag per passenger for a silver
  economy booking).  Baggage wording now covers suitcases, luggage, packing,
  carry-on and backpacks, and the caller's identity and membership are stated
  whenever the caller is known, so the solver quotes the profile instead of
  guessing an id.

* The index summarised round trips as "JFK-JFK"/"IAH-IAH", so the solver relayed
  a route the caller did not recognise and lost turns matching it.  The route is
  taken from the outbound direction (IAH-SEA rt), and the guidance refers to
  "the list above" only on turns where the list is actually emitted.

The standing "state the one change once" line is now suppressed on turns whose
newest message is a question rather than a change request, where it used to push
the solver to re-state or re-confirm a change instead of answering.

Only an explicit request for a *new* booking suppresses the reservation card; a
request about an existing booking ("change my booking") keeps it.

Shared state: H2 owns `h2_pending` / `h2_block`; this layer reads them, consumes
`h2_pending` when the caller's newest message points at another reservation or
refuses that action (recorded as `h4_void`), and keeps its guidance short enough
that the next-step line is never truncated.  The assembly budget is wide enough
that the caller's identity and allowance facts cannot displace the reservation
index on an "all my reservations" turn.  State is episode-local and reset by the
runtime.
"""
from datetime import datetime, timedelta
from json import loads
from re import IGNORECASE, compile

TARGET_LAYER = 'h4'

CURRENT_TIME = datetime(2024, 5, 15, 15, 0, 0)
HARNESS_PREFIXES = ('[h2]', '[h3]', '[h4]', '[h5]')
FLOWN_STATES = ('landed', 'flying')
WORD_BUDGET = 170
INDEX_LIMIT = 8
BAG_ALLOWANCE = {
    'regular': {'basic_economy': 0, 'economy': 1, 'business': 2},
    'silver': {'basic_economy': 1, 'economy': 2, 'business': 3},
    'gold': {'basic_economy': 2, 'economy': 3, 'business': 4},
}
RID_TOKEN = compile(r"\b[A-Za-z0-9]{6}\b")
UID_TOKEN = compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,3}\b")
ACTION_VERBS = {
    'cancel_reservation': r'(?:cancel\w*|cancellation)',
    'update_reservation_flights': (r'(?:change|changes|modify|modification|upgrade|upgrades|'
                                   r'update|rebook)'),
    'update_reservation_baggages': r'(?:bag|bags|baggage|baggages|add)',
    'update_reservation_passengers': r'(?:passenger|passengers|name)',
    'book_reservation': r'(?:book|booking|reserve|reservation)',
    'send_certificate': r'(?:certificate|compensation|compensate|voucher|goodwill)',
}
NEG_WORD = r"(?:do not|does not|don'?t|won'?t|will not|never|no|cannot|can'?t|not|stop)"
NEG_FILL = (r"(?:(?:ever|really|actually|just|please|want|wanna|need|needed|have|has|had|to|be|"
            r"being|say|saying|go|going|a|an|the|any|this|that|it|them|him|her|me|my|your|our|"
            r"us|i|you|we|they|for|with|about|of|on|at|again|now|yet|even|still|longer)\s+){0,4}")
DENIALS = dict((name, compile(NEG_WORD + r"\s+" + NEG_FILL + verbs + r"\b", IGNORECASE))
               for name, verbs in ACTION_VERBS.items())
NOT_WHAT = compile(
    r"\b(?:cancel\w*|cancellation|change|changes|modification|upgrade|update|rebook|booking|"
    r"reservation)\s+(?:is|was|are|would be)\s+not\b", IGNORECASE)
LEAD_NO = compile(
    r"^\s*(?:no\b|nope\b|not\b|never\b|don'?t\b|do not\b|stop\b|hold on\b|hold off\b|wait\b|"
    r"i (?:do not|don'?t|won'?t|cannot|can'?t)\b|please (?:don'?t|stop)\b|not yet\b|no thanks\b)",
    IGNORECASE)
CONFIRM_START = compile(
    r"^\s*(?:(?:yes|yeah|yep|yup|absolutely|correct|confirmed|agreed)\b"
    r"|(?:go ahead|please do|proceed|do it|i confirm|i agree|"
    r"that(?:'s| is) (?:right|correct|fine))\b"
    r"|(?:ok(?:ay)?|sure|perfect|great|alright)\s*[,.!\-]?\s*"
    r"(?:please|go ahead|proceed|do it|yes|confirm)\b)", IGNORECASE)
SHORT_YES = compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok|okay|absolutely|correct|confirmed|agreed|do it|"
    r"go ahead|please do|proceed)\s*[.!]?\s*$", IGNORECASE)
CONFIRM_ANY = compile(
    r"\b(?:yes,?\s*please|please (?:do|proceed|go ahead)(?!\s+not)|go ahead(?!\s+not)|"
    r"you (?:can|may) proceed|i confirm|i agree|please proceed|"
    r"would like you to proceed)\b", IGNORECASE)
# A confirmation that is not the first sentence still counts, but never when the
# same message is asking about the action or attaching a condition to it.
STRONG_QUERY = compile(
    r"\b(?:how much|how many|price|prices|cost|costs|quote|estimate|fare|fares|whether|"
    r"is it possible|would it be|would that be|check if|check whether|let me know|tell me|"
    r"show me|compare|what (?:would|will|is|are)|do i (?:have|need)|am i)\b", IGNORECASE)
STRONG_CONDITION = compile(
    r"\b(?:only if|unless|as long as|provided that|provided|assuming|in case|contingent|"
    r"subject to|not until)\b", IGNORECASE)
NEGATED_YES = compile(
    r"\b(?:do not|don'?t|does not|doesn'?t|will not|won'?t|cannot|can'?t|not|never)\s+"
    r"(?:\w+\s+){0,3}(?:confirm|agree|proceed|go ahead|do it)\b", IGNORECASE)
QUERY = compile(
    r"\b(?:check|checking|see|seeing|find out|whether|if|how much|how many|"
    r"what (?:would|will|is|are)|price|prices|cost|costs|quote|estimate|fare|fares|"
    r"confirm (?:that|if|whether|the)|let me know|tell me|show me|wait|compare)\b", IGNORECASE)
CONDITION = compile(
    r"\b(?:unless|until|before|after|once|provided|assuming|only if|as long as|in case|first)\b",
    IGNORECASE)
ACTION_WORDS = compile(
    r"\b(?:cancel|change|modify|update|upgrade|bag|baggage|baggages|reschedul\w*|move|"
    r"transfer|switch)\b", IGNORECASE)
# An informational question about the policy or the records versus a request that
# actually asks for a booking-database change.
ASK = compile(
    r"\?|\b(?:what|which|when|where|who|whom|whose|why|how|can you|could you|would you|do i|"
    r"does|is it|are there|did|will you)\b", IGNORECASE)
ACTION_REQUEST = compile(
    r"\b(?:please|can you|could you|would you|i want|i would like|i'?d like|i need|we need|"
    r"go ahead|do it|proceed|let'?s)\b[^.\n]{0,70}\b(?:cancel|change|changes|modify|"
    r"modification|update|upgrade|rebook|reschedul\w*|move|switch|add|book|reserve|remove|"
    r"include|transfer)\b", IGNORECASE)
PAY_TOPIC = compile(
    r"\b(?:pay|pays|paid|payment|payments|certificate|certificates|gift ?cards?|credit cards?|"
    r"card|refund|refunded|charge|charged|balance|balances|cost|costs|price|prices|fare|fares|"
    r"cover|covered|afford|budget|receipt)\b", IGNORECASE)
BAG_TOPIC = compile(
    r"\b(?:bag|bags|baggage|baggages|suitcase|suitcases|luggage|carry-on|backpack|backpacks|"
    r"checked|pack|packing|packed)\b", IGNORECASE)
LIST_ASK = compile(
    r"\b(?:all (?:my|of my|the)|every(?: one)?|other (?:reservations?|flights?|bookings?)|"
    r"more (?:reservations?|flights?|bookings?)|list (?:of )?(?:my|the|all)|"
    r"show me (?:all|my|the)|which reservations?|how many reservations?)\b", IGNORECASE)
CONFIRM_ASK = compile(
    r"\b(?:confirm|confirmation|proceed|would you like me to|shall i|do you want me to|"
    r"reply|yes/no|yes or no)\b", IGNORECASE)
NEW_BOOK = compile(
    r"\b(?:book (?:a|an|the|new|me|it|flight|flights|two|three|another)|booking a|"
    r"new (?:reservation|booking|flight|flights)|make a (?:new )?(?:reservation|booking)|"
    r"reserve (?:a|me|new)|create .{0,20}reservation|start a new (?:reservation|booking))\b",
    IGNORECASE)


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


def _head(text):
    for index, char in enumerate(text):
        if char in '.!?\n':
            return text[:index]
    return text


def _reply(text, tool=None):
    """Read the caller's newest message as a reply to a prepared write.

    Returns 'yes' only for an unconditional go-ahead, 'no' for a refusal of that
    action, otherwise 'unclear'.  Kept byte-identical to the H2 reader so the
    gate and the guidance can never disagree.
    """
    if not isinstance(text, str) or not text.strip():
        return 'unclear'
    body = text.strip()
    if LEAD_NO.match(body):
        return 'no'
    denial = DENIALS.get(tool) if isinstance(tool, str) else None
    if NOT_WHAT.search(body) or (denial is not None and denial.search(body)):
        return 'no'
    head = _head(body)
    if QUERY.search(head) or CONDITION.search(head):
        return 'unclear'
    if SHORT_YES.match(body):
        return 'yes'
    if CONFIRM_START.search(body):
        return 'yes'
    if CONFIRM_ANY.search(body) and len(body.split()) <= 12:
        return 'yes'
    if (CONFIRM_ANY.search(body) and not NEGATED_YES.search(body)
            and not STRONG_QUERY.search(body) and not STRONG_CONDITION.search(body)):
        return 'yes'
    return 'unclear'


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
    """Outbound route of the reservation, with a round trip marked as such."""
    flights = res.get('flights')
    legs = [leg for leg in flights if isinstance(leg, dict)] if isinstance(flights, list) else []
    if not legs:
        return 'route?'
    origin = legs[0].get('origin')
    destination = legs[0].get('destination')
    for index in range(len(legs) - 1):
        first, second = legs[index], legs[index + 1]
        if (isinstance(first.get('origin'), str)
                and first.get('origin') == second.get('destination')
                and isinstance(first.get('destination'), str)
                and first.get('destination') == second.get('origin')):
            return '%s-%s rt' % (origin, first.get('destination'))
    if len(legs) > 1:
        return '%s-%s/%s legs' % (origin, legs[-1].get('destination'), len(legs))
    return '%s-%s' % (origin, destination)


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


def _passengers(res):
    passengers = res.get('passengers')
    parts = []
    if isinstance(passengers, list):
        for item in passengers[:2]:
            if isinstance(item, dict):
                parts.append('%s %s %s' % (item.get('first_name'), item.get('last_name'),
                                           item.get('dob')))
        if len(passengers) > 2:
            parts.append('+%s' % (len(passengers) - 2))
    return ', '.join(parts)


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
    pax = _passengers(res)
    parts = ['DB facts %s: user %s (%s); cabin %s; %s; %s pax%s; insurance %s; checked bags on '
             'record %s (%s non-free);'
             % (rid, uid, membership, res.get('cabin'), _legs_text(db, res), count,
                ': ' + pax if pax else '', res.get('insurance'), res.get('total_baggages'),
                res.get('nonfree_baggages'))]
    if not _refunded(res):
        parts.append('paid $%s;' % _payments(res)[0])
    per = _allowance(membership, res.get('cabin'))
    if per is not None and count:
        parts.append('free allowance %s checked bag(s) for %s passenger(s) (%s per passenger, $50 '
                     'for every bag beyond that);' % (per * count, count, per))
    parts.append('created %s;' % str(res.get('created_at'))[:10])
    parts.append('cancel: %s.' % _cancel_note(db, res))
    return ' '.join(parts)


def _method_ids(record):
    methods = record.get('payment_methods') if isinstance(record, dict) else None
    if isinstance(methods, dict):
        found = []
        for key in methods:
            if not isinstance(key, str):
                continue
            entry = methods.get(key)
            amount = entry.get('amount') if isinstance(entry, dict) else None
            if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                found.append('%s $%s' % (key, int(amount)))
            else:
                found.append(key)
        return found
    if isinstance(methods, list):
        found = []
        for item in methods:
            if isinstance(item, dict) and isinstance(item.get('id'), str):
                found.append(item['id'])
        return found
    return None


def _caller_line(db, uid, latest):
    """The caller's identity, plus the profile facts their own question turns on."""
    record = _user_record(db, uid)
    if not isinstance(record, dict):
        return ''
    membership = record.get('membership')
    head = ('Caller %s (%s membership).'
            % (uid, membership if isinstance(membership, str) else 'unknown'))
    text = latest if isinstance(latest, str) else ''
    money = bool(PAY_TOPIC.search(text))
    bags = bool(BAG_TOPIC.search(text))
    booking = bool(NEW_BOOK.search(text))
    parts = []
    if money or booking:
        ids = _method_ids(record)
        if ids:
            shown = ids[:5]
            if len(ids) > 5:
                shown.append('+%s more' % (len(ids) - 5))
            parts.append('payment methods: %s.' % ', '.join(shown))
    if bags or booking:
        row = BAG_ALLOWANCE.get(membership) if isinstance(membership, str) else None
        if isinstance(row, dict):
            parts.append('free checked bags per passenger: %s basic economy, %s economy, %s business '
                         '(%s member, each extra bag $50).'
                         % (row.get('basic_economy'), row.get('economy'), row.get('business'),
                            membership))
    if not parts:
        return head
    return head + ' ' + ' '.join(parts)


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


def _index_text(uid, entries, limit):
    used = entries[:limit]
    if not used:
        return ''
    return 'Reservations of caller %s: %s.' % (uid, '; '.join(used))


def _awaiting_execution(messages):
    """The solver listed details and asked for confirmation, and has not acted since."""
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


def _guidance(db, state, messages, latest, rid, uid, have_index, booking):
    pending = state.get('h2_pending') if isinstance(state, dict) else None
    blocked = state.get('h2_block') if isinstance(state, dict) else None
    live = pending if isinstance(pending, dict) and pending.get('tool') else None
    if live is None and isinstance(blocked, dict) and blocked.get('tool'):
        live = blocked
    tool = live.get('tool') if isinstance(live, dict) else None
    key = live.get('key') if isinstance(live, dict) else None
    reply = _reply(latest, tool)
    named = _named_reservations(db, latest)
    if isinstance(pending, dict) and pending.get('tool'):
        if named and key not in named:
            if isinstance(state, dict):
                state['h2_pending'] = None
                state['h4_void'] = {'tool': tool, 'key': key, 'to': named[0]}
            return ("Next action: the caller's latest message is about %s, not %s, so the prepared "
                    "%s on %s is void - serve %s instead." % (named[0], key, tool, key, named[0]))
        if reply == 'no':
            if isinstance(state, dict):
                state['h2_pending'] = None
                state['h4_void'] = {'tool': tool, 'key': key, 'to': 'refused'}
            return ("Next action: the caller has just refused %s for %s, so it is void and must not "
                    "be sent or offered again; serve the request in the caller's newest message."
                    % (tool, key))
        if reply == 'yes':
            return ("Next action: the caller's newest message is the explicit yes for %s %s, so "
                    "make exactly that call now and do not send the request again." % (tool, key))
        return ("Next action: %s %s is still waiting for the caller's explicit yes. Do not execute "
                "it and do not send the same request again - answer what the caller just asked, or "
                "state the single policy condition that decides it." % (tool, key))
    if isinstance(blocked, dict) and blocked.get('tool'):
        reason = blocked.get('reason')
        if reason in ('refused', 'mismatch'):
            return ("Next action: the prepared %s on %s was dropped before execution (%s) - do not "
                    "offer or repeat it; serve the request in the caller's newest message."
                    % (tool, key, reason))
        if reason == 'confirm':
            return ("Next action: the %s on %s is still unconfirmed and the same request has "
                    "already been sent once - do not send it again; answer the caller's newest "
                    "message, or state plainly the one condition that blocks that change."
                    % (tool, key))
        if reason in ('records', 'repeat'):
            return ("Next action: the %s on %s was not sent because it did not match the records - "
                    "correct the figures from the records above and only then offer it again."
                    % (tool, key))
        return ("Next action: the %s on %s was replaced before execution (policy: %s); do not "
                "repeat it - state that reason once." % (tool, key, reason))
    if _awaiting_execution(messages) and reply == 'yes' and rid is not None:
        return ('Next action: the caller already answered yes to the details you listed and nothing '
                'has been done yet, so carry out that change now instead of sending the same request '
                'again, or state plainly which policy condition blocks it.')
    if booking:
        return ('Next action: this is a new booking, so no existing reservation is in play - answer '
                'with the itinerary, the fare and the charge from the search results and the caller '
                'profile, then list the exact details once for the caller\'s yes.')
    if LIST_ASK.search(latest) and uid is not None and have_index:
        return ('Next action: the caller is asking about all of their reservations - answer from the '
                'list above (id, route, dates, cabin) instead of one reservation only.')
    if rid is not None and ACTION_WORDS.search(latest):
        if ASK.search(latest) and not ACTION_REQUEST.search(latest):
            return ('Next action: the caller is asking a question about %s, so answer it from the '
                    'record facts above (membership, cabin, bags, payments, cancellation '
                    'condition); do not restate a change and do not repeat a confirmation you '
                    'already sent.' % rid)
        return ('Next action: state the one change to %s with its exact details once; if the '
                'caller\'s newest message asks for something else (a price, a cheaper option, a '
                'different cabin), serve that first instead of re-stating the same change.' % rid)
    if rid is None and uid is not None and ACTION_WORDS.search(latest):
        if have_index:
            return ('Next action: which reservation is in play is not established - match the '
                    'caller\'s route and dates against the list above, or ask which one, before '
                    'quoting facts or acting.')
        return ('Next action: which reservation is in play is not established and no list is '
                'available - ask the caller for the reservation id or the route and dates before '
                'quoting facts or acting.')
    if rid is None and uid is None and ACTION_WORDS.search(latest):
        return ('Next action: no reservation is identified yet - get the caller\'s user id or '
                'reservation id before quoting facts or acting.')
    return ''


def _assemble(card, caller, index, guide):
    """Keep the next-step line intact; drop the least important facts first."""
    for parts in ((card, caller, index, guide), (card, caller, guide), (card, guide)):
        text = ' '.join([part for part in parts if part])
        if len(text.split()) <= WORD_BUDGET:
            return text
    if not guide:
        return ''
    room = WORD_BUDGET - len(guide.split()) - 1
    if room > 10:
        return ' '.join([' '.join(card.split()[:room]), guide])
    return guide


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
        shared = self.state if isinstance(self.state, dict) else None
        uid = _caller_id(db, messages)
        rid = _subject(db, messages, uid)
        named = _named_reservations(db, latest)
        booking = bool(NEW_BOOK.search(latest)) and not named
        if booking:
            rid = None
            entries = []
        else:
            entries = _index_entries(db, uid) if uid is not None else []
            if rid is not None and not LIST_ASK.search(latest):
                entries = []
        card = ''
        if rid is not None:
            record = _reservation(db, rid)
            if record is not None:
                card = _card(db, record)
        caller = _caller_line(db, uid, latest) if uid is not None else ''
        index = _index_text(uid, entries, INDEX_LIMIT) if entries else ''
        guide = _guidance(db, shared, messages, latest, rid, uid, bool(index), booking)
        text = _assemble(card, caller, index, guide)
        return [text] if text else []

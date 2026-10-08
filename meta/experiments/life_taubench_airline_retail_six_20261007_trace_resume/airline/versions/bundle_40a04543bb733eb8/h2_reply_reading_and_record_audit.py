"""H2: read the caller's reply, audit the prepared write against the bound records.

Mechanism selected from public TRAIN traces (no hidden answers, no evaluator state).

1. Reply reading.  A booking-database write is dispatched only when the caller's
   newest message is an unconditional go-ahead for that action.  A message that
   asks for a price or a feasibility check ("yes, please go ahead and *check if*
   upgrading is possible"), states a condition ("if it is $650 or less, go
   ahead"), or refuses the action ("no, please don't cancel", "that is not what I
   asked") is not that confirmation.  H4 uses the identical reader, so the two
   layers cannot contradict each other any more; the observed TRAIN loops came
   from H2 vetoing a write while H4 told the solver to execute it.

2. Record audit.  Before dispatch the prepared call is checked against the bound
   business records: every payment id must exist in the booking user's profile, a
   booking's payment must equal the record prices of the listed flights plus $50
   per non-free bag (plus insurance), and the non-free count must match the
   membership/cabin allowance.  A mismatch is replaced by a message carrying the
   record figures instead of the raw tool error the environment returns
   ("Payment method ... not found", "Payment amount does not add up").

3. Repetition guard.  A write identical to an earlier call that already returned
   an error in this episode is replaced by that recorded error rather than being
   dispatched again.

Eligibility verdicts (cancellation conditions, certificate conditions), the
target-bound confirmation check and the permissive default for anything the
bound records cannot establish are retained from the previous revision.

Shared state: this layer owns `h2_pending` (a write waiting for the caller's yes)
and `h2_block` (the last write replaced before execution, with its reason); both
are rewritten on every gate decision and cleared on a clean pass.  H4 reads them
for next-step guidance.
"""
from datetime import datetime, timedelta
from json import dumps, loads
from re import IGNORECASE, compile

TARGET_LAYER = 'h2'

# Clock stated by the public policy (<policy>: current time 2024-05-15 15:00 EST).
CURRENT_TIME = datetime(2024, 5, 15, 15, 0, 0)
HARNESS_PREFIXES = ('[h2]', '[h3]', '[h4]', '[h5]')
WRITE_TOOLS = ('book_reservation', 'cancel_reservation', 'update_reservation_flights',
               'update_reservation_baggages', 'update_reservation_passengers',
               'send_certificate')
CHANGE_TOOLS = ('cancel_reservation', 'update_reservation_flights',
                'update_reservation_baggages', 'update_reservation_passengers')
CANCEL_TOOL = 'cancel_reservation'
CERTIFICATE_TOOL = 'send_certificate'
BOOK_TOOL = 'book_reservation'
FLOWN_STATES = ('landed', 'flying')
BAG_ALLOWANCE = {
    'regular': {'basic_economy': 0, 'economy': 1, 'business': 2},
    'silver': {'basic_economy': 1, 'economy': 2, 'business': 3},
    'gold': {'basic_economy': 2, 'economy': 3, 'business': 4},
}
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
QUERY = compile(
    r"\b(?:check|checking|see|seeing|find out|whether|if|how much|how many|"
    r"what (?:would|will|is|are)|price|prices|cost|costs|quote|estimate|fare|fares|"
    r"confirm (?:that|if|whether|the)|let me know|tell me|show me|wait|compare)\b", IGNORECASE)
CONDITION = compile(
    r"\b(?:unless|until|before|after|once|provided|assuming|only if|as long as|in case|first)\b",
    IGNORECASE)
HEALTH = compile(
    r"\b(?:sick|ill|illness|unwell|flu|fever|covid|health|medical|doctor|hospital|"
    r"surgery|injur\w*|migraine|nauseous|diagnos\w*|emergency surgery)\b", IGNORECASE)
WEATHER = compile(
    r"\b(?:weather|storm\w*|snow\w*|hurricane|tornado\w*|flood\w*|blizzard|typhoon|"
    r"grounded)\b", IGNORECASE)
UNCOVERED = compile(
    r"\b(?:change of plans?|plans? (?:have |has )?changed|changed my plans?|birthday|"
    r"wedding|anniversary|no longer need|don'?t need|not going|celebrat\w*|"
    r"vacation plans?|mistake)\b", IGNORECASE)
COMPENSATION_ASK = compile(
    r"\b(?:compensat\w*|voucher|certificate|goodwill|gesture)\b", IGNORECASE)
DELAY = compile(r"\bdelay(?:ed|s)?\b", IGNORECASE)
REQUEST_CHANGE = compile(
    r"\b(?:want|wanted|need|needed|like|prefer|please|can|could|help)\b[^.\n]{0,60}"
    r"\b(?:change|modify|cancel|move|reschedul\w*|switch|upgrade|rebook|different flight|"
    r"another flight|later flight|earlier flight|new flight)\b", IGNORECASE)
REFUSE_CHANGE = compile(
    r"\b(?:not|never|don'?t|won'?t|do not)[^.\n]{0,60}"
    r"\b(?:cancel|change|changes|modify|modification|alter)\b", IGNORECASE)
RID_TOKEN = compile(r"\b[A-Za-z0-9]{6}\b")


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


def _user_texts(history):
    texts = []
    if isinstance(history, list):
        for item in history:
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
    action, otherwise 'unclear'.  Kept byte-identical in the H4 artifact so the
    gate and the guidance cannot disagree.
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
    return 'unclear'


def _identity(name, args):
    rid = args.get('reservation_id')
    if isinstance(rid, str) and rid:
        return rid
    user_id = args.get('user_id')
    return user_id if isinstance(user_id, str) and user_id else name


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


def _payload(content):
    if not isinstance(content, str) or not content.lstrip().startswith('{'):
        return None
    try:
        value = loads(content)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _named_reservations(db, text):
    """Reservation ids the caller's own message refers to, in order of mention."""
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


def _subject_record(db, history):
    """Reservation identified by a caller id, a reservation-scoped call or result."""
    if not isinstance(history, list):
        return None
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        role = item.get('role')
        content = item.get('content')
        rid = None
        if role == 'user':
            if isinstance(content, str) and not _is_harness_text(content):
                named = _named_reservations(db, content)
                if named:
                    rid = named[0]
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
        if rid is not None:
            return _reservation(db, rid)
    return None


def _membership(db, user_id):
    record = _user_record(db, user_id)
    if not isinstance(record, dict):
        return None
    membership = record.get('membership')
    return membership if isinstance(membership, str) else None


def _allowance(membership, cabin):
    row = BAG_ALLOWANCE.get(membership) if isinstance(membership, str) else None
    if not isinstance(row, dict):
        return None
    value = row.get(cabin)
    return value if isinstance(value, int) else None


def _method_ids(record):
    methods = record.get('payment_methods') if isinstance(record, dict) else None
    if isinstance(methods, dict):
        return [key for key in methods if isinstance(key, str)]
    if isinstance(methods, list):
        found = []
        for item in methods:
            if isinstance(item, dict) and isinstance(item.get('id'), str):
                found.append(item['id'])
        return found
    return None


def _requested_methods(name, args):
    if name == BOOK_TOOL:
        found = []
        items = args.get('payment_methods')
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and isinstance(item.get('payment_id'), str):
                    found.append(item['payment_id'])
        return found
    value = args.get('payment_id')
    return [value] if isinstance(value, str) else []


def _subject_user(db, name, args):
    if name == BOOK_TOOL:
        value = args.get('user_id')
        return value if isinstance(value, str) else None
    record = _reservation(db, args.get('reservation_id'))
    value = record.get('user_id') if isinstance(record, dict) else None
    return value if isinstance(value, str) else None


def _price(db, flight_number, date, cabin):
    flights = db.get('flights') if isinstance(db, dict) else None
    if not isinstance(flights, dict) or not isinstance(flight_number, str):
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
    prices = day.get('prices')
    if not isinstance(prices, dict):
        return None
    value = prices.get(cabin)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _fare_total(db, args):
    cabin = args.get('cabin')
    flights = args.get('flights')
    passengers = args.get('passengers')
    if not isinstance(cabin, str) or not isinstance(flights, list) or not flights:
        return None
    if not isinstance(passengers, list) or not passengers:
        return None
    total = 0
    for leg in flights:
        if not isinstance(leg, dict):
            return None
        price = _price(db, leg.get('flight_number'), leg.get('date'), cabin)
        if price is None:
            return None
        total += price
    return total * len(passengers)


def _payment_sum(items):
    if not isinstance(items, list):
        return None
    total = 0
    for item in items:
        if not isinstance(item, dict):
            return None
        amount = item.get('amount')
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            return None
        total += int(amount)
    return total


def _records_text(db, name, args):
    """A repair message when the prepared call contradicts the records, else None."""
    uid = _subject_user(db, name, args)
    requested = _requested_methods(name, args)
    if requested and uid is not None:
        known = _method_ids(_user_record(db, uid))
        if known is not None:
            missing = [pid for pid in requested if pid not in known]
            if missing:
                return ("Nothing has been changed: the profile of user %s does not contain the "
                        "payment method %s, so that call would be refused and nothing would be "
                        "booked or changed. The payment methods in that profile are: %s. Use one of "
                        "those ids exactly as written, with the amount it carries."
                        % (uid, ', '.join(missing), ', '.join(known[:6]) if known else 'none'))
    if name != BOOK_TOOL:
        return None
    fare = _fare_total(db, args)
    if fare is None:
        return None
    passengers = args.get('passengers')
    pax = len(passengers) if isinstance(passengers, list) else 0
    if not pax:
        return None
    cabin = args.get('cabin')
    total_bags = args.get('total_baggages')
    nonfree = args.get('nonfree_baggages')
    insurance = str(args.get('insurance')).lower() == 'yes'
    per = _allowance(_membership(db, uid), cabin)
    if isinstance(per, int) and isinstance(total_bags, int) and isinstance(nonfree, int):
        free = per * pax
        wanted = max(0, total_bags - free)
        if nonfree != wanted:
            return ("Nothing has been booked. User %s gets %s free checked bag(s) per %s passenger, "
                    "so %s passenger(s) already include %s free bag(s); %s total bag(s) therefore "
                    "means %s non-free bag(s) at $50 each, not %s. The listed flights also cost $%s "
                    "per passenger in %s. Re-list those exact figures and take the caller's "
                    "decision again."
                    % (uid, per, cabin, pax, free, total_bags, wanted, nonfree, fare, cabin))
    paid = _payment_sum(args.get('payment_methods'))
    if paid is None or not isinstance(nonfree, int):
        return None
    due = fare + 50 * nonfree
    insured = due + 30 * pax
    if paid == due or (insurance and paid == insured):
        return None
    return ("Nothing has been booked. The records price the listed itinerary in %s at $%s for %s "
            "passenger(s) plus $50 for each of the %s non-free bag(s)%s, so the payment must add up "
            "to exactly $%s. The payment methods listed add up to $%s. Adjust the amounts to $%s and "
            "send the call again."
            % (cabin, fare, pax, nonfree,
               ' plus $30 insurance per passenger ($%s)' % (30 * pax) if insurance else '',
               insured if insurance else due, paid, insured if insurance else due))


def _repeat_text(name, identity, error):
    return ("Nothing has been sent. That exact %s for %s already returned an error in this "
            "conversation: \"%s\". Sending the identical call again cannot succeed - correct the "
            "arguments from the records (payment ids and amounts, flight prices, bag counts) or take "
            "a different step, then call it again."
            % (name.replace('_', ' '), identity, error))


def _refused_text(name, identity):
    return ("Understood - I will not go ahead with %s for %s, and nothing has been changed. Tell me "
            "what you would like done instead and I will check it against the policy and the records."
            % (name.replace('_', ' '), identity))


def _call_key(name, args):
    try:
        return name + '|' + dumps(args, sort_keys=True)
    except Exception:
        return None


def _identical_failure(history, name, args):
    """The error text of an identical earlier call in this episode, else None."""
    if not isinstance(history, list):
        return None
    key = _call_key(name, args)
    if key is None:
        return None
    seen = []
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        for call_name, call_args in _tool_calls(item):
            seen.append((index, _call_key(call_name, call_args)))
    for index, seen_key in reversed(seen):
        if seen_key != key:
            continue
        for item in history[index + 1:]:
            if not isinstance(item, dict) or item.get('role') != 'tool':
                continue
            content = item.get('content')
            if isinstance(content, str) and content.lstrip().lower().startswith('error'):
                return content.strip()[:180]
            break
        return None
    return None


def _itinerary(res):
    legs = []
    if isinstance(res, dict):
        flights = res.get('flights')
        if isinstance(flights, list):
            for leg in flights:
                if isinstance(leg, dict):
                    legs.append((leg.get('flight_number'), leg.get('date'),
                                 leg.get('origin'), leg.get('destination')))
    return legs


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


def _passenger_count(res):
    if not isinstance(res, dict):
        return 0
    passengers = res.get('passengers')
    return len(passengers) if isinstance(passengers, list) else 0


def _paid_total(res):
    total = 0
    if isinstance(res, dict):
        history = res.get('payment_history')
        if isinstance(history, list):
            for entry in history:
                if isinstance(entry, dict):
                    amount = entry.get('amount')
                    if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                        total += amount
    return int(total)


def _legs_text(res):
    parts = []
    for flight_number, date, origin, destination in _itinerary(res):
        parts.append('%s %s-%s on %s' % (flight_number, origin, destination, date))
    return '; '.join(parts) if parts else 'no flight segments on record'


def _arg_legs(flights):
    parts = []
    if isinstance(flights, list):
        for leg in flights:
            if isinstance(leg, dict):
                parts.append('%s on %s' % (leg.get('flight_number'), leg.get('date')))
    return '; '.join(parts) if parts else 'as listed'


def _arg_payments(payments):
    parts = []
    if isinstance(payments, list):
        for item in payments:
            if isinstance(item, dict):
                parts.append('%s $%s' % (item.get('payment_id'), item.get('amount')))
    return '; '.join(parts) if parts else 'as listed'


def _itinerary_state(db, res):
    flown = False
    cancelled = False
    for flight_number, date, _origin, _destination in _itinerary(res):
        status = _leg_status(db, flight_number, date)
        if status in FLOWN_STATES:
            flown = True
        elif status == 'cancelled':
            cancelled = True
        elif status is None and isinstance(date, str) and date < '2024-05-15':
            flown = True
    return flown, cancelled


def _has_change_action(history):
    if not isinstance(history, list):
        return False
    for item in history:
        if not isinstance(item, dict):
            continue
        for name, _args in _tool_calls(item):
            if name in CHANGE_TOOLS:
                return True
    return False


def _cancel_verdict(db, args, text):
    record = _reservation(db, args.get('reservation_id'))
    if record is None:
        return 'unknown'
    flown, cancelled = _itinerary_state(db, record)
    if flown:
        return 'flown'
    if cancelled:
        return 'eligible'
    if str(record.get('cabin')).lower() == 'business':
        return 'eligible'
    delta = _created_delta(record.get('created_at'))
    if delta is not None and delta <= timedelta(hours=24):
        return 'eligible'
    insured = str(record.get('insurance')).lower() == 'yes'
    covered = bool(HEALTH.search(text) or WEATHER.search(text))
    if insured and covered:
        return 'eligible'
    if insured and not UNCOVERED.search(text):
        return 'unknown'
    return 'ineligible'


def _certificate_verdict(db, args, history, text):
    if not COMPENSATION_ASK.search(text):
        return 'not_requested'
    membership = _membership(db, args.get('user_id'))
    record = _subject_record(db, history)
    if membership == 'regular':
        if record is not None:
            insured = str(record.get('insurance')).lower() == 'yes'
            business = str(record.get('cabin')).lower() == 'business'
            if not insured and not business:
                return 'not_eligible'
        elif not _has_qualifying_reservation(db, args.get('user_id')):
            return 'not_eligible'
    if DELAY.search(text):
        changed = _has_change_action(history)
        asked = bool(REQUEST_CHANGE.search(text))
        refused = bool(REFUSE_CHANGE.search(text))
        if not changed and (refused or not asked):
            return 'delay_without_change'
    return None


def _has_qualifying_reservation(db, user_id):
    record = _user_record(db, user_id)
    rids = record.get('reservations') if isinstance(record, dict) else None
    if not isinstance(rids, list):
        return True
    for rid in rids:
        res = _reservation(db, rid)
        if res is None:
            continue
        if str(res.get('insurance')).lower() == 'yes':
            return True
        if str(res.get('cabin')).lower() == 'business':
            return True
    return False


def _action_body(db, name, args):
    rid = args.get('reservation_id')
    record = _reservation(db, rid) if isinstance(rid, str) else None
    if name == CANCEL_TOOL:
        body = 'cancel reservation %s' % rid
        if record is not None:
            body += (' (%s; cabin %s; %s passenger(s); refund of $%s to the original '
                     'payment method)' % (_legs_text(record), record.get('cabin'),
                                          _passenger_count(record), _paid_total(record)))
    elif name == 'update_reservation_flights':
        body = ('update reservation %s to cabin %s with flights %s for %s passenger(s) and '
                'payment %s' % (rid, args.get('cabin'), _arg_legs(args.get('flights')),
                                _passenger_count(record) if record is not None else 'the listed',
                                args.get('payment_id')))
        if record is not None:
            body += ' (currently cabin %s with %s)' % (record.get('cabin'), _legs_text(record))
    elif name == 'update_reservation_baggages':
        body = ('update reservation %s to %s total checked bags (%s non-free, $50 each) with '
                'payment %s' % (rid, args.get('total_baggages'), args.get('nonfree_baggages'),
                                args.get('payment_id')))
        if record is not None:
            body += ' (currently %s total, %s non-free)' % (record.get('total_baggages'),
                                                            record.get('nonfree_baggages'))
    elif name == 'update_reservation_passengers':
        names = []
        passengers = args.get('passengers')
        if isinstance(passengers, list):
            for item in passengers:
                if isinstance(item, dict):
                    names.append('%s %s (%s)' % (item.get('first_name'), item.get('last_name'),
                                                 item.get('dob')))
        body = 'update reservation %s passengers to %s' % (rid if rid else 'the reservation',
                                                           '; '.join(names) if names else 'as listed')
    elif name == BOOK_TOOL:
        passengers = args.get('passengers')
        count = len(passengers) if isinstance(passengers, list) else 'the listed number of'
        body = ('book a %s flight %s to %s in %s cabin for %s passenger(s): flights %s; payment %s; '
                '%s total checked bags (%s non-free); insurance %s'
                % (args.get('flight_type'), args.get('origin'), args.get('destination'),
                   args.get('cabin'), count, _arg_legs(args.get('flights')),
                   _arg_payments(args.get('payment_methods')), args.get('total_baggages'),
                   args.get('nonfree_baggages'), args.get('insurance')))
    elif name == CERTIFICATE_TOOL:
        body = 'send a $%s certificate to user %s' % (args.get('amount'), args.get('user_id'))
    else:
        body = name
    return body


def _confirmation_text(db, name, args, terse=False):
    body = _action_body(db, name, args)
    if terse:
        return ('Nothing has been changed yet. I still need your explicit yes for exactly this: %s. '
                'Reply "yes" to proceed, or tell me what to change and I will adjust it first.' % body)
    return ('Before I make any change, here are the exact details: %s. Nothing has been changed '
            'yet - please reply with an explicit "yes" and I will proceed with exactly that.' % body)


def _mismatch_text(name, target, other):
    return ('Nothing has been changed. Your latest message refers to reservation %s, while the '
            'action I had prepared was %s for reservation %s, so I stopped before touching the '
            'wrong booking. Tell me what you want done to %s - or which reservation you mean - and '
            'I will list those exact details for your confirmation first.'
            % (other, name.replace('_', ' '), target, other))


def _cancel_flown_text(args):
    return ('I cannot cancel reservation %s: part of that itinerary has already been flown, so '
            'this is outside what I can do here. Nothing has been cancelled. I can transfer you to '
            'a human agent who can review it - would you like me to do that?'
            % args.get('reservation_id'))


def _cancel_denied_text(db, args, text):
    rid = args.get('reservation_id')
    record = _reservation(db, rid)
    reasons = []
    if record is not None:
        cabin = record.get('cabin')
        if str(cabin).lower() != 'business':
            reasons.append('the cabin is %s, not business' % cabin)
        insured = str(record.get('insurance')).lower() == 'yes'
        if not insured:
            reasons.append('the reservation has no travel insurance')
        elif not (HEALTH.search(text) or WEATHER.search(text)):
            reasons.append('the travel insurance covers health or weather reasons only, and that '
                           'is not the reason given')
        created = record.get('created_at')
        delta = _created_delta(created)
        if delta is not None and delta > timedelta(hours=24):
            reasons.append('it was booked on %s, more than 24 hours before 2024-05-15 15:00'
                           % str(created)[:10])
        flown, cancelled = _itinerary_state(db, record)
        if not cancelled:
            reasons.append('no flight in it was cancelled by the airline')
    detail = '; '.join(reasons) if reasons else 'none of the policy conditions is satisfied'
    return ('I cannot cancel reservation %s: %s. A reservation can only be cancelled within 24 '
            'hours of booking, when the airline cancelled the flight, when it is a business '
            'reservation, or with travel insurance plus a health or weather reason. Nothing has '
            'been cancelled. If you would like, I can transfer you to a human agent, or help with '
            'something else.' % (rid, detail))


def _certificate_text(verdict, db, args, history):
    if verdict == 'not_requested':
        return ('I can only send a certificate when you explicitly ask for compensation, so I will '
                'not add one on my own. Nothing has been sent. Tell me if you would like '
                'compensation and I will check what the policy allows.')
    if verdict == 'not_eligible':
        record = _subject_record(db, history)
        membership = _membership(db, args.get('user_id'))
        cabin = record.get('cabin') if record is not None else 'unknown'
        insurance = record.get('insurance') if record is not None else 'unknown'
        return ('I cannot add a certificate here: certificates are only possible for silver or gold '
                'members, reservations with travel insurance, or business-class reservations, and '
                'this is a %s member booking in %s class with insurance "%s". Nothing has been '
                'sent. I can transfer you to a human agent if you would like this reviewed.'
                % (membership if isinstance(membership, str) else 'regular', cabin, insurance))
    return ('I cannot add a certificate for the delay alone: the policy allows a goodwill '
            'certificate after a delayed flight only when the reservation is also changed or '
            'cancelled, and you asked to keep the booking unchanged. Nothing has been sent. I can '
            'transfer you to a human agent if you would like this reviewed.')


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

    def h2(self, message, history):
        if not isinstance(message, dict):
            return message
        writes = [call for call in _tool_calls(message) if call[0] in WRITE_TOOLS]
        if not writes:
            return message
        try:
            return self._gate(message, history, writes)
        except Exception:
            return message

    def _replace(self, shared, tool, key, reason, text):
        if shared is not None:
            if reason == 'confirm':
                shared['h2_pending'] = {'tool': tool, 'key': key}
            else:
                shared['h2_pending'] = None
            shared['h2_block'] = {'tool': tool, 'key': key, 'reason': reason}
        return {'role': 'assistant', 'content': text}

    def _allow(self, shared):
        if shared is not None:
            shared['h2_pending'] = None
            shared['h2_block'] = None

    def _gate(self, message, history, writes):
        db = _business_db(self.env)
        texts = _user_texts(history)
        joined = '\n'.join(texts)
        latest = texts[-1] if texts else ''
        shared = self.state if isinstance(self.state, dict) else None
        pending = shared.get('h2_pending') if shared is not None else None
        named = _named_reservations(db, latest)
        for name, args in writes:
            key = _identity(name, args)
            target = args.get('reservation_id')
            if (isinstance(target, str) and named and target not in named
                    and _reservation(db, target) is not None):
                return self._replace(shared, name, key, 'mismatch',
                                     _mismatch_text(name, target, named[0]))
            problem = _records_text(db, name, args)
            if problem is not None:
                return self._replace(shared, name, key, 'records', problem)
            error = _identical_failure(history, name, args)
            if error is not None:
                return self._replace(shared, name, key, 'repeat', _repeat_text(name, key, error))
            reason, text = self._judge(name, args, db, joined, history, latest, pending, key)
            if reason is None:
                self._allow(shared)
                continue
            return self._replace(shared, name, key, reason, text)
        return message

    def _judge(self, name, args, db, joined, history, latest, pending, key):
        if name == CANCEL_TOOL:
            verdict = _cancel_verdict(db, args, joined)
            if verdict == 'flown':
                return 'cancel_flown', _cancel_flown_text(args)
            if verdict == 'ineligible':
                return 'cancel_ineligible', _cancel_denied_text(db, args, joined)
        if name == CERTIFICATE_TOOL:
            verdict = _certificate_verdict(db, args, history, joined)
            if verdict is not None:
                return verdict, _certificate_text(verdict, db, args, history)
        reply = _reply(latest, name)
        same = (isinstance(pending, dict) and pending.get('tool') == name
                and pending.get('key') == key)
        if reply == 'no' and same:
            return 'refused', _refused_text(name, key)
        if reply == 'yes':
            return None, None
        return 'confirm', _confirmation_text(db, name, args, same)

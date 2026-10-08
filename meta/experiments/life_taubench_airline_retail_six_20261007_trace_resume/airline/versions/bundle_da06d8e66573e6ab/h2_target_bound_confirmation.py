"""H2: verify public business-state facts, the exact target and explicit confirmation.

Mechanism selected from TRAIN evidence (not from hidden answers).  The observed
failures cluster on actions that executed although the public policy did not
permit them or the user had not confirmed *that* action:

* cancelling reservations that were booked more than 24h earlier, are not
  business, have no insurance and were not cancelled by the airline;
* changing cabin/bags while the user was still asking for a price;
* sending a goodwill certificate the policy does not allow;
* and, after the caller redirected the conversation to another reservation,
  executing the stale action anyway because an affirmative sentence anywhere in
  the newest message counted as confirmation for whatever call was prepared.

This hook replaces such a pending action with an assistant message (no
environment action is dispatched).  A write is allowed through only when the
caller's newest message is a confirmation AND does not point at a different
reservation than the one the call targets; otherwise the prepared action is
replaced by a message that states the failing policy condition, lists the exact
action details and asks for the explicit "yes", or - when the caller's newest
message names another reservation - says that the prepared action was dropped so
the wrong booking cannot be touched.  Every verdict the bound business DB cannot
establish stays permissive, and a failed verification never changes the action.

Shared state: this layer owns `h2_pending` (a write waiting for the caller's yes)
and `h2_block` (a write replaced before execution); both are cleared or rewritten
on every gate decision.  H4 reads them for next-step guidance and clears
`h2_pending` when the caller redirects to a different reservation.
"""
from datetime import datetime, timedelta
from json import loads
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
FLOWN_STATES = ('landed', 'flying')

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
NEGATION = compile(
    r"\b(?:do not|don'?t|no longer|not going to|won'?t|never mind|nevermind|"
    r"changed my mind|hold (?:on|off)|not yet)\b", IGNORECASE)
HEALTH = compile(
    r"\b(?:sick|ill|illness|unwell|flu|fever|covid|health|medical|doctor|hospital|"
    r"surgery|injur\w*|migraine|nauseous|diagnos\w*|emergency surgery)\b", IGNORECASE)
WEATHER = compile(
    r"\b(?:weather|storm\w*|snow\w*|hurricane|tornado\w*|flood\w*|blizzard|typhoon|"
    r"grounded)\b", IGNORECASE)
UNCOVERED = compile(
    r"\b(?:change of plans?|plans? (?:have |has )?changed|changed my plans?|birthday|"
    r"wedding|anniversary|no longer need|don'?t need|not going|celebrat\w*|"
    r"vacation plans?)\b", IGNORECASE)
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


def _confirmed(text):
    if not isinstance(text, str) or not text.strip():
        return False
    if CONFIRM_START.search(text):
        return True
    if NEGATION.search(text) and not CONFIRM_START.search(text):
        return False
    return bool(CONFIRM_ANY.search(text))


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


def _confirmation_text(db, name, args):
    rid = args.get('reservation_id')
    record = _reservation(db, rid) if isinstance(rid, str) else None
    if name == 'cancel_reservation':
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
    elif name == 'book_reservation':
        passengers = args.get('passengers')
        count = len(passengers) if isinstance(passengers, list) else 'the listed number of'
        body = ('book a %s flight %s to %s in %s cabin for %s passenger(s): flights %s; payment %s; '
                '%s total checked bags (%s non-free); insurance %s'
                % (args.get('flight_type'), args.get('origin'), args.get('destination'),
                   args.get('cabin'), count, _arg_legs(args.get('flights')),
                   _arg_payments(args.get('payment_methods')), args.get('total_baggages'),
                   args.get('nonfree_baggages'), args.get('insurance')))
    elif name == 'send_certificate':
        body = 'send a $%s certificate to user %s' % (args.get('amount'), args.get('user_id'))
    else:
        body = name
    return ('Before I make any change, here are the exact details: %s. Nothing has been changed '
            'yet - please reply with an explicit "yes" and I will proceed with exactly that.'
            % body)


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

    def _remember(self, shared, name, identity, reason):
        if shared is None:
            return
        shared['h2_pending'] = None
        shared['h2_block'] = {'tool': name, 'key': identity, 'reason': reason}

    def _gate(self, message, history, writes):
        db = _business_db(self.env)
        texts = _user_texts(history)
        joined = '\n'.join(texts)
        latest = texts[-1] if texts else ''
        confirmed = _confirmed(latest)
        shared = self.state if isinstance(self.state, dict) else None
        named = _named_reservations(db, latest)
        for name, args in writes:
            identity = _identity(name, args)
            target = args.get('reservation_id')
            if (isinstance(target, str) and named and target not in named
                    and _reservation(db, target) is not None):
                self._remember(shared, name, identity, 'the confirmation named another reservation')
                return {'role': 'assistant', 'content': _mismatch_text(name, target, named[0])}
            reason, text = self._judge(name, args, db, joined, history, confirmed)
            if reason is None:
                if shared is not None:
                    shared['h2_pending'] = None
                continue
            if shared is not None:
                if reason == 'confirm':
                    shared['h2_pending'] = {'tool': name, 'key': identity}
                    shared['h2_block'] = None
                else:
                    shared['h2_pending'] = None
                    shared['h2_block'] = {'tool': name, 'key': identity, 'reason': reason}
            return {'role': 'assistant', 'content': text}
        return message

    def _judge(self, name, args, db, text, history, confirmed):
        if name == CANCEL_TOOL:
            verdict = _cancel_verdict(db, args, text)
            if verdict == 'flown':
                return 'cancel_flown', _cancel_flown_text(args)
            if verdict == 'ineligible':
                return 'cancel_ineligible', _cancel_denied_text(db, args, text)
        if name == CERTIFICATE_TOOL:
            verdict = _certificate_verdict(db, args, history, text)
            if verdict is not None:
                return verdict, _certificate_text(verdict, db, args, history)
        if not confirmed:
            return 'confirm', _confirmation_text(db, name, args)
        return None, None

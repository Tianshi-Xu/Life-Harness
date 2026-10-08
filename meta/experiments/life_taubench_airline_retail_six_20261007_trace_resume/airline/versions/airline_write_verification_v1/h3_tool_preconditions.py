"""H3: carry the write preconditions inside the model-visible tool descriptions.

Only descriptions change; names, schemas, order and dispatch stay fixed. The
observed TRAIN failures are mostly unauthorised or ineligible writes, so each
mutating tool is annotated with the condition the public policy attaches to it.
This is a reminder that travels with every solver call, complementing the H2 gate
and the H4 state card.
"""
TARGET_LAYER = 'h3'

PRECONDITIONS = {
    'book_reservation': (' Only call after listing the full itinerary, passengers, payment methods '
                         'and total price and receiving the user explicit confirmation.'),
    'cancel_reservation': (' Only call after the user confirms, and only when the booking is within '
                           'the last 24 hours, the airline cancelled the flight, the cabin is '
                           'business, or there is travel insurance plus a health or weather reason.'),
    'update_reservation_flights': (' Only call after listing the exact new flights, dates, cabin, '
                                   'passengers and payment method and receiving the user explicit '
                                   'confirmation; cabin must stay the same across every segment and '
                                   'basic-economy flights cannot be changed.'),
    'update_reservation_baggages': (' Only call after listing the updated total and non-free bag '
                                    'counts, the $50 per extra bag charge and the payment method, '
                                    'and after the user confirms; bags can be added, never removed.'),
    'update_reservation_passengers': (' Only call after listing every passenger (all names and dates '
                                      'of birth) and receiving the user explicit confirmation; the '
                                      'number of passengers cannot change.'),
    'send_certificate': (' Only call when the user explicitly asks for compensation and the policy '
                         'allows it (silver or gold member, travel insurance, or business cabin), '
                         'with the confirmation the policy requires.'),
}


class Harness:
    def h3(self, tools):
        if not isinstance(tools, list):
            return tools
        patched = []
        for tool in tools:
            if not isinstance(tool, dict):
                patched.append(tool)
                continue
            function = tool.get('function')
            note = PRECONDITIONS.get(function.get('name')) if isinstance(function, dict) else None
            if not note:
                patched.append(tool)
                continue
            item = dict(tool)
            copied = dict(function)
            copied['description'] = str(copied.get('description') or '') + note
            item['function'] = copied
            patched.append(item)
        return patched

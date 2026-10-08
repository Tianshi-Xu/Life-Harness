"""H5: cold-start operating protocol extracted from the public airline policy.

Payload-only: the text is appended to the first solver message once per episode.
It restates the public policy's decision rules that the observed TRAIN failures
broke - confirmation before every booking-database write, the four cancellation
conditions, the compensation conditions and amounts, and the membership-based
checked-bag allowance - so the first action of an episode already follows them.
Only the public policy text and the caller's messages are used; no hidden answer,
evaluation state or private per-task artefact is referenced.
"""
from re import IGNORECASE, compile

TARGET_LAYER = 'h5'

AIRLINE = compile(r"Airline Agent Policy|airline agent", IGNORECASE)
PROTOCOL = (
    "Airline operating protocol (from the <policy> above - apply it exactly):\n"
    "1. One tool call per turn. Never state that a reservation was updated, "
    "cancelled, booked or compensated before its tool result confirms it; quote "
    "amounts and identifiers from the records.\n"
    "2. Before ANY call that changes the booking database (book, cancel, flights, "
    "baggages, passengers, certificate): first list the exact details - reservation "
    "id, flight numbers and dates, cabin, passengers, payment id, refund or charge - "
    "and obtain the user's explicit 'yes'. A request to change something is not that "
    "confirmation, and a price question must be answered before any change is made.\n"
    "3. Cancellation is allowed only if the booking was made within the last 24 "
    "hours, the airline cancelled the flight, the cabin is business, or the user has "
    "travel insurance and the reason is health or weather. Otherwise deny and explain "
    "the condition that fails; if part of the itinerary has already flown, transfer to "
    "a human agent.\n"
    "4. Compensation certificates: only when the user explicitly asks AND the booking "
    "user is silver/gold, or the reservation has travel insurance, or it is business "
    "class. A delayed-flight complaint qualifies only if the reservation is also "
    "changed or cancelled ($50 x passengers); a cancelled flight in a reservation "
    "qualifies at $100 x passengers. Never offer compensation otherwise.\n"
    "5. Free checked bags per passenger are 0/1/2 for regular, 1/2/3 for silver and "
    "2/3/4 for gold members over basic economy/economy/business; each extra bag is "
    "$50. Answer bag and membership questions from the user profile records, not from "
    "what the caller claims.\n"
    "6. Transfer to a human agent only when the request is outside the policy or the "
    "user asks for one, and look the reservation up first."
)


def _first_content(messages):
    if not isinstance(messages, list):
        return ''
    for item in messages:
        if isinstance(item, dict) and isinstance(item.get('content'), str):
            return item['content']
    return ''


class Harness:
    state = None
    context = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h5(self, messages):
        try:
            if not AIRLINE.search(_first_content(messages)):
                return []
        except Exception:
            return []
        return [PROTOCOL]

"""H5: cold-start skill for the telecom duty split and the lookup order."""
TARGET_LAYER = 'h5'

SKILL = (
    "Telecom duty split: your tools act only on the account (customer lookup, bills, lines, "
    "roaming, refueling, transfer). The phone actions listed in the support manual (status bar, "
    "network status, SIM, APN, speed test, MMS, airplane mode, mobile data, roaming, data saver, "
    "network mode, SIM reseat, APN reset, reboot) are performed by the customer, never by you: ask "
    "the customer to do one step at a time and report the result. Always identify the customer "
    "first: pass the phone number to get_customer_by_phone exactly as the customer states it, "
    "dashes included (for example 555-123-2002); a name lookup also needs the customer's date of "
    "birth. Then check the line status and the bills before concluding that the account is unknown "
    "or that the problem is on the device."
)


class Harness:
    state = None
    context = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h5(self, messages):
        if not isinstance(messages, list) or not messages:
            return []
        return [SKILL]

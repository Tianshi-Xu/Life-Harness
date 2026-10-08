"""H5: cold-start skill for the telecom duty split, lookup order and account check."""
TARGET_LAYER = 'h5'

SKILL = (
    "Telecom duty split: your tools act only on the account (customer lookup, line details, bills, "
    "data usage, roaming, refueling, line suspension, payment requests, transfer). The phone actions "
    "listed in the support manual (status bar, network status, SIM, APN, speed test, MMS, airplane "
    "mode, mobile data, roaming, data saver, network mode, SIM reseat, APN reset, reboot) are "
    "performed by the customer, never by you: ask the customer to do one step at a time and report "
    "the result. Always identify the customer first: pass the phone number to get_customer_by_phone "
    "exactly as the customer states it, dashes included (for example 555-123-2002); a name lookup "
    "also needs the customer's date of birth. Then check the account with your own tool calls before "
    "concluding that the problem is on the device: the line attached to the number the customer gave "
    "(get_details_by_id) including its status, contract end date and roaming flag, its data usage "
    "against the plan limit (get_data_usage) and the customer's bills (get_bills_for_customer). Do "
    "not call tools that are not in your tool list and do not promise a check you did not make."
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

"""H2: keep customer-only phone actions out of the environment and fix lookups.

Two rules:
* the first request for a customer phone action becomes a customer-facing
  instruction (the environment can never execute it). A repeat is vetoed through
  `_harness_control` with corrective feedback naming the account tools; the same
  correction is also carried in the message content so the veto is observable in
  the returned action while the host keeps using `_harness_control`.
* get_customer_by_phone is dispatched with the stored phone literal when the
  proposed digits uniquely match one stored customer or line number.

Clean account-tool actions and unknown or ambiguous numbers pass through
byte-identical.
"""
from copy import deepcopy
from json import dumps, loads

TARGET_LAYER = 'h2'

LEAD = 'Please do this on your phone and tell me the result: '

REREQUEST = ("I already asked you to do that step. Please confirm the result you see now, and tell "
             "me if anything changed.")

# Phone actions that the public support manual assigns to the customer.
CUSTOMER_STEPS = {
    'check_status_bar':
        "look at the top of your phone screen (the status bar) and report the icons you see, "
        "including Airplane Mode, signal bars, network type, mobile data, data saver, Wi-Fi, VPN "
        "and battery",
    'check_network_status':
        "open your phone's network status screen (or read the status bar) and report Airplane "
        "Mode, signal strength, network type (2G/3G/4G/5G), mobile data on or off, and data "
        "roaming on or off",
    'check_network_mode_preference':
        "open Settings and report your phone's preferred network mode (2G, 3G, 4G or 5G)",
    'check_sim_status':
        "open your phone's settings and report whether the SIM status is Active, Missing or Locked",
    'check_data_restriction_status':
        "open Settings and report whether Data Saver mode is on and whether background data is "
        "restricted",
    'check_apn_settings':
        "open Settings and report the APN name and the MMSC URL shown there",
    'check_wifi_status':
        "open Settings and report whether Wi-Fi is on and which network you are connected to, "
        "if any",
    'check_wifi_calling_status':
        "open Settings and report whether Wi-Fi Calling is on",
    'check_vpn_status':
        "open Settings and report whether a VPN is connected and what performance it shows",
    'check_installed_apps':
        "open Settings and report the apps installed on your phone",
    'check_app_status':
        "open Settings and report the permissions and background data settings of the app in "
        "question",
    'check_app_permissions':
        "open Settings and report the permissions currently granted to your messaging app "
        "(storage and SMS)",
    'run_speed_test':
        "run a speed test on your phone and report the download speed it shows",
    'can_send_mms':
        "try to send a picture message (MMS) and report whether your messaging app can send it",
    'toggle_airplane_mode':
        "turn Airplane Mode off (or on) from your phone's quick settings",
    'set_network_mode_preference':
        "set your preferred network mode to 4G/5G preferred in Settings",
    'reseat_sim_card':
        "remove your SIM card, insert it again properly and report what the phone says",
    'toggle_data':
        "turn mobile data on (or off) in your phone's settings",
    'toggle_roaming':
        "turn Data Roaming on (or off) in your phone's settings",
    'toggle_data_saver_mode':
        "turn Data Saver mode on (or off) in your phone's settings",
    'set_apn_settings':
        "open Settings and set the APN settings exactly as I describe",
    'reset_apn_settings':
        "reset your APN settings to the default values in Settings, then restart your phone so "
        "they apply",
    'toggle_wifi':
        "turn Wi-Fi on (or off) in your phone's settings",
    'toggle_wifi_calling':
        "turn Wi-Fi Calling off (or on) in your phone's settings",
    'connect_vpn':
        "connect your VPN in your phone's settings",
    'disconnect_vpn':
        "disconnect your VPN in your phone's settings",
    'grant_app_permission':
        "grant the requested permission to your messaging app in Settings",
    'reboot_device':
        "restart your phone now and report what the network shows when it comes back on",
    'check_payment_request':
        "open your payment requests and report what you see",
    'make_payment':
        "complete the payment for the bill from your payment requests",
}

REPEAT = (
    "Do not call {names} again: the customer already performed or was asked for that step and "
    "their report is in the conversation. Phone actions belong to the customer, not to you. Use "
    "your own account tools instead, for example get_customer_by_phone, get_customer_by_name, "
    "get_bills_for_customer, get_details_by_id, get_data_usage, send_payment_request, resume_line, "
    "refuel_data, enable_roaming or transfer_to_human_agents."
)


def _digits(value):
    return ''.join(part for part in value if part.isdigit())


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

    def h2(self, message, history):
        action = deepcopy(message)
        self.counts()
        calls = action.get('tool_calls') or []
        customer = [call for call in calls if self.customer_step(call)]
        if customer:
            rest = [call for call in calls if not self.customer_step(call)]
            if rest:
                action['tool_calls'] = rest
                return self.repair_phone(action)
            repeats = [self.customer_step(call) for call in customer
                       if self.already_requested(self.customer_step(call))]
            if repeats:
                feedback = REPEAT.format(names=', '.join(repeats))
                action['content'] = REREQUEST
                action['_harness_control'] = {
                    'action': 'suppress_tool',
                    'messages': [{'role': 'user', 'content': feedback}],
                }
                return action
            name = self.customer_step(customer[0])
            self.record_request(name)
            return {'role': 'assistant', 'content': LEAD + CUSTOMER_STEPS[name] + '.'}
        return self.repair_phone(action)

    def customer_step(self, call):
        """The phone-action name of a call, or '' when the call is not one."""
        if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
            return ''
        name = call['function'].get('name')
        if not isinstance(name, str) or name not in CUSTOMER_STEPS:
            return ''
        available = self.state.get('agent_tool_names') if isinstance(self.state, dict) else None
        if isinstance(available, list) and name in available:
            return ''
        return name

    def requested(self):
        value = self.state.get('user_side_requests') if isinstance(self.state, dict) else None
        return value if isinstance(value, dict) else None

    def counts(self):
        value = self.requested()
        if value is None and isinstance(self.state, dict):
            self.state['user_side_requests'] = {}
            value = self.state['user_side_requests']
        return value

    def already_requested(self, name):
        counts = self.requested()
        return bool(name and counts and isinstance(counts.get(name), int) and counts.get(name) > 0)

    def record_request(self, name):
        counts = self.counts()
        if counts is not None:
            counts[name] = (counts[name] if isinstance(counts.get(name), int) else 0) + 1

    def repair_phone(self, action):
        for call in action.get('tool_calls') or []:
            if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                continue
            function = call['function']
            if function.get('name') != 'get_customer_by_phone':
                continue
            arguments = function.get('arguments')
            if not isinstance(arguments, str):
                continue
            try:
                parsed = loads(arguments)
            except ValueError:
                continue
            if not isinstance(parsed, dict) or not isinstance(parsed.get('phone_number'), str):
                continue
            stored = self.stored_phone(parsed['phone_number'])
            if stored:
                parsed['phone_number'] = stored
                function['arguments'] = dumps(parsed)
        return action

    def stored_phone(self, number):
        db = self.env.get('db') if isinstance(self.env, dict) else None
        if not isinstance(db, dict):
            return None
        wanted = _digits(number)
        if not wanted:
            return None
        found = {}
        for section in ('customers', 'lines'):
            rows = db.get(section)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                value = row.get('phone_number')
                if isinstance(value, str) and _digits(value) == wanted:
                    found[value] = True
        if len(found) == 1:
            stored = sorted(found)[0]
            return stored if stored != number else None
        return None

from ahe_native.environment_rpc import invoke

def get_customer_by_phone(**arguments):
    return invoke('get_customer_by_phone', arguments)

def get_customer_by_id(**arguments):
    return invoke('get_customer_by_id', arguments)

def get_customer_by_name(**arguments):
    return invoke('get_customer_by_name', arguments)

def get_details_by_id(**arguments):
    return invoke('get_details_by_id', arguments)

def suspend_line(**arguments):
    return invoke('suspend_line', arguments)

def resume_line(**arguments):
    return invoke('resume_line', arguments)

def get_bills_for_customer(**arguments):
    return invoke('get_bills_for_customer', arguments)

def send_payment_request(**arguments):
    return invoke('send_payment_request', arguments)

def get_data_usage(**arguments):
    return invoke('get_data_usage', arguments)

def enable_roaming(**arguments):
    return invoke('enable_roaming', arguments)

def disable_roaming(**arguments):
    return invoke('disable_roaming', arguments)

def transfer_to_human_agents(**arguments):
    return invoke('transfer_to_human_agents', arguments)

def refuel_data(**arguments):
    return invoke('refuel_data', arguments)

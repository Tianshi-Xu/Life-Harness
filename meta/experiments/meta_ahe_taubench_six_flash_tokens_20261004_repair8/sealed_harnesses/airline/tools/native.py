from ahe_native.environment_rpc import invoke

def book_reservation(**arguments):
    return invoke('book_reservation', arguments)

def calculate(**arguments):
    return invoke('calculate', arguments)

def cancel_reservation(**arguments):
    return invoke('cancel_reservation', arguments)

def get_reservation_details(**arguments):
    return invoke('get_reservation_details', arguments)

def get_user_details(**arguments):
    return invoke('get_user_details', arguments)

def list_all_airports(**arguments):
    return invoke('list_all_airports', arguments)

def search_direct_flight(**arguments):
    return invoke('search_direct_flight', arguments)

def search_onestop_flight(**arguments):
    return invoke('search_onestop_flight', arguments)

def send_certificate(**arguments):
    return invoke('send_certificate', arguments)

def transfer_to_human_agents(**arguments):
    return invoke('transfer_to_human_agents', arguments)

def update_reservation_baggages(**arguments):
    return invoke('update_reservation_baggages', arguments)

def update_reservation_flights(**arguments):
    return invoke('update_reservation_flights', arguments)

def update_reservation_passengers(**arguments):
    return invoke('update_reservation_passengers', arguments)

def get_flight_status(**arguments):
    return invoke('get_flight_status', arguments)

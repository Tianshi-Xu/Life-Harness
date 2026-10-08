from ahe_native.environment_rpc import invoke

def calculate(**arguments):
    return invoke('calculate', arguments)

def cancel_pending_order(**arguments):
    return invoke('cancel_pending_order', arguments)

def exchange_delivered_order_items(**arguments):
    return invoke('exchange_delivered_order_items', arguments)

def find_user_id_by_name_zip(**arguments):
    return invoke('find_user_id_by_name_zip', arguments)

def find_user_id_by_email(**arguments):
    return invoke('find_user_id_by_email', arguments)

def get_order_details(**arguments):
    return invoke('get_order_details', arguments)

def get_product_details(**arguments):
    return invoke('get_product_details', arguments)

def get_item_details(**arguments):
    return invoke('get_item_details', arguments)

def get_user_details(**arguments):
    return invoke('get_user_details', arguments)

def list_all_product_types(**arguments):
    return invoke('list_all_product_types', arguments)

def modify_pending_order_address(**arguments):
    return invoke('modify_pending_order_address', arguments)

def modify_pending_order_items(**arguments):
    return invoke('modify_pending_order_items', arguments)

def modify_pending_order_payment(**arguments):
    return invoke('modify_pending_order_payment', arguments)

def modify_user_address(**arguments):
    return invoke('modify_user_address', arguments)

def return_delivered_order_items(**arguments):
    return invoke('return_delivered_order_items', arguments)

def transfer_to_human_agents(**arguments):
    return invoke('transfer_to_human_agents', arguments)

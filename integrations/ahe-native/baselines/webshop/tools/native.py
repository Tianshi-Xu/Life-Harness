from ahe_native.environment_rpc import invoke

def search_action(**arguments):
    return invoke('search_action', arguments)

def click_action(**arguments):
    return invoke('click_action', arguments)

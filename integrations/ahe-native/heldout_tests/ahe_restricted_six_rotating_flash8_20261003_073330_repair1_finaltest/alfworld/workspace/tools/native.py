from ahe_native.environment_rpc import invoke

def take_action(**arguments):
    return invoke('take_action', arguments)

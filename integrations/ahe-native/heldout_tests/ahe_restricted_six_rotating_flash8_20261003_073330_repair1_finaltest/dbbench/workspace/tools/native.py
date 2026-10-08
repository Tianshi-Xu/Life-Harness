from ahe_native.environment_rpc import invoke

def execute_sql(**arguments):
    return invoke('execute_sql', arguments)

def commit_final_answer(**arguments):
    return invoke('commit_final_answer', arguments)

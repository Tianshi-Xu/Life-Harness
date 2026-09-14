"""No model calls: real MySQL lifecycle, isolation and original task execution."""
import asyncio
import contextlib
import sys
import json
from pathlib import Path
from types import SimpleNamespace
import yaml
from src.server.tasks.dbbench.task import DBBenchTask
from src.server.tasks.dbbench.interaction import MySQLDatabase
from src.server.tasks.dbbench.environment import DBBenchEnvironmentDelegation
from src.server.tasks.native_environment import create_controller


class ScriptedSession:
    def __init__(self):
        self.history = []
        self.turn = 0
    def inject(self, item):
        self.history.append(item)
    async def action(self):
        self.turn += 1
        name, args = ('execute_sql', {'query': 'SELECT 1'}) if self.turn == 1 else ('commit_final_answer', {'answers': ['1']})
        return SimpleNamespace(messages=[{'role': 'assistant', 'tool_calls': [{'id': f'smoke-{self.turn}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}])


async def main():
    controller = create_controller('native_mysql', DBBenchEnvironmentDelegation(''), port=13306)
    left, right = MySQLDatabase(controller), MySQLDatabase(controller)
    try:
        await asyncio.gather(left.initialize(), right.initialize())
        assert left.database != right.database
        await left.batch_execute(['CREATE TABLE transport_probe (value INT)', 'INSERT INTO transport_probe VALUES (7)'])
        assert await left.execute('SELECT value FROM transport_probe') == '[(7,)]'
        assert await right.execute('SHOW TABLES') == '[]'
        version = await left.execute('SELECT VERSION()')
    finally:
        await asyncio.gather(left.delete(), right.delete())
    assert not controller.sessions
    config = yaml.safe_load(Path('.native/configs/dbbench.yaml').read_text())
    params = {**config['default']['parameters'], **config['dbbench-std']['parameters']}
    task = DBBenchTask(**params)
    results = []
    try:
        for index in task.get_indices()[:3]:
            session = ScriptedSession()
            with contextlib.redirect_stdout(sys.stderr):
                result = await task.start_sample(index, session)
            assert 'error' not in result.status.value.lower(), result
            assert session.turn == 2, (index, session.turn, result)
            assert any(isinstance(x, dict) and x.get('role') == 'tool' and x.get('content') == '[(1,)]' for x in session.history), session.history
            results.append({'index': index, 'status': result.status.value, 'result': result.result})
        assert not task.env_controller.sessions
    finally:
        if task.env_controller_background_task:
            task.env_controller_background_task.cancel()
    print(json.dumps({'mysql_version': version, 'database_isolation': True, 'cleanup': True, 'available_standard_tasks': len(task.get_indices()), 'scripted_task_smokes': results}, indent=2, default=str))


if __name__ == '__main__':
    asyncio.run(main())

"""Reset and step actual benchmark games without any model calls."""
import json
import os
from pathlib import Path
import yaml
from src.server.tasks.alfworld.environment import AlfworldEnvWrapper

root = Path.cwd()
os.environ['ALFWORLD_DATA'] = str(root / 'data/alfworld')
config = yaml.safe_load((root / 'src/server/tasks/alfworld/configs/base_config.yaml').read_text())
wrapper = AlfworldEnvWrapper(config)
games = json.loads((root / 'data/alfworld/new_std.json').read_text())
results = []
for kind, files in games.items():
    if not files:
        continue
    path = root / 'data/alfworld' / files[0]
    env = wrapper.create_env(str(path))
    try:
        obs, info = wrapper.reset_env(env)
        actions = info['admissible_commands'][0]
        assert actions and obs[0]
        next_obs, reward, done, next_info = wrapper.step_env(env, actions[0])
        assert next_obs[0]
        results.append({'kind': kind, 'game': files[0], 'action': actions[0], 'observation_chars': len(next_obs[0])})
    finally:
        wrapper.close_env(env)
print(json.dumps({'standard_games': sum(map(len, games.values())), 'reset_step_close': results}, indent=2))

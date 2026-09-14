"""Generate native profiles without changing checked-in benchmark settings."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / '.native/configs'
out.mkdir(parents=True, exist_ok=True)
for task in ('alfworld', 'dbbench'):
    config = yaml.safe_load((ROOT / f'configs/tasks/{task}.yaml').read_text())
    params = config['default']['parameters']
    params['concurrency'] = 1
    if task == 'alfworld':
        for key in ('data_path', 'config_path', 'prompts_path'):
            params[key] = str(ROOT / params[key].removeprefix('/app/'))
    else:
        params['env_driver'] = 'native_mysql'
        params['env_options'] = {'host': '127.0.0.1', 'port': 13306}
        params['db_password'] = ''  # private dedicated loopback test instance
    (out / f'{task}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    assignment = {
        'import': str(ROOT / f'configs/assignments/{task}.yaml'),
        'definition': {'task': {'overwrite': {'parameters': {
            'controller_address': 'http://127.0.0.1:15020/api',
        }}}},
        'concurrency': {'task': {f'{task}-std': 1}, 'agent': {'qwen3-4b-instruct': 1}},
        'output': f'outputs/native/{task}/{{TIMESTAMP}}',
    }
    (out / f'assign-{task}.yaml').write_text(yaml.safe_dump(assignment, sort_keys=False))
print(out)

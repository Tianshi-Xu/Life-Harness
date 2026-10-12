"""Run the native Tau evaluation CLI with host-only loopback connections."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main(request_path):
    request = json.loads(Path(request_path).read_text())
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('native_tau_eval', root / 'TauBench/scripts/eval_harness.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module._set_nl_assertions_llm

    def judge(model, extra_args=None):
        original(model, extra_args={**(extra_args or {}), 'api_base': request['judge_base']})
    module._set_nl_assertions_llm = judge
    if request.get('resume_trials'):
        run_tasks = module.run_tasks

        def resume(config, *args, **kwargs):
            config.auto_resume = True
            return run_tasks(config, *args, **kwargs)

        module.run_tasks = resume
    sys.argv = ['eval_harness', '--domain', request['domain'], '--split', request['split'],
        '--task-id-file', request['task_id_file'], '--trials', str(request['trials']),
        '--concurrency', str(request['concurrency']), '--harness-plugin', request['candidate'],
        '--agent-llm', 'openai/qwen3-4b', '--agent-api-base', request['solver_base'],
        '--agent-max-tokens', '4096', '--user-llm', request['user_model'],
        '--user-api-base', request['user_base'], '--user-api-key-env', 'OPENAI_API_KEY',
        '--user-max-tokens', '100000' if request['domain'] == 'telecom' else '4096',
        '--save-to', request['save_to']]
    if request['domain'] != 'telecom':
        sys.argv.append('--user-enable-thinking')
    if request['domain'] == 'retail':
        sys.argv.append('--nl')
    os.environ['OPENAI_API_KEY'] = 'EMPTY'
    os.environ['AGENT_API_KEY'] = 'EMPTY'
    module.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    main(parser.parse_args().request)

"""Pause the exact predecessor only at a fully recorded API/checkpoint boundary."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/mnt/data/xts/harness/Life-Harness/meta/experiments/life_agentbench_six_flash_tokens_20261004')
PID = 2890786

def clean_boundary():
    rows = [json.loads(line) for line in (ROOT/'flash_requests.jsonl').read_text().splitlines()]
    starts, finishes = set(), set()
    for row in rows:
        identifier = row.get('request_id', row.get('id'))
        event = row.get('event', row.get('type'))
        if event == 'started': starts.add(identifier)
        if event == 'finished':
            if not row.get('accounting_complete'):
                return False
            finishes.add(identifier)
    if not starts or starts != finishes:
        return False
    for domain in ('alfworld', 'dbbench', 'webshop'):
        checkpoints = sorted((ROOT/domain/'optimizer_sessions').glob('round_*/agent_checkpoint.json'))
        if checkpoints and json.loads(checkpoints[-1].read_text()).get('inflight'):
            return False
    return True

def main():
    expected = str(ROOT/'runner.py').encode()
    command = Path(f'/proc/{PID}/cmdline').read_bytes().split(b'\0')
    if expected not in command:
        raise RuntimeError('Refuse to signal an unrelated process')
    deadline = time.monotonic() + 240
    next_report = 0
    while time.monotonic() < deadline:
        try:
            ready = clean_boundary()
        except (OSError, ValueError):
            ready = False
        if ready:
            os.kill(PID, signal.SIGSTOP)
            time.sleep(0.1)
            if clean_boundary():
                print(json.dumps({'paused_pid':PID,'api_ledger_complete':True,'checkpoint_inflight':False}), flush=True)
                return
            os.kill(PID, signal.SIGCONT)
        if time.monotonic() >= next_report:
            print(json.dumps({'phase':'waiting_for_recorded_request_boundary','pid':PID}), flush=True)
            next_report = time.monotonic()+20
        time.sleep(0.03)
    raise RuntimeError('No safe boundary yet; predecessor left running, not force-killed')

if __name__ == '__main__':
    main()

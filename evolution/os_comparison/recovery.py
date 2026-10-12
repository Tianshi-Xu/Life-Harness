"""Explicitly authorized timeout estimates; successful API receipts stay measured."""
import hashlib
import json
from pathlib import Path
import statistics
from .protocol import digest, save

AUTHORIZATION = 'OS timeout recovery policy: retain unknown provider usage and label estimates separately.'


def estimate_timeout(ledger, experiment, number):
    starts = {r['request_id']: r for r in ledger.rows if r['event'] == 'started'}
    ends = {r['request_id']: r for r in ledger.rows if r['event'] == 'finished'}
    scoped = [i for i,s in starts.items() if s['experiment'] == experiment and s['domain'] == 'os'
              and s['round'] == number and s['role'] == 'life_optimizer']
    if any(i not in ends for i in scoped):
        raise RuntimeError('wait for outstanding requests before timeout recovery')
    path = ledger.path.with_suffix('.estimates.json')
    records = json.loads(path.read_text()) if path.exists() else []
    existing = {r['request_id'] for r in records}
    measured = [i for i in scoped if ends[i].get('usage') is not None]
    unknown = [i for i in scoped if ends[i].get('usage') is None and i not in existing]
    for ident in unknown:
        start, end = starts[ident], ends[ident]
        if end.get('error_type') != 'ReadTimeout' or end.get('http_status') is not None or end.get('transport') is not None:
            raise RuntimeError('only undelivered OS Life ReadTimeout may be estimated here')
        prior = [i for i in measured if starts[i]['utc_unix'] < start['utc_unix']]
        if not prior:
            raise RuntimeError('no same-round measured reference for timeout estimate')
        last = max(prior, key=lambda i: starts[i]['utc_unix'])
        # Latency includes queueing and reasoning; this is an explicit rough estimate.
        rate = statistics.median(ends[i]['usage']['output_tokens'] /
            max(1, ends[i]['utc_unix'] - starts[i]['utc_unix']) for i in prior)
        output = min(100000, max(1, round(rate * (end['utc_unix'] - start['utc_unix']))))
        prompt = ends[last]['usage']['input_tokens']
        fingerprint = hashlib.sha256(json.dumps([start,end], sort_keys=True,separators=(',',':')).encode()).hexdigest()
        records.append({'request_id': ident, 'receipt_sha256': fingerprint, 'user_authorized': True,
            'exception_kind': 'authorized_os_read_timeout', 'authorization_quote': AUTHORIZATION,
            'basis': 'Input approximated by latest measured same-round prompt; output approximated by median previous output-token/wall-second rate multiplied by timeout duration. Queueing, reasoning and gateway billing are uncertain.',
            'reference_request_id': last, 'input_token_range': [0,262144], 'output_token_range': [0,100000],
            'usage': {'prompt_tokens':prompt,'completion_tokens':output,'total_tokens':prompt+output}})
    save(path,records)
    ledger.publish()
    return [r for r in records if r['request_id'] in scoped]


def recover_checkpoint(checkpoint, ledger, experiment, number, archive):
    checkpoint, archive = Path(checkpoint), Path(archive)
    saved = json.loads(checkpoint.read_text())
    if not saved.get('inflight'):
        raise RuntimeError('recovery requires the original interrupted checkpoint')
    records = estimate_timeout(ledger,experiment,number)
    if len(records) != 1:
        raise RuntimeError('exactly one original timeout required')
    group = [g for g in ledger.summary()['groups'].values() if g['experiment']==experiment
             and g['round']==number and g['role']=='life_optimizer'][0]
    if saved['usage']['total_tokens'] != group['total_tokens'] or saved['requests'] != group['attempts']:
        raise RuntimeError('checkpoint and original successful API receipts differ')
    sandbox = checkpoint.parent/'workspace'
    if {str(p.relative_to(sandbox)):digest(p) for p in sandbox.rglob('*') if p.is_file()} != saved['workspace_hashes']:
        raise RuntimeError('checkpoint workspace changed')
    archive.mkdir(parents=True,exist_ok=True)
    before = archive/'agent_checkpoint_before_timeout_retry.json'
    if before.exists():
        raise FileExistsError('recovery checkpoint already archived')
    before.write_bytes(checkpoint.read_bytes())
    after = {**saved,'inflight':False}
    save(checkpoint,after)
    save(archive/'timeout_retry_receipt.json', {'authorization_quote':AUTHORIZATION,
        'checkpoint_before_sha256':digest(before),'checkpoint_after_sha256':digest(checkpoint),
        'retained_successful_calls':group['known_usage_calls'],'estimated_failed_requests':records,
        'legacy_delivery_flag_correction':'Original delivered_to_client=True was initialized before headers; http_status=None and transport=None prove no upstream response was received. Raw receipt retained.',
        'messages_changed':False,'workspace_changed':False,'repairs_changed':False,
        'paid_successful_calls_replayed':0,'retry_limit_per_completion':1,'upstream_read_timeout_seconds':1800})

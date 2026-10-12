"""Host-only raw API receipts. Cache and reasoning counts are never added twice."""
import json
import threading
import time
from pathlib import Path

from common import save


def normalize(usage):
    if not isinstance(usage, dict):
        raise ValueError('API response did not provide usage')
    inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if any(type(x) is not int or x < 0 for x in (inp, out)) or inp + out == 0:
        raise ValueError('API token fields are missing, invalid, or zero placeholders')
    total = usage.get('total_tokens', inp + out)
    if type(total) is not int or total != inp + out:
        raise ValueError('API total_tokens does not equal input plus output')
    details = usage.get('prompt_tokens_details') or {}
    cached = details.get('cached_tokens', usage.get('prompt_cache_hit_tokens'))
    if cached is not None and (type(cached) is not int or not 0 <= cached <= inp):
        raise ValueError('Invalid cache token accounting')
    reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    if reasoning is not None and (type(reasoning) is not int or not 0 <= reasoning <= out):
        raise ValueError('Invalid reasoning token accounting')
    return {'input_tokens': inp, 'output_tokens': out, 'total_tokens': total,
            'cached_input_tokens': cached,
            'uncached_input_tokens': inp - cached if cached is not None else None,
            'reasoning_tokens': reasoning}


class Ledger:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()

    def record(self, domain, metadata, status, result=None, error=None):
        result = result or {}
        row = {**metadata, 'timestamp': time.time(), 'domain': domain,
               'http_status': status, 'raw_usage': result.get('usage'),
               'response_model': result.get('model'), 'response_id': result.get('id')}
        if status == 200:
            try:
                row.update(normalize(result.get('usage')), accounting='complete')
            except ValueError as exc:
                row.update(accounting='missing_or_invalid', error=str(exc))
        else:
            row.update(accounting='not_provided', error=error or 'non-success response')
        with self.lock:
            path = self.root / domain / 'flash_requests.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as file:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
                file.flush()
        return row

    def rows(self, domain=None, iteration=None):
        paths = ([self.root / domain / 'flash_requests.jsonl'] if domain
                 else sorted(self.root.glob('*/flash_requests.jsonl')))
        rows = []
        with self.lock:
            for path in paths:
                if path.exists():
                    rows += [json.loads(line) for line in path.read_text().splitlines() if line]
        return [r for r in rows if iteration is None or r.get('round') == iteration]

    def summarize(self, domain=None, iteration=None):
        rows = self.rows(domain, iteration)
        complete = [r for r in rows if r['accounting'] == 'complete']
        # Errors/timeouts must not be disguised as zero-token API calls.
        uncertain = [r for r in rows if r['accounting'] != 'complete'
                     and r['http_status'] != 429]
        cache_known = [r for r in complete if r['cached_input_tokens'] is not None]
        return {'recorded_upstream_attempts': len(rows), 'successful_calls': len(complete),
                'accounting_complete': not uncertain,
                'calls_with_unknown_usage': len(uncertain),
                **{key: sum(r[key] for r in complete) for key in
                   ('input_tokens', 'output_tokens', 'total_tokens')},
                'cached_input_tokens_known': sum(r['cached_input_tokens'] for r in cache_known),
                'calls_with_cache_details': len(cache_known),
                'by_role': {role: {key: sum(r[key] for r in complete if r.get('role') == role)
                                  for key in ('input_tokens', 'output_tokens', 'total_tokens')}
                            for role in sorted({r.get('role', 'unknown') for r in rows})}}

    def report(self, domains):
        overall = self.summarize()
        body = {'budget_type': 'six optimization rounds; measured token usage, no preset token ceiling',
                'overall': overall,
                'domains': {d: {'total': self.summarize(d),
                                'rounds': {str(i): self.summarize(d, i) for i in range(1, 7)}}
                            for d in domains}}
        save(self.root / 'TOKEN_USAGE.json', body)
        return body

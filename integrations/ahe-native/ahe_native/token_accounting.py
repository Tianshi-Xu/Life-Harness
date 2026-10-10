"""Durable API-side usage receipts. Never save prompts, responses or secrets."""
import base64
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

ROLES = {"debugger", "evolve", "life_optimizer", "user_simulator", "judge"}


def scoped_url(url, *, experiment, domain, round, role):
    scope = dict(experiment=str(experiment), domain=str(domain), round=int(round), role=str(role))
    if round < 1 or role not in ROLES:
        raise ValueError("invalid accounting scope")
    encoded = base64.urlsafe_b64encode(json.dumps(scope, separators=(",", ":")).encode()).decode().rstrip("=")
    return url.rstrip("/") + "/accounting/" + encoded


def decode_scope(path):
    if path == "/v1/chat/completions":
        return {"experiment": "unscoped", "domain": "unknown", "round": None, "role": "unknown"}
    prefix, suffix = "/v1/accounting/", "/chat/completions"
    if not path.startswith(prefix) or not path.endswith(suffix):
        raise ValueError("unsupported route")
    encoded = path[len(prefix):-len(suffix)]
    if len(encoded) > 2048:
        raise ValueError("oversized accounting scope")
    scope = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    if (set(scope) != {"experiment", "domain", "round", "role"}
            or type(scope["round"]) is not int or scope["round"] < 1
            or scope["role"] not in ROLES
            or any(not isinstance(scope[k], str) or not scope[k] or len(scope[k]) > 200
                   or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in scope[k])
                   for k in ("experiment", "domain"))):
        raise ValueError("invalid accounting scope")
    return scope


def numeric_usage(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if "token" not in key.lower() and "cache" not in key.lower():
            continue
        if type(item) is int and item >= 0:
            result[key] = item
        elif isinstance(item, dict):
            result[key] = numeric_usage(item)
    return result


def canonical_usage(value):
    prompt = value.get("prompt_tokens", value.get("input_tokens"))
    output = value.get("completion_tokens", value.get("output_tokens"))
    if type(prompt) is not int or type(output) is not int or min(prompt, output) < 0:
        return None
    total = value.get("total_tokens", prompt + output)
    if total != prompt + output:
        return None
    return {"input_tokens": prompt, "output_tokens": output, "total_tokens": total}


class TokenLedger:
    def __init__(self, path, summary_path=None):
        self.path = Path(path)
        self.summary_path = Path(summary_path) if summary_path else self.path.with_name("TOKEN_USAGE.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []

    def append(self, row):
        with self.lock:
            with self.path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.rows.append(row)

    def start(self, scope, *, model, attempt, stream):
        ident = uuid.uuid4().hex
        self.append({"event": "started", "request_id": ident, "utc_unix": time.time(),
                     **scope, "model": model, "attempt": attempt, "stream": bool(stream)})
        return ident

    def finish(self, ident, *, status, usage=None, error_type=None, delivered=True, transport=None):
        raw = numeric_usage(usage)
        measured = canonical_usage(raw)
        row = {"event": "finished", "request_id": ident, "utc_unix": time.time(),
                     "http_status": status, "raw_usage": raw, "usage": measured,
                     "accounting_complete": measured is not None,
                     "error_type": error_type, "delivered_to_client": delivered}
        if transport is not None:
            row["transport"] = transport
        self.append(row)

    def estimates(self, starts, ends):
        """Explicit historical exceptions only; raw receipts remain unchanged."""
        path = self.path.with_suffix('.estimates.json')
        result = {}
        for row in json.loads(path.read_text()) if path.is_file() else []:
            ident = row['request_id']
            start, end = starts.get(ident), ends.get(ident)
            fingerprint = hashlib.sha256(json.dumps([start, end], sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()
            usage = canonical_usage(numeric_usage(row.get('usage')))
            allowed = (end is not None and start is not None and (
                (end.get('http_status') == 200 and start.get('role') == 'evolve') or
                (end.get('http_status') is None and end.get('error_type') == 'ReadTimeout'
                 and start.get('domain') == 'os' and start.get('role') == 'life_optimizer'
                 and row.get('exception_kind') == 'authorized_os_read_timeout'
                 and row.get('authorization_quote') and row.get('output_token_range') == [0, 100000])))
            if (ident in result or start is None or end is None or end.get('usage') is not None
                    or not allowed
                    or row.get('user_authorized') is not True or not row.get('basis')
                    or fingerprint != row.get('receipt_sha256') or usage is None or usage['total_tokens'] <= 0):
                raise ValueError('invalid or changed historical token estimate receipt')
            result[ident] = usage
        return result

    def summary(self):
        with self.lock:
            starts = {r["request_id"]: r for r in self.rows if r["event"] == "started"}
            ends = {r["request_id"]: r for r in self.rows if r["event"] == "finished"}
        estimates = self.estimates(starts, ends)
        estimated = dict(estimated_usage_calls=0, estimated_input_tokens=0,
                         estimated_output_tokens=0, estimated_total_tokens=0)
        total = dict(attempts=len(starts), known_usage_calls=0, unknown_usage_calls=0,
                     pending_calls=0, input_tokens=0, output_tokens=0, total_tokens=0,
                     rate_limit_rejections_without_usage=0, **estimated)
        groups = {}
        for ident, start in starts.items():
            key = "|".join(str(start[k]) for k in ("experiment", "domain", "round", "role"))
            group = groups.setdefault(key, {**{k:start[k] for k in ("experiment", "domain", "round", "role")},
                "attempts": 0, "known_usage_calls": 0, "unknown_usage_calls": 0, "pending_calls": 0,
                "rate_limit_rejections_without_usage": 0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, **estimated})
            group["attempts"] += 1
            end = ends.get(ident)
            field = "pending_calls" if end is None else "unknown_usage_calls" if end["usage"] is None else "known_usage_calls"
            total[field] += 1
            group[field] += 1
            if end and end["usage"] is None and end["http_status"] == 429:
                total["rate_limit_rejections_without_usage"] += 1
                group["rate_limit_rejections_without_usage"] += 1
            if end and end["usage"] is not None:
                for name in ("input_tokens", "output_tokens", "total_tokens"):
                    total[name] += end["usage"][name]
                    group[name] += end["usage"][name]
            if ident in estimates:
                for value in (total, group):
                    value['estimated_usage_calls'] += 1
                    for name in ('input_tokens', 'output_tokens', 'total_tokens'):
                        value['estimated_' + name] += estimates[ident][name]
        for value in [total, *groups.values()]:
            value["accounting_complete"] = value["unknown_usage_calls"] == value["pending_calls"] == 0
            # Keep absent receipts unknown; distinguish explicit rate-limit
            # rejections from successful/ambiguous calls with missing usage.
            value["inference_accounting_complete"] = (value["pending_calls"] == 0 and
                value["unknown_usage_calls"] == value["rate_limit_rejections_without_usage"])
            value['accounting_usable_with_estimates'] = (value['pending_calls'] == 0 and
                value['unknown_usage_calls'] == value['rate_limit_rejections_without_usage'] + value['estimated_usage_calls'])
            value['budget_tokens_including_estimates'] = value['total_tokens'] + value['estimated_total_tokens']
        return {"schema_version": 1, "unit": "API input plus output tokens; cached input included once",
                "overall": total, "groups": groups}

    def publish(self):
        target = self.summary_path
        temporary = target.with_suffix(".tmp")
        with self.lock:
            value = self.summary()
            temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
            temporary.replace(target)
        return value

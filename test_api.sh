#!/usr/bin/env bash
# Minimal ModelRouter connectivity probe.
#
# Examples:
#   /mnt/workspace/xts/skill/test_ap/test_api.sh
#   TEST_ALL=1 /mnt/workspace/xts/skill/test_ap/test_api.sh
#   TEST_ALL=1 CONCURRENCY=5 /mnt/workspace/xts/skill/test_ap/test_api.sh
#   MR_HOST=https://routify.alibaba-inc.com TEST_ALL=1 /mnt/workspace/xts/skill/test_ap/test_api.sh
#   PROTOCOL=anthropic MODEL=mr.claude-opus-4-7 /mnt/workspace/xts/skill/test_ap/test_api.sh
#   PROTOCOL=responses MODEL=mr.gpt-5.5 /mnt/workspace/xts/skill/test_ap/test_api.sh

export MR_API_KEY="sk-856330ce991f4080b9cc5c2614cefcb7"

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY

MR_HOST="${MR_HOST:-https://routify-pub.alibaba-inc.com}"
PROTOCOL="$(printf '%s' "${PROTOCOL:-anthropic}" | tr '[:upper:]' '[:lower:]')"
MODEL="${MODEL:-mr.claude-opus-4-7}"
PROMPT="${PROMPT:-Reply with exactly: ok}"
STREAM="${STREAM:-false}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-10}"
MAX_TIME="${MAX_TIME:-60}"
CONCURRENCY="${CONCURRENCY:-${PARALLEL:-5}}"
STRIP_MR_PREFIX="${STRIP_MR_PREFIX:-1}"

truthy() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

normalize_model_name() {
    local value="$1"
    if truthy "$STRIP_MR_PREFIX"; then
        value="${value#mr.}"
    fi
    printf '%s' "$value"
}

infer_protocol() {
    local value
    value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    case "$value" in
        *claude*) printf '%s\n' anthropic ;;
        *gemini*) printf '%s\n' gemini ;;
        *gpt-5*|gpt-*) printf '%s\n' responses ;;
        *) printf '%s\n' openai ;;
    esac
}

if truthy "${TEST_ALL:-0}" || [ "$PROTOCOL" = "all" ]; then
    if [ -n "${MODELS:-}" ]; then
        read -r -a MODEL_LIST <<<"$(printf '%s' "$MODELS" | tr ',' ' ')"
    else
        MODEL_LIST=(
            "mr.claude-opus-4-8"
            # "mr.gemini-3-flash-preview"
            # "mr.gemini-3.1-pro-preview"
            # "mr.gpt-5.4-pro"
            # "mr.gpt-5.5"
            "mr.deepseek-v4-pro"
        )
    fi

    python3 - "$MR_API_KEY" "$MR_HOST" "$PROMPT" "$MAX_TOKENS" "$CONNECT_TIMEOUT" "$MAX_TIME" "$CONCURRENCY" "$STRIP_MR_PREFIX" "${MODEL_LIST[@]}" <<'PY'
import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

api_key, host, prompt = sys.argv[1], sys.argv[2].rstrip('/'), sys.argv[3]
max_tokens = int(sys.argv[4])
connect_timeout = int(sys.argv[5])
max_time = int(sys.argv[6])
concurrency = int(sys.argv[7])
strip_mr = sys.argv[8].lower() in {'1', 'true', 'yes', 'on'}
models = sys.argv[9:]


def normalize(model: str) -> str:
    return model[3:] if strip_mr and model.startswith('mr.') else model


def infer_protocol(model: str) -> str:
    lower = model.lower()
    if 'claude' in lower:
        return 'anthropic'
    if 'gemini' in lower:
        return 'gemini'
    if lower.startswith('gpt-') or 'gpt-5' in lower:
        return 'responses'
    return 'openai'


def build_request(protocol: str, model: str):
    if protocol == 'anthropic':
        url = f'{host}/protocol/anthropic/v1/messages'
        payload = {
            'model': model,
            'max_tokens': max_tokens,
            'stream': False,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    elif protocol == 'gemini':
        url = f'{host}/protocol/vertex/v1beta/models/{model}:generateContent'
        payload = {
            'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'maxOutputTokens': max_tokens},
        }
        headers = {'x-goog-api-key': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    elif protocol == 'responses':
        url = f'{host}/protocol/openai/v1/responses'
        payload = {
            'model': model,
            'stream': False,
            'max_output_tokens': max_tokens,
            'input': [{'role': 'user', 'content': prompt}],
        }
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    else:
        url = f'{host}/protocol/openai/v1/chat/completions'
        payload = {
            'model': model,
            'stream': False,
            'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    return url, json.dumps(payload, ensure_ascii=False).encode(), headers


def parse_result(http_status: int | None, body: bytes, transport_error: str = ''):
    text = body.decode('utf-8', errors='replace') if body else ''
    if transport_error:
        return False, transport_error
    try:
        obj = json.loads(text)
    except Exception:
        sample = 'non_json_response'
        if text.strip():
            sample += ': ' + text.strip()[:120].replace('\n', ' ')
        return False, sample
    if isinstance(obj, dict):
        rr = obj.get('routify_response')
        if isinstance(rr, dict) and rr.get('success') is False:
            return False, str(rr.get('error_message') or rr.get('status') or 'routify_error')
        if obj.get('error'):
            return False, str(obj.get('error'))[:200]
    if http_status is not None and http_status >= 400:
        return False, f'http_{http_status}'
    return True, ''


def one_request(protocol: str, model: str):
    url, data, headers = build_request(protocol, model)
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    started = time.time()
    status = None
    body = b''
    error = ''
    try:
        with urllib.request.urlopen(req, timeout=max_time) as resp:
            status = resp.status
            body = resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            body = exc.read()
        except Exception:
            body = b''
    except Exception as exc:
        error = f'{exc.__class__.__name__}: {exc}'
    elapsed = time.time() - started
    ok, err = parse_result(status, body, error)
    return {'ok': ok, 'elapsed': elapsed, 'status': status, 'error': err, 'url': url}

print('==========================================')
print('ModelRouter batch connectivity probe')
print(f'  count       : {len(models)}')
print(f'  concurrency : {concurrency} per model')
print(f'  mr_host     : {host}')
print(f'  max_tokens  : {max_tokens}')
print('==========================================')

for original in models:
    model = normalize(original)
    protocol = infer_protocol(model)
    print('\n==========================================')
    print(f'Testing  : {original}')
    print(f'Protocol : {protocol}')
    print(f'Name     : {model}')
    print('==========================================')
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda _: one_request(protocol, model), range(concurrency)))
    oks = [r for r in results if r['ok']]
    errs = [r for r in results if not r['ok']]
    lat = [r['elapsed'] for r in oks]
    print(f'Reqs     : {len(results)} (ok={len(oks)}, error={len(errs)})')
    if lat:
        print(f'Min      : {min(lat):.3f}s')
        print(f'Max      : {max(lat):.3f}s')
        print(f'Avg      : {statistics.mean(lat):.3f}s')
        print(f'Median   : {statistics.median(lat):.3f}s')
    for i, r in enumerate(results, 1):
        suffix = f" error={r['error']}" if r['error'] else ''
        status = r['status'] if r['status'] is not None else '-'
        print(f"  req {i}: {r['elapsed']:.3f}s http={status} [{'ok' if r['ok'] else 'error'}]{suffix}")
    if errs:
        print(f"First error: {errs[0]['error']}")
PY
    exit 0
fi

MODEL_NORM="$(normalize_model_name "$MODEL")"
if [ "$PROTOCOL" = "auto" ]; then
    PROTOCOL="$(infer_protocol "$MODEL_NORM")"
fi

python3 - "$MR_API_KEY" "$MR_HOST" "$PROTOCOL" "$MODEL_NORM" "$PROMPT" "$STREAM" "$MAX_TOKENS" "$CONNECT_TIMEOUT" "$MAX_TIME" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

api_key, host, protocol, model, prompt, stream_s, max_tokens_s, connect_timeout_s, max_time_s = sys.argv[1:]
host = host.rstrip('/')
max_tokens = int(max_tokens_s)
max_time = int(max_time_s)
stream = stream_s.lower() in {'1', 'true', 'yes', 'on'}

if protocol in {'anthropic', 'claude'}:
    protocol = 'anthropic'
    url = f'{host}/protocol/anthropic/v1/messages'
    payload = {'model': model, 'max_tokens': max_tokens, 'stream': stream, 'messages': [{'role': 'user', 'content': prompt}]}
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
elif protocol in {'gemini', 'vertex'}:
    protocol = 'gemini'
    action = 'streamGenerateContent?alt=sse' if stream else 'generateContent'
    url = f'{host}/protocol/vertex/v1beta/models/{model}:{action}'
    payload = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'generationConfig': {'maxOutputTokens': max_tokens}}
    headers = {'x-goog-api-key': f'Bearer {api_key}', 'Content-Type': 'application/json'}
elif protocol in {'responses', 'openai-responses'}:
    protocol = 'responses'
    url = f'{host}/protocol/openai/v1/responses'
    payload = {'model': model, 'stream': stream, 'max_output_tokens': max_tokens, 'input': [{'role': 'user', 'content': prompt}]}
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
elif protocol in {'openai', 'chat', 'chat-completions'}:
    protocol = 'openai'
    url = f'{host}/protocol/openai/v1/chat/completions'
    payload = {'model': model, 'stream': stream, 'max_tokens': max_tokens, 'messages': [{'role': 'user', 'content': prompt}]}
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
else:
    raise SystemExit(f'Unsupported PROTOCOL={protocol}')

print('==========================================')
print('ModelRouter connectivity probe')
print(f'  protocol : {protocol}')
print(f'  model    : {model}')
print(f'  url      : {url}')
print(f'  stream   : {str(stream).lower()}')
print(f'  prompt   : {prompt}')
print('==========================================')

req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers, method='POST')
started = time.time()
status = None
body = b''
try:
    with urllib.request.urlopen(req, timeout=max_time) as resp:
        status = resp.status
        body = resp.read()
except urllib.error.HTTPError as exc:
    status = exc.code
    body = exc.read()
except Exception as exc:
    print(f'ERROR: {exc.__class__.__name__}: {exc}')
    raise SystemExit(1)
elapsed = time.time() - started
print('------------------------------------------')
print(f'  http_status : {status}')
print(f'  elapsed     : {elapsed:.3f}s')
print('------------------------------------------')
text = body.decode('utf-8', errors='replace')
try:
    print(json.dumps(json.loads(text), ensure_ascii=False, indent=2))
except Exception:
    print(text)
PY


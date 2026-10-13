"""Optional Flash token accounting; never apply the solver's context budget."""
import copy
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path

# Flash output budget. None means no local context cap;
# the ModelRouter/provider's actual context limit still applies.
CONTEXT_LIMIT = None
OUTPUT_LIMIT = 60000
TOKEN_MARGIN = 0
TOKENIZER_ROOT = Path(os.environ.get('LIFE_FLASH_TOKENIZER',
    str(Path(__file__).resolve().parents[2] / '.cache/DeepSeek-V4-Flash-tokenizer')))
TOKENIZER_REVISION = '60d8d70770c6776ff598c94bb586a859a38244f1'
TOKENIZER_HASHES = {
    'tokenizer.json': '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf',
    'tokenizer_config.json': '6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547',
    'encoding/encoding_dsv4.py': 'bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0',
}

class ContextLimit(ValueError):
    """Normal request/episode budget exhaustion, not an infrastructure outage."""

class ProposalFailure(ValueError):
    """Normal failed proposal; controller records it and retains the incumbent."""
    def __init__(self, code, detail):
        self.code = code
        super().__init__(detail)

def context_fingerprint():
    for name, expected in TOKENIZER_HASHES.items():
        actual = hashlib.sha256((TOKENIZER_ROOT/name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Pinned public tokenizer changed: ' + name)
    return {'context_limit': CONTEXT_LIMIT, 'output_limit': OUTPUT_LIMIT,
            'token_margin': TOKEN_MARGIN, 'tokenizer_repository': 'deepseek-ai/DeepSeek-V4-Flash',
            'tokenizer_revision': TOKENIZER_REVISION, 'tokenizer_sha256': TOKENIZER_HASHES}

class FlashCounter:
    def __init__(self):
        context_fingerprint()
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_ROOT/'tokenizer.json'))
        spec = importlib.util.spec_from_file_location('life_dsv4_encoding', TOKENIZER_ROOT/'encoding/encoding_dsv4.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.encode_messages = module.encode_messages

    def count(self, messages, tools):
        values = copy.deepcopy(messages)
        if not values or values[0].get('role') != 'system':
            values.insert(0, {'role': 'system', 'content': ''})
        values[0]['tools'] = copy.deepcopy(tools)
        prompt = self.encode_messages(values, thinking_mode='thinking', drop_thinking=False)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False).ids)

    def guard(self, messages, tools):
        """Compatibility accounting helper, without a local budget rejection."""
        return self.count(messages, tools)

def verify_usage(result):
    usage = result.get('usage', {})
    prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if type(prompt) is not int or type(completion) is not int or prompt < 0 or completion < 0:
        raise RuntimeError('API token usage missing or invalid; cannot audit usage')

@lru_cache(maxsize=1)
def flash_counter():
    return FlashCounter()

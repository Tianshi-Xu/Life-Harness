"""Local token counting only, no network or LLM calls; shared frozen Qwen template."""
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[3]
tokenizer = AutoTokenizer.from_pretrained(os.environ.get('LIFE_SOLVER_TOKENIZER', 'Qwen/Qwen3-4B'), local_files_only=True)
for line in sys.stdin:
    try:
        value = json.loads(line)
        tokens = tokenizer.apply_chat_template(value['messages'], tools=value['tools'], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        print(json.dumps({'length': len(tokens)}), flush=True)
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), flush=True)

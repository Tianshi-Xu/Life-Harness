"""Download and verify the exact public tokenizer required by Life evolve."""
import hashlib
import os
from pathlib import Path
import urllib.request

from evolution.life_optimizer_v25.context import TOKENIZER_ROOT, TOKENIZER_REVISION, TOKENIZER_HASHES


def main():
    for name, expected in TOKENIZER_HASHES.items():
        target = TOKENIZER_ROOT / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/{TOKENIZER_REVISION}/{name}'
        temporary = target.with_name(target.name + '.download')
        urllib.request.urlretrieve(url, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            temporary.unlink()
            raise RuntimeError('Tokenizer checksum mismatch: ' + name)
        os.replace(temporary, target)
    print(TOKENIZER_ROOT)


if __name__ == '__main__':
    main()

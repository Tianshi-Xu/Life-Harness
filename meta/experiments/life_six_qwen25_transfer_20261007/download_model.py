"""Download public official Qwen weights, verifying ModelScope SHA256 metadata."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests

DEST = Path('/mnt/data/xts/models/Qwen2.5-14B-Instruct')
API = 'https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-14B-Instruct/repo'


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(row):
    name = row['Path']
    if Path(name).name != name:
        raise ValueError('Unexpected repository path')
    target = DEST / name
    if target.exists() and target.stat().st_size == row['Size'] and checksum(target) == row['Sha256']:
        print(json.dumps({'file': name, 'status': 'verified_existing'}), flush=True)
        return
    part = target.with_name(name + '.download')
    for attempt in range(3):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset < row['Size']:
                url = API + '?Revision=master&FilePath=' + quote(name)
                with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                                  stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Invalid download range')
                    with part.open('ab' if append else 'wb') as stream:
                        for block in response.iter_content(8 * 1024 * 1024):
                            stream.write(block)
            if part.stat().st_size != row['Size'] or checksum(part) != row['Sha256']:
                raise ValueError('Official file size or SHA256 mismatch: ' + name)
            part.replace(target)
            print(json.dumps({'file': name, 'status': 'downloaded_verified', 'bytes': row['Size']}), flush=True)
            return
        except Exception as exc:
            print(json.dumps({'file': name, 'attempt': attempt + 1, 'error_type': type(exc).__name__}), flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


if __name__ == '__main__':
    response = requests.get(API + '/files?Revision=master&Recursive=true', timeout=30)
    response.raise_for_status()
    value = response.json()
    if not value['Success']:
        raise RuntimeError('Official file metadata unavailable')
    files = [r for r in value['Data']['Files'] if r['Type'] == 'blob' and r['Path'] != '.gitattributes']
    DEST.mkdir(parents=True, exist_ok=True)
    receipt = Path(__file__).with_name('MODEL_DOWNLOAD_RECEIPT.json')
    receipt.write_text(json.dumps({'repository': 'Qwen/Qwen2.5-14B-Instruct', 'provider': 'ModelScope',
                                  'files': files}, indent=2) + '\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, files))
    index = json.loads((DEST / 'model.safetensors.index.json').read_text())
    if len(set(index['weight_map'].values())) != 8:
        raise ValueError('Unexpected standard 14B weight inventory')
    print(json.dumps({'status': 'complete', 'verified_files': len(files)}), flush=True)

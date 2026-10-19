"""Download the pinned Windows CUDA wheel from the official canonical host."""
import hashlib
import html
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
WHEEL_NAME = 'torch-2.11.0+cu128-cp312-cp312-win_amd64.whl'
FOLDER = PROJECT / '.tmp' / 'cuda_wheels'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(1024*1024), b''):
            value.update(piece)
    return value.hexdigest()


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen('https://download.pytorch.org/whl/cu128/torch/', timeout=60) as response:
        index = response.read().decode()
    links = [html.unescape(link) for link in re.findall(r'href="([^"]+)"', index)]
    matches = [link for link in links if urllib.parse.unquote(urllib.parse.urlsplit(link).path).endswith('/'+WHEEL_NAME)]
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one official pinned wheel entry, got {len(matches)}')
    original = urllib.parse.urlsplit(matches[0])
    expected = urllib.parse.parse_qs(original.fragment).get('sha256', [''])[0]
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise RuntimeError('Official index did not provide a SHA256')
    # The package index currently uses a mirror host which may return 403.
    # Same official wheel path on canonical pytorch.org is verified with its index hash.
    url = urllib.parse.urlunsplit(('https', 'download.pytorch.org', original.path, '', ''))
    destination = FOLDER / WHEEL_NAME
    if destination.is_file() and digest(destination) == expected:
        print(f'Already verified: {destination}', flush=True)
        return
    temporary = destination.with_suffix('.whl.part')
    last = 0
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
        total = int(response.headers.get('Content-Length') or 0)
        downloaded = 0
        while piece := response.read(4*1024*1024):
            out.write(piece)
            downloaded += len(piece)
            if time.monotonic()-last >= 15:
                print(f'CUDA wheel: {downloaded/1e6:.0f} / {total/1e6:.0f} MB', flush=True)
                last = time.monotonic()
    if digest(temporary) != expected:
        raise RuntimeError('Wheel integrity failure; refusing installation')
    temporary.replace(destination)
    print(f'Official SHA256 verified: {destination}', flush=True)


if __name__ == '__main__':
    main()

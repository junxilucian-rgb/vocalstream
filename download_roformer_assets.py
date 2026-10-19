"""Download a fixed public vocal checkpoint, validate SHA256, preserve provenance."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT / 'model_cache' / 'mel_roformer'
REVISION = 'ac9b0614ab3cd7f77219e18ba494dfd93956c348'
EXPECTED_SHA = '87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e'
URL = f'https://huggingface.co/KimberleyJSN/melbandroformer/resolve/{REVISION}/MelBandRoformer.ckpt'
CONFIG_REVISION = '25f44ffb55ee3c301281bba21b2d6d311cb69ae2'
CONFIG_URL = f'https://raw.githubusercontent.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/{CONFIG_REVISION}/configs/config_vocals_mel_band_roformer.yaml'


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_DIR / 'MelBandRoformer.ckpt'
    if destination.is_file() and file_hash(destination) == EXPECTED_SHA:
        print('Checkpoint already present and SHA256 verified', flush=True)
    else:
        temporary = destination.with_suffix('.ckpt.part')
        request = urllib.request.Request(URL, headers={'User-Agent': 'VocalStream-academic-project/0.1'})
        started = time.monotonic()
        last_notice = 0
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as out:
            total = int(response.headers.get('Content-Length') or 0)
            downloaded = 0
            while block := response.read(1024*1024):
                out.write(block)
                downloaded += len(block)
                if time.monotonic()-last_notice >= 15:
                    print(f'Checkpoint: {downloaded/1e6:.1f} MB' + (f' / {total/1e6:.1f} MB' if total else ''), flush=True)
                    last_notice = time.monotonic()
        actual = file_hash(temporary)
        if actual != EXPECTED_SHA:
            raise RuntimeError(f'Checkpoint SHA256 mismatch: {actual}; partial file retained for diagnosis')
        temporary.replace(destination)
        print(f'Checkpoint SHA256 verified; download {time.monotonic()-started:.1f}s', flush=True)
    metadata = {'model': 'KimberleyJensen MelBandRoformer vocal checkpoint',
                'url': URL, 'hf_revision': REVISION, 'sha256': EXPECTED_SHA,
                'model_card': 'https://huggingface.co/KimberleyJSN/melbandroformer',
                'publisher_declared_weight_license': 'MIT',
                'training_data': 'extra community training data; full provenance not disclosed',
                'exact_paper_checkpoint': False, 'causal': False}
    request = urllib.request.Request(CONFIG_URL, headers={'User-Agent': 'VocalStream-academic-project/0.1'})
    with urllib.request.urlopen(request, timeout=60) as response:
        config_bytes = response.read()
    (MODEL_DIR/'config.yaml').write_bytes(config_bytes)
    metadata['config_url'] = CONFIG_URL
    metadata['config_sha256'] = hashlib.sha256(config_bytes).hexdigest()
    metadata['implementation'] = 'MIT lucidrains BS-RoFormer==0.3.9'
    metadata['implementation_commit'] = '93a07dda7867d4acd8f5bd49ae3f33e1fcbfd8cf'
    (MODEL_DIR/'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

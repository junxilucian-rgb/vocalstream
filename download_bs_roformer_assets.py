"""Fetch the professor-recommended BS architecture's public HQ checkpoint.

Weights are ZFTurbo's community release, not the original paper's weights.
Asset hashes follow the professor-linked OpenMIRLab registry; source is pinned
to the checkpoint publisher's release commit and imported in a local namespace.
"""
import ast
import hashlib
import json
from pathlib import Path
import time
import urllib.request

PROJECT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT / 'model_cache' / 'bs_roformer_hq'
VENDOR = PROJECT / 'vocalstream' / 'vendor' / 'zfturbo_bsroformer'
BASE = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/'
SOURCE_COMMIT = 'aef04b2e52fb3beaf25e333199f5a7236e628e7b'
SOURCE_FILES = {
    'bs_roformer.py': ('models/bs_roformer/bs_roformer.py',
                      '93408c7254c60c48e47be0657a64745065396b0b1c6da4e02c75aca57eb62bf3',
                      '7c4b9aaf2075011621a87cd78368b46db1065133f11baf7cd27839bc44733002'),
    'attend.py': ('models/bs_roformer/attend.py',
                  '0459d799ade55541df2994b0becf7aec12214491360c5a06e346f6d615eaed15',
                  '0459d799ade55541df2994b0becf7aec12214491360c5a06e346f6d615eaed15'),
    'LICENSE': ('LICENSE',
                '3282dc057695ef5b9a64909a7092ca40b2c292c232580fc6ace6e5d665cc0207',
                '3282dc057695ef5b9a64909a7092ca40b2c292c232580fc6ace6e5d665cc0207'),
}
ASSETS = {
    'model.ckpt': ('model_bs_roformer_ep_17_sdr_9.6568.ckpt', '3e9daecd70aaed5b5a0d1f861cc4d77eaa45afb3fc6301b1cf32c1be0f5868fb'),
    'config.yaml': ('config_bs_roformer_384_8_2_485100.yaml', 'd8afb980318d0c08b9c2e24a7adc00d4f3150320c127a7e4de861800d1321939')
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'VocalStream-research'})
    return urllib.request.urlopen(request, timeout=120)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    commit = SOURCE_COMMIT
    VENDOR.mkdir(parents=True, exist_ok=True)
    provenance = {'architecture': 'BSRoformer', 'checkpoint_release': 'v1.0.12',
                  'publisher': 'ZFTurbo/Music-Source-Separation-Training', 'source_commit': commit,
                  'training_data_statement': 'Publisher says trained on MUSDB18HQ; exact training/validation list not supplied',
                  'exact_original_paper_checkpoint': False, 'weight_license': 'not specified in the release',
                  'registry_source': 'https://github.com/openmirlab/bs-roformer-infer', 'source_files': {}, 'assets': {}}
    for name, (relative, upstream_hash, local_hash) in SOURCE_FILES.items():
        url = f'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{commit}/{relative}'
        destination = VENDOR/name
        if destination.exists():
            if sha(destination) != local_hash:
                raise ValueError(f'Bundled publisher source changed: {name}; restore it before downloading assets')
        else:
            with get(url) as response:
                original = response.read()
            if hashlib.sha256(original).hexdigest() != upstream_hash:
                raise ValueError(f'Pinned upstream source hash mismatch: {name}')
            local = original
            if name == 'bs_roformer.py':
                old = b'from models.bs_roformer.attend import Attend'
                if original.count(old) != 1:
                    raise ValueError('Unexpected upstream import layout')
                local = original.replace(old, b'from .attend import Attend')
            if hashlib.sha256(local).hexdigest() != local_hash:
                raise ValueError(f'Pinned adapted source hash mismatch: {name}')
            if name.endswith('.py'):
                ast.parse(local.decode('utf-8'))
            destination.write_bytes(local)
        provenance['source_files'][name] = {'url': url, 'upstream_sha256': upstream_hash,
                                           'local_sha256': local_hash}
    if not (VENDOR/'__init__.py').exists():
        (VENDOR/'__init__.py').write_text('from .bs_roformer import BSRoformer\n', encoding='utf-8')
    for local_name, (remote_name, expected) in ASSETS.items():
        destination = MODEL_DIR/local_name
        url = BASE+remote_name
        if not destination.is_file() or sha(destination) != expected:
            temporary = destination.with_suffix(destination.suffix+'.part')
            notice = 0
            downloaded = 0
            with get(url) as response, temporary.open('wb') as output:
                total = int(response.headers.get('Content-Length') or 0)
                while block := response.read(1024*1024):
                    output.write(block)
                    downloaded += len(block)
                    if time.monotonic()-notice > 15:
                        print(f'{local_name}: {downloaded/1e6:.1f} / {total/1e6:.1f} MB', flush=True)
                        notice = time.monotonic()
            if sha(temporary) != expected:
                raise ValueError(f'Asset SHA256 mismatch: {local_name}')
            temporary.replace(destination)
        print(f'{local_name}: SHA256 verified', flush=True)
        provenance['assets'][local_name] = {'url': url, 'sha256': expected, 'bytes': destination.stat().st_size}
    (MODEL_DIR/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Assets ready: {MODEL_DIR}', flush=True)


if __name__ == '__main__':
    main()

"""Optional setup helper; run inside an activated Python 3.12 environment.

The README's explicit pip commands are preferred. Normal setup never rewrites
the recorded environment lock. ``--phase freeze`` records the active runtime
only when explicitly requested, into --output or results/runtime_lock.txt.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['torch', 'dependencies', 'freeze'], required=True)
    parser.add_argument('--output', type=Path, help='Freeze destination; default results/runtime_lock.txt')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('Use Python 3.12 for the recorded project environment')
    temp_folder = PROJECT / '.tmp' / 'roformer_setup'
    temp_folder.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_folder)
    for key in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[key] = str(temp_folder)
    os.environ['PIP_CACHE_DIR'] = str(PROJECT / '.tmp' / 'pip_cache')
    if args.phase == 'torch':
        subprocess.run([sys.executable, '-m', 'pip', 'install',
                        'torch==2.11.0', 'torchaudio==2.11.0',
                        '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
        return
    if args.phase == 'dependencies':
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(PROJECT)], check=True)
        return
    output = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    import torch
    # A wheel installed from disk appears as a machine-specific file:/// URL.
    # Keep the exact public version while making the environment lock portable.
    output = re.sub(r'^torch\s+@\s+.*$', 'torch==' + str(torch.__version__), output, flags=re.MULTILINE)
    import torchaudio
    output = re.sub(r'^torchaudio\s+@\s+.*$', 'torchaudio==' + str(torchaudio.__version__), output, flags=re.MULTILINE)
    if '+cu128' in str(torch.__version__):
        output = '--extra-index-url https://download.pytorch.org/whl/cu128\n' + output
    target = args.output or PROJECT / 'results' / 'runtime_lock.txt'
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output, encoding='utf-8')
    info = {'python': sys.version, 'torch': str(torch.__version__),
            'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    target.with_suffix('.environment.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'Runtime lock: {target}')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()

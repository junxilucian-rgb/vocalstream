"""Create tiny non-musical stems ONLY to check train/evaluation plumbing.

Never report its scores as MUSDB18 performance. No copyrighted audio is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def create(root, seconds=2.0, seed=17):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing fixture: {root}")
    sr = 44100
    t = np.arange(round(sr * seconds)) / sr
    rng = np.random.default_rng(seed)
    for subset, names in (("train", ["SYNTHETIC_train_a", "SYNTHETIC_valid_b"]),
                          ("test", ["SYNTHETIC_test_c"])):
        for i, name in enumerate(names):
            envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 1.1 * t)
            tone = np.sin(2 * np.pi * (240 + i * 15) * t)
            vocals = 0.12 * envelope * (tone + .3 * np.sin(2 * np.pi * 480 * t))
            bass = .10 * np.sin(2 * np.pi * 80 * t)
            other = .06 * np.sin(2 * np.pi * 1100 * t)
            drums = .01 * rng.standard_normal(len(t))
            arrays = {"vocals": vocals, "bass": bass, "other": other, "drums": drums}
            arrays = {key: np.stack([value, value * .92], axis=1).astype('float32')
                      for key, value in arrays.items()}
            arrays["mixture"] = sum(arrays.values())
            folder = root / subset / name
            folder.mkdir(parents=True, exist_ok=True)
            for key, audio in arrays.items():
                sf.write(folder / (key + '.wav'), audio, sr, subtype='FLOAT')
    metadata = {"synthetic_fixture": True, "purpose": "engineering smoke check, NOT human singing or MUSDB18 audio",
                "seed": seed, "duration": seconds, "sample_rate": sr}
    (root / 'SYNTHETIC_ONLY.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results' / 'synthetic_fixture')
    parser.add_argument('--seconds', type=float, default=2.0)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be > 0')
    print(f'SYNTHETIC ONLY: {create(args.out, args.seconds)}')


if __name__ == '__main__':
    main()

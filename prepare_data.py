"""Inspect a MUSDB18-HQ directory before training/evaluation. No downloading."""

import argparse
import json
from pathlib import Path

from vocalstream.data import discover_hq, load_segment, dataset_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "hq_manifest.json")
    args = parser.parse_args()
    metadata = dataset_info(args.root)
    report = {"dataset": metadata['kind'], "dataset_metadata": metadata, "root": str(args.root.resolve()), "subsets": {}, "warnings": []}
    for subset, expected in (("train", 100), ("test", 50)):
        tracks = discover_hq(args.root, subset)
        entries = []
        for track in tracks:
            _, segment = load_segment(track, duration=min(1.0, track.frames / track.sample_rate))
            entries.append({**track.as_dict(), "first_second_alignment_check": segment})
        report["subsets"][subset] = entries
        if len(tracks) != expected:
            report["warnings"].append(f"{subset}: {len(tracks)} tracks, full official dataset expects {expected}; treat as pilot")
    train = {t["name"] for t in report["subsets"]["train"]}
    test = {t["name"] for t in report["subsets"]["test"]}
    if train & test:
        raise ValueError(f"Track names appear in both train and test: {sorted(train & test)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {args.out}")
    for subset in ("train", "test"):
        print(f"{subset}: {len(report['subsets'][subset])} stereo tracks, 44100 Hz")
    for warning in report["warnings"]:
        print(f"PILOT: {warning}")


if __name__ == "__main__":
    main()

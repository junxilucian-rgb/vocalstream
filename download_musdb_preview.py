"""Prepare the official seven-second MUSDB18 preview as WAV stem folders.

This is a compressed preview, NOT MUSDB18-HQ and NOT a formal benchmark.
The download URL and five audio-stream IDs follow sigsep-mus-db's mus.yaml.
No musdb, stempeg or ffprobe import is needed.  Decoding uses FFmpeg, supplied
by imageio-ffmpeg by default, and soundfile checks every resulting stem.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
import zipfile

import soundfile as sf


PROJECT_DIR = Path(__file__).resolve().parent
PREVIEW_URL = (
    "https://github.com/sigsep/sigsep-mus-db/releases/download/"
    "v0.4.0/MUSDB18-7-STEMS.zip"
)
DATASET_KIND = "MUSDB18-7-compressed-preview"
STREAMS = {"mixture": 0, "drums": 1, "bass": 2, "other": 3, "vocals": 4}
SAMPLE_RATE = 44100
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_ARCHIVE_ITEMS = 1000
TRACK_MARKER = "_TRACK_INFO.json"
RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def contained_path(root: Path, path: Path) -> Path:
    """Reject paths/symlinks escaping the selected output directory."""
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes output directory: {path}") from error
    # Preserve the original final component so subsequent .is_symlink() checks
    # can reject an existing stem/metadata link even when its target is inside
    # the output root.  Resolution above is solely the containment check.
    return path.absolute()


def atomic_json(path: Path, value: dict) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symlink: {path}")
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(path: Path) -> None:
    """Download only the fixed official URL, with a hard size limit."""
    partial = path.with_suffix(path.suffix + ".part")
    if path.is_symlink() or partial.is_symlink():
        raise ValueError("Refusing a symlink for the archive or partial download.")
    request = Request(PREVIEW_URL, headers={"User-Agent": "VocalStream-preview/1.0"})
    for attempt in range(1, 4):
        try:
            print(f"Downloading official compressed preview (attempt {attempt}/3)...", flush=True)
            with urlopen(request, timeout=120) as response, partial.open("wb") as output:
                declared = response.headers.get("Content-Length")
                total = int(declared) if declared else None
                if total is not None and total > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive exceeds the 2 GiB download limit.")
                size = 0
                next_notice = 10 * 1024**2
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Archive exceeds the 2 GiB download limit.")
                    output.write(chunk)
                    if size >= next_notice:
                        percent = f" ({100 * size / total:.0f}%)" if total else ""
                        print(f"  Downloaded {size / 1024**2:.1f} MiB{percent}", flush=True)
                        next_notice = size + 10 * 1024**2
                if total is not None and size != total:
                    raise OSError(f"Incomplete download: received {size} of {total} bytes.")
            if not zipfile.is_zipfile(partial):
                raise ValueError("Downloaded file is not a ZIP archive.")
            os.replace(partial, path)
            return
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as error:
            if attempt == 3:
                raise RuntimeError(f"Official preview download failed: {error}") from error
            print(f"  Download interrupted: {error}. Retrying...", flush=True)
            time.sleep(2 * attempt)


def safe_member_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    """Check all archive entries, including those we will not extract."""
    name = info.filename
    if "\\" in name or "\x00" in name or name.startswith("/"):
        raise ValueError(f"Unsafe ZIP member name: {name!r}")
    parts = tuple(name.rstrip("/").split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe ZIP member name: {name!r}")
    for part in parts:
        if any(character in part for character in ':<>"|?*') or part.endswith((" ", ".")):
            raise ValueError(f"ZIP member cannot safely be named on Windows: {name!r}")
        if part.split(".", 1)[0].upper() in RESERVED_WINDOWS_NAMES:
            raise ValueError(f"Reserved Windows ZIP member name: {name!r}")
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP symlink is not allowed: {name!r}")
    if info.flag_bits & 1:
        raise ValueError(f"Encrypted ZIP member is not allowed: {name!r}")
    return parts


def archive_tracks(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str, str]]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_ITEMS:
        raise ValueError("Archive exceeds the 1000-entry limit.")
    if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("Archive exceeds the 2 GiB unpacked-size limit.")
    tracks = []
    seen_names: set[str] = set()
    seen_tracks: set[tuple[str, str]] = set()
    for info in infos:
        parts = safe_member_parts(info)
        normal_name = "/".join(parts).casefold()
        if normal_name in seen_names:
            raise ValueError(f"Duplicate or case-colliding ZIP entry: {info.filename}")
        seen_names.add(normal_name)
        if info.is_dir():
            continue
        if info.file_size > 0 and info.file_size / max(1, info.compress_size) > 1000:
            raise ValueError(f"Suspicious ZIP compression ratio: {info.filename}")
        filename = parts[-1]
        if not filename.lower().endswith(".stem.mp4"):
            continue
        # The official release can have an enclosing folder; no arbitrary
        # archive paths are used as output paths.  Only train/test are accepted.
        if len(parts) < 2 or parts[-2] not in {"train", "test"}:
            raise ValueError(f"Unexpected preview stem location: {info.filename}")
        subset = parts[-2]
        song = filename[: -len(".stem.mp4")]
        if not song or song.endswith((" ", ".")):
            raise ValueError(f"Invalid preview song name: {filename}")
        if song.split(".", 1)[0].upper() in RESERVED_WINDOWS_NAMES:
            raise ValueError(f"Reserved preview song name: {song}")
        identity = (subset, song.casefold())
        if identity in seen_tracks:
            raise ValueError(f"Duplicate preview track: {subset}/{song}")
        seen_tracks.add(identity)
        tracks.append((info, subset, song))
    if not tracks:
        raise ValueError("Archive contains no train/test *.stem.mp4 preview tracks.")
    if len(tracks) > 150:
        raise ValueError(f"Unexpected track count: {len(tracks)} (maximum 150).")
    return sorted(tracks, key=lambda entry: (entry[1], entry[2].casefold()))


def locate_ffmpeg(explicit: str | None) -> tuple[str, str]:
    if explicit:
        executable = shutil.which(explicit) or explicit
    else:
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError(
                "Install imageio-ffmpeg in this environment, or pass --ffmpeg PATH."
            ) from error
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    completed = subprocess.run(
        [executable, "-version"], capture_output=True, text=True, timeout=30,
        encoding="utf-8", errors="replace", check=True,
    )
    return executable, completed.stdout.splitlines()[0]


def validate_stems(directory: Path) -> dict:
    lengths = []
    for stem in STREAMS:
        path = directory / f"{stem}.wav"
        if path.is_symlink():
            raise ValueError(f"Stem is a symlink: {path}")
        info = sf.info(str(path))
        if info.samplerate != SAMPLE_RATE or info.channels != 2:
            raise ValueError(f"Invalid sample rate or channel count: {path}")
        if info.format not in {"WAV", "WAVEX"} or info.subtype != "FLOAT":
            raise ValueError(f"Expected an uncompressed float WAV: {path}")
        if not 6 * SAMPLE_RATE <= info.frames <= 8 * SAMPLE_RATE:
            raise ValueError(f"Expected an approximately seven-second preview: {path}")
        # Read every sample to reject truncated/corrupt WAVs.  This also ensures
        # that headers claiming a larger frame count cannot pass a skip check.
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        if len(audio) != info.frames or rate != SAMPLE_RATE:
            raise ValueError(f"Truncated stem: {path}")
        import numpy as np
        if not np.isfinite(audio).all():
            raise ValueError(f"Non-finite samples in stem: {path}")
        lengths.append(info.frames)
    if len(set(lengths)) != 1:
        raise ValueError(f"Stem frame counts do not match: {directory}")
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": 2,
        "frames": lengths[0],
        "duration_seconds": lengths[0] / SAMPLE_RATE,
        "wav_subtype": "FLOAT",
    }


def existing_track_valid(directory: Path, member_name: str, archive_sha: str) -> bool:
    if not directory.exists():
        return False
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Refusing an existing non-directory/symlink: {directory}")
    marker = directory / TRACK_MARKER
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(
            f"Existing track has no trusted preview marker; refusing to overwrite: {directory}"
        )
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if (
        saved.get("dataset_kind") != DATASET_KIND
        or saved.get("archive_member") != member_name
        or saved.get("archive_sha256") != archive_sha
    ):
        raise ValueError(f"Existing track belongs to a different source: {directory}")
    try:
        validate_stems(directory)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def decode_track(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
    cache: Path,
    ffmpeg: str,
    archive_sha: str,
) -> dict:
    staging = cache / ("decode-" + uuid4().hex)
    staging.mkdir()
    source = staging / "input.stem.mp4"
    try:
        # Copy only the selected regular MP4 member, under a fixed local name.
        # ZipExtFile verifies the member CRC during this bounded copy.
        copied = 0
        with archive.open(info) as incoming, source.open("xb") as outgoing:
            while True:
                chunk = incoming.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > info.file_size or copied > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("Archive member exceeded its declared size.")
                outgoing.write(chunk)
        if copied != info.file_size:
            raise ValueError(f"Truncated archive member: {info.filename}")
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", str(source)]
        for stem, stream_id in STREAMS.items():
            command.extend([
                "-map", f"0:a:{stream_id}", "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE),
                "-c:a", "pcm_f32le", "-map_metadata", "-1", "-f", "wav",
                str(staging / f"{stem}.wav"),
            ])
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        if completed.returncode:
            raise RuntimeError(f"FFmpeg failed for {info.filename}:\n{completed.stderr[-4000:]}")
        audio_info = validate_stems(staging)
        marker = {
            "dataset_kind": DATASET_KIND,
            "formal_benchmark": False,
            "archive_member": info.filename,
            "archive_sha256": archive_sha,
            "decoded_at_utc": utc_now(),
            **audio_info,
        }
        atomic_json(staging / TRACK_MARKER, marker)
        source.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.replace(staging, destination)
        else:
            # Only a previously generated, matching marker reaches here.
            # Preserve any extra files; replace our five stems and marker only.
            for filename in [*(f"{stem}.wav" for stem in STREAMS), TRACK_MARKER]:
                target = destination / filename
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise ValueError(f"Refusing to replace a non-file/symlink: {target}")
            for filename in [*(f"{stem}.wav" for stem in STREAMS), TRACK_MARKER]:
                os.replace(staging / filename, destination / filename)
            staging.rmdir()
        return audio_info
    finally:
        # Delete only the fixed files created in this uniquely named staging
        # directory, never a recursive tree or archive-supplied path.
        if staging.exists():
            for filename in ["input.stem.mp4", *(f"{stem}.wav" for stem in STREAMS), TRACK_MARKER]:
                path = staging / filename
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            if not any(staging.iterdir()):
                staging.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", "--out", type=Path,
        default=PROJECT_DIR / "data" / "musdb18_preview",
        help="Output root (default: project/data/musdb18_preview).",
    )
    parser.add_argument("--archive", type=Path, help="Use an already downloaded official ZIP.")
    parser.add_argument("--ffmpeg", help="Explicit FFmpeg executable; default: imageio-ffmpeg.")
    parser.add_argument("--limit-tracks", type=int, help="Decode only the first N tracks for a small pilot.")
    args = parser.parse_args()
    if args.limit_tracks is not None and args.limit_tracks <= 0:
        parser.error("--limit-tracks must be positive.")
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset_marker = contained_path(root, root / "_DATASET_INFO.json")
    if dataset_marker.is_symlink():
        raise ValueError("Refusing a symlink for the dataset marker.")
    if dataset_marker.exists():
        saved = json.loads(dataset_marker.read_text(encoding="utf-8"))
        if saved.get("kind") != DATASET_KIND:
            raise ValueError("Output folder already contains a different dataset marker.")
    cache = contained_path(root, root / ".cache")
    if cache.is_symlink():
        raise ValueError("Refusing a symlink for the preview cache directory.")
    cache.mkdir(parents=True, exist_ok=True)
    ffmpeg, version = locate_ffmpeg(args.ffmpeg)
    archive_path = (
        args.archive.expanduser().resolve() if args.archive
        else contained_path(root, cache / "MUSDB18-7-STEMS.zip")
    )
    if not archive_path.exists():
        if args.archive:
            raise FileNotFoundError(f"Local archive does not exist: {archive_path}")
        download_archive(archive_path)
    if not archive_path.is_file() or archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Expected a ZIP file no larger than 2 GiB.")
    digest = sha256_file(archive_path)
    integrity_file = contained_path(root, cache / "archive_integrity.json")
    if integrity_file.is_symlink():
        raise ValueError("Refusing a symlink for the local integrity record.")
    if integrity_file.exists():
        prior = json.loads(integrity_file.read_text(encoding="utf-8"))
        if prior.get("sha256") != digest:
            raise ValueError("Archive SHA-256 differs from the previous local integrity record.")
    atomic_json(integrity_file, {
        "original_url": PREVIEW_URL,
        "sha256": digest,
        "bytes": archive_path.stat().st_size,
        "sha256_status": "locally computed integrity record; no publisher checksum asserted",
        "checked_at_utc": utc_now(),
    })
    print("This is the lossy seven-second preview, not MUSDB18-HQ or a formal benchmark.", flush=True)
    print(f"FFmpeg: {version}", flush=True)
    prepared = skipped = 0
    subsets = {"train": 0, "test": 0}
    with zipfile.ZipFile(archive_path) as archive:
        all_tracks = archive_tracks(archive)
        tracks = all_tracks[: args.limit_tracks] if args.limit_tracks else all_tracks
        metadata = {
            "kind": DATASET_KIND,
            "formal_benchmark": False,
            "preview_seconds": 7,
            "decoded_lossy_audio": True,
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "wav_subtype": "FLOAT",
            "audio_stream_mapping": STREAMS,
            "original_url": PREVIEW_URL,
            "archive_sha256": digest,
            "archive_bytes": archive_path.stat().st_size,
            "sha256_status": "locally computed integrity record; no publisher checksum asserted",
            "ffmpeg_version": version,
            "archive_track_count": len(all_tracks),
            "selected_track_count": len(tracks),
            "complete": False,
            "updated_at_utc": utc_now(),
            "usage_note": (
                "Tooling and pilot experiments only. WAV decoding does not restore AAC bandwidth "
                "or make this MUSDB18-HQ. Do not compare preview scores with full-test paper scores. "
                "Obtain formal data and check its terms at https://sigsep.github.io/datasets/musdb.html."
            ),
        }
        atomic_json(dataset_marker, metadata)
        for index, (info, subset, song) in enumerate(tracks, start=1):
            destination = contained_path(root, root / subset / song)
            if existing_track_valid(destination, info.filename, digest):
                skipped += 1
            else:
                decode_track(archive, info, destination, cache, ffmpeg, digest)
                prepared += 1
            subsets[subset] += 1
            if index == 1 or index % 10 == 0 or index == len(tracks):
                print(
                    f"[{index}/{len(tracks)}] {subset}/{song} "
                    f"(decoded {prepared}, validated and skipped {skipped})",
                    flush=True,
                )
        metadata.update({
            "complete": len(tracks) == len(all_tracks),
            "selected_tracks_prepared": len(tracks),
            "subset_counts_prepared_this_run": subsets,
            "updated_at_utc": utc_now(),
        })
        atomic_json(dataset_marker, metadata)
    print(f"Prepared {len(tracks)} preview tracks under {root}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

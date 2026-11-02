"""Verify published evidence without importing models or accessing audio.

Hashes establish consistency with the committed index, not independent proof of
scientific validity. Historical artifact paths are logical references: their
audio/log files deliberately do not have to exist in a public checkout.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sys


class EvidenceError(ValueError):
    """A published record is missing, inconsistent, or exposes a local path."""


@dataclass(frozen=True)
class AuditSummary:
    studies: int
    jobs: int
    json_files: int


URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
ABSOLUTE_PATH = re.compile(
    r"(?:^|[\s=\"'(\[])(?:[A-Za-z]:[\\/]|\\\\|/[^\s]|file://)", re.IGNORECASE
)
PATH_KEYS = {"path", "artifact", "log", "public_report", "report"}
UNDECLARED_HASH = object()


def relative_path(base: Path, value: object, location: str) -> Path:
    """Resolve a portable relative path and reject traversal/symlink escapes."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EvidenceError(f"{location}: expected a nonempty relative path")
    normalized = value.replace("\\", "/")
    if (PureWindowsPath(value).drive or normalized.startswith("/")
            or ":" in normalized or ".." in normalized.split("/")):
        raise EvidenceError(f"{location}: path must stay inside the repository")
    result = (base / normalized).resolve()
    if not result.is_relative_to(base.resolve()):
        raise EvidenceError(f"{location}: path escapes its declared directory")
    return result


def check_strings(value: object, root: Path, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            check_strings(key, root, f"{location}.<key>")
            child = f"{location}.{key}"
            if isinstance(item, str) and (key in PATH_KEYS or key.endswith("_path")):
                relative_path(root, item, child)
            check_strings(item, root, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_strings(item, root, f"{location}[{index}]")
    elif isinstance(value, str) and ABSOLUTE_PATH.search(URL.sub("<url>", value)):
        # Do not echo the sensitive value into CI output.
        raise EvidenceError(f"{location}: absolute machine path found")


def object_value(value: object, location: str) -> dict:
    if not isinstance(value, dict):
        raise EvidenceError(f"{location}: expected a JSON object")
    return value


def load_record(path: Path, root: Path, location: str,
                expected_hash: object = UNDECLARED_HASH) -> dict:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"{location}: cannot read evidence file") from exc
    if expected_hash is not UNDECLARED_HASH:
        if (not isinstance(expected_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)):
            raise EvidenceError(f"{location}: invalid SHA-256 declaration")
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise EvidenceError(f"{location}: SHA-256 mismatch")
    try:
        record = object_value(json.loads(raw), location)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{location}: invalid JSON") from exc
    check_strings(record, root, location)
    return record


def same_count(value: object, expected: int, location: str) -> None:
    if type(value) is not int or value != expected:
        raise EvidenceError(f"{location}: declared job count does not match {expected}")


def verify(root: Path) -> AuditSummary:
    root = root.resolve()
    reports = relative_path(root, "reports", "reports")
    index = load_record(reports / "index.json", root, "index")
    if index.get("schema_version") != 1:
        raise EvidenceError("index: unsupported schema version")
    studies = object_value(index.get("studies"), "index.studies")
    if not studies:
        raise EvidenceError("index.studies: no studies declared")
    seen_files = {reports / "index.json"}
    total_jobs = 0
    for name, entry in studies.items():
        location = f"study {name}"
        entry = object_value(entry, location)
        path = relative_path(reports, entry.get("path"), location)
        if path in seen_files:
            raise EvidenceError(f"{location}: duplicate evidence file")
        seen_files.add(path)
        if "sha256" not in entry:
            raise EvidenceError(f"{location}: missing SHA-256 declaration")
        study = load_record(path, root, location, entry["sha256"])
        if study.get("publication_schema_version") != 1:
            raise EvidenceError(f"{location}: unsupported publication schema")
        if not study.get("study_id") or study["study_id"] != entry.get("original_study_id"):
            raise EvidenceError(f"{location}: study identity mismatch")
        execution = object_value(study.get("execution"), f"{location}.execution")
        if execution.get("status") != "complete":
            raise EvidenceError(f"{location}: execution is not complete")
        jobs = object_value(study.get("jobs"), f"{location}.jobs")
        if not jobs:
            raise EvidenceError(f"{location}: no completed jobs")
        same_count(entry.get("job_count"), len(jobs), f"{location}.index")
        same_count(execution.get("job_count"), len(jobs), f"{location}.execution")
        protocol = object_value(study.get("protocol"), f"{location}.protocol")
        configurations = [protocol, protocol.get("configuration", {})]
        context = object_value(execution.get("context", {}), f"{location}.context")
        configurations.append(context.get("configuration", {}))
        for config in configurations:
            config = object_value(config, f"{location}.configuration")
            if "expected_jobs" in config:
                same_count(config["expected_jobs"], len(jobs), f"{location}.expected_jobs")
        for job_name, job in jobs.items():
            label = f"{location}, job {job_name}"
            job = object_value(job, label)
            if job.get("status") != "complete":
                raise EvidenceError(f"{label}: job is not complete")
            relative_path(root, job.get("artifact"), f"{label}.artifact")
            relative_path(root, job.get("log"), f"{label}.log")
            public_path = relative_path(reports, job.get("public_report"), label)
            if public_path in seen_files:
                raise EvidenceError(f"{label}: duplicate evidence file")
            seen_files.add(public_path)
            if "public_report_sha256" not in job:
                raise EvidenceError(f"{label}: missing public report SHA-256")
            load_record(public_path, root, label, job["public_report_sha256"])
        total_jobs += len(jobs)
    return AuditSummary(len(studies), total_jobs, len(seen_files))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root (defaults to this script's checkout)")
    args = parser.parse_args(argv)
    try:
        result = verify(args.root)
    except EvidenceError as exc:
        print(f"Evidence verification FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Evidence verified: {result.studies} studies, {result.jobs} complete jobs, "
          f"{result.json_files} JSON files; hashes and public paths valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

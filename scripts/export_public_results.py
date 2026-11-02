"""Export existing experiment evidence without audio, weights or local paths.

This is a publication utility, not an evaluation or model runner. Original
evidence is read only. Hashes of original files refer to the private originals;
hashes of sanitized files refer to the actual files distributed in this repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PureWindowsPath

STUDIES = {
    "head_optimization": ("head_optimization_20261019", "head_optimization_analysis.json", 16),
    "streaming_study": ("streaming_study_20261019", "streaming_study_analysis.json", 13),
    "refinement": ("refinement_20261012", "analysis.json", 18),
}
ABSOLUTE_WINDOWS = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def export(source_project: Path, destination: Path, *, studies=None,
           include_figures=True, merge_index=False) -> None:
    source_project = source_project.resolve()
    destination = destination.resolve()
    if source_project == destination and include_figures:
        raise ValueError("Publication destination must differ from the original project")
    studies = STUDIES if studies is None else studies
    results = source_project / "results"
    output = destination / "reports"
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    publication_index = read_json(index_path) if merge_index and index_path.exists() else {"schema_version": 1, "studies": {}}
    if merge_index:
        for item in publication_index["studies"].values():
            existing = (output / item["path"]).resolve()
            if not existing.is_relative_to(output) or digest(existing) != item["sha256"]:
                raise ValueError("Existing publication index does not match its evidence")
    exported_jobs = 0

    for public_name, (folder, analysis_name, expected_jobs) in studies.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", public_name) or Path(folder).name != folder:
            raise ValueError("Study name/folder must be a simple publication identifier")
        original = results / folder
        run_path = original / "run.json"
        run = read_json(run_path)
        if run["status"] != "complete" or len(run["jobs"]) != expected_jobs:
            raise ValueError(f"Incomplete study: {folder}")
        # Original project location can differ from the location of this export.
        commands = [j.get("command", []) for j in run["jobs"].values()]
        old_project = str(PureWindowsPath(commands[0][1]).parent)
        old_python = commands[0][0]
        config = run.get("context", {}).get("configuration", {})
        old_dataset = config.get("root", "")

        def sanitize(value):
            if isinstance(value, dict):
                return {sanitize(k): sanitize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [sanitize(v) for v in value]
            if not isinstance(value, str):
                return value
            text = value
            if text == old_python:
                return "python"
            for prefix, replacement in ((old_project, ""), (old_dataset, "MUSDB18-HQ")):
                if not prefix:
                    continue
                if text.lower() == prefix.lower():
                    return replacement or "."
                if text.lower().startswith(prefix.lower() + "\\") or text.lower().startswith(prefix.lower() + "/"):
                    return ((replacement + "/") if replacement else "") + text[len(prefix) + 1:].replace("\\", "/")
            # A small number of runtime/cache paths are not experiment artifacts.
            if ABSOLUTE_WINDOWS.match(text):
                return "<local-path-omitted>"
            if ABSOLUTE_WINDOWS.search(text):
                raise ValueError("Embedded absolute path requires explicit review")
            return text

        analysis_path = original / analysis_name
        protocol_path = original / "protocol.json"
        if config.get("window_inference_study") == "compare":
            analysis = read_json(analysis_path)
            generation = analysis["analysis_generation"]
            if (analysis.get("status") != "complete"
                    or read_json(protocol_path) != run["context"]
                    or analysis["configuration"] != config
                    or generation["run_record_sha256"] != digest(run_path)
                    or generation["protocol_sha256"] != digest(protocol_path)
                    or analysis["inference_source_hashes"] != run["context"]["source_hashes"]):
                raise ValueError("Matched-study analysis does not identify the completed inference run")
        public_jobs = {}
        job_output = output / public_name / "jobs"
        job_output.mkdir(parents=True, exist_ok=True)
        for job_id, job in run["jobs"].items():
            if job["status"] != "complete":
                raise ValueError(f"Incomplete job: {job_id}")
            # Locate by logical suffix, never copy a path selected by run metadata.
            suffix = PureWindowsPath(job["artifact"]).parts
            marker = suffix.index(folder)
            report_path = original.joinpath(*suffix[marker + 1:])
            if digest(report_path) != job["artifact_sha256"]:
                raise ValueError(f"Original report hash mismatch: {job_id}")
            public_report = sanitize(read_json(report_path))
            report_out = job_output / f"{job_id}.json"
            report_out.write_bytes(encode(public_report))
            manifest = sanitize(job)
            manifest["public_report"] = report_out.relative_to(output).as_posix()
            manifest["public_report_sha256"] = digest(report_out)
            manifest["original_report_sha256"] = job["artifact_sha256"]
            public_jobs[job_id] = manifest

        public = {
            "publication_schema_version": 1,
            "study_id": folder,
            "data_minimization": {
                "absolute_paths": "Replaced by logical project-relative or MUSDB18-HQ/ references; unknown runtime/cache paths omitted.",
                "logical_artifact_paths": "Identify historical artifacts; audio and console logs are not included or downloadable from those paths.",
                "included": "Numeric measurements, per-track errors, block timings, experiment configuration, software/hardware metadata, source/input/output/checkpoint SHA-256 hashes.",
                "excluded": ["audio", "model weights", "console logs", "private correspondence", "absolute local paths"],
                "numeric_measurements_modified": False,
                "formal_test_evaluation": False,
            },
            "original_evidence_sha256": {
                "analysis": digest(analysis_path),
                "protocol": digest(protocol_path),
                "run": digest(run_path),
            },
            "analysis": sanitize(read_json(analysis_path)),
            "protocol": sanitize(read_json(protocol_path)),
            "execution": {
                k: sanitize(run[k]) for k in ("identity", "created_utc", "updated_utc", "status", "context", "job_count") if k in run
            },
            "jobs": public_jobs,
        }
        target = output / f"{public_name}.json"
        target.write_bytes(encode(public))
        publication_index["studies"][public_name] = {
            "path": target.name,
            "sha256": digest(target),
            "job_count": len(public_jobs),
            "original_study_id": folder,
        }
        exported_jobs += len(public_jobs)

    if include_figures:
        assets = destination / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        for folder, stem in (("head_optimization_20261019", "head_optimization_overview"), ("refinement_20261012", "refinement_overview")):
            for extension in ("png", "svg"):
                source = results / folder / "figures" / f"{stem}.{extension}"
                target = assets / source.name
                shutil.copyfile(source, target)
                if extension == "svg" and ABSOLUTE_WINDOWS.search(target.read_text(encoding="utf-8")):
                    raise ValueError("Absolute path found in exported SVG")
    index_path.write_bytes(encode(publication_index))
    print(f"Exported {exported_jobs} existing job reports; no model execution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-project", type=Path, help="Export the three historical studies")
    source_group.add_argument("--run", type=Path, help="Append one completed local streaming study under results/")
    parser.add_argument("--study-name", default="window_comparison", help="Public name for --run")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.run is not None:
        run = args.run.expanduser().resolve()
        if run.parent.name != "results":
            parser.error("--run must be a direct study folder under a project's results/ directory")
        config = read_json(run / "protocol.json")["configuration"]
        if config.get("window_inference_study") != "compare" or config.get("expected_jobs") != 25:
            parser.error("--run currently supports the declared 25-job matched window comparison")
        export(run.parent.parent, args.destination,
               studies={args.study_name: (run.name, "streaming_study_analysis.json", 25)},
               include_figures=False, merge_index=True)
    else:
        export(args.source_project, args.destination, merge_index=True)

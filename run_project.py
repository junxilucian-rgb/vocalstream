"""Prepare, execute and freeze the vocal separation experiment protocol.

`prepare` uses only the standard library and works before audio is downloaded.
`run` trains on train songs and evaluates reserved validation songs only.
`final` is a separate command requiring a frozen HQ development checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parent


@contextmanager
def exclusive_run(folder):
    """OS-managed lock, released even when the process exits unexpectedly."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process is using this experiment: {folder}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def validate_config(config):
    if config.get("schema_version") != 1 or config.get("sample_rate") != 44100:
        raise ValueError("Expected schema 1 and 44100 Hz")
    if len(set(config["validation_tracks"])) != 14:
        raise ValueError("Pin exactly the 14 official validation song names")
    if config["threads"] < 1 or not config["candidates"]:
        raise ValueError("threads and candidates must be positive/nonempty")
    names = []
    for item in config["candidates"] + config["references"]:
        name = item["name"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid experiment name: {name}")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique")
    for candidate in config["candidates"]:
        if candidate["n_fft"] <= candidate["hop_length"] or candidate["hop_length"] <= 0 or candidate["hidden_size"] <= 0:
            raise ValueError("Invalid causal model dimensions")
    for profile in config["profiles"].values():
        for key in ("steps", "batch_size", "crop_seconds", "validate_every", "timing_repeats"):
            if not math.isfinite(profile[key]) or profile[key] <= 0:
                raise ValueError(f"Invalid positive profile field: {key}")
        for key in ("steps", "batch_size", "validate_every", "timing_repeats"):
            if isinstance(profile[key], bool) or not isinstance(profile[key], int):
                raise ValueError(f"{key} must be an integer")
        limit = profile["validation_track_limit"]
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("validation_track_limit must be a positive integer or null")
        if profile["evaluation_seconds"] is not None and (not math.isfinite(profile["evaluation_seconds"]) or profile["evaluation_seconds"] <= 0):
            raise ValueError("evaluation_seconds must be positive or null")
        condition_names = [c["name"] for c in profile["conditions"]]
        if not condition_names or len(condition_names) != len(set(condition_names)) or not any(c["mode"] == "mixture" for c in profile["conditions"]):
            raise ValueError("Condition names must be unique")
    for condition in config["final_protocol"]["conditions"] + [c for p in config["profiles"].values() for c in p["conditions"]]:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", condition["name"]) or condition["mode"] not in ("mixture", "bleed", "noise", "reverb", "clean"):
            raise ValueError("Invalid condition name or mode")
    if config["final_protocol"]["subset"] != "test" or config["final_protocol"]["expected_tracks"] != 50 or config["final_protocol"]["evaluation_seconds"] is not None:
        raise ValueError("Final protocol must use all 50 whole test tracks")
    if not isinstance(config["final_protocol"]["timing_repeats"], int) or config["final_protocol"]["timing_repeats"] < 1:
        raise ValueError("Final timing repeats must be positive")
    final_names = [c["name"] for c in config["final_protocol"]["conditions"]]
    if len(final_names) != len(set(final_names)) or not any(c["mode"] == "mixture" for c in config["final_protocol"]["conditions"]):
        raise ValueError("Final conditions must include mixture and have unique names")


def source_snapshot():
    paths = [*PROJECT.glob("*.py"), *PROJECT.joinpath("vocalstream").rglob("*.py")]
    return {str(p.relative_to(PROJECT)): digest(p) for p in sorted(paths)}


def runtime_snapshot():
    packages = {}
    for name in ("torch", "torchaudio", "numpy", "soundfile", "musdb", "museval", "openunmix", "BS-RoFormer", "librosa"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages}


def prepare(args):
    config = read_json(args.config)
    validate_config(config)
    if args.cpu_only:
        config["training_device"] = "cpu"
        config["causal_evaluation_device"] = "cpu"
        for reference in config["references"]:
            reference["device"] = "cpu"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = {"schema_version": 1, "created_utc": stamp(), "root": str(args.root.resolve()),
            "profile": args.profile, "configuration": config,
            "purpose": "Preparation only; no data read and no model execution", "output_directory": str(out)}
    write_json(out / "plan.json", plan)
    (out / "NEXT_STEP.md").write_text(
        "# 已准备实验计划\n\n数据与环境准备好后，使用同一套 Python 执行：\n\n"
        f'```powershell\npython "{PROJECT / "run_project.py"}" run --plan "{out / "plan.json"}"\n```\n\n'
        "pilot 用于预览流程；development 要求完整 HQ 的 100 首 train。程序只在 train 内保留验证曲目，"
        "完成后再选择配置并 freeze；final 才读取 test。首次运行失败可使用同一命令继续，失败步骤将新建 attempt 目录。\n",
        encoding="utf-8")
    print(f"Prepared (no audio/model execution): {out / 'plan.json'}")


def resolve_data(root, config, profile):
    from vocalstream.data import dataset_info, discover_hq, STEMS
    from train_causal import _official_validation_names
    if set(config["validation_tracks"]) != set(_official_validation_names()):
        raise ValueError("Validation names differ from the installed musdb official split")
    tracks = discover_hq(root, "train")
    available = {t.name for t in tracks}
    validation = sorted(available & set(config["validation_tracks"]))
    metadata = dataset_info(root)
    if profile == "development":
        if metadata["kind"] != "MUSDB18-HQ" or not metadata.get("formal_benchmark") or len(tracks) != 100 or len(validation) != 14:
            raise ValueError("development requires official HQ provenance, 100 train songs and all 14 reserved validation songs")
    if not validation or not available - set(validation):
        raise ValueError("Need both training songs and at least one official validation song; download the full public preview or HQ")
    selected = validation[:config["profiles"][profile]["validation_track_limit"]]
    return {"metadata": metadata, "training_tracks": sorted(available - set(validation)),
            "validation_tracks": validation, "evaluation_tracks": selected,
            "train_track_inventory": [t.as_dict() for t in tracks],
            "audio_file_inventory": [{"track": t.name, "stem": stem, "bytes": (t.path / f"{stem}.wav").stat().st_size,
                                      "mtime_ns": (t.path / f"{stem}.wav").stat().st_mtime_ns} for t in tracks for stem in STEMS],
            "inventory_integrity_scope": "format, file size and modification time; not full audio content hashes",
            "test_set_accessed": False}


def preflight(config):
    import torch
    devices = [config["training_device"], config["causal_evaluation_device"], *[r["device"] for r in config["references"]]]
    if any(d.startswith("cuda") for d in devices) and not torch.cuda.is_available():
        raise ValueError("This plan needs CUDA. Use the prepared GPU Python, or prepare a separate plan with --cpu-only")
    required = ["torch", "torchaudio", "numpy", "soundfile", "musdb", "museval", "openunmix"]
    if any(r["system"] == "melroformer" for r in config["references"]):
        required.extend(["BS-RoFormer", "librosa"])
        for name in ("MelBandRoformer.ckpt", "config.yaml", "provenance.json"):
            if not (PROJECT / "model_cache" / "mel_roformer" / name).is_file():
                raise FileNotFoundError(f"Missing RoFormer asset: {name}; run download_roformer_assets.py")
    packages = runtime_snapshot()["packages"]
    missing = [name for name in required if packages.get(name) is None]
    if missing:
        raise ValueError(f"Missing project dependencies: {missing}")


class JobRunner:
    def __init__(self, out, identity, context):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "run.json"
        self.state = read_json(self.path) if self.path.exists() else {
            "identity": identity, "created_utc": stamp(), "status": "prepared", "context": context, "jobs": {}}
        if self.state["identity"] != identity or self.state["context"] != context:
            raise ValueError("Plan, data inventory, code or runtime changed. Prepare a new run to preserve reproducibility")
        self.save()

    def save(self):
        self.state["updated_utc"] = stamp()
        write_json(self.path, self.state)

    def job(self, name, kind, command_builder, **metadata):
        previous = self.state["jobs"].get(name)
        if previous and previous["status"] == "complete":
            artifact = Path(previous["artifact"])
            if not artifact.is_file() or digest(artifact) != previous["artifact_sha256"]:
                raise ValueError(f"Completed artifact changed/missing: {artifact}")
            print(f"Already complete: {name}", flush=True)
            return artifact
        attempt = 1 if previous is None else previous["attempt"] + 1
        folder = self.out / name / f"attempt_{attempt:02d}"
        while folder.exists():
            attempt += 1
            folder = self.out / name / f"attempt_{attempt:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        output = folder / "output"
        command = command_builder(output)
        row = {"status": "running", "kind": kind, "attempt": attempt, "command": command,
               "started_utc": stamp(), "log": str(folder / "console.log"), **metadata}
        self.state["jobs"][name] = row
        self.save()
        print(f"Running: {name} (log: {row['log']})", flush=True)
        environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
        try:
            with (folder / "console.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                finally:
                    process.stdout.close()
            if code:
                raise RuntimeError(f"{name} exited with {code}; see {row['log']}")
            artifact = output / ("best.pt" if kind == "training" else "report.json")
            if kind == "evaluation" and read_json(artifact)["status"] != "complete":
                raise ValueError("Evaluation did not complete")
            row.update(status="complete", artifact=str(artifact), artifact_sha256=digest(artifact), finished_utc=stamp())
            self.save()
            return artifact
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            self.save()
            raise


def evaluate(runner, root, config, protocol, models, tracks=None, final=False):
    if not final and not tracks:
        raise ValueError("Validation evaluation needs explicit held-out track names")
    for condition in protocol["conditions"]:
        for model in models:
            repeats = protocol["timing_repeats"] if model["system"] == "causal" else 1
            for repeat in range(1, repeats + 1):
                def command(output, model=model, condition=condition):
                    values = [sys.executable, str(PROJECT / "evaluate_project.py"), "--root", str(root),
                              "--subset", "test" if final else "train", "--system", model["system"],
                              "--device", model["device"], "--threads", str(config["threads"]),
                              "--seed", str(config["seed"]), "--condition", condition["mode"], "--out", str(output)]
                    if protocol["evaluation_seconds"] is not None:
                        values += ["--duration", str(protocol["evaluation_seconds"])]
                    for track in tracks or []:
                        values += ["--track", track]
                    for key in ("checkpoint", "model", "niter", "overlap", "precision"):
                        if key in model:
                            values += ["--" + key, str(model[key])]
                    if model.get("tta"):
                        values += ["--tta"]
                    for key in ("bleed_db", "snr_db", "rt60"):
                        if key in condition:
                            values += ["--" + key.replace("_", "-"), str(condition[key])]
                    if repeat == 1 and not final:
                        values += ["--save-audio"]
                    if final and repeat == 1 and condition["mode"] == "mixture" and protocol["bsseval_for_mixture"]:
                        values += ["--bsseval"]
                    return values
                runner.job(f"eval_{model['name']}_{condition['name']}_r{repeat}", "evaluation", command,
                           model_id=model["name"], condition=condition["name"], repeat=repeat)


def make_run_report(runner):
    lines = ["# 项目实验结果索引", "", "每行均来自完成的报告；不同设备与计时范围不能直接归因于模型速度差异。", "",
             "| 模型 | 条件 | 设备 | 次数 | 曲目数 | SI-SDR中位数 dB | 原始报告 |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    reports = []
    for row in runner.state["jobs"].values():
        if row["kind"] != "evaluation" or row["status"] != "complete":
            continue
        report = read_json(row["artifact"])
        score = report["aggregate"]["si_sdr_track_median_db"]
        text_score = "未测/无有效参考" if score is None else f"{score:.3f}"
        relative = Path(row["artifact"]).relative_to(runner.out).as_posix()
        lines.append(f"| {row['model_id']} | {row['condition']} | {report['configuration']['device']} | {row['repeat']} | "
                     f"{report['aggregate']['track_count']} | {text_score} | [report.json]({relative}) |")
        reports.append((row, report))
    lines += ["", "重复计时行使用相同歌曲，不是独立质量样本。各权重的训练数据和预算可能不同；结果并非仅架构差异。",
              "", "## 同条件同设备详细表", ""]
    from summarize_runs import signature, summarize
    groups = {}
    for row, report in reports:
        groups.setdefault(signature(report), []).append(Path(row["artifact"]))
    for index, paths in enumerate(groups.values(), 1):
        target = runner.out / f"comparison_{index:02d}.md"
        summarize(paths, target)
        lines.append(f"- [详细比较 {index}]({target.name})：质量、计时范围、逐块 RTR、超时和延迟。")
    lines += ["", "## 数据与执行信息", "", f"数据标识：`{runner.state['context']['data']['metadata']['kind']}`。",
              f"执行状态：`{runner.state['status']}`；完整参数、命令与日志见 [run.json](run.json)。",
              "完整 test 之前须固定配置与权重；预览、片段、合成信号不能作为论文完整基准成绩。"]
    (runner.out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_plan(args):
    plan = read_json(args.plan)
    config, root = plan["configuration"], Path(plan["root"])
    validate_config(config)
    data = resolve_data(root, config, plan["profile"])
    preflight(config)
    out = args.plan.resolve().parent
    context = {"plan": plan, "data": data, "source_hashes": source_snapshot(), "runtime": runtime_snapshot()}
    runner = JobRunner(out, digest(args.plan), context)
    profile = config["profiles"][plan["profile"]]
    runner.state["status"] = "running"
    runner.state.pop("error", None)
    runner.save()
    try:
        models = list(config["references"])
        for candidate in config["candidates"]:
            def command(output, candidate=candidate):
                values = [sys.executable, str(PROJECT / "train_causal.py"), "--root", str(root), "--output", str(output),
                          "--device", config["training_device"], "--threads", str(config["threads"]), "--seed", str(config["seed"])]
                for key in ("steps", "batch_size", "crop_seconds", "validate_every"):
                    values += ["--" + key.replace("_", "-"), str(profile[key])]
                for key in ("n_fft", "hop_length", "hidden_size"):
                    values += ["--" + key.replace("_", "-"), str(candidate[key])]
                return values + ["--validation-tracks", *data["validation_tracks"]]
            checkpoint = runner.job(f"train_{candidate['name']}", "training", command, model_id=candidate["name"])
            models.append({"name": candidate["name"], "system": "causal", "device": config["causal_evaluation_device"], "checkpoint": str(checkpoint)})
        evaluate(runner, root, config, profile, models, data["evaluation_tracks"])
        runner.state["status"] = "complete"
        make_run_report(runner)
    except BaseException as error:
        runner.state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        runner.save()
    print(f"Validation workflow complete: {out / 'RESULTS.md'}")


def freeze(args):
    if not args.reason.strip():
        raise ValueError("Record a validation-based selection reason")
    run_path = args.run / "run.json" if args.run.is_dir() else args.run
    run = read_json(run_path)
    context = run["context"]
    plan = context["plan"]
    if run["status"] != "complete" or plan["profile"] != "development" or context["data"]["metadata"]["kind"] != "MUSDB18-HQ":
        raise ValueError("Freeze requires a completed HQ development run, not a preview/pilot")
    if context["source_hashes"] != source_snapshot() or context["runtime"] != runtime_snapshot():
        raise ValueError("Code/runtime changed since validation; repeat validation in a new run before freezing")
    for job in run["jobs"].values():
        if job["status"] != "complete" or not Path(job["artifact"]).is_file() or digest(job["artifact"]) != job["artifact_sha256"]:
            raise ValueError("A validation artifact is incomplete, missing or changed")
    row = run["jobs"].get(f"train_{args.candidate}")
    if not row or row["status"] != "complete" or digest(row["artifact"]) != row["artifact_sha256"]:
        raise ValueError("Candidate checkpoint is missing, changed or not trained in this run")
    target = run_path.parent / "frozen_selection.json"
    if target.exists():
        raise FileExistsError("Selection already frozen; use a new development run for a new protocol")
    value = {"created_utc": stamp(), "development_run": str(run_path.resolve()), "development_run_sha256": digest(run_path),
             "root": plan["root"], "configuration": plan["configuration"], "candidate": args.candidate,
             "checkpoint": row["artifact"], "checkpoint_sha256": row["artifact_sha256"],
             "reason": args.reason, "source_hashes": context["source_hashes"], "runtime": context["runtime"],
             "test_used_for_selection": False, "data": context["data"]}
    write_json(target, value)
    print(f"Selection frozen before test: {target}")


def final(args):
    frozen = read_json(args.frozen)
    config = frozen["configuration"]
    validate_config(config)
    if digest(frozen["checkpoint"]) != frozen["checkpoint_sha256"] or digest(frozen["development_run"]) != frozen["development_run_sha256"]:
        raise ValueError("Frozen checkpoint/development record changed")
    if source_snapshot() != frozen["source_hashes"] or runtime_snapshot() != frozen["runtime"]:
        raise ValueError("Use the validated code and Python environment for final evaluation")
    preflight(config)
    from vocalstream.data import discover_hq, dataset_info
    root = Path(frozen["root"])
    data = resolve_data(root, config, "development")
    if data != frozen["data"]:
        raise ValueError("Development data inventory changed after selection")
    test_tracks = discover_hq(root, "test")
    if len(test_tracks) != 50 or {t.name for t in test_tracks} & set(data["training_tracks"] + data["validation_tracks"]):
        raise ValueError("Final test requires 50 separate official test songs")
    context = {"frozen": frozen, "data": {"metadata": dataset_info(root), "test_tracks": [t.as_dict() for t in test_tracks]},
               "source_hashes": source_snapshot(), "runtime": runtime_snapshot(), "test_set_accessed": True}
    runner = JobRunner(args.out.resolve(), digest(args.frozen), context)
    runner.state["status"] = "running"
    runner.state.pop("error", None)
    runner.save()
    models = [*config["references"], {"name": frozen["candidate"], "system": "causal", "device": config["causal_evaluation_device"], "checkpoint": frozen["checkpoint"]}]
    try:
        evaluate(runner, root, config, config["final_protocol"], models, final=True)
        runner.state["status"] = "complete"
        make_run_report(runner)
    except BaseException as error:
        runner.state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        runner.save()
    print(f"Frozen test evaluation complete: {runner.out / 'RESULTS.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Write plan without reading audio or loading ML dependencies")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--profile", choices=["pilot", "development"], default="pilot")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=PROJECT / "project_config.json")
    p.add_argument("--cpu-only", action="store_true", help="Also runs the large reference on CPU; can be slow")
    p.set_defaults(action=prepare)
    p = commands.add_parser("run", help="Execute/resume train and validation jobs from a prepared plan")
    p.add_argument("--plan", type=Path, required=True)
    p.set_defaults(action=execute_plan)
    p = commands.add_parser("freeze", help="Record a validation-based selection before accessing test")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--reason", required=True, help="Explain quality/RTR/latency tradeoff using validation results")
    p.set_defaults(action=freeze)
    p = commands.add_parser("final", help="Evaluate the frozen candidate and references on all HQ test tracks")
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(action=final)
    args = parser.parse_args()
    if args.command == "run":
        with exclusive_run(args.plan.resolve().parent):
            args.action(args)
    elif args.command == "final":
        with exclusive_run(args.out.resolve()):
            args.action(args)
    elif args.command == "freeze":
        with exclusive_run(args.run if args.run.is_dir() else args.run.parent):
            args.action(args)
    else:
        args.action(args)


if __name__ == "__main__":
    main()

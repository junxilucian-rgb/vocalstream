"""One reporting protocol for an offline baseline and a trained causal model."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vocalstream.baselines import OpenUnmixBaseline
from vocalstream.conditions import make_condition
from vocalstream.data import discover_hq, load_segment, dataset_info
from vocalstream.metrics import separation_metrics, timing_summary

PROJECT = Path(__file__).resolve().parent


def environment():
    packages = {}
    for package in ('torch', 'torchaudio', 'openunmix', 'soundfile', 'numpy', 'museval', 'BS-RoFormer', 'librosa', 'einops'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER'),
            "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(), "versions": packages,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_hash(audio):
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--root', type=Path, help='MUSDB18-HQ root with train/test folders')
    source.add_argument('--input', type=Path, help='One local WAV or FLAC, qualitative demo only')
    parser.add_argument('--system', choices=['openunmix', 'causal', 'melroformer', 'bsroformer'], default='openunmix')
    parser.add_argument('--chunk-seconds', type=float, help='BS-RoFormer context experiment; omit for the published 11-second context')
    parser.add_argument('--bs-heads', choices=['all', 'vocals'], default='all',
                        help='BS-RoFormer inference: all four heads or only the strictly loaded vocal head')
    parser.add_argument('--model', choices=['umx', 'umxhq', 'umxl'], default='umxhq')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--overlap', type=int, choices=[2, 4, 8], default=8)
    parser.add_argument('--precision', choices=['fp32', 'amp'], default='fp32')
    parser.add_argument('--tta', action='store_true', help='MelRoformer channel/polarity test-time averaging')
    parser.add_argument('--niter', type=int, default=0)
    parser.add_argument('--subset', choices=['train', 'test'], default='test')
    parser.add_argument('--track', action='append', help='Exact track name; repeat to select multiple')
    parser.add_argument('--limit', type=int, help='Pilot: first N sorted tracks')
    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--duration', type=float, help='Omit to evaluate the whole track')
    parser.add_argument('--condition', choices=['mixture', 'clean', 'noise', 'reverb', 'bleed'], default='mixture')
    parser.add_argument('--snr-db', type=float, default=10)
    parser.add_argument('--bleed-db', type=float, default=-20)
    parser.add_argument('--rt60', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1, help='Fix PyTorch CPU threads for fair comparisons')
    parser.add_argument('--bsseval', action='store_true', help='Optional slow museval v4 vocal/accompaniment metric')
    parser.add_argument('--save-audio', action='store_true', help='Save FLOAT WAVs; no clipping/normalization')
    parser.add_argument('--save-audio-limit', type=int, help='Export full input/reference/estimate bundles for only the first N tracks')
    parser.add_argument('--save-estimates', action='store_true', help='Keep every estimated vocal WAV for offline post-processing analysis')
    parser.add_argument('--out', type=Path, help='New output directory, must not already exist')
    args = parser.parse_args()
    if args.chunk_seconds is not None and args.system != 'bsroformer':
        parser.error('--chunk-seconds is supported only by BS-RoFormer')
    if args.bs_heads != 'all' and args.system != 'bsroformer':
        parser.error('--bs-heads vocals is supported only by BS-RoFormer')
    if not np.isfinite(args.start) or args.start < 0 or args.threads <= 0:
        parser.error('start must be finite >= 0, threads must be > 0')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('duration must be finite and > 0')
    if args.limit is not None and args.limit <= 0:
        parser.error('limit must be > 0')
    if args.save_audio_limit is not None and (args.save_audio_limit<=0 or not args.save_audio):
        parser.error('--save-audio-limit requires --save-audio and a positive value')
    if args.system == 'causal' and not args.checkpoint:
        parser.error('causal separation requires --checkpoint; random weights are not a separation model')
    if args.input and (args.condition != 'mixture' or args.bsseval):
        parser.error('controlled conditions / BSSEval require HQ reference stems')
    if args.bsseval and args.condition != 'mixture':
        parser.error('BSSEval vocal/accompaniment is only reported for standard mixture, not restoration tasks')
    return args


def inputs(args, expected_rate):
    if args.root:
        tracks = discover_hq(args.root, args.subset, expected_rate)
        if args.track:
            requested = set(args.track)
            absent = requested - {t.name for t in tracks}
            if absent:
                raise ValueError(f'Unknown track names: {sorted(absent)}')
            tracks = [t for t in tracks if t.name in requested]
        if args.limit:
            tracks = tracks[:args.limit]
        for track in tracks:
            arrays, segment = load_segment(track, args.start, args.duration)
            if args.condition == 'mixture':
                audio, ref = arrays['mixture'], arrays['vocals']
                metadata = {"mode": "mixture", "task": "music_vocal_source_separation",
                            "comparison_group": "music_vocal_source_separation",
                            "input": "official_mixture.wav", "reference": "official_vocals.wav",
                            "reference_is_stem_not_guaranteed_dry_close_mic": True}
            else:
                audio, ref, metadata = make_condition(arrays['vocals'], arrays['accompaniment'],
                    expected_rate, mode=args.condition, seed=args.seed, snr_db=args.snr_db,
                    bleed_db=args.bleed_db, rt60_seconds=args.rt60)
            yield track.name, audio, ref, arrays['accompaniment'], segment, metadata
    else:
        info = sf.info(args.input)
        if info.samplerate != expected_rate or info.channels != 2:
            raise ValueError(f'Input must be stereo {expected_rate} Hz. Found {info.channels} channels {info.samplerate} Hz; resample explicitly first')
        begin = round(args.start * expected_rate)
        if begin >= info.frames:
            raise ValueError('start is beyond the input')
        end = info.frames if args.duration is None else min(info.frames, begin + round(args.duration * expected_rate))
        audio, _ = sf.read(args.input, start=begin, stop=end, always_2d=True, dtype='float32')
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Empty or invalid input')
        segment = {"start_sample": begin, "samples": len(audio), "duration_seconds": len(audio)/expected_rate,
                   "input_sha256": _hash(args.input)}
        yield args.input.stem, audio, None, None, segment, {"task": "qualitative_demo", "reference": None}


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    out = args.out or PROJECT / 'results' / 'evaluations' / stamp
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {out}')
    out.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {"status": "running", "created_utc": stamp, "configuration": config,
              "environment": None, "tracks": [], "end_to_end_audio_latency_measured": False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        report['stage'] = 'model_setup'
        report['environment'] = environment()
        if args.system == 'openunmix':
            separator = OpenUnmixBaseline(args.model, args.niter, args.device)
            model_info = separator.metadata()
            rate = separator.sample_rate
        elif args.system == 'melroformer':
            from vocalstream.roformer import MelRoFormerBaseline
            separator = MelRoFormerBaseline(device=args.device, overlap=args.overlap,
                                           precision=args.precision, tta=args.tta)
            model_info = separator.metadata()
            rate = separator.sample_rate
        elif args.system == 'bsroformer':
            from vocalstream.bsroformer import BSRoFormerBaseline
            separator = BSRoFormerBaseline(device=args.device, overlap=args.overlap, precision=args.precision,
                                          tta=args.tta, chunk_seconds=args.chunk_seconds, heads=args.bs_heads)
            model_info = separator.metadata()
            rate = separator.sample_rate
        else:
            from vocalstream.causal import load_checkpoint, benchmark_stream
            model, causal_config, checkpoint = load_checkpoint(args.checkpoint, device=args.device)
            rate = model.config.sample_rate
            model_info = {"name": "CausalMaskGRU educational baseline", "streaming": True,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "checkpoint_sha256": _hash(args.checkpoint), "checkpoint": str(args.checkpoint),
                          "training_dataset": checkpoint.get('training', checkpoint.get('data', {})),
                          "causal_config": vars(causal_config), "latency": causal_config.latency_info(),
                          "trained_optimizer_steps": model.trained_optimizer_steps,
                          "timing_scope": "sum of per-hop analysis + GRU mask + overlap-add and final flush; excludes load, I/O, transfer, loop assembly and warmup"}
        report['model'] = model_info
        report['stage'] = 'dataset_setup'
        report['dataset_metadata'] = dataset_info(args.root) if args.root else {'kind': 'local_audio_without_reference', 'formal_benchmark': False}
        report['dataset'] = report['dataset_metadata']['kind']
        report['benchmark_protocol'] = 'pilot_or_demo'
        if args.root:
            full_count = len(discover_hq(args.root, args.subset, rate))
            if args.subset == 'test' and full_count == 50 and not args.track and not args.limit and not args.duration and args.start == 0 and args.condition == 'mixture' and report['dataset'] == 'MUSDB18-HQ' and report['dataset_metadata'].get('formal_benchmark', False):
                report['benchmark_protocol'] = 'full_test_set_whole_tracks'
        report['stage'] = 'separation_and_evaluation'
        for index, (name, audio, reference, backing, segment, condition) in enumerate(inputs(args, rate)):
            if args.system == 'causal' and name in checkpoint.get('training', {}).get('training_tracks', []):
                raise ValueError(f'{name} was used to train this checkpoint. Evaluate held-out validation/test songs instead')
            print(f'{index+1}: {name} ({len(audio)/rate:.2f}s)', flush=True)
            call_started = time.perf_counter()
            if args.system in ('openunmix', 'melroformer', 'bsroformer'):
                estimate, seconds = separator.separate(audio)
                call_seconds = time.perf_counter() - call_started
                timing = {"inference_seconds": seconds, "inference_rtf": seconds/(len(audio)/rate),
                          "scope": model_info['timing_scope'], "block_rtr": None}
                if args.system in ('melroformer', 'bsroformer'):
                    runtime = separator.metadata()
                    timing['model_runtime'] = runtime
                    timing['adapter_processing_seconds'] = runtime['last_run']['processing_seconds']
                    timing['adapter_processing_rtf'] = runtime['last_run']['processing_rtf']
                    timing['adapter_processing_scope'] = runtime['processing_timing_scope']
            else:
                # API returns an aligned waveform and measured per-hop records.
                waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
                output_tensor, stream_report = benchmark_stream(waveform, model, device=args.device)
                estimate = output_tensor.numpy().T.copy()
                call_seconds = time.perf_counter() - call_started
                block_times = stream_report['block_processing_seconds']
                deadline = causal_config.hop_length / rate
                timing = {"stream": stream_report,
                          "block_timing": timing_summary(block_times, [deadline] * len(block_times)),
                          "inference_rtf": stream_report['whole_file_rtf_including_flush'],
                          "scope": model_info['timing_scope']}
            timing.update({
                "processing_call_seconds": call_seconds,
                "processing_call_rtf": call_seconds / (len(audio) / rate),
                "processing_call_scope": "one separation call from in-memory input to CPU waveform; includes array preparation, device transfers and output assembly; excludes model loading, file I/O, quality metrics and audio-device buffers",
                "processing_call_includes_warmup": args.system == 'causal',
                "model_warmup_applied": model_info.get('warmup_applied', args.system in ('causal', 'openunmix')),
            })
            if estimate.shape != audio.shape or not np.isfinite(estimate).all():
                raise ValueError('Separator produced invalid or misaligned output')
            quality = separation_metrics(reference, estimate, audio) if reference is not None else {"status": "not_measured", "reason": "no time-aligned vocal reference"}
            segment['processed_input_float32_sha256'] = _audio_hash(audio)
            segment['reference_float32_sha256'] = _audio_hash(reference) if reference is not None else None
            row = {"track": name, "segment": segment, "condition": condition,
                   "quality": quality, "timing": timing,
                   "input_peak": float(np.max(np.abs(audio))), "estimate_peak": float(np.max(np.abs(estimate)))}
            if args.bsseval:
                from vocalstream.bsseval import evaluate_vocal_accompaniment
                begin = time.perf_counter()
                row['bsseval_v4'] = evaluate_vocal_accompaniment(reference, backing, estimate, audio, rate)
                row['bsseval_v4']['evaluation_seconds'] = time.perf_counter() - begin
            full_audio_export = args.save_audio and (args.save_audio_limit is None or index < args.save_audio_limit)
            if full_audio_export or args.save_estimates:
                folder = out / f'track_{index+1:03d}'
                folder.mkdir()
                sf.write(folder/'vocals_estimate.wav', estimate, rate, subtype='FLOAT')
                if full_audio_export:
                    sf.write(folder/'input.wav', audio, rate, subtype='FLOAT')
                if full_audio_export and reference is not None:
                    sf.write(folder/'vocals_reference.wav', reference, rate, subtype='FLOAT')
                row['audio_directory'] = str(folder)
                row['audio_export_scope'] = 'input_reference_estimate' if full_audio_export else 'estimate_only'
            report['tracks'].append(row)
            (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        report['status'] = 'complete'
        scores = [r['quality'].get('si_sdr_db') for r in report['tracks']]
        scores = [v for v in scores if v is not None]
        report['aggregate'] = {"si_sdr_track_median_db": float(np.median(scores)) if scores else None,
                               "valid_si_sdr_tracks": len(scores), "track_count": len(report['tracks']),
                               "si_sdr_aggregation": "median of one whole-segment joint-stereo score per selected track"}
        bss_scores = [r.get('bsseval_v4', {}).get('vocals_sdr_median_db') for r in report['tracks']]
        bss_scores = [v for v in bss_scores if v is not None]
        report['aggregate']['bsseval_v4_median_tracks_median_frames_vocals_sdr_db'] = float(np.median(bss_scores)) if bss_scores else None
        report['aggregate']['valid_bsseval_v4_sdr_tracks'] = len(bss_scores)
        report['stage'] = 'complete'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        report['error'] = 'KeyboardInterrupt: run interrupted by the user'
        raise
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    fields = ['track', 'si_sdr_db', 'si_sdri_db', 'inference_rtf', 'processing_call_rtf',
              'inference_scope', 'processing_call_includes_warmup',
              'mean_rtr', 'p95_rtr', 'max_rtr', 'deadline_miss_count',
              'global_residual_sdr_db', 'normalized_waveform_rmse', 'reference_projection_gain']
    with (out/'tracks.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report['tracks']:
            timing = row['timing']
            block = timing.get('block_timing', timing)
            writer.writerow({"track": row['track'], **{k: row['quality'].get(k) for k in fields[1:3]},
                             "inference_rtf": timing.get('inference_rtf'),
                             "processing_call_rtf": timing.get('processing_call_rtf'),
                             "inference_scope": timing.get('scope'),
                             "processing_call_includes_warmup": timing.get('processing_call_includes_warmup'),
                             **{k: block.get(k) for k in fields[7:11]},
                             **{k: row['quality'].get(k) for k in fields[11:]}})
    print(f'Report: {out / "report.json"}')
    print(f'Protocol: {report["benchmark_protocol"]}, dataset: {report["dataset"]}')


if __name__ == '__main__':
    main()

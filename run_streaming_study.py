"""Declare and run a short bounded-future scheduling study on development clips.

This is one timing observation per setting, not an audio-device benchmark.
The opt-in compare mode reruns nested OLA and single-forward inference under
the same bounded-future schedules; historical reports are never reused as a
new timing baseline.
Every worker loads and warms up its own pinned pretrained model.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys

from project_paths import resolve_dataset_root

from run_project import (PROJECT, JobRunner, digest, exclusive_run, read_json,
                         runtime_snapshot, write_json)


TRACKS = ['Actions - One Minute Smile', 'Alexander Ross - Goodbye Bolero', 'ANiMAL - Rockshow']
SETTINGS = [
    {'name': 'emit20_look0', 'emit_seconds': .020, 'lookahead_seconds': .000},
    {'name': 'emit20_look40', 'emit_seconds': .020, 'lookahead_seconds': .040},
    {'name': 'emit20_look160', 'emit_seconds': .020, 'lookahead_seconds': .160},
    {'name': 'emit80_look40', 'emit_seconds': .080, 'lookahead_seconds': .040},
]


def study_settings(mode="ola"):
    """Keep historical OLA names while giving each new policy a unique name."""
    modes = ("ola", "single") if mode == "compare" else (mode,)
    if any(value not in ("ola", "single") for value in modes):
        raise ValueError("Unknown window inference mode")
    return [dict(setting, name=setting["name"] if policy == "ola" else "single_" + setting["name"],
                 window_inference=policy)
            for setting in SETTINGS for policy in modes]


def inference_sources():
    # A separately written analysis/report generator cannot change inference
    # identity while workers are running. Only used inference sources are pinned.
    names = ['run_streaming_study.py', 'project_paths.py', 'run_project.py', 'evaluate_project.py',
             'offline_stream_demo.py', 'vocalstream/__init__.py',
             'vocalstream/streaming.py', 'vocalstream/window_inference.py', 'vocalstream/bsroformer.py',
             'vocalstream/roformer.py', 'vocalstream/baselines.py',
             'vocalstream/data.py', 'vocalstream/metrics.py', 'vocalstream/conditions.py']
    names += [p.relative_to(PROJECT).as_posix()
              for p in (PROJECT/'vocalstream/vendor/zfturbo_bsroformer').glob('*.py')]
    return {name: digest(PROJECT/name) for name in sorted(names)}


def gpu_snapshot():
    try:
        value = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'], text=True, timeout=15)
        return {'columns': 'name,temp_C,power_W,graphics_MHz,memory_MHz,utilization_percent,memory_used_MiB',
                'value': value.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {'unavailable_reason': str(error)}


def stream_job_name(index, setting):
    return f'stream_track{index:02d}_{setting["name"]}'


def offline_command(output, config):
    args = [sys.executable, str(PROJECT/'evaluate_project.py'),
            '--root', config['root'], '--subset', 'train', '--system', 'bsroformer',
            '--bs-heads', 'vocals', '--precision', 'amp', '--chunk-seconds', '2',
            '--overlap', '2', '--device', 'cuda', '--threads', '1', '--seed', '2026',
            '--condition', 'mixture', '--start', str(config['start_seconds']),
            '--duration', str(config['duration_seconds']), '--save-estimates', '--out', str(output)]
    for name in config['tracks']:
        args += ['--track', name]
    return args


def streaming_command(output, config, track, setting):
    return [sys.executable, str(PROJECT/'offline_stream_demo.py'),
            '--input', str(track.path/'mixture.wav'), '--reference', str(track.path/'vocals.wav'),
            '--output', str(output), '--start', str(config['start_seconds']),
            '--duration', str(config['duration_seconds']), '--context-seconds', '2',
            '--emit-seconds', str(setting['emit_seconds']),
            '--lookahead-seconds', str(setting['lookahead_seconds']),
            '--device', 'cuda', '--precision', 'amp', '--heads', 'vocals',
            '--window-inference', setting.get('window_inference', 'ola')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Dataset path; defaults to MUSDB_ROOT or data/musdb18-hq')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=float, default=60.0)
    parser.add_argument('--duration', type=float, default=4.0)
    parser.add_argument('--window-inference', choices=('ola', 'single', 'compare'), default='ola',
                        help='compare reruns both policies with matched schedules in the same environment')
    args = parser.parse_args()
    if not math.isfinite(args.start) or args.start < 0 or not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('start must be finite >=0; duration must be finite >0')
    sample_count = round(args.duration*44100)
    if sample_count < 1:
        parser.error('duration is shorter than one sample')
    settings = study_settings(args.window_inference)
    job_count = 1 + len(TRACKS) * len(settings)
    from vocalstream.data import dataset_info, discover_hq, load_segment
    from evaluate_project import _audio_hash

    root = resolve_dataset_root(args.root)
    official = read_json(PROJECT/'project_config.json')['validation_tracks']
    if not set(TRACKS) <= set(official):
        raise ValueError('Declared songs must belong to the reserved official development split')
    inventory = discover_hq(root, 'train')
    tracks = {track.name: track for track in inventory if track.name in TRACKS}
    if set(tracks) != set(TRACKS):
        raise ValueError('Missing one or more declared development tracks')
    metadata = dataset_info(root)
    if len(inventory) != 100 or metadata['kind'] != 'MUSDB18-HQ':
        raise ValueError('This study requires the complete MUSDB18-HQ train inventory')
    config = {'schema_version': 2, 'root': str(root), 'tracks': TRACKS,
              'start_seconds': args.start, 'duration_seconds': args.duration,
              'sample_rate': 44100, 'requested_samples': sample_count,
              'device': 'cuda', 'precision': 'amp', 'heads': 'vocals',
              'context_seconds': 2.0, 'overlap': 2, 'threads': 1, 'seed': 2026,
              'overlap_scope': 'offline reference and ola policy; single policy uses one full-context forward',
              'settings': settings, 'window_inference_study': args.window_inference,
              'model_forward_counts_recorded': True, 'timing_trials_per_track_setting': 1,
              'expected_jobs': job_count,
              'scope': 'three short official development clips; simulated file streaming, no training/test/device I/O',
              'quality_protocol': 'whole clip plus one identical contiguous real-context interval shared by all streaming settings and the offline reference; undefined if empty'}
    segments = []
    for name in TRACKS:
        track = tracks[name]
        arrays, info = load_segment(track, args.start, args.duration)
        if info['samples'] != sample_count:
            raise ValueError(f'The requested segment is shorter than declared: {name}')
        segments.append({'track': name, 'segment': info,
                         'mixture_path': str((track.path/'mixture.wav').resolve()),
                         'reference_path': str((track.path/'vocals.wav').resolve()),
                         'mixture_float32_sha256': _audio_hash(arrays['mixture']),
                         'vocals_float32_sha256': _audio_hash(arrays['vocals'])})
    assets = {name: digest(PROJECT/'model_cache/bs_roformer_hq'/name)
              for name in ('model.ckpt', 'config.yaml', 'provenance.json')}
    context = {'configuration': config, 'source_hashes': inference_sources(),
               'runtime': runtime_snapshot(), 'model_asset_sha256': assets,
               'data': {'metadata': metadata, 'segments': segments,
                        'train_inventory_count': len(inventory), 'test_set_accessed': False,
                        'pretrained_training_split_verified': False}}
    out = args.out.expanduser().resolve()
    with exclusive_run(out):
        protocol = out/'protocol.json'
        if protocol.exists() and read_json(protocol) != context:
            raise ValueError('Protocol/data/source/runtime changed; select a new output directory')
        if not protocol.exists():
            # Refuse to adopt an unrelated folder as an experiment directory.
            if any(path.name != '.run.lock' for path in out.iterdir()):
                raise ValueError('Output is not an empty/new study folder')
            write_json(protocol, context)
        runner = JobRunner(out, digest(protocol), context)
        runner.state.update(status='running', job_count=job_count)
        runner.state.pop('error', None)
        runner.save()
        jobs = [('offline_amp2s', 'offline', None, None)]
        # Rotate setting order by track; still only one observation per setting.
        for index, name in enumerate(TRACKS, 1):
            order = settings[index-1:] + settings[:index-1]
            if index % 2 == 0:
                order = list(reversed(order))
            jobs += [(stream_job_name(index, setting), 'streaming', name, setting) for setting in order]
        try:
            for name, stage, track, setting in jobs:
                if inference_sources() != context['source_hashes']:
                    raise ValueError('Inference code changed during the run; use a new output folder')
                if stage == 'offline':
                    command = lambda output: offline_command(output, config)
                else:
                    command = lambda output, t=track, s=setting: streaming_command(output, config, tracks[t], s)
                previously_complete = runner.state['jobs'].get(name, {}).get('status') == 'complete'
                runner.job(name, 'evaluation', command, stage=stage, track=track,
                           setting=setting, repeat=1, gpu_before=None if previously_complete else gpu_snapshot(),
                           inference_source_hashes=context['source_hashes'])
                if not previously_complete:
                    runner.state['jobs'][name]['gpu_after'] = gpu_snapshot()
                runner.save()
                if inference_sources() != context['source_hashes']:
                    raise ValueError('Inference code changed during a worker; preserve evidence and use a new folder')
            if {name: digest(PROJECT/'model_cache/bs_roformer_hq'/name) for name in assets} != assets:
                raise ValueError('Model assets changed during the run')
            runner.state['status'] = 'complete'
        except BaseException as error:
            runner.state.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            runner.save()
    print(f'Streaming study complete: {out/"run.json"}', flush=True)
    print(f'Build its analysis with: python build_streaming_study_report.py --run "{out}"', flush=True)


if __name__ == '__main__':
    main()

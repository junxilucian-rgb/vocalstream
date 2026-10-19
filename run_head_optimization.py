"""Run a declared inference-only vocal-head ablation on development audio."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

from project_paths import resolve_dataset_root

from run_project import (PROJECT, JobRunner, digest, exclusive_run, read_json,
                         runtime_snapshot, write_json)
from run_refinement import gpu_snapshot


def inference_sources():
    # Only files used by these jobs, rather than unrelated streaming/plotting
    # work which may be developed concurrently. Vendor source is also verified
    # against the original provenance inside the separator constructor.
    names = ['run_head_optimization.py', 'project_paths.py', 'run_project.py', 'run_refinement.py',
             'evaluate_project.py', 'vocalstream/__init__.py', 'vocalstream/baselines.py',
             'vocalstream/bsroformer.py', 'vocalstream/roformer.py',
             'vocalstream/data.py', 'vocalstream/conditions.py', 'vocalstream/metrics.py']
    names += [p.relative_to(PROJECT).as_posix()
              for p in (PROJECT/'vocalstream/vendor/zfturbo_bsroformer').glob('*.py')]
    return {name: digest(PROJECT/name) for name in sorted(names)}


def validate(config, official):
    if config.get('schema_version') != 1 or config['device'] not in ('cpu', 'cuda'):
        raise ValueError('Expected schema 1 and explicit cpu/cuda')
    for name in ('threads', 'timing_repeats'):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f'Invalid {name}')
    if config['timing_repeats'] < 3 or config['overlap'] != 2:
        raise ValueError('This ablation requires three repeats and overlap=2')
    for name in ('start_seconds', 'duration_seconds'):
        if not math.isfinite(config[name]) or config[name] < (0 if name == 'start_seconds' else 1e-9):
            raise ValueError(f'Invalid {name}')
    expected = [{'name': 'all_fp32', 'heads': 'all', 'precision': 'fp32'},
                {'name': 'vocal_fp32', 'heads': 'vocals', 'precision': 'fp32'},
                {'name': 'all_amp', 'heads': 'all', 'precision': 'amp'},
                {'name': 'vocal_amp', 'heads': 'vocals', 'precision': 'amp'}]
    if config['models'] != expected:
        raise ValueError('Keep the four declared single-factor comparisons')
    if config['device'] != 'cuda':
        raise ValueError('Declared AMP comparison requires CUDA')
    if not config['timing_tracks'] or not set(config['timing_tracks']) <= set(official):
        raise ValueError('Timing audio must belong to the reserved validation set')
    if len(official) != 14 or len(set(official)) != 14:
        raise ValueError('Need the 14 distinct official validation tracks')
    for key in ('max_normalized_waveform_rmse', 'max_absolute_per_song_si_sdr_delta_db'):
        value = config['numerical_comparison'][key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid numerical tolerance: {key}')


def command(output, config, model, tracks, save_estimates):
    args = [sys.executable, str(PROJECT/'evaluate_project.py'),
            '--root', str(Path(config['root']).resolve()), '--subset', 'train',
            '--system', 'bsroformer', '--bs-heads', model['heads'],
            '--precision', model['precision'], '--overlap', str(config['overlap']),
            '--device', config['device'], '--threads', str(config['threads']),
            '--seed', str(config['seed']), '--condition', 'mixture',
            '--start', str(config['start_seconds']), '--duration', str(config['duration_seconds']),
            '--out', str(output)]
    for name in sorted(tracks):
        args += ['--track', name]
    if save_estimates:
        args += ['--save-estimates']
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'head_optimization_experiments.json')
    parser.add_argument('--root', type=Path, help='Dataset path; overrides MUSDB_ROOT and config root')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    config['root'] = str(resolve_dataset_root(args.root, config.get('root')))
    official = read_json(PROJECT/'project_config.json')['validation_tracks']
    validate(config, official)
    from vocalstream.data import dataset_info, discover_hq, load_segment
    from evaluate_project import _audio_hash
    root = Path(config['root']).resolve()
    tracks = discover_hq(root, 'train')
    if len(tracks) != 100 or not set(official) <= {t.name for t in tracks}:
        raise ValueError('Require complete MUSDB18-HQ train and its official validation songs')
    segments = []
    for track in tracks:
        if track.name not in official:
            continue
        arrays, info = load_segment(track, config['start_seconds'], config['duration_seconds'])
        if info['duration_seconds'] != config['duration_seconds']:
            raise ValueError(f'Segment shorter than declared: {track.name}')
        segments.append({'track': track.name, 'segment': info,
                         'mixture_sha256': _audio_hash(arrays['mixture']),
                         'vocals_sha256': _audio_hash(arrays['vocals'])})
    context = {'configuration': config, 'source_hashes': inference_sources(),
               'runtime': runtime_snapshot(),
               'validation_tracks': official,
               'data': {'metadata': dataset_info(root), 'segments': segments,
                        'test_set_accessed': False, 'pretrained_training_split_verified': False}}
    out = args.out.resolve()
    with exclusive_run(out):
        path = out/'protocol.json'
        if path.exists() and read_json(path) != config:
            raise ValueError('Protocol changed; choose a new output directory')
        if not path.exists():
            write_json(path, config)
        runner = JobRunner(out, digest(path), context)
        jobs = [(f"quality_{m['name']}", m, official, True, 1, 'quality') for m in config['models']]
        # Balance the timing order across repeats, separating cold starts and
        # model warmup (inside each worker) from all reported processing times.
        orders = [config['models'], list(reversed(config['models'])),
                  config['models'][2:] + config['models'][:2]]
        for repeat in range(1, config['timing_repeats'] + 1):
            for model in orders[(repeat - 1) % len(orders)]:
                jobs.append((f"timing_r{repeat}_{model['name']}", model,
                             config['timing_tracks'], False, repeat, 'timing'))
        runner.state.update(status='running', job_count=len(jobs))
        runner.state.pop('error', None)
        runner.save()
        try:
            for name, model, selected, save, repeat, stage in jobs:
                if inference_sources() != context['source_hashes']:
                    raise ValueError('Inference source changed during run; preserve this run and create a new one')
                runner.job(name, 'evaluation',
                           lambda output, m=model, t=selected, s=save: command(output, config, m, t, s),
                           model_id=model['name'], condition='mixture', repeat=repeat,
                           stage=stage, gpu_before=gpu_snapshot())
                runner.state['jobs'][name]['gpu_after'] = gpu_snapshot()
                runner.save()
            runner.state['status'] = 'complete'
        except BaseException as error:
            runner.state.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            runner.save()
    print(f'Head ablation complete: {out/"run.json"}', flush=True)


if __name__ == '__main__':
    main()

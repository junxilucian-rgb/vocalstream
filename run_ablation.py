"""Train a fixed grid, evaluate validation songs, never use test to select it."""
import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from vocalstream.data import dataset_info

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--crop-seconds', type=float, default=2)
    parser.add_argument('--eval-seconds', type=float, default=15)
    parser.add_argument('--grid', type=Path, default=PROJECT / 'experiments.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.steps <= 0 or args.crop_seconds <= 0 or args.eval_seconds <= 0:
        parser.error('steps and durations must be > 0')
    grid = json.loads(args.grid.read_text(encoding='utf-8'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    out = args.out or PROJECT / 'results' / 'ablations' / stamp
    if out.exists():
        raise FileExistsError('Choose a new output directory; existing runs are preserved')
    out.mkdir(parents=True)
    run = {'grid': grid, 'root': str(args.root.resolve()), 'status': 'running', 'results': [],
           'test_set_accessed': False, 'steps_per_model': args.steps,
           'dataset_metadata': dataset_info(args.root)}
    run['dataset'] = run['dataset_metadata']['kind']

    def save():
        (out/'ablation.json').write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    try:
        for experiment in grid['experiments']:
            name = experiment['name']
            if name != Path(name).name or name in ('.', '..'):
                raise ValueError('Experiment name must be a simple folder name')
            folder = out / name
            folder.mkdir()
            train = folder / 'training'
            commands = [[sys.executable, str(PROJECT/'train_causal.py'), '--root', str(args.root.resolve()),
                         '--output', str(train), '--steps', str(args.steps), '--batch-size', '1',
                         '--crop-seconds', str(args.crop_seconds), '--validate-every', str(min(25, args.steps)),
                         '--seed', str(grid['seed']), '--device', args.device, '--threads', '1',
                         '--n-fft', str(experiment['n_fft']), '--hop-length', str(experiment['hop_length']),
                         '--hidden-size', str(experiment['hidden_size'])]]
            print(f'Training: {name}', flush=True)
            subprocess.run(commands[0], check=True)
            log = json.loads((train/'training_log.json').read_text(encoding='utf-8'))
            validation_names = log['training']['validation_tracks']
            evaluation = [sys.executable, str(PROJECT/'evaluate_project.py'), '--root', str(args.root.resolve()),
                          '--system', 'causal', '--checkpoint', str(train/'best.pt'), '--subset', 'train',
                          '--duration', str(args.eval_seconds), '--out', str(folder/'validation'),
                          '--device', args.device, '--threads', '1']
            for track in validation_names:
                evaluation.extend(['--track', track])
            commands.append(evaluation)
            subprocess.run(evaluation, check=True)
            report = json.loads((folder/'validation'/'report.json').read_text(encoding='utf-8'))
            run['results'].append({'experiment': experiment, 'commands': commands,
                                   'evaluation_report': str(folder/'validation'/'report.json'),
                                   'validation_quality': report['aggregate'], 'latency': log['latency']})
            save()
        run['status'] = 'complete'
    except Exception as exc:
        run['status'] = 'failed'
        run['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    print(f'Validation ablation: {out / "ablation.json"}')
    print('Freeze the chosen configuration before a separate full test-set evaluation.')


if __name__ == '__main__':
    main()

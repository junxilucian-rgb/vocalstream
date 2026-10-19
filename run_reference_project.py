"""Current project: two pretrained architectures, with BS-RoFormer ablations.

run --stage pilot|ablation|validation; freeze after full validation;
run --stage test --frozen ... evaluates the selected setting plus Open-Unmix.
"""
import argparse
import json
from pathlib import Path
import re
import sys

from project_paths import resolve_dataset_root

from run_project import (PROJECT, JobRunner, digest, exclusive_run, make_run_report,
                         read_json, runtime_snapshot, source_snapshot, stamp, write_json)


def validate(config):
    if config.get('schema_version') != 1 or config['device'] not in ('cpu', 'cuda') or config['threads'] < 1:
        raise ValueError('Invalid reference configuration')
    names = [e['name'] for e in config['experiments']]
    if len(names) != len(set(names)) or any(not re.fullmatch(r'[a-z][a-z0-9_]*', n) for n in names):
        raise ValueError('Experiment names must be unique safe names')
    for experiment in config['experiments']:
        if experiment['system'] not in ('openunmix', 'bsroformer'):
            raise ValueError('Current project accepts Open-Unmix and BS-RoFormer only')
        if 'bs_heads' in experiment and (experiment['system'] != 'bsroformer' or experiment['bs_heads'] not in ('all', 'vocals')):
            raise ValueError('bs_heads requires BS-RoFormer and all/vocals')
    for selection in config['stages'].values():
        if not selection or set(selection)-set(names):
            raise ValueError('Invalid experiment selection')
    if config['pilot_start'] < 0 or config['pilot_duration'] <= 0 or not config['pilot_tracks']:
        raise ValueError('Invalid pilot audio selection')


def dataset(root, config, stage):
    from vocalstream.data import dataset_info, discover_hq
    metadata = dataset_info(root)
    if metadata['kind'] != 'MUSDB18-HQ' or not metadata.get('formal_benchmark'):
        raise ValueError('This reference project uses the full HQ dataset')
    split = read_json(PROJECT/'project_config.json')['validation_tracks']
    train = discover_hq(root, 'train')
    if len(train) != 100 or not set(split).issubset({t.name for t in train}):
        raise ValueError('Expected 100 train songs including all 14 official validation names')
    if not set(config['pilot_tracks']).issubset(set(split)):
        raise ValueError('Pilot tracks must belong to the reserved validation list')
    tracks = config['pilot_tracks'] if stage in ('pilot', 'ablation') else split
    if stage == 'test':
        test = discover_hq(root, 'test')
        if len(test) != 50 or {t.name for t in test} & {t.name for t in train}:
            raise ValueError('Expected 50 separate test songs')
        tracks = [t.name for t in test]
    return {'metadata': metadata, 'evaluation_tracks': sorted(tracks), 'reserved_validation_tracks': sorted(split),
            'pretrained_bs_training_split_verified': False, 'test_set_accessed': stage == 'test'}


def build_command(output, root, config, stage, experiment, tracks):
    command = [sys.executable, str(PROJECT/'evaluate_project.py'), '--root', str(root),
               '--subset', 'test' if stage == 'test' else 'train', '--system', experiment['system'],
               '--device', config['device'], '--threads', str(config['threads']), '--seed', str(config['seed']), '--out', str(output)]
    if stage in ('pilot', 'ablation'):
        command += ['--start', str(config['pilot_start']), '--duration', str(config['pilot_duration']), '--save-audio']
    # Passing no --track on final test activates the whole official test protocol.
    if stage != 'test':
        for track in tracks:
            command += ['--track', track]
    if stage == 'test':
        command += ['--bsseval']
    for key in ('model', 'niter', 'overlap', 'precision', 'chunk_seconds', 'bs_heads'):
        if key in experiment:
            command += ['--'+key.replace('_', '-'), str(experiment[key])]
    return command


def verify_reused(path, root, config, stage, experiment, tracks):
    if stage not in ('pilot', 'ablation'):
        raise ValueError('Reuse is only for pilot/ablation reports')
    report = read_json(path)
    c = report['configuration']
    expected = {'root': str(root), 'subset': 'train', 'system': experiment['system'], 'device': config['device'],
                'threads': config['threads'], 'seed': config['seed'], 'start': config['pilot_start'],
                'duration': config['pilot_duration'], 'condition': 'mixture', 'bsseval': False}
    expected.update({key: value for key, value in experiment.items() if key not in ('name', 'system')})
    if report['status'] != 'complete' or any(c.get(k) != v for k,v in expected.items()) or sorted(t['track'] for t in report['tracks']) != tracks:
        raise ValueError(f'Existing report does not match this experiment: {path}')
    current = runtime_snapshot()['packages']
    for name, version in report['environment']['versions'].items():
        if name in current and current[name] != version:
            raise ValueError('Existing report uses a different runtime')
    return report


def run(args):
    config = read_json(args.config)
    config['root'] = str(resolve_dataset_root(args.root, config.get('root')))
    validate(config)
    root = Path(config['root']).resolve()
    experiments = {e['name']: e for e in config['experiments']}
    selected = config['stages'].get(args.stage)
    if args.stage == 'test':
        if not args.frozen:
            raise ValueError('Test requires --frozen from completed validation')
        frozen = read_json(args.frozen)
        if frozen['configuration'] != config or frozen['source_hashes'] != source_snapshot() or frozen['runtime'] != runtime_snapshot():
            raise ValueError('Frozen configuration/code/runtime changed')
        if digest(frozen['validation_run']) != frozen['validation_run_sha256']:
            raise ValueError('Frozen validation record changed')
        selected = ['openunmix_hq', frozen['selected_experiment']]
    if not selected:
        raise ValueError('No experiments selected')
    data = dataset(root, config, args.stage)
    context = {'configuration': config, 'stage': args.stage, 'data': data,
               'source_hashes': source_snapshot(), 'runtime': runtime_snapshot()}
    identity = json.dumps(context, sort_keys=True)
    with exclusive_run(args.out):
        runner = JobRunner(args.out.resolve(), identity, context)
        runner.state.update(status='running')
        runner.state.pop('error', None)
        runner.save()
        try:
            reused = {}
            for value in args.reuse or []:
                key, path = value.split('=', 1)
                if key not in selected or key in reused:
                    raise ValueError('Reuse requires a unique selected experiment ID')
                reused[key] = Path(path).resolve()
            for name in selected:
                experiment = experiments[name]
                if name in reused and name not in runner.state['jobs']:
                    path = reused[name]
                    verify_reused(path, root, config, args.stage, experiment, data['evaluation_tracks'])
                    runner.state['jobs'][name] = {'kind':'evaluation','status':'complete','attempt':0,
                        'artifact':str(path),'artifact_sha256':digest(path), 'model_id':name,'condition':'mixture','repeat':1,
                        'command':build_command(path.parent,root,config,args.stage,experiment,data['evaluation_tracks']),
                        'execution_note':'existing complete report verified against selected configuration and runtime'}
                    runner.save()
                runner.job(name, 'evaluation', lambda out,e=experiment: build_command(out,root,config,args.stage,e,data['evaluation_tracks']),
                           model_id=name,condition='mixture',repeat=1)
            runner.state['status']='complete'
            make_run_report(runner)
        except BaseException as error:
            runner.state.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            runner.save()
    print(f'Reference project: {args.out.resolve()/"RESULTS.md"}')


def freeze(args):
    if not args.reason.strip():
        raise ValueError('Record a validation-based reason')
    path = args.run/'run.json' if args.run.is_dir() else args.run
    state = read_json(path)
    context=state['context']
    if state['status']!='complete' or context['stage']!='validation':
        raise ValueError('Freeze requires full 14-song validation, not three-song pilot')
    if context['source_hashes']!=source_snapshot() or context['runtime']!=runtime_snapshot():
        raise ValueError('Code/runtime changed after validation')
    experiments={e['name']:e for e in context['configuration']['experiments']}
    if args.experiment not in state['jobs'] or experiments[args.experiment]['system']!='bsroformer':
        raise ValueError('Select a BS-RoFormer setting evaluated in this run')
    for job in state['jobs'].values():
        if job['status']!='complete' or digest(job['artifact'])!=job['artifact_sha256']:
            raise ValueError('Validation artifact incomplete or changed')
    target=path.parent/'frozen_reference.json'
    if target.exists():
        raise FileExistsError('Selection is already frozen')
    write_json(target,{'created_utc':stamp(),'configuration':context['configuration'],
        'source_hashes':context['source_hashes'],'runtime':context['runtime'],
        'selected_experiment':args.experiment,'reason':args.reason,
        'validation_run':str(path.resolve()),'validation_run_sha256':digest(path),'test_used_for_selection':False})
    print(f'Frozen inference setting: {target}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('run')
    p.add_argument('--stage',choices=['pilot','ablation','validation','test'],default='pilot')
    p.add_argument('--root',type=Path,help='Dataset path; overrides MUSDB_ROOT and config root')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=PROJECT/'reference_experiments.json')
    p.add_argument('--reuse',action='append',help='Existing pilot/ablation experiment_id=report.json')
    p.add_argument('--frozen',type=Path)
    p.set_defaults(action=run)
    p=commands.add_parser('freeze')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--experiment',required=True)
    p.add_argument('--reason',required=True)
    p.set_defaults(action=freeze)
    args=parser.parse_args()
    args.action(args)


if __name__=='__main__':
    main()

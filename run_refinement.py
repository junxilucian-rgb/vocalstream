"""Run the declared 14-song development and vocal-preservation experiments."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys

from project_paths import resolve_dataset_root

from run_project import (PROJECT, JobRunner, digest, exclusive_run, make_run_report,
                         read_json, runtime_snapshot, source_snapshot, write_json)


def gpu_snapshot():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu,memory.used',
                                          '--format=csv,noheader,nounits'], text=True, timeout=15)
        return {'columns':'name,temp_C,power_W,graphics_MHz,memory_MHz,utilization_percent,memory_used_MiB', 'value':output.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {'unavailable_reason':str(error)}


def validate(config, official_tracks):
    import math
    if config.get('schema_version')!=1 or config['device'] not in ('cuda','cpu') or config['threads']<1:
        raise ValueError('Invalid refinement configuration')
    if not math.isfinite(config['start_seconds']) or config['start_seconds']<0 or not math.isfinite(config['duration_seconds']) or config['duration_seconds']<=0:
        raise ValueError('Invalid fixed segment')
    if config['timing_repeats']<3 or not isinstance(config['timing_repeats'],int):
        raise ValueError('At least three independent timing repeats are required')
    if not config['timing_tracks'] or not set(config['timing_tracks']).issubset(set(official_tracks)):
        raise ValueError('Timing songs must be reserved validation songs')
    names=[m['name'] for m in config['models']]
    if len(names)!=len(set(names)) or any(not re.fullmatch(r'[a-z][a-z0-9_]*',name) for name in names) or any(m['system'] not in ('openunmix','bsroformer') for m in config['models']):
        raise ValueError('Only the two pretrained model families are supported')
    for selection in ('robustness_models','timing_models'):
        if not config[selection] or set(config[selection])-set(names):
            raise ValueError('Unknown selected model')
    for condition in config['conditions']:
        if not re.fullmatch(r'[a-z][a-z0-9_]*',condition['name']) or condition['mode'] not in ('bleed','clean'):
            raise ValueError('Invalid controlled condition')
        if condition['mode']=='bleed' and (not math.isfinite(condition['bleed_db']) or not -120<=condition['bleed_db']<=60):
            raise ValueError('Invalid bleed amplitude gain')


def command(output, config, model, tracks, condition, save_audio):
    cmd=[sys.executable,str(PROJECT/'evaluate_project.py'),'--root',str(Path(config['root']).resolve()),
         '--subset','train','--start',str(config['start_seconds']),'--duration',str(config['duration_seconds']),
         '--device',config['device'],'--threads',str(config['threads']),'--seed',str(config['seed']),
         '--system',model['system'],'--condition',condition['mode'],'--out',str(output)]
    for name in sorted(tracks):
        cmd+=['--track',name]
    for key in ('model','niter','precision','overlap','chunk_seconds'):
        if key in model:
            cmd+=['--'+key.replace('_','-'),str(model[key])]
    if condition['mode']=='bleed':
        cmd+=['--bleed-db',str(condition['bleed_db'])]
    if save_audio:
        cmd+=['--save-audio','--save-audio-limit','3','--save-estimates']
    elif len(tracks)>3 and model['system']=='bsroformer':
        cmd+=['--save-estimates']
    return cmd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'refinement_experiments.json')
    parser.add_argument('--root',type=Path,help='Dataset path; overrides MUSDB_ROOT and config root')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    config=read_json(args.config)
    config['root']=str(resolve_dataset_root(args.root,config.get('root')))
    official=read_json(PROJECT/'project_config.json')['validation_tracks']
    validate(config,official)
    from vocalstream.data import dataset_info, discover_hq, load_segment
    from evaluate_project import _audio_hash
    root=Path(config['root']).resolve()
    tracks=discover_hq(root,'train')
    metadata=dataset_info(root)
    if len(tracks)!=100 or metadata['kind']!='MUSDB18-HQ' or not set(official).issubset({t.name for t in tracks}):
        raise ValueError('Expected complete HQ train with the official validation songs')
    segments=[]
    for track in tracks:
        if track.name in official:
            arrays, info=load_segment(track,config['start_seconds'],config['duration_seconds'])
            if info['duration_seconds']!=config['duration_seconds']:
                raise ValueError(f'Segment is shorter than declared: {track.name}')
            segments.append({'track':track.name,'segment':info,'mixture_sha256':_audio_hash(arrays['mixture']),
                             'vocals_sha256':_audio_hash(arrays['vocals']),'backing_sha256':_audio_hash(arrays['accompaniment'])})
    context={'configuration':config,'data':{'metadata':metadata,'segments':segments,'test_set_accessed':False,
             'pretrained_training_split_verified':False},'source_hashes':source_snapshot(),'runtime':runtime_snapshot()}
    out=args.out.resolve()
    with exclusive_run(out):
        plan_path=out/'protocol.json'
        if plan_path.exists() and read_json(plan_path)!=config:
            raise ValueError('Protocol changed; create a new output directory')
        if not plan_path.exists():
            write_json(plan_path,config)
        runner=JobRunner(out,digest(plan_path),context)
        jobs=[]
        models={m['name']:m for m in config['models']}
        for model in config['models']:
            jobs.append((f"expanded_{model['name']}",model,official,{'mode':'mixture'},False,1,'expanded_quality'))
        for condition in config['conditions']:
            for name in config['robustness_models']:
                jobs.append((f"{condition['name']}_{name}",models[name],official,condition,True,1,'robustness'))
        # Interleave FP32/AMP trials to reduce bias from a single model order.
        for repeat in range(1,config['timing_repeats']+1):
            order=config['timing_models'][::1 if repeat%2 else -1]
            for name in order:
                jobs.append((f"timing_r{repeat}_{name}",models[name],config['timing_tracks'],{'mode':'mixture'},False,repeat,'timing_repeat'))
        runner.state.update(status='running',job_count=len(jobs))
        runner.state.pop('error',None)
        runner.save()
        try:
            for name,model,selected,condition,save_audio,repeat,stage in jobs:
                runner.job(name,'evaluation',lambda output,m=model,t=selected,c=condition,a=save_audio:command(output,config,m,t,c,a),
                    model_id=model['name'],condition=condition.get('name','mixture'),repeat=repeat,stage=stage,gpu_before=gpu_snapshot())
                runner.state['jobs'][name]['gpu_after']=gpu_snapshot()
                runner.save()
            runner.state['status']='complete'
            make_run_report(runner)
        except BaseException as error:
            runner.state.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            runner.save()
    print(f'Refinement complete: {out/"RESULTS.md"}')


if __name__=='__main__':
    main()

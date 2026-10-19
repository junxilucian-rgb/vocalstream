"""Post-hoc fixed dry/wet blending using cached estimates, no model retraining.

The reference is used only for evaluation. Every fixed lambda uses only input
and model output at inference. No per-song/condition oracle is deployed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import numpy as np
import soundfile as sf
from vocalstream.conditions import make_condition
from vocalstream.data import discover_hq, load_segment
from vocalstream.metrics import separation_metrics
from evaluate_project import _audio_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(); out=args.run.resolve()
    state=json.loads((out/'run.json').read_text())
    if state['status']!='complete': raise ValueError('Finish the fixed protocol before post-hoc analysis')
    config=state['context']['configuration']; tracks={t.name:t for t in discover_hq(config['root'],'train')}
    lambdas=[0,.25,.5,.75,1]; conditions=[{'name':'mixture','mode':'mixture'},*config['conditions']]
    results={}
    for condition in conditions:
        job='expanded_bs_amp' if condition['mode']=='mixture' else condition['name']+'_bs_amp'
        report=json.loads(Path(state['jobs'][job]['artifact']).read_text())
        scores={str(lam):[] for lam in lambdas}; per_song=[]
        for row in report['tracks']:
            arrays,segment=load_segment(tracks[row['track']],config['start_seconds'],config['duration_seconds'])
            if condition['mode']=='mixture':
                audio,ref=arrays['mixture'],arrays['vocals']
            else:
                audio,ref,_=make_condition(arrays['vocals'],arrays['accompaniment'],44100,mode=condition['mode'],
                                           seed=config['seed'],bleed_db=condition.get('bleed_db',-20))
            if _audio_hash(audio)!=row['segment']['processed_input_float32_sha256'] or _audio_hash(ref)!=row['segment']['reference_float32_sha256']:
                raise ValueError('Source samples changed since model evaluation')
            estimate,rate=sf.read(Path(row['audio_directory'])/'vocals_estimate.wav',always_2d=True,dtype='float32')
            if rate!=44100 or estimate.shape!=audio.shape: raise ValueError('Cached estimate is not aligned')
            detail={'track':row['track'],'scores':{}}
            for lam in lambdas:
                output=(1-lam)*audio+lam*estimate
                quality=separation_metrics(ref,output,audio)
                detail['scores'][str(lam)]=quality
                scores[str(lam)].append(quality)
            per_song.append(detail)
        aggregate={}
        for lam,values in scores.items():
            aggregate[lam]={key:float(np.median([v[key] for v in values if v[key] is not None]))
                            for key in ('si_sdr_db','si_sdri_db','normalized_waveform_rmse')}
        results[condition['name']]={'condition':condition,'aggregate':aggregate,'per_song':per_song}
    payload={'scope':'post-hoc exploratory analysis on the same development clips; no independent confirmation',
             'rule':'output=(1-lambda)*input+lambda*BS_output; fixed lambda, no reference access at inference',
             'lambda_meaning':{'0':'unchanged input baseline','1':'full BS output'},'lambdas':lambdas,'conditions':results,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'selection_status':'No adaptive gate and no universal lambda frozen. Condition-specific maxima are exploratory, not deployable oracle decisions.'}
    (out/'blend_analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 探索性后处理：输入与分离结果的固定混合','',
           '这项分析在受控实验结束后提出，使用已经保存的输出，不重新运行或训练网络。它属于同一开发数据上的后验探索，需要新的数据确认。','',
           '`输出 = (1−λ) × 输入 + λ × BS人声估计`。每个固定 λ 在推理时只使用输入与模型输出，不读取参考答案。λ=0 是不处理，λ=1 是原模型输出。','',
           '| 输入条件 | λ=0 | λ=.25 | λ=.5 | λ=.75 | λ=1 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,result in results.items():
        lines.append('| '+name+' | '+' | '.join(f"{result['aggregate'][str(lam)]['si_sdr_db']:.3f}" for lam in lambdas)+' |')
    lines+=['','表中为 14 首片段的 SI-SDR 中位数。纯人声 λ=0 的完美一致分数受数值上限限制，应同时看 NRMSE=0。',
            '','这项实验用于检查一种可能的保真取舍：弱污染时保留更多输入可能减少网络损伤；正常混音时同样操作可能留下更多伴奏。',
            '不能根据已知参考答案或条件标签给每首歌曲挑最优 λ，再把那个成绩当成部署结果。下一步若设计自适应门控，必须只根据实际可获取的输入特征决定，并在独立数据上评价。',
            '','这是一种已有的干湿混合思路，当前贡献是可复现的场景实验与失败分析。具体方向值得请教授判断。']
    (out/'EXPLORATORY_BLEND.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(out/'EXPLORATORY_BLEND.md')


if __name__=='__main__': main()

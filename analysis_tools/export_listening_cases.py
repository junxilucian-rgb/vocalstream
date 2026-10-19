"""Create reference/A/B listening packs for declared boundary-case selection."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
import soundfile as sf
from vocalstream.data import discover_hq, load_segment
from vocalstream.conditions import make_condition
from evaluate_project import _audio_hash


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True); args=p.parse_args()
    root=args.run.resolve(); data=json.loads((root/'analysis.json').read_text()); state=json.loads((root/'run.json').read_text())
    cfg=data['configuration']; tracks={t.name:t for t in discover_hq(cfg['root'],'train')}
    cases=list(data['failure_cases'])
    um=json.loads(Path(state['jobs']['bleed_minus30_openunmix_hq']['artifact']).read_text())
    worst=min(um['tracks'],key=lambda r:r['quality']['si_sdri_db'])
    if not any(c['condition']['name']=='bleed_minus30' and c['track']==worst['track'] for c in cases):
        cases.append({'condition':{'name':'bleed_minus30','mode':'bleed','bleed_db':-30},'track':worst['track'],
                      'selection':'lowest Open-Unmix SI-SDR improvement in the weakest-bleed condition'})
    folder=root/'listening_cases'; folder.mkdir(exist_ok=True); keys=[]; rng=random.Random(2026)
    lines=['# 边界案例试听：参考、A、B','',
           '这些片段按最大原声误差或最低改善选取，用于检查困难情况，不是代表性听感抽样。最低改善仍可能是正值；不能把所有边界案例都称为失败。',
           'A/B 标签按固定种子打乱；对应关系保存在 answer_key.json，建议试听并记录后再查看。本项目尚未收集人工听感评分。','']
    for i,case in enumerate(cases,1):
        c=case['condition']; stem=f'case_{i:02d}'; target=folder/stem; target.mkdir(exist_ok=True)
        arrays,_=load_segment(tracks[case['track']],cfg['start_seconds'],cfg['duration_seconds'])
        audio,ref,_=make_condition(arrays['vocals'],arrays['accompaniment'],44100,mode=c['mode'],seed=cfg['seed'],bleed_db=c.get('bleed_db',-20))
        sf.write(target/'input.wav',audio,44100,subtype='FLOAT'); sf.write(target/'reference.wav',ref,44100,subtype='FLOAT')
        model_order=['openunmix_hq','bs_amp']; rng.shuffle(model_order); mapping={}
        for label,model in zip(('A','B'),model_order):
            report=json.loads(Path(state['jobs'][c['name']+'_'+model]['artifact']).read_text())
            row=next(r for r in report['tracks'] if r['track']==case['track'])
            if row['segment']['processed_input_float32_sha256']!=_audio_hash(audio) or row['segment']['reference_float32_sha256']!=_audio_hash(ref):
                raise ValueError('Case input/reference does not match cached inference')
            output,rate=sf.read(Path(row['audio_directory'])/'vocals_estimate.wav',always_2d=True,dtype='float32')
            if rate!=44100 or output.shape!=ref.shape: raise ValueError('Case alignment mismatch')
            sf.write(target/f'{label}.wav',output,44100,subtype='FLOAT'); mapping[label]=model
        keys.append({'case':stem,'track':case['track'],'condition':c,'mapping':mapping,
                     'selection':case.get('selection','highest BS original-waveform error or lowest BS improvement within this condition')})
        lines += [f"## {stem}: {case['track']}",'',f"条件：{c['name']}；原始 60–90 秒，保留原增益与时间对齐。",'',
                  f'- [输入]({stem}/input.wav)',f'- [参考人声]({stem}/reference.wav)',
                  f'- [输出 A]({stem}/A.wav)',f'- [输出 B]({stem}/B.wav)','']
    lines += ['先关注伴奏残留，再关注人声齿音、轻声、尾音与响度变化。相同播放音量对照参考。',
              '听感记录表为 listening_ratings.csv；空表表示尚未收集评分，不能据此宣称主观优势。']
    (folder/'LISTENING_GUIDE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (folder/'answer_key.json').write_text(json.dumps(keys,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/'listening_ratings.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['case','listener','preferred_A_or_B_or_tie','A_vocal_damage_1_to_5','B_vocal_damage_1_to_5','A_bleed_1_to_5','B_bleed_1_to_5','comments'])
        for row in keys: writer.writerow([row['case'],'','','','','','',''])
    print(folder/'LISTENING_GUIDE.md')


if __name__=='__main__': main()

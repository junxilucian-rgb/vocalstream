"""Build song-paired statistics, repeat timing and the professor review brief."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

from run_project import read_json, write_json


def bootstrap_median_ci(values, seed=2026, draws=10000):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):
        return None
    rng=np.random.default_rng(seed)
    medians=np.median(rng.choice(values,(draws,len(values)),replace=True),axis=1)
    return {'median':float(np.median(values)),'low':float(np.quantile(medians,.025)),
            'high':float(np.quantile(medians,.975)),'n':len(values),
            'scope':'exploratory song bootstrap percentile interval, not a guarantee of unseen-data generalization'}


def report_summary(report):
    rows=report['tracks']
    def median(key):
        values=[r['quality'].get(key) for r in rows]
        values=[v for v in values if v is not None]
        return float(np.median(values)) if values else None
    seconds=sum(r['timing']['processing_call_seconds'] for r in rows)
    duration=sum(r['segment']['duration_seconds'] for r in rows)
    memories=[r['timing'].get('model_runtime',{}).get('last_run',{}).get('cuda_peak_allocated_bytes') for r in rows]
    memories=[v for v in memories if v is not None]
    return {'si_sdr_median':median('si_sdr_db'),'si_sdri_median':median('si_sdri_db'),
            'raw_error_sdr_median':median('global_residual_sdr_db'),'nrmse_median':median('normalized_waveform_rmse'),
            'projection_gain_median':median('reference_projection_gain'),'processing_seconds':seconds,
            'audio_seconds':duration,'rtf':seconds/duration,'peak_allocated_GB':max(memories)/1e9 if memories else None,
            'valid_score_count':sum(r['quality'].get('si_sdr_db') is not None for r in rows),'track_count':len(rows)}


def paired_delta(a,b):
    ar={r['track']:r for r in a['tracks']}; br={r['track']:r for r in b['tracks']}
    if ar.keys()!=br.keys():
        raise ValueError('Pairing needs identical song sets')
    values=[]
    for name in sorted(ar):
        for key in ('start_sample','samples','processed_input_float32_sha256','reference_float32_sha256'):
            if ar[name]['segment'].get(key)!=br[name]['segment'].get(key):
                raise ValueError('Pairing needs identical processed audio and reference hashes')
        x,y=ar[name]['quality']['si_sdr_db'],br[name]['quality']['si_sdr_db']
        if x is not None and y is not None:
            values.append(x-y)
    result=bootstrap_median_ci(values)
    if result is not None:
        result.update(wins=sum(v>0 for v in values),losses=sum(v<0 for v in values))
    return result


def fmt(value,digits=3):
    return '未测' if value is None else f'{value:.{digits}f}'


def build(out):
    out=Path(out).resolve(); state=read_json(out/'run.json')
    if state['status']!='complete':
        raise ValueError('Complete the declared experiment run first')
    config=state['context']['configuration']
    reports={name:read_json(j['artifact']) for name,j in state['jobs'].items()}
    if any(r['status']!='complete' for r in reports.values()):
        raise ValueError('All reports must be complete')
    summaries={name:report_summary(r) for name,r in reports.items()}
    pairs={name:paired_delta(reports[f'expanded_{name}'],reports['expanded_bs_fp32'])
           for name in ('openunmix_hq','bs_amp','bs_2s')}
    timing={}
    for model in config['timing_models']:
        observations=[summaries[f'timing_r{i}_{model}']['rtf'] for i in range(1,config['timing_repeats']+1)]
        timing[model]={'repeat_rtfs':observations,'median_rtf':float(np.median(observations)),
                       'min_rtf':min(observations),'max_rtf':max(observations),
                       'n_independent_runs':len(observations)}
    timing['amp_speedup_ratio']=timing['bs_fp32']['median_rtf']/timing['bs_amp']['median_rtf']
    lines=['# 教授评阅用：BS-RoFormer 推理与串音实验','',
           '范围：MUSDB18-HQ 官方验证名单的 14 首歌，每首固定 60–90 秒。预训练模型实验，无自建模型训练、无 test 推理。',
           '硬件：RTX 4060 Laptop GPU，固定同一环境、CPU 线程 1。BS 公共 checkpoint 的完整训练名单未披露，开发歌可能在其训练数据中。',
           '本报告不是整曲或官方测试集基准，也不是现场超低延迟部署验证。','',
           '## 1. 扩大覆盖后的分离与计算成本','',
           '| 设置 | 有效歌曲 | SI-SDR中位数 dB | SI-SDRi中位数 dB | 调用RTF | 峰值已分配显存 GB |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    labels={'openunmix_hq':'Open-Unmix','bs_fp32':'BS 11s FP32','bs_amp':'BS 11s AMP','bs_2s':'BS 2s FP32'}
    for model in labels:
        s=summaries[f'expanded_{model}']
        lines.append(f"| {labels[model]} | {s['valid_score_count']}/{s['track_count']} | {fmt(s['si_sdr_median'])} | {fmt(s['si_sdri_median'])} | {fmt(s['rtf'])} | {fmt(s['peak_allocated_GB'])} |")
    lines+=['','## 2. 按歌曲配对，而非只比较总中位数','',
            '| 相对 BS FP32 的SI-SDR变化 | 配对中位数 dB | 探索性95%区间 | 上升/下降歌曲 |',
            '| --- | ---: | --- | --- |']
    for model,p in pairs.items():
        if p is not None:
            lines.append(f"| {labels[model]} − BS FP32 | {p['median']:.6f} | [{p['low']:.6f}, {p['high']:.6f}] | {p['wins']}/{p['losses']} |")
    lines+=['','区间使用 14 首歌曲为重采样单位、种子 2026、10000 次 bootstrap；属于有限开发歌曲的探索性分析。三个片段试验的结论现在接受更广歌曲检查，不能据此证明独立测试集泛化。',
            '','## 3. 独立重复速度实验','',
            '每次重新加载并预热模型，计时排除加载、预热、读写和指标计算。FP32/AMP 顺序交替；每次使用固定三首歌，共 90 秒。',
            '| 设置 | 三次调用RTF | 中位数 | 最小—最大 |','| --- | --- | ---: | --- |']
    for model in config['timing_models']:
        t=timing[model]
        lines.append(f"| {labels[model]} | {', '.join(f'{v:.4f}' for v in t['repeat_rtfs'])} | {t['median_rtf']:.4f} | {t['min_rtf']:.4f}—{t['max_rtf']:.4f} |")
    lines += ['',f"本次重复测量中位数之比为 {timing['amp_speedup_ratio']:.2f} 倍。不是实时延迟或跨硬件保证。GPU 温度/功耗/频率快照见 run.json。",'',
              '## 4. 受控串音：输入已以人声为主时会怎样','',
              '输入 = 原始人声 stem + gain × (drums+bass+other)。−30/−20/−10 dB 是原伴奏增益，不是固定输入信干比；实际比值逐曲记录。',
              '| 伴奏增益 | 模型 | 输入SI-SDR中位数 | 输出SI-SDR中位数 | SI-SDRi中位数 | 改善歌曲数 |','| --- | --- | ---: | ---: | ---: | ---: |']
    robustness={}
    for c in config['conditions']:
        if c['mode']!='bleed': continue
        for model in config['robustness_models']:
            key=f"{c['name']}_{model}"; r=reports[key]; s=summaries[key]
            input_scores=[t['quality']['mixture_si_sdr_db'] for t in r['tracks'] if t['quality']['mixture_si_sdr_db'] is not None]
            improved=sum(t['quality']['si_sdri_db'] is not None and t['quality']['si_sdri_db']>0 for t in r['tracks'])
            robustness[key]={'input_si_sdr_median':float(np.median(input_scores)) if input_scores else None,
                             'improved_song_count':improved,**s}
            lines.append(f"| {c['bleed_db']} dB | {labels[model]} | {fmt(robustness[key]['input_si_sdr_median'])} | {fmt(s['si_sdr_median'])} | {fmt(s['si_sdri_median'])} | {improved}/{s['track_count']} |")
    lines+=['','若输出比输入差，说明该设置在这个污染条件下引入的损伤超过去除干扰的收益；必须保留这些负结果。此处没有真实麦克风响应或声学反馈闭环。','',
            '## 5. 输入只有原始人声时的保留能力','',
            '| 模型 | SI-SDR中位数 dB | 未拟合增益的波形NRMSE | 参考投影增益 |','| --- | ---: | ---: | ---: |']
    for model in config['robustness_models']:
        s=summaries[f'clean_vocal_{model}']
        lines.append(f"| {labels[model]} | {fmt(s['si_sdr_median'])} | {fmt(s['nrmse_median'],4)} | {fmt(s['projection_gain_median'],4)} |")
    lines+=['','NRMSE = ||输出−参考||/||参考||，越低越好，0 表示没有变化；不拟合增益，能识别纯音量变化。投影增益接近 1 表示整体增益较接近参考。原 stem 可以自带混响和制作效果，参考并非物理干声。',
            '','## 6. 边界案例与试听','',
            '下列为 BS 的最高原声误差或最低改善片段；最低改善仍可能为正，不把所有边界片段都称为失败。另一个 Open-Unmix 弱串音负例及匿名 A/B 包见 listening_cases。','']
    cases=[]
    for c in config['conditions']:
        key=f"{c['name']}_bs_amp"; report=reports[key]
        field='normalized_waveform_rmse' if c['mode']=='clean' else 'si_sdri_db'
        valid=[r for r in report['tracks'] if r['quality'].get(field) is not None]
        worst=max(valid,key=lambda r:r['quality'][field]) if c['mode']=='clean' else min(valid,key=lambda r:r['quality'][field])
        case={'condition':c,'track':worst['track'],'quality':worst['quality'],'row':worst,'report':state['jobs'][key]['artifact']}
        cases.append(case)
        lines.append(f"- {c['name']}：`{worst['track']}`；{field} = {worst['quality'][field]:.4f}。估计文件：{worst.get('audio_directory')}。")
    lines+=['','## 7. 请教授判断的具体问题','',
            '1. BS 在理想配比串音中全部片段改善，但这不能保证真实录音表现。下一步最值得加哪种污染或真实场景，才能形成范围合适的学生研究问题？',
            '2. 当前“人声+衰减伴奏”模拟应怎样增加房间/麦克风响应？是否有带对齐参考的真实污染录音或合适的房间响应？',
            '3. 哪些来源明确的预训练权重或独立评价数据适合验证泛化？如需进一步贴合超低延迟岗位，是否有可运行的因果基线？',
            '','## 复现与原始证据','',
            '协议见 protocol.json；命令、代码/环境哈希、输入样本哈希、GPU 状态和每步日志见 run.json。全部结果均来自完成的 report.json，重复计时不增加独立歌曲样本量。',
            '逐报告索引与同条件比较见 [RESULTS.md](RESULTS.md)。']
    write_json(out/'analysis.json',{'summaries':summaries,'paired_quality':pairs,'timing_repeats':timing,'robustness':robustness,'failure_cases':cases,
               'analysis_generation':{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                      'run_record_sha256':hashlib.sha256((out/'run.json').read_bytes()).hexdigest(),'numpy':np.__version__},
               'configuration':config,'scope':'14 fixed development segments, no test inference; public pretrained split unknown'})
    (out/'PROFESSOR_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(out/'PROFESSOR_REVIEW.md')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    build(p.parse_args().run)


if __name__=='__main__':
    main()

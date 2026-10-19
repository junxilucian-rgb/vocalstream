"""Create a readable table only for runs on identical audio/conditions/hardware."""
import argparse
import json
from pathlib import Path


def number(value):
    return '未测' if value is None else f'{value:.3f}'


def duration_weighted_rtf(tracks, field):
    """Do not silently turn a missing timer in older reports into zero time."""
    seconds = sum(row['segment']['duration_seconds'] for row in tracks)
    values = [row['timing'].get(field) for row in tracks]
    if not tracks or seconds <= 0 or any(value is None for value in values):
        return None
    return sum(value * row['segment']['duration_seconds']
               for row, value in zip(tracks, values)) / seconds


def signature(report):
    config = report['configuration']
    return json.dumps({
        'dataset': report['dataset'], 'root': config.get('root'),
        'input': config.get('input'), 'subset': config['subset'],
        'device': config['device'], 'threads': config['threads'],
        'condition': config['condition'], 'snr_db': config['snr_db'],
        'bleed_db': config['bleed_db'], 'rt60': config['rt60'], 'seed': config['seed'],
        'environment': report['environment'],
        'tracks': [(r['track'], r['segment']['start_sample'], r['segment']['samples'],
                    r['segment'].get('input_sha256'),
                    r['segment'].get('processed_input_float32_sha256'),
                    r['segment'].get('reference_float32_sha256')) for r in report['tracks']]
    }, sort_keys=True)


def summarize(paths, out):
    reports = [json.loads(Path(p).read_text(encoding='utf-8')) for p in paths]
    if any(r['status'] != 'complete' for r in reports):
        raise ValueError('Only complete runs can be compared')
    if len({signature(r) for r in reports}) != 1:
        raise ValueError('Audio segments, conditions, device, environment or threads differ; rerun under identical settings')
    first = reports[0]
    lines = ['# 人声分离实验报告', '', f"数据：{first['dataset']}；协议：{first['benchmark_protocol']}。", '',
             'SI-SDR 是整片段、每声道去均值、联合立体声投影；BSSEval SDR 为逐帧中位数再跨曲目取中位数。', '',
             '模型/逐块计时 RTF 的范围因系统而异，详见下方计时说明；处理调用 RTF 覆盖从内存音频到返回波形的整个调用。', '',
             '| 系统 | 参数量 | SI-SDR dB | BSSEval SDR dB | 模型/逐块计时 RTF | 处理调用 RTF | 块 RTR p95 | 块 RTR 最大 | 超时块 | 样本位移 ms | 首样本可用 ms |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    if first['dataset'] == 'synthetic_engineering_fixture':
        lines[2:2] = ['**仅工程流程检查：这些是人工信号，分数不代表真实音乐人声分离效果。**', '']
    elif 'preview' in first['dataset'].lower():
        lines[2:2] = ['**公开 7 秒预览实验：属于压缩音源解码后的短片段，不能作为完整 MUSDB18-HQ 测试集论文分数。**', '']
    elif first.get('dataset_metadata', {}).get('formal_benchmark') is False:
        lines[2:2] = ['**该数据来源标记为非正式基准；按演示或流程实验解释结果。**', '']
    timing_notes = []
    for report in reports:
        model = report['model']
        label = model['name'] + (f" niter={model['niter']}" if 'niter' in model else '')
        if model['streaming']:
            cfg = model['causal_config']
            label += f" FFT={cfg['n_fft']}, hop={cfg['hop_length']}, hidden={cfg['hidden_size']}"
        elif model.get('architecture') in ('BSRoformer', 'MelBandRoformer'):
            label += f" context={model['chunk_seconds']:g}s, overlap={model['overlap']}, {model['precision']}, TTA={model['tta']}"
            if model.get('architecture') == 'BSRoformer':
                label += f", heads={model.get('inference_heads', 'all')}"
        tracks = report['tracks']
        # Pool block records rather than taking the average of percentiles.
        block = None
        if model['streaming']:
            from vocalstream.metrics import timing_summary
            times = [t for r in tracks for t in r['timing']['stream']['block_processing_seconds']]
            durations = [r['timing']['stream']['input_block_duration_ms']/1000
                         for r in tracks for _ in r['timing']['stream']['block_processing_seconds']]
            block = timing_summary(times, durations)
        latency = model.get('latency', {})
        cells = [label, str(model['parameters']), number(report['aggregate']['si_sdr_track_median_db']),
                 number(report['aggregate'].get('bsseval_v4_median_tracks_median_frames_vocals_sdr_db')),
                 number(duration_weighted_rtf(tracks, 'inference_rtf')),
                 number(duration_weighted_rtf(tracks, 'processing_call_rtf')),
                 number(block['p95_rtr'] if block else None),
                 number(block['max_rtr'] if block else None), str(block['deadline_miss_count']) if block else '不适用',
                 number(latency.get('algorithmic_latency_ms')), number(latency.get('first_sample_ready_ms'))]
        lines.append('| ' + ' | '.join(cells) + ' |')
        scopes = sorted({r['timing'].get('scope', model.get('timing_scope', '未记录')) for r in tracks})
        timing_notes.append(f'- {label}：模型/逐块计时范围为 {"; ".join(scopes)}。')
        if any(r['timing'].get('processing_call_includes_warmup') for r in tracks):
            timing_notes.append('  处理调用 RTF 包含该系统每条音频开始前的独立预热；逐块 RTR 排除预热。')
        if model.get('warmup_applied') is False:
            timing_notes.append('  模型未进行预热，首条音频可能包含首次执行的额外开销。')
    lines.extend(['', 'RTF 小于 1 仅说明平均吞吐快于音频播放速度；离线模型没有因此成为因果流式模型。',
                  '流式逐块计时排除模型加载、文件读写及声卡缓冲；端到端音频延迟尚未测量。',
                  '处理调用 RTF 包含数组准备、传输、输出组装及调用内发生的预热；不含模型加载、文件读写、质量计算或声卡缓冲。旧报告缺少该字段时显示“未测”。',
                  '所有整段 RTF 按音频时长加权。上述耗时为一次运行的观测，正式结论应重复测量，并注明设备与线程数。', '',
                  '## 计时范围与预热', '', *timing_notes, '', '## 原始报告', ''])
    for path in paths:
        lines.append(f'- {Path(path).resolve()}')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Summary: {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.reports, args.out)


if __name__ == '__main__':
    main()

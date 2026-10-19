"""Verify and summarize the declared four-way vocal-head inference experiment.

This is CPU-only post-processing. It never instantiates a separator or accesses
the MUSDB test split. WAV comparisons use unshifted, unscaled float32 samples;
error accumulation is float64. Numerical checks are not perceptual equivalence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from run_project import digest, read_json, write_json


PROJECT = Path(__file__).resolve().parent
MODELS = (
    ('all_fp32', 'all', 'fp32', '四分支 FP32'),
    ('vocal_fp32', 'vocals', 'fp32', '仅人声 FP32'),
    ('all_amp', 'all', 'amp', '四分支 AMP'),
    ('vocal_amp', 'vocals', 'amp', '仅人声 AMP'),
)
CHECKPOINT_SHA256 = '3e9daecd70aaed5b5a0d1f861cc4d77eaa45afb3fc6301b1cf32c1be0f5868fb'
CONFIG_SHA256 = 'd8afb980318d0c08b9c2e24a7adc00d4f3150320c127a7e4de861800d1321939'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f'Invalid finite number: {name}')
    require(not positive or value > 0, f'Expected positive number: {name}')
    return float(value)


def inside(path, folder):
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(folder.resolve()), f'Artifact escaped run directory: {resolved}')
    return resolved


def array_digest(audio):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def track_map(rows):
    mapped = {row['track']: row for row in rows}
    require(len(mapped) == len(rows), 'Duplicate track in report')
    return mapped


def verify_report(path, job, model, expected_tracks, config, context):
    require(job.get('status') == 'complete', f'Incomplete job: {path}')
    file_sha = digest(path)
    require(file_sha == job['artifact_sha256'], f'Report content changed: {path}')
    report = read_json(path)
    require(report.get('status') == 'complete', f'Incomplete report: {path}')
    name, heads, precision, _ = model
    c = report['configuration']
    required = {'system': 'bsroformer', 'subset': 'train', 'bs_heads': heads,
                'precision': precision, 'overlap': config['overlap'],
                'device': config['device'], 'threads': config['threads'],
                'seed': config['seed'], 'condition': 'mixture',
                'start': config['start_seconds'], 'duration': config['duration_seconds'],
                'tta': False, 'chunk_seconds': None}
    for key, value in required.items():
        require(c.get(key) == value, f'{name}: report configuration mismatch for {key}')
    require(Path(c['root']).resolve() == Path(config['root']).resolve(), 'Dataset root mismatch')
    require(report.get('end_to_end_audio_latency_measured') is False, 'Unexpected latency protocol')
    m = report['model']
    for key, value in {'architecture': 'BSRoformer', 'checkpoint_sha256': CHECKPOINT_SHA256,
                       'config_sha256': CONFIG_SHA256, 'inference_heads': heads,
                       'precision': precision, 'sample_rate': 44100, 'chunk_seconds': 11.0,
                       'overlap': 2, 'tta': False, 'head_selection_after_strict_load': True,
                       'state_dict_loading': 'weights_only=True; strict=True',
                       'architecture_commit': 'aef04b2e52fb3beaf25e333199f5a7236e628e7b',
                       'streaming': False, 'causal': False, 'warmup_applied': True,
                       'network_outputs_all_four_stems': heads == 'all'}.items():
        require(m.get(key) == value, f'{name}: model metadata mismatch for {key}')
    source_key = 'vocalstream/vendor/zfturbo_bsroformer/bs_roformer.py'
    require(m['vendor_sha256'] == context['source_hashes'][source_key], 'Vendor source hash mismatch')
    require(job.get('model_id') == name, 'Job model identity mismatch')
    rows = track_map(report['tracks'])
    require(set(rows) == set(expected_tracks), f'{name}: unexpected evaluated song set')
    declared = {row['track']: row for row in context['data']['segments']}
    for track, row in rows.items():
        segment, source = row['segment'], declared[track]
        for key in ('start_sample', 'samples', 'duration_seconds'):
            require(segment[key] == source['segment'][key], f'{track}: segment mismatch for {key}')
        require(segment['processed_input_float32_sha256'] == source['mixture_sha256'],
                f'{track}: processed mixture hash mismatch')
        require(segment['reference_float32_sha256'] == source['vocals_sha256'],
                f'{track}: reference hash mismatch')
        finite(row['quality']['si_sdr_db'], f'{name}/{track}/SI-SDR')
        finite(row['quality']['si_sdri_db'], f'{name}/{track}/SI-SDRi')
        finite(row['timing']['processing_call_seconds'], f'{name}/{track}/call seconds', positive=True)
        require(row['timing']['processing_call_includes_warmup'] is False, 'Timing includes warmup')
        require(row['timing']['model_warmup_applied'] is True, 'Timing lacks declared warmup')
        runtime = row['timing']['model_runtime']
        for key in ('parameters', 'checkpoint_sha256', 'config_sha256', 'inference_heads', 'precision'):
            require(runtime[key] == m[key], f'{track}: runtime metadata changed for {key}')
        last = runtime['last_run']
        for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'):
            finite(last[key], f'{track}/{key}', positive=True)
    return report, file_sha


def summarize(report):
    rows, model = report['tracks'], report['model']
    seconds = sum(float(row['timing']['processing_call_seconds']) for row in rows)
    audio_seconds = sum(float(row['segment']['duration_seconds']) for row in rows)
    network_seconds = sum(float(row['timing']['inference_seconds']) for row in rows)
    allocated = max(row['timing']['model_runtime']['last_run']['cuda_peak_allocated_bytes'] for row in rows)
    reserved = max(row['timing']['model_runtime']['last_run']['cuda_peak_reserved_bytes'] for row in rows)
    parameters, full = int(model['parameters']), int(model['full_checkpoint_parameters'])
    require(full >= parameters > 0, 'Invalid retained parameter counts')
    require(model['removed_parameters'] == full - parameters, 'Removed parameter count mismatch')
    return {'track_count': len(rows),
            'si_sdr_median_db': float(np.median([row['quality']['si_sdr_db'] for row in rows])),
            'si_sdri_median_db': float(np.median([row['quality']['si_sdri_db'] for row in rows])),
            'parameters': parameters, 'full_checkpoint_parameters': full,
            'removed_parameters': full - parameters,
            'parameter_reduction_percent': 100.0 * (full - parameters) / full,
            'original_mask_head_parameters': model['original_mask_head_parameters'],
            'peak_allocated_bytes': int(allocated), 'peak_reserved_bytes': int(reserved),
            'peak_allocated_GB': allocated / 1e9, 'peak_reserved_GB': reserved / 1e9,
            'processing_call_seconds': seconds, 'audio_seconds': audio_seconds,
            'processing_call_rtf': seconds / audio_seconds,
            'network_forward_seconds': network_seconds, 'network_forward_rtf': network_seconds / audio_seconds,
            'load_seconds': model['load_seconds'], 'warmup_seconds': model['warmup_seconds']}


def load_estimates(report, report_path):
    arrays, manifest = {}, {}
    for row in report['tracks']:
        track = row['track']
        require(row.get('audio_export_scope') == 'estimate_only', f'{track}: expected FLOAT estimate export')
        path = inside(Path(row['audio_directory']) / 'vocals_estimate.wav', report_path.parent)
        info = sf.info(path)
        require(info.format in ('WAV', 'WAVEX') and info.subtype == 'FLOAT', f'Expected unnormalized FLOAT WAV: {path}')
        require(info.samplerate == 44100 and info.channels == 2 and info.frames == row['segment']['samples'],
                f'WAV is not aligned stereo 44100 Hz: {path}')
        audio, rate = sf.read(path, always_2d=True, dtype='float32')
        require(rate == 44100 and audio.shape == (info.frames, 2) and np.isfinite(audio).all(),
                f'Invalid waveform: {path}')
        require(float(np.max(np.abs(audio))) == row['estimate_peak'], f'WAV peak differs from original report: {path}')
        arrays[track] = audio
        manifest[track] = {'path': str(path), 'wav_file_sha256': digest(path),
                           'float32_samples_sha256': array_digest(audio),
                           'frames': info.frames, 'channels': info.channels,
                           'sample_rate': rate, 'subtype': info.subtype,
                           'input_float32_sha256': row['segment']['processed_input_float32_sha256'],
                           'reference_float32_sha256': row['segment']['reference_float32_sha256']}
    return arrays, manifest


def compare(candidate, baseline, reports, arrays, config, same_precision):
    candidate_rows = track_map(reports[candidate]['tracks'])
    baseline_rows = track_map(reports[baseline]['tracks'])
    tolerance = config['numerical_comparison']
    rows = []
    for track in sorted(baseline_rows):
        # No mean removal, gain fitting, normalization or alignment search in
        # these waveform comparisons. SI-SDR deltas use the recorded scores
        # against the independently hashed MUSDB vocal reference.
        reference = arrays[baseline][track].astype(np.float64)
        residual = arrays[candidate][track].astype(np.float64) - reference
        energy = float(np.sum(reference * reference))
        require(energy > 0, f'Cannot normalize output error against silent baseline: {track}')
        error_energy = float(np.sum(residual * residual))
        nrmse = math.sqrt(error_energy / energy)
        si_delta = float(candidate_rows[track]['quality']['si_sdr_db'] - baseline_rows[track]['quality']['si_sdr_db'])
        row = {'track': track, 'max_absolute_sample_error': float(np.max(np.abs(residual))),
               'waveform_rmse': math.sqrt(error_energy / residual.size),
               'normalized_waveform_rmse': nrmse, 'relative_l2_error': nrmse,
               'candidate_si_sdr_db': candidate_rows[track]['quality']['si_sdr_db'],
               'baseline_si_sdr_db': baseline_rows[track]['quality']['si_sdr_db'],
               'si_sdr_delta_db': si_delta,
               'samplewise_equal': bool(np.array_equal(arrays[candidate][track], arrays[baseline][track]))}
        if same_precision:
            row['within_predeclared_numerical_tolerances'] = (
                nrmse <= tolerance['max_normalized_waveform_rmse']
                and abs(si_delta) <= tolerance['max_absolute_per_song_si_sdr_delta_db'])
        rows.append(row)
    result = {'candidate': candidate, 'baseline': baseline, 'same_precision_head_comparison': same_precision,
              'track_count': len(rows), 'samplewise_equal_track_count': sum(row['samplewise_equal'] for row in rows),
              'max_absolute_sample_error': max(row['max_absolute_sample_error'] for row in rows),
              'max_waveform_rmse': max(row['waveform_rmse'] for row in rows),
              'median_normalized_waveform_rmse': float(np.median([row['normalized_waveform_rmse'] for row in rows])),
              'max_normalized_waveform_rmse': max(row['normalized_waveform_rmse'] for row in rows),
              'si_sdr_paired_delta_median_db': float(np.median([row['si_sdr_delta_db'] for row in rows])),
              'max_absolute_si_sdr_delta_db': max(abs(row['si_sdr_delta_db']) for row in rows),
              'tracks': rows}
    if same_precision:
        result['predeclared_tolerances'] = tolerance
        result['within_predeclared_numerical_tolerances'] = all(row['within_predeclared_numerical_tolerances'] for row in rows)
        result['passing_track_count'] = sum(row['within_predeclared_numerical_tolerances'] for row in rows)
    return result


def fmt(value, digits=4):
    return f'{value:.{digits}f}'


def markdown(analysis):
    quality, timing, comparisons = analysis['quality'], analysis['timing'], analysis['comparisons']
    gpu = analysis['environment']['gpu']
    lines = ['# BS-RoFormer 仅人声输出分支优化：实测报告', '',
             f'硬件：{gpu}。同一环境、CPU 线程 1，MUSDB18-HQ 官方验证名单 14 首歌，每首固定 60–90 秒。',
             '使用同一社区预训练 checkpoint、11 秒输入上下文、overlap=2、无 TTA。四组质量实验及每组 3 次独立计时共 16 个任务均已完成；没有训练或 test 推理。', '',
             '## 1. 四组结果', '',
             '| 设置 | SI-SDR中位数 dB | SI-SDRi中位数 dB | 保留参数 M | 峰值 allocated / reserved GB | 重复计时RTF中位数 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, _, _, label in MODELS:
        q, t = quality[name], timing[name]
        lines.append(f"| {label} | {q['si_sdr_median_db']:.6f} | {q['si_sdri_median_db']:.6f} | {q['parameters']/1e6:.6f} | {q['peak_allocated_GB']:.4f} / {q['peak_reserved_GB']:.4f} | {t['median_rtf']:.6f} |")
    q = quality['vocal_fp32']
    lines += ['', f"仅人声模式保留 {q['parameters']:,} 个参数，移除 {q['removed_parameters']:,} 个参数（{q['parameter_reduction_percent']:.2f}%）。共享主干和人声权重不变；checkpoint 文件没有缩小或改写。",
              '显存是质量实验中逐片段的 PyTorch allocator 峰值，GB=10^9 bytes；reserved 包含缓存，不等于实际张量占用或系统 nvidia-smi 数值。', '',
              '## 2. 同精度数值检查', '',
              '预先声明的逐曲阈值：未拟合增益的波形 NRMSE ≤ 1e-5，并且相对于同精度四分支输出的 |SI-SDR变化| ≤ 0.001 dB。两项必须逐曲同时满足。',
              '比较直接使用已保存的 FLOAT WAV；不做归一化、时移搜索、增益拟合或裁剪。NRMSE = ||候选−基线||₂ / ||基线||₂，与 relative L2 相同。', '',
              '| 比较 | 最大样本绝对误差 | 最大波形RMSE | 最大NRMSE | SI-SDR最大绝对变化 dB | 逐曲通过 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key in ('vocal_fp32_vs_all_fp32', 'vocal_amp_vs_all_amp'):
        c = comparisons[key]
        lines.append(f"| {c['candidate']} − {c['baseline']} | {c['max_absolute_sample_error']:.9g} | {c['max_waveform_rmse']:.9g} | {c['max_normalized_waveform_rmse']:.9g} | {c['max_absolute_si_sdr_delta_db']:.9g} | {c['passing_track_count']}/{c['track_count']} |")
    passed = all(comparisons[key]['within_predeclared_numerical_tolerances'] for key in ('vocal_fp32_vs_all_fp32', 'vocal_amp_vs_all_amp'))
    lines += ['', '结论：' + ('两种精度均在全部 14 个片段上满足预先声明的数值容差。' if passed else '至少一组未能在全部片段上满足预先声明的数值容差；应保留并调查未通过记录。'),
              f"直接采样相同的片段数：FP32 {comparisons['vocal_fp32_vs_all_fp32']['samplewise_equal_track_count']}/14，AMP {comparisons['vocal_amp_vs_all_amp']['samplewise_equal_track_count']}/14。这只描述当前保存的输出，不保证其他输入、硬件或运行环境也相同，且不替代主观听感评价。", '',
              '## 3. AMP 与组合优化相对 FP32 的变化', '',
              '下列跨精度比较记录误差，但不套用仅为“同精度分支选择”预先声明的通过阈值。', '',
              '| 候选 − all_fp32 | SI-SDR逐曲差值中位数 dB | SI-SDR最大绝对变化 dB | 最大NRMSE |',
              '| --- | ---: | ---: | ---: |']
    for key in ('all_amp_vs_all_fp32', 'vocal_amp_vs_all_fp32'):
        c = comparisons[key]
        lines.append(f"| {c['candidate']} | {c['si_sdr_paired_delta_median_db']:.9g} | {c['max_absolute_si_sdr_delta_db']:.9g} | {c['max_normalized_waveform_rmse']:.9g} |")
    lines += ['', '逐曲 SI-SDR 差值的中位数不一定等于两组 SI-SDR 中位数相减。完整逐曲波形误差、评分和输出哈希见 head_optimization_analysis.json。', '',
              '## 4. 三次独立计时', '',
              '每次任务使用全新进程、重新加载和预热；固定三首歌，共 90 秒。各重复轮次改变设置顺序。计时从内存音频进入分离调用到 CPU 波形返回，含准备、传输和拼接；排除加载、预热、文件读写、指标和音频设备缓冲。', '',
              '| 设置 | RTF第1次 | RTF第2次 | RTF第3次 | 中位数 | 最小—最大 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, _, _, label in MODELS:
        t = timing[name]
        repeats = ' | '.join(f'{value:.6f}' for value in t['repeat_rtfs'])
        lines.append(f"| {label} | {repeats} | {t['median_rtf']:.6f} | {t['min_rtf']:.6f}—{t['max_rtf']:.6f} |")
    lines += ['', '| 优化比较 | 基线RTF中位数 / 候选RTF中位数 | 调用耗时减少 |', '| --- | ---: | ---: |']
    for value in analysis['speedups'].values():
        lines.append(f"| {value['candidate']} 相对 {value['baseline']} | {value['median_rtf_ratio']:.4f}× | {value['processing_time_reduction_percent']:.2f}% |")
    lines += ['', '以上为同硬件、固定工作负载的吞吐速度；重复计时不会增加独立歌曲样本量。GPU 前后快照、完整进程命令及原始日志见 run.json。', '',
              '## 5. 适用范围与限制', '',
              '- 使用 ZFTurbo 社区权重，并非原论文发布的同一 checkpoint；其完整训练歌曲名单未披露，不能把这些结果称为独立留出泛化。',
              '- 只验证 14 首歌的固定 30 秒片段，尚未涵盖整曲、真实麦克风串音或声学反馈闭环。',
              '- 分支选择没有移除双向注意力或未来上下文；RTF 小于 1 不代表超低延迟。算法延迟、首个输出等待时间和音频设备端到端延迟均未测。',
              '- 显存和速度变化可能受 GPU 频率、温度、allocator 缓存、软件版本和其他进程影响。',
              '- 同精度阈值结果无论通过与否均完整保留；不据此宣称数学等价、主观音质等价或跨设备保证。', '',
              '## 6. 复现与证据', '',
              '在项目目录中使用已安装项目依赖的 Python。新实验必须使用新的输出目录；既有目录保留原始证据。', '',
              '```powershell',
              'python run_head_optimization.py --config head_optimization_experiments.json --out results/head_optimization_new',
              'python build_head_optimization_report.py --run results/head_optimization_new',
              '```', '',
              'protocol.json 保存预先声明的容差；run.json 固定推理代码、环境和输入哈希。构建报告前核对全部 16 份 report.json 的内容 SHA、各设置输入和参考哈希、参数及音频形状。分析文件另记生成器 SHA 和每个 WAV 的文件/采样内容 SHA，报告生成器不属于推理源代码快照。', '',
              '原始数据索引、逐曲误差、质量汇总、每次计时及速度比：head_optimization_analysis.json。']
    return '\n'.join(lines) + '\n'


def build(out):
    out = Path(out).resolve()
    run_path = out / 'run.json'
    state = read_json(run_path)
    require(state.get('status') == 'complete', 'Complete the declared experiment before reporting')
    context, config = state['context'], state['context']['configuration']
    require(read_json(out / 'protocol.json') == config, 'Protocol/configuration mismatch')
    require(digest(out / 'protocol.json') == state['identity'], 'Protocol SHA differs from run identity')
    require(config['models'] == [{'name': name, 'heads': heads, 'precision': precision}
                                 for name, heads, precision, _ in MODELS], 'Unexpected model variants')
    require(config['timing_repeats'] == 3 and state.get('job_count') == 16,
            'This report requires exactly four quality jobs and twelve timing jobs')
    require(config['numerical_comparison']['max_normalized_waveform_rmse'] == 1e-5
            and config['numerical_comparison']['max_absolute_per_song_si_sdr_delta_db'] == .001,
            'Numerical thresholds differ from the declared experiment')
    require(context['data']['test_set_accessed'] is False, 'Test-set access is outside this experiment')
    validation = context['validation_tracks']
    require(len(validation) == 14 and len(set(validation)) == 14, 'Need 14 distinct development tracks')
    require(set(validation) == set(read_json(PROJECT / 'project_config.json')['validation_tracks']),
            'Development song set differs from the official reserved list')
    require(len(config['timing_tracks']) == 3 and len(set(config['timing_tracks'])) == 3
            and set(config['timing_tracks']) <= set(validation), 'Need the three declared development timing songs')
    segments = context['data']['segments']
    require(len(segments) == 14 and {row['track'] for row in segments} == set(validation),
            'Predeclared audio hashes do not cover the development songs exactly')
    expected = {f'quality_{m[0]}' for m in MODELS}
    expected |= {f'timing_r{repeat}_{m[0]}' for repeat in range(1, 4) for m in MODELS}
    require(set(state['jobs']) == expected, 'Missing or unexpected experiment jobs')
    reports, report_index = {}, {}
    for model in MODELS:
        for stage, repeat in [('quality', 1)] + [('timing', r) for r in range(1, 4)]:
            key = f'quality_{model[0]}' if stage == 'quality' else f'timing_r{repeat}_{model[0]}'
            job = state['jobs'][key]
            require(job.get('stage') == stage and job.get('repeat') == repeat, f'Incorrect stage/repeat: {key}')
            path = inside(job['artifact'], out)
            tracks = validation if stage == 'quality' else config['timing_tracks']
            report, sha = verify_report(path, job, model, tracks, config, context)
            require(report['configuration']['save_estimates'] is (stage == 'quality'), 'Estimate export protocol mismatch')
            reports[key] = report
            report_index[key] = {'path': str(path), 'report_sha256': sha, 'stage': stage, 'repeat': repeat}
    environments = [report['environment'] for report in reports.values()]
    require(all(value == environments[0] for value in environments), 'Runtime environments differ between jobs')
    quality_reports = {model[0]: reports[f'quality_{model[0]}'] for model in MODELS}
    quality = {name: summarize(report) for name, report in quality_reports.items()}
    full_parameters = quality['all_fp32']['parameters']
    head_parameters = quality['all_fp32']['original_mask_head_parameters']
    require(len(head_parameters) == 4, 'Need four original output-head parameter counts')
    for name, heads, _, _ in MODELS:
        q = quality[name]
        require(q['full_checkpoint_parameters'] == full_parameters
                and q['original_mask_head_parameters'] == head_parameters,
                f'Checkpoint parameter accounting differs: {name}')
        expected_removed = 0 if heads == 'all' else sum(head_parameters[:3])
        require(q['removed_parameters'] == expected_removed, f'Incorrect removed-head parameter count: {name}')
    arrays, waveform_manifest = {}, {}
    for model in MODELS:
        name = model[0]
        path = Path(report_index[f'quality_{name}']['path'])
        arrays[name], waveform_manifest[name] = load_estimates(quality_reports[name], path)
    comparisons = {}
    for candidate, baseline, same_precision in [('vocal_fp32', 'all_fp32', True),
                                                 ('all_amp', 'all_fp32', False),
                                                 ('vocal_amp', 'all_fp32', False),
                                                 ('vocal_amp', 'all_amp', True)]:
        key = f'{candidate}_vs_{baseline}'
        comparisons[key] = compare(candidate, baseline, quality_reports, arrays, config, same_precision)
    timing = {}
    for name, _, _, _ in MODELS:
        observations = [summarize(reports[f'timing_r{repeat}_{name}']) for repeat in range(1, 4)]
        require(all(row['parameters'] == quality[name]['parameters'] for row in observations),
                f'Parameter count differs between quality and timing: {name}')
        rtfs = [row['processing_call_rtf'] for row in observations]
        timing[name] = {'n_independent_runs': 3, 'repeat_rtfs': rtfs,
                        'median_rtf': float(np.median(rtfs)), 'min_rtf': min(rtfs), 'max_rtf': max(rtfs),
                        'runs': [{'repeat': repeat, 'job': f'timing_r{repeat}_{name}', **row}
                                 for repeat, row in enumerate(observations, 1)]}
    speedups = {}
    for key, baseline, candidate in [('head_selection_fp32', 'all_fp32', 'vocal_fp32'),
                                     ('head_selection_amp', 'all_amp', 'vocal_amp'),
                                     ('amp_four_heads', 'all_fp32', 'all_amp'),
                                     ('amp_vocal_only', 'vocal_fp32', 'vocal_amp'),
                                     ('combined', 'all_fp32', 'vocal_amp')]:
        ratio = timing[baseline]['median_rtf'] / timing[candidate]['median_rtf']
        speedups[key] = {'baseline': baseline, 'candidate': candidate,
                         'median_rtf_ratio': ratio, 'processing_time_reduction_percent': 100.0 * (1 - 1 / ratio),
                         'repeat_rtf_ratios': [a / b for a, b in zip(timing[baseline]['repeat_rtfs'], timing[candidate]['repeat_rtfs'])],
                         'definition': 'ratio of independent-run median call RTFs; not latency reduction'}
    analysis = {'schema_version': 1, 'configuration': config, 'quality': quality,
                'comparisons': comparisons, 'timing': timing, 'speedups': speedups,
                'waveform_manifest': waveform_manifest, 'report_index': report_index,
                'environment': environments[0],
                'scope': '14 fixed official development excerpts; inference-only, no training or test inference; not measured latency',
                'inference_source_hashes': context['source_hashes'],
                'analysis_generation': {'created_utc': datetime.now(timezone.utc).isoformat(),
                                        'source_path': str(Path(__file__).resolve()),
                                        'source_sha256': digest(Path(__file__)), 'run_record_sha256': digest(run_path),
                                        'protocol_sha256': digest(out / 'protocol.json'),
                                        'numpy': np.__version__, 'soundfile': sf.__version__,
                                        'separate_from_inference_source_hashes': True}}
    text = markdown(analysis)
    write_json(out / 'head_optimization_analysis.json', analysis)
    (out / 'HEAD_OPTIMIZATION_RESULTS.md').write_text(text, encoding='utf-8')
    print(out / 'HEAD_OPTIMIZATION_RESULTS.md')
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed run_head_optimization output directory')
    build(parser.parse_args().run)


if __name__ == '__main__':
    main()

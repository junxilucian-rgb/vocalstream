"""Analyze a completed short file-stream study on identical target intervals.

Raw worker reports remain unchanged. Vocal references are read here only after
all thirteen inference jobs and their report hashes have been verified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from run_project import PROJECT, digest, exclusive_run, read_json, write_json
from run_streaming_study import SETTINGS, TRACKS, stream_job_name
from vocalstream.metrics import separation_metrics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_relative_to(root), f'Artifact outside declared experiment: {path}')
    return path


def array_digest(audio):
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def median(values):
    values = [value for value in values if value is not None]
    return float(np.median(values)) if values else None


def score_delta(candidate, baseline):
    if candidate is None or baseline is None:
        return None
    return candidate-baseline


def check_model(model, context):
    config = context['configuration']
    require(model['checkpoint_sha256'] == context['model_asset_sha256']['model.ckpt'], 'Checkpoint mismatch')
    require(model['config_sha256'] == context['model_asset_sha256']['config.yaml'], 'Model config mismatch')
    require(model['inference_heads'] == 'vocals' and model['output_stem_order'] == ['vocals'], 'Expected vocal-only inference')
    require(model['precision'] == 'amp' and model['device'] == 'cuda', 'Expected AMP/CUDA')
    require(model['chunk_samples'] == 88200 and model['overlap'] == 2 and model['tta'] is False,
            'Expected two-second context, overlap two, no TTA')
    require(model['warmup_applied'] is True and model['sample_rate'] == config['sample_rate'], 'Warmup/rate mismatch')


def read_wave(path, frames, out, expected_hash=None):
    path = inside(path, out)
    info = sf.info(path)
    require(info.samplerate == 44100 and info.channels == 2 and info.frames == frames and info.subtype == 'FLOAT',
            f'Expected aligned stereo FLOAT output: {path}')
    if expected_hash is not None:
        require(digest(path) == expected_hash, f'Output hash changed: {path}')
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    require(rate == 44100 and np.isfinite(audio).all(), f'Invalid output samples: {path}')
    return audio, {'path': str(path), 'file_sha256': digest(path),
                   'float32_samples_sha256': array_digest(audio), 'frames': frames,
                   'sample_rate': info.samplerate, 'channels': info.channels, 'subtype': info.subtype}


def verify_emissions(report, setting, frames):
    stream, emissions = report['stream'], report['emissions']
    rate = 44100
    require(stream['sample_rate'] == rate and stream['context_samples'] == 88200, 'Stream sample rate/context mismatch')
    require(stream['emit_samples'] == round(setting['emit_seconds']*rate)
            and stream['lookahead_samples'] == round(setting['lookahead_seconds']*rate), 'Stream schedule mismatch')
    require(stream['input_samples'] == stream['output_samples'] == frames and stream['finished'] is True,
            'Stream input/output coverage mismatch')
    require(stream['network_architecture_causal'] is False
            and stream['hardware_end_to_end_latency_measured'] is False,
            'This experiment does not establish architecture causality or device latency')
    cursor = 0
    steady = []
    for index, row in enumerate(emissions):
        begin, end = row['output_start_sample'], row['output_end_sample_exclusive']
        require(row['emission_index'] == index and begin == cursor and begin < end <= frames,
                'Emissions must cover output once, consecutively, without gaps')
        require(row['output_samples'] == end-begin, 'Emission output length mismatch')
        require(np.isfinite(row['processing_seconds']) and row['processing_seconds'] >= 0,
                'Invalid processing time')
        if not row['eof_flush']:
            require(row['required_real_input_sample_boundary'] == end + stream['lookahead_samples'],
                    'Unexpected future input dependency')
        if (not row['eof_flush'] and row['left_zero_padding_samples'] == 0
                and row['right_zero_padding_samples'] == 0):
            if steady:
                require(steady[-1]['output_end_sample_exclusive'] == begin, 'Steady emissions are not contiguous')
            steady.append(row)
        cursor = end
    require(cursor == frames, 'Emissions do not cover the entire clip')
    expected_intervals = [[row['output_start_sample'], row['output_end_sample_exclusive']] for row in steady]
    require(report['steady_state_quality_intervals_clip_relative_samples'] == expected_intervals,
            'Worker steady interval metadata mismatch')
    return steady


def timing_summary(rows):
    if not rows:
        return {'emission_count': 0, 'undefined_reason': 'No unpadded non-EOF emissions'}
    times = np.array([row['processing_seconds'] for row in rows], dtype=np.float64)
    queues = np.array([row['virtual_queue_wait_seconds'] for row in rows], dtype=np.float64)
    seconds = sum(row['output_audio_seconds'] for row in rows)
    return {'emission_count': len(rows), 'processing_seconds_p50': float(np.percentile(times, 50)),
            'processing_seconds_p95': float(np.percentile(times, 95)), 'processing_seconds_max': float(times.max()),
            'total_processing_seconds': float(times.sum()), 'total_output_audio_seconds': seconds,
            'processing_rtf_vs_emitted_audio': float(times.sum()/seconds),
            'compute_over_nominal_period_fraction': float(np.mean([row['compute_exceeds_nominal_emit_period'] for row in rows])),
            'virtual_deadline_miss_fraction': float(np.mean([row['virtual_deadline_missed'] for row in rows])),
            'virtual_queue_wait_seconds_p50': float(np.percentile(queues, 50)),
            'virtual_queue_wait_seconds_p95': float(np.percentile(queues, 95)),
            'virtual_queue_wait_seconds_max': float(queues.max()),
            'virtual_first_boundary_ready_delay_seconds_p95': float(np.percentile(
                [row['virtual_ready_delay_from_first_output_boundary_seconds'] for row in rows], 95))}


def fmt(value, digits=3):
    return '未定义' if value is None else f'{value:.{digits}f}'


def markdown(analysis):
    config = analysis['configuration']
    interval = analysis['common_quality_interval_clip_relative_samples']
    if interval:
        interval_text = f"[{interval[0]/44100:.3f}, {interval[1]/44100:.3f}) 秒，共 {(interval[1]-interval[0])/44100:.3f} 秒"
    else:
        interval_text = '不存在；片段没有所有配置共同的无填充稳态区间，因此不报告公共区间分数'
    lines = ['# BS-RoFormer 有限未来信息文件流实验', '',
             f"三首官方开发歌曲，每首从 {config['start_seconds']:g} 秒开始，长度 {config['requested_samples']/44100:g} 秒。",
             '固定同一社区预训练权重、人声单分支、AMP、2 秒输入上下文和 overlap=2。完成 12 次流式短实验和 1 次含三首歌曲的离线参照任务。没有训练、test 推理或音频设备回放。', '',
             '## 1. 实验改变了什么', '',
             '滚动保留过去输入，每次只接收下一小包数据；在指定未来信息到达后，重新调用离线分离器并截取一小段人声。内部仍使用离线重叠拼接，且每次重新计算整个窗口，没有神经状态缓存。',
             '等待输入的时间与计算时间分别报告。下面的“可用等待”从输出块第一个样本的时间边界算起，为输出块长度 E 加未来上下文 L；从输出块末端边界算起则为 L。过去缓存不计作未来等待。', '',
             '| 配置 | 输出块 E ms | 未来信息 L ms | 首边界可用等待 E+L ms | 末边界可用等待 L ms |',
             '| --- | ---: | ---: | ---: | ---: |']
    for setting in SETTINGS:
        e, l = setting['emit_seconds']*1000, setting['lookahead_seconds']*1000
        lines.append(f"| {setting['name']} | {e:g} | {l:g} | {e+l:g} | {l:g} |")
    lines += ['', '这是输入依赖的名义等待，不包含计算、虚拟队列或设备延迟。L=0 表示输出块结束前的输入仍可用于同块早期样本，不等于逐样本零延迟或模型架构已经变成因果。', '',
              '## 2. 在相同目标样本上比较质量', '',
              f'所有流式配置与离线参照共同评分区间（相对片段起点）：{interval_text}。',
              '先对每首歌曲、每个配置在完全相同的目标区间计算一次 joint-stereo SI-SDR，再取三首中位数。整段分数包括起点零历史和结尾 EOF；公共区间排除所有流式配置的启动左填充与 EOF。不同配置各自的稳态分数不用于相互排名。', '',
              '| 设置 | 整段 SI-SDR 中位数 dB | 公共区间 SI-SDR 中位数 dB | 公共区间相对离线逐曲差值中位数 dB |',
              '| --- | ---: | ---: | ---: |']
    for name in ['offline_amp2s'] + [setting['name'] for setting in SETTINGS]:
        row = analysis['quality_summary'][name]
        lines.append(f"| {name} | {fmt(row['whole_clip_si_sdr_median_db'])} | {fmt(row['common_interval_si_sdr_median_db'])} | {fmt(row['common_interval_paired_delta_vs_offline_median_db'])} |")
    lines += ['', '离线参照也是 2 秒上下文，不是此前 11 秒高质量设置；它能使用未来音频并进行离线重叠拼接。短区间和三首中位数仅用于解释调度影响，不能当作完整音乐分离基准。', '',
              '## 3. 无填充稳态输出块的计算与虚拟排队', '',
              '| 配置 | 稳态块数 | 单块耗时 p50 / p95 ms | 计算超出块周期 | 虚拟截止时间错过 | 虚拟队列 p95 / 最大 s | 工作量 RTF |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for setting in SETTINGS:
        name = setting['name']
        row = analysis['steady_emission_timing'][name]
        if not row['emission_count']:
            lines.append(f'| {name} | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |')
            continue
        lines.append(f"| {name} | {row['emission_count']} | {1000*row['processing_seconds_p50']:.2f} / {1000*row['processing_seconds_p95']:.2f} | {100*row['compute_over_nominal_period_fraction']:.1f}% | {100*row['virtual_deadline_miss_fraction']:.1f}% | {row['virtual_queue_wait_seconds_p95']:.3f} / {row['virtual_queue_wait_seconds_max']:.3f} | {row['processing_rtf_vs_emitted_audio']:.3f} |")
    lines += ['', '耗时分位数合并三首歌的无填充、非 EOF 输出块；这些块高度相关，不是独立重复实验。每种设置每首歌只有一次运行。加载、预热、文件读写和音频设备缓冲不计入上述计算时间。',
              '“计算超出块周期”只比较当前块计算耗时与 E；“虚拟截止时间错过”还包含之前积累的工作。虚拟单工作线程按 finish=max(输入到达时间, 上一块完成时间)+实测计算耗时调度，截止时间定义为输入到达时间+E。排队可以从启动阶段累积，即使表中只统计稳态块。',
              '该调度是用测得计算时间推演的音频时钟，没有真实等待、声卡或系统调度抖动。不能把虚拟队列数值称为实际设备延迟。', '',
              '## 4. 结果范围与复现', '',
              '- 该流程限制每次推理可访问的输入范围，但没有修改双向注意力，也没有将模型训练成因果网络。',
              '- 重算窗口的计算浪费、输出块边界没有交叉淡化、短片段零历史/EOF 均可能影响结果。',
              '- 权重完整训练歌曲名单未披露，开发片段不构成独立留出泛化证明。',
              '- 三首短片段、一次计时仅用于找出可继续研究的问题；没有测量端到端音频设备延迟。',
              '- 输出均为 FLOAT WAV；不做增益拟合、归一化、时移搜索或峰值裁剪。全部原始逐块记录保留在 worker report.json。', '',
              '```powershell',
              'python run_streaming_study.py --root data/musdb18-hq --out results/streaming_study_new',
              'python build_streaming_study_report.py --run results/streaming_study_new',
              '```', '',
              'protocol.json 固定模型、源文件、运行环境及输入/参考片段哈希。run.json 记录 13 个任务的命令、日志和 GPU 前后快照；streaming_study_analysis.json 保存逐曲整段及公共区间分数、波形哈希和分析程序哈希。报告生成器与推理源文件分开记录。']
    return '\n'.join(lines)+'\n'


def build(out):
    out = Path(out).resolve()
    state = read_json(out/'run.json')
    require(state.get('status') == 'complete', 'Complete all inference before analyzing this study')
    context = state['context']
    config = context['configuration']
    require(read_json(out/'protocol.json') == context and state['identity'] == digest(out/'protocol.json'),
            'Protocol identity/context changed')
    require(config['tracks'] == TRACKS and config['settings'] == SETTINGS and config['expected_jobs'] == 13,
            'Unexpected study tracks/settings')
    require(config['timing_trials_per_track_setting'] == 1 and context['data']['test_set_accessed'] is False,
            'Expected one-trial development-only experiment')
    for name, expected_hash in context['source_hashes'].items():
        require(digest(PROJECT/name) == expected_hash, f'Inference/metric source changed since run: {name}')
    expected_jobs = {'offline_amp2s'} | {stream_job_name(index, setting)
        for index in range(1, 4) for setting in SETTINGS}
    require(state['job_count'] == 13 and set(state['jobs']) == expected_jobs, 'Expected exactly thirteen declared jobs')
    reports, report_index = {}, {}
    for name, job in state['jobs'].items():
        require(job.get('status') == 'complete' and job.get('repeat') == 1, f'Incomplete/unexpected job: {name}')
        require(job['inference_source_hashes'] == context['source_hashes'], f'Source hash context differs: {name}')
        path = inside(job['artifact'], out)
        require(digest(path) == job['artifact_sha256'], f'Report changed: {name}')
        report = read_json(path)
        require(report.get('status') == 'complete', f'Worker report incomplete: {name}')
        check_model(report['model'], context)
        reports[name] = report
        report_index[name] = {'path': str(path), 'report_sha256': digest(path)}
    # All report completion and integrity checks above happen before reference
    # waveform reading/scoring below. No test inventory or audio is opened here.
    segments = {row['track']: row for row in context['data']['segments']}
    require(set(segments) == set(TRACKS), 'Input manifest differs from declared tracks')
    offline = reports['offline_amp2s']
    c = offline['configuration']
    require(c['subset'] == 'train' and set(c['track']) == set(TRACKS) and c['system'] == 'bsroformer'
            and c['bs_heads'] == 'vocals' and c['precision'] == 'amp' and c['chunk_seconds'] == 2
            and c['overlap'] == 2 and c['save_estimates'] is True and c['condition'] == 'mixture'
            and c['start'] == config['start_seconds'] and c['duration'] == config['duration_seconds']
            and c['seed'] == 2026 and c['threads'] == 1,
            'Offline reference configuration differs')
    require(offline['environment']['torch_threads'] == 1
            and offline['environment']['versions']['torch'] == context['runtime']['packages']['torch'],
            'Offline runtime differs from the declared environment')
    require(len(offline['tracks']) == 3, 'Expected three offline reference rows')
    offline_rows = {row['track']: row for row in offline['tracks']}
    require(set(offline_rows) == set(TRACKS), 'Offline track set mismatch')
    waves, waveform_manifest, steady_by_setting = {}, {}, {setting['name']: [] for setting in SETTINGS}
    ranges = []
    for index, track in enumerate(TRACKS, 1):
        manifest = segments[track]
        segment = manifest['segment']
        frames = config['requested_samples']
        require(segment['samples'] == frames, 'Declared clip frame count mismatch')
        row = offline_rows[track]
        require(row['segment']['start_sample'] == segment['start_sample']
                and row['segment']['samples'] == frames
                and row['segment']['processed_input_float32_sha256'] == manifest['mixture_float32_sha256']
                and row['segment']['reference_float32_sha256'] == manifest['vocals_float32_sha256'],
                f'Offline input/reference hash or interval mismatch: {track}')
        waves[track], waveform_manifest[track] = {}, {}
        audio, proof = read_wave(Path(row['audio_directory'])/'vocals_estimate.wav', frames, out)
        waves[track]['offline_amp2s'], waveform_manifest[track]['offline_amp2s'] = audio, proof
        for setting in SETTINGS:
            key = stream_job_name(index, setting)
            report, job = reports[key], state['jobs'][key]
            require(job['track'] == track and job['setting'] == setting and job['stage'] == 'streaming', 'Job mapping mismatch')
            require(Path(report['input_path']).resolve() == Path(manifest['mixture_path']).resolve()
                    and Path(report['reference_path']).resolve() == Path(manifest['reference_path']).resolve(),
                    f'Unexpected stream input/reference paths: {key}')
            require(report['clip_start_sample'] == segment['start_sample'] and report['clip_samples'] == frames,
                    'Streaming clip interval differs')
            require(report['random_seed'] == 2026 and report['torch_threads'] == 1, 'Stream seed/thread mismatch')
            require(report['torch_version'] == context['runtime']['packages']['torch'], 'Torch runtime mismatch')
            require(Path(report['python_executable']).resolve() == Path(context['runtime']['executable']).resolve(),
                    'Stream worker Python executable differs')
            worker_sources = {source_name.replace('\\', '/'): sha for source_name, sha in report['source_files_sha256'].items()}
            expected_worker_sources = {name: sha for name, sha in context['source_hashes'].items()
                                       if name in ('offline_stream_demo.py', 'vocalstream/streaming.py',
                                                   'vocalstream/bsroformer.py', 'vocalstream/roformer.py',
                                                   'vocalstream/metrics.py')
                                       or name.startswith('vocalstream/vendor/zfturbo_bsroformer/')}
            require(worker_sources == expected_worker_sources, 'Streaming worker source snapshot differs')
            steady = verify_emissions(report, setting, frames)
            steady_by_setting[setting['name']].extend(steady)
            if steady:
                ranges.append([steady[0]['output_start_sample'], steady[-1]['output_end_sample_exclusive']])
            else:
                ranges.append(None)
            audio, proof = read_wave(report['output_path'], frames, out, report['output_sha256'])
            waves[track][setting['name']], waveform_manifest[track][setting['name']] = audio, proof
    common = None
    if all(interval is not None for interval in ranges):
        begin, end = max(interval[0] for interval in ranges), min(interval[1] for interval in ranges)
        if begin < end:
            common = [begin, end]
    per_track = []
    source_audio_manifest = []
    for track in TRACKS:
        manifest = segments[track]
        start, frames = manifest['segment']['start_sample'], config['requested_samples']
        mixture_path, reference_path = Path(manifest['mixture_path']), Path(manifest['reference_path'])
        require(mixture_path.is_relative_to(Path(config['root'])/'train')
                and reference_path.is_relative_to(Path(config['root'])/'train'), 'Only train/development sources allowed')
        # Read aligned references only at this post-inference stage, without
        # consulting them to choose model inputs, schedules, or parameters.
        mixture, rate = sf.read(mixture_path, start=start, frames=frames, dtype='float32', always_2d=True)
        reference, ref_rate = sf.read(reference_path, start=start, frames=frames, dtype='float32', always_2d=True)
        require(rate == ref_rate == 44100 and mixture.shape == reference.shape == (frames, 2)
                and np.isfinite(mixture).all() and np.isfinite(reference).all(), 'Invalid aligned source segment')
        require(array_digest(mixture) == manifest['mixture_float32_sha256']
                and array_digest(reference) == manifest['vocals_float32_sha256'], 'Source clip hashes changed')
        mixture_sha, reference_sha = digest(mixture_path), digest(reference_path)
        for index, name in enumerate(TRACKS, 1):
            if name != track:
                continue
            for setting in SETTINGS:
                report = reports[stream_job_name(index, setting)]
                require(report['input_sha256'] == mixture_sha and report['reference_sha256'] == reference_sha,
                        'Source full WAV hashes changed between streaming jobs')
        source_audio_manifest.append({'track': track, 'mixture_file_sha256': mixture_sha,
                                      'reference_file_sha256': reference_sha,
                                      'mixture_clip_float32_sha256': array_digest(mixture),
                                      'reference_clip_float32_sha256': array_digest(reference)})
        scores = {}
        for name, estimate in waves[track].items():
            whole = separation_metrics(reference, estimate, mixture)
            if name == 'offline_amp2s':
                saved_score = offline_rows[track]['quality']['si_sdr_db']
                require((whole['si_sdr_db'] is None and saved_score is None)
                        or (whole['si_sdr_db'] is not None and saved_score is not None
                            and abs(whole['si_sdr_db']-saved_score) < 1e-8),
                        'Recomputed offline quality differs from worker report')
            common_score = separation_metrics(reference[common[0]:common[1]], estimate[common[0]:common[1]],
                                               mixture[common[0]:common[1]]) if common else None
            scores[name] = {'quality_whole_clip': whole, 'quality_common_real_context_interval': common_score}
        for name in scores:
            scores[name]['common_interval_si_sdr_delta_vs_offline_db'] = score_delta(
                scores[name]['quality_common_real_context_interval']['si_sdr_db'],
                scores['offline_amp2s']['quality_common_real_context_interval']['si_sdr_db']) if common else None
        per_track.append({'track': track, 'clip_start_sample_in_song': start,
                          'clip_samples': frames, 'common_interval_clip_relative_samples': common, 'settings': scores})
    quality_summary = {}
    for name in ['offline_amp2s']+[setting['name'] for setting in SETTINGS]:
        quality_summary[name] = {
            'track_count': 3,
            'valid_whole_clip_si_sdr_tracks': sum(row['settings'][name]['quality_whole_clip']['si_sdr_db'] is not None for row in per_track),
            'valid_common_interval_si_sdr_tracks': sum(
                row['settings'][name]['quality_common_real_context_interval'] is not None
                and row['settings'][name]['quality_common_real_context_interval']['si_sdr_db'] is not None for row in per_track),
            'whole_clip_si_sdr_median_db': median([row['settings'][name]['quality_whole_clip']['si_sdr_db'] for row in per_track]),
            'common_interval_si_sdr_median_db': median([
                row['settings'][name]['quality_common_real_context_interval']['si_sdr_db']
                if row['settings'][name]['quality_common_real_context_interval'] else None for row in per_track]),
            'common_interval_paired_delta_vs_offline_median_db': median([
                row['settings'][name]['common_interval_si_sdr_delta_vs_offline_db'] for row in per_track])}
    analysis = {'schema_version': 1, 'status': 'complete', 'configuration': config,
                'common_quality_interval_clip_relative_samples': common,
                'common_interval_convention': 'start inclusive, end exclusive; one identical contiguous target interval across all tracks/settings and offline output',
                'quality_summary': quality_summary, 'tracks': per_track,
                'steady_emission_timing': {name: timing_summary(rows) for name, rows in steady_by_setting.items()},
                'nominal_availability_seconds': {setting['name']: setting['emit_seconds']+setting['lookahead_seconds'] for setting in SETTINGS},
                'waveform_manifest': waveform_manifest, 'source_audio_manifest': source_audio_manifest,
                'report_index': report_index, 'inference_source_hashes': context['source_hashes'],
                'runtime': context['runtime'],
                'limitations': ['Three short development clips only; one timing trial per track/setting.',
                                'No causal architecture conversion or audio-device end-to-end latency measurement.',
                                'Virtual serial queue follows measured compute; it is not observed hardware latency.',
                                'Pretrained training-song membership undisclosed; no independent generalization claim.'],
                'analysis_generation': {'created_utc': datetime.now(timezone.utc).isoformat(),
                                        'source_path': str(Path(__file__).resolve()),
                                        'source_sha256': digest(Path(__file__)),
                                        'run_record_sha256': digest(out/'run.json'),
                                        'protocol_sha256': digest(out/'protocol.json'),
                                        'numpy': np.__version__, 'soundfile': sf.__version__,
                                        'separate_from_inference_source_hashes': True}}
    text = markdown(analysis)
    write_json(out/'streaming_study_analysis.json', analysis)
    (out/'STREAMING_STUDY_RESULTS.md').write_text(text, encoding='utf-8')
    print(out/'STREAMING_STUDY_RESULTS.md')
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    with exclusive_run(args.run.resolve()):
        build(args.run)


if __name__ == '__main__':
    main()

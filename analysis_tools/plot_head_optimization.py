"""Standalone scientific plots from the completed head-optimization analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    source = args.run/'head_optimization_analysis.json'
    analysis = json.loads(source.read_text(encoding='utf-8'))
    names = ['all_fp32', 'vocal_fp32', 'all_amp', 'vocal_amp']
    labels = ['4 heads\nFP32', 'Vocal head\nFP32', '4 heads\nAMP', 'Vocal head\nAMP']
    colors = ['#68778D', '#449B88', '#8BA3C5', '#16745E']
    q, t = analysis['quality'], analysis['timing']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    x = np.arange(4)
    values = [[q[n]['parameters']/1e6 for n in names],
              [q[n]['peak_allocated_GB'] for n in names],
              [q[n]['si_sdr_median_db'] for n in names],
              [t[n]['median_rtf'] for n in names]]
    titles = ['A  Active model parameters', 'B  Peak allocated GPU memory',
              'C  Vocal separation quality', 'D  Processing cost (3 independent runs)']
    ylabels = ['Parameters (million)', 'GPU memory (GB, decimal)',
               'SI-SDR median across 14 excerpts (dB)', 'Processing call RTF (lower is faster)']
    formats = ['.2f', '.3f', '.6f', '.4f']
    for index, (ax, data) in enumerate(zip(axes.flat, values)):
        ax.bar(x, data, color=colors, width=.64)
        ax.set_xticks(x, labels)
        ax.set_title(titles[index], loc='left', fontweight='bold', pad=12)
        ax.set_ylabel(ylabels[index])
        ax.set_ylim(0, max(data)*1.25)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.18)
        for position, value in zip(x, data):
            ax.text(position, value+max(data)*.035, format(value, formats[index]),
                    ha='center', va='bottom', fontsize=9)
        if index == 3:
            for position, name in zip(x, names):
                rtfs = t[name]['repeat_rtfs']
                ax.scatter(position+np.linspace(-.14, .14, len(rtfs)), rtfs,
                           s=25, color='#152F3F', zorder=4)
    reduction = q['vocal_amp']['parameter_reduction_percent']
    combined = analysis['speedups']['combined']['median_rtf_ratio']
    fig.suptitle(f'BS-RoFormer vocal inference: {reduction:.1f}% fewer parameters; {combined:.2f}x combined throughput',
                 fontsize=15, fontweight='bold', y=.985)
    fig.text(.065, .045,
             'Quality/memory: 14 development excerpts, 30 s each. Timing: 3 songs x 30 s, 3 fresh-process runs per setting.\n'
             'RTX 4060 Laptop GPU; 11 s context; overlap=2; no TTA. Community checkpoint training-song list is unknown.\n'
             'RTF measures computation throughput. Algorithmic and audio-device latency were not measured here.',
             fontsize=9, color='#3C4D59', va='bottom')
    fig.tight_layout(rect=(.025, .14, .99, .94), h_pad=2.2, w_pad=2.8)
    folder = args.run/'figures'
    folder.mkdir(exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(folder/f'head_optimization_overview.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    metadata = {'analysis_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
                'timing_points': '3 independent runs for each setting, not 3 extra quality samples'}
    (folder/'plot_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(folder/'head_optimization_overview.png')


if __name__ == '__main__':
    main()

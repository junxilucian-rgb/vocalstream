"""Plot a completed matched window study; does not execute separation models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    source = args.run / "streaming_study_analysis.json"
    analysis = json.loads(source.read_text(encoding="utf-8"))
    config = analysis["configuration"]
    if analysis["status"] != "complete" or config.get("window_inference_study") != "compare":
        raise ValueError("A completed matched window comparison is required")
    pairs = analysis["matched_window_policy_comparisons"]
    if len(pairs) != 4 or analysis["common_quality_interval_clip_relative_samples"] is None:
        raise ValueError("Four paired schedules and a shared unpadded quality interval are required")
    settings = {row["name"]: row for row in config["settings"]}
    labels = [f"{1000*settings[p['ola_setting']]['emit_seconds']:g} / "
              f"{1000*settings[p['ola_setting']]['lookahead_seconds']:g}" for p in pairs]
    colors = {"ola": "#6A7D95", "single": "#16866B"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x = np.arange(len(pairs))
    width = 0.34
    bars = {}
    for policy, offset in (("ola", -width/2), ("single", width/2)):
        names = [pair[f"{policy}_setting"] for pair in pairs]
        timing = [analysis["steady_emission_timing"][name] for name in names]
        p50 = np.array([row["processing_seconds_p50"]*1000 for row in timing])
        p95 = np.array([row["processing_seconds_p95"]*1000 for row in timing])
        bars[policy] = axes[0, 0].bar(x+offset, p50, width=width, color=colors[policy], label=policy)
        axes[0, 0].errorbar(x+offset, p50, yerr=[np.zeros(len(x)), p95-p50],
                           fmt="none", ecolor="#253447", capsize=3)
        axes[0, 1].bar(x+offset, [row["processing_rtf_vs_emitted_audio"] for row in timing],
                       width=width, color=colors[policy], label=policy)
        quality = [analysis["quality_summary"][name]["common_interval_si_sdr_median_db"] for name in names]
        axes[1, 0].bar(x+offset, quality, width=width, color=colors[policy], label=policy)
    for position, pair in zip(x, pairs):
        budget = settings[pair["ola_setting"]]["emit_seconds"]*1000
        budget_line, = axes[0, 0].plot([position-.39, position+.39], [budget, budget],
                                      color="#B34F3B", linewidth=2)
    axes[0, 0].set_title("A  Per-block compute: median and p95", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Processing time (ms)")
    axes[0, 1].set_title("B  Computation per emitted audio duration", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Steady-state workload RTF")
    rtf_line = axes[0, 1].axhline(1, color="#B34F3B", linestyle="--", linewidth=1.5)
    offline = analysis["quality_summary"]["offline_amp2s"]["common_interval_si_sdr_median_db"]
    offline_line = axes[1, 0].axhline(offline, color="#805EA3", linestyle="--", linewidth=1.5)
    axes[1, 0].set_title("C  Quality on identical target samples", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Median SI-SDR across 3 songs (dB)")
    for position, pair in zip(x, pairs):
        deltas = [row["settings"][pair["single_setting"]]["quality_common_real_context_interval"]["si_sdr_db"]
                  - row["settings"][pair["ola_setting"]]["quality_common_real_context_interval"]["si_sdr_db"]
                  for row in analysis["tracks"]]
        axes[1, 1].scatter(position+np.linspace(-.12, .12, len(deltas)), deltas,
                           color="#253447", s=26, zorder=3)
        axes[1, 1].plot([position-.28, position+.28], [np.median(deltas)]*2,
                        color=colors["single"], linewidth=3,
                        label="Median paired change" if position == 0 else None)
    axes[1, 1].axhline(0, color="#B34F3B", linestyle="--", linewidth=1)
    axes[1, 1].set_title("D  Paired quality change per song", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("SI-SDR: single minus ola (dB)")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Output E / lookahead L (ms)")
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.17)
    begin, end = analysis["common_quality_interval_clip_relative_samples"]
    fig.suptitle("BS-RoFormer stream contexts: nested OLA vs one forward", fontsize=15, fontweight="bold", y=.985)
    fig.legend([bars["ola"], bars["single"], budget_line, rtf_line, offline_line],
               ["ola", "single", "Output-period budget", "RTF = 1", "Offline 2 s reference"],
               loc="upper center", bbox_to_anchor=(.5, .945), ncol=5, frameon=False, fontsize=9)
    fig.text(.055, .025,
             f"3 development clips, {config['requested_samples']/44100:g} s each; 1 timing observation per song/setting; blocks are correlated.\n"
             f"Shared target interval: [{begin/44100:.3f}, {end/44100:.3f}) s. Vocal-only AMP; 2 s context; same checkpoint.\n"
             "Each context: ola 2 network calls, single 1. No neural cache, output crossfade, or audio-device latency measurement.",
             fontsize=9, color="#3C4D59", va="bottom")
    fig.tight_layout(rect=(.015, .13, .99, .915), h_pad=2.3, w_pad=2.5)
    folder = args.run / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"window_comparison_overview.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    metadata = {"analysis_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                "timing_scope": "pooled steady blocks from one run per song/setting; upper whiskers are p95, not confidence intervals",
                "quality_scope": "three independent songs; short shared target interval; unknown pretraining overlap"}
    (folder / "plot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(folder / "window_comparison_overview.png")


if __name__ == "__main__":
    main()

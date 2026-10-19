"""Create a teaching spectrogram comparison from user-supplied WAV files.

Requires numpy, soundfile and Pillow. The plot uses a shared dB scale referenced to the
mixture, so the panels can be compared visually. It is for learning/listening
support, not an objective separation score.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_RATE_EXPECTED = 44100
N_FFT = 4096
HOP_LENGTH = 1024
MAX_FREQUENCY = 8000
DB_FLOOR = -60.0


def read_audio(path: Path, start_seconds: float, duration_seconds: float):
    info = sf.info(path)
    start = round(start_seconds * info.samplerate)
    frames = round(duration_seconds * info.samplerate)
    audio, sample_rate = sf.read(path, start=start, frames=frames, dtype='float32', always_2d=True)
    if len(audio) != frames:
        raise ValueError(f'Requested excerpt extends past the audio end: {path}')
    return audio, sample_rate


def calculate_power_db(audio: np.ndarray, sample_rate: int, reference_power: float):
    if sample_rate != SAMPLE_RATE_EXPECTED:
        raise ValueError(f"Expected {SAMPLE_RATE_EXPECTED} Hz, got {sample_rate} Hz")
    if len(audio) < N_FFT:
        raise ValueError("Audio excerpt is shorter than one STFT window")

    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT, axis=0)[::HOP_LENGTH]
    window = np.hanning(N_FFT).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, None, :], axis=-1)
    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate)
    keep = frequencies <= MAX_FREQUENCY
    power = power[:, keep].T
    power_db = 10.0 * np.log10(np.maximum(power, 1e-20) / reference_power)
    return np.clip(power_db, DB_FLOOR, 0.0)


def colorize(values: np.ndarray):
    stops = np.array(
        [
            [12, 8, 28],
            [62, 37, 112],
            [35, 112, 150],
            [78, 177, 118],
            [245, 224, 76],
        ],
        dtype=np.float32,
    )
    scaled = np.clip(values, 0.0, 1.0) * (len(stops) - 1)
    low = np.floor(scaled).astype(np.int32)
    high = np.minimum(low + 1, len(stops) - 1)
    blend = (scaled - low)[..., None]
    rgb = stops[low] * (1.0 - blend) + stops[high] * blend
    return np.uint8(np.clip(rgb, 0, 255))


def load_font(size: int):
    for font_path in ('DejaVuSans.ttf', r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixture', type=Path, required=True)
    parser.add_argument('--estimate-a', type=Path, required=True, help='Already cropped first estimate')
    parser.add_argument('--estimate-b', type=Path, help='Already cropped second estimate')
    parser.add_argument('--residual', type=Path, help='Already cropped accompaniment estimate')
    parser.add_argument('--label-a', default='Estimated vocals A')
    parser.add_argument('--label-b', default='Estimated vocals B')
    parser.add_argument('--start', type=float, default=0.0, help='Mixture crop start; estimates start at zero')
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--output', type=Path, default=PROJECT_DIR/'visualizations'/'stft_comparison.png')
    args = parser.parse_args()
    if not np.isfinite(args.start) or args.start < 0 or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('start must be finite >=0; duration must be finite >0')
    panels = [('Mixture', args.mixture, args.start), (args.label_a, args.estimate_a, 0.0)]
    if args.estimate_b:
        panels.append((args.label_b, args.estimate_b, 0.0))
    if args.residual:
        panels.append(('Estimated accompaniment', args.residual, 0.0))
    loaded = []
    mixture_peak_power = None
    for label, path, start_seconds in panels:
        audio, sample_rate = read_audio(path, start_seconds, args.duration)
        if mixture_peak_power is None:
            frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT, axis=0)[::HOP_LENGTH]
            window = np.hanning(N_FFT).astype(np.float32)
            spectrum = np.fft.rfft(frames * window[None, None, :], axis=-1)
            mixture_peak_power = float(np.max(np.mean(np.abs(spectrum) ** 2, axis=1)))
        db = calculate_power_db(audio, sample_rate, mixture_peak_power)
        loaded.append((label, db))

    width = 1280
    plot_left = 120
    plot_width = 1040
    plot_height = 145
    row_gap = 48
    top = 115
    row_step = plot_height + row_gap
    height = top + row_step * len(loaded) + 95
    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    title_font = load_font(27)
    label_font = load_font(19)
    small_font = load_font(15)

    draw.text((plot_left, 22), f"STFT view | {args.start:g} to {args.start + args.duration:g} seconds", fill=(24, 32, 44), font=title_font)
    draw.text(
        (plot_left, 58),
        "4096-sample window, 1024-sample hop | stereo power averaged | shared dB scale",
        fill=(75, 84, 96),
        font=small_font,
    )

    for panel_index, (label, db) in enumerate(loaded):
        row_top = top + panel_index * row_step
        draw.text((plot_left, row_top - 26), label, fill=(28, 38, 52), font=label_font)
        normalized = (db - DB_FLOOR) / (0.0 - DB_FLOOR)
        heat = Image.fromarray(colorize(np.flipud(normalized)), mode="RGB")
        heat = heat.resize((plot_width, plot_height), Image.Resampling.BILINEAR)
        image.paste(heat, (plot_left, row_top))
        draw.rectangle(
            (plot_left, row_top, plot_left + plot_width, row_top + plot_height),
            outline=(95, 105, 118),
            width=1,
        )

        for frequency in (0, 4000, 8000):
            y = row_top + plot_height - int(frequency / MAX_FREQUENCY * plot_height)
            draw.line((plot_left - 5, y, plot_left, y), fill=(55, 63, 75), width=1)
            draw.text((plot_left - 54, y - 8), f"{frequency // 1000}k", fill=(55, 63, 75), font=small_font)

        for seconds in np.linspace(0, args.duration, 4):
            x = plot_left + int(seconds / args.duration * plot_width)
            draw.line((x, row_top + plot_height, x, row_top + plot_height + 4), fill=(55, 63, 75), width=1)
            draw.text((x - 9, row_top + plot_height + 6), f'{seconds:g}', fill=(55, 63, 75), font=small_font)

    legend_y = height - 35
    draw.text((plot_left, legend_y), "Relative level: -60 dB", fill=(55, 63, 75), font=small_font)
    gradient = np.linspace(0, 1, 300, dtype=np.float32)[None, :]
    legend = Image.fromarray(colorize(np.repeat(gradient, 18, axis=0)), mode="RGB")
    image.paste(legend, (plot_left + 190, legend_y - 1))
    draw.text((plot_left + 500, legend_y), "0 dB (mixture peak)", fill=(55, 63, 75), font=small_font)
    draw.text((plot_left + 760, legend_y), "Time within excerpt (seconds)", fill=(55, 63, 75), font=small_font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()

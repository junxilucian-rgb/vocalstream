"""Dependency bounds, packet delivery, EOF and virtual scheduling without audio hardware."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from vocalstream.streaming import BoundedFutureSeparator


class RecordingSeparator:
    def __init__(self, sample_rate=1000, use_context_mean=False):
        self.sample_rate = sample_rate
        self.windows = []
        self.use_context_mean = use_context_mean
        self.last_run = None
        self.window_inference_mode = "single"

    def separate(self, window):
        self.windows.append(window.copy())
        self.last_run = {"model_forward_passes": 1, "processing_seconds": 0.001,
                         "cuda_peak_allocated_bytes": None}
        if self.use_context_mean:
            return np.broadcast_to(window.mean(axis=0), window.shape).copy(), 0.001
        return window.copy(), 0.001


class StreamingTests(unittest.TestCase):
    def make_stream(self, model=None, **kwargs):
        model = model or RecordingSeparator()
        options = dict(sample_rate=1000, context_seconds=.010,
                       emit_seconds=.004, lookahead_seconds=.002)
        options.update(kwargs)
        return BoundedFutureSeparator(model, **options)

    def test_irregular_packets_reconstruct_once_and_pad_only_missing_context(self):
        samples = np.arange(1, 24, dtype=np.float32)
        audio = np.column_stack([samples, -samples])
        original = audio.copy()
        model = RecordingSeparator()
        stream = self.make_stream(model)
        emissions, cursor, packet_index = [], 0, 0
        while cursor < len(audio):
            count = min((1, 4, 2, 3)[packet_index % 4], stream.next_input_frames,
                        len(audio)-cursor)
            emissions.extend(stream.push(audio[cursor:cursor+count]))
            cursor += count
            packet_index += 1
        emissions.extend(stream.finish())
        np.testing.assert_array_equal(np.concatenate([item.audio for item in emissions]), audio)
        np.testing.assert_array_equal(audio, original)
        self.assertEqual(len(model.windows), len(emissions))
        for window, emission in zip(model.windows, emissions):
            record = emission.record
            begin, end = record["context_start_sample"], record["context_end_sample_exclusive"]
            expected = np.zeros((10, 2), dtype=np.float32)
            real_begin, real_end = max(0, begin), min(len(audio), end)
            expected[real_begin-begin:real_end-begin] = audio[real_begin:real_end]
            np.testing.assert_array_equal(window, expected)
            self.assertLessEqual(record["real_context_end_sample_exclusive"],
                                 record["input_received_sample_boundary"])
            self.assertEqual(record["packetization_wait_samples"], 0)
        self.assertEqual(emissions[-1].record["output_samples"], 3)
        self.assertTrue(emissions[-1].record["eof_flush"])
        self.assertEqual(stream.summary()["output_samples"], 23)
        json.dumps(stream.summary(), allow_nan=False)

    def test_extra_delivered_packet_samples_do_not_leak_beyond_lookahead(self):
        prefix = np.arange(16, dtype=np.float32).reshape(8, 2)
        changed = prefix.copy()
        changed[6:] += 100000
        observations = []
        for audio in (prefix, changed):
            model = RecordingSeparator(use_context_mean=True)
            stream = self.make_stream(model)
            self.assertEqual(stream.push(audio[:4]), [])
            emitted = stream.push(audio[4:8])
            self.assertEqual(len(emitted), 1)
            record = emitted[0].record
            self.assertEqual(record["required_real_input_sample_boundary"], 6)
            self.assertEqual(record["input_received_sample_boundary"], 8)
            self.assertEqual(record["packetization_wait_samples"], 2)
            self.assertEqual(record["real_context_end_sample_exclusive"], 6)
            observations.append((emitted[0].audio, model.windows[0]))
        np.testing.assert_array_equal(observations[0][0], observations[1][0])
        np.testing.assert_array_equal(observations[0][1], observations[1][1])

    def test_virtual_queue_uses_output_budget_and_separates_startup(self):
        model = RecordingSeparator(sample_rate=8)
        stream = self.make_stream(model, sample_rate=8, context_seconds=.5,
                                  emit_seconds=.25, lookahead_seconds=0)
        with patch("vocalstream.streaming.time.perf_counter", side_effect=[0, .375, 1, 1.375, 2, 2.375]):
            for _ in range(3):
                stream.push(np.ones((2, 2), dtype=np.float32))
        self.assertEqual(stream.finish(), [])
        self.assertEqual([row["virtual_worker_finish_seconds"] for row in stream.records], [.625, 1, 1.375])
        self.assertEqual([row["virtual_queue_wait_seconds"] for row in stream.records], [0, .125, .25])
        self.assertTrue(all(row["virtual_deadline_missed"] for row in stream.records))
        summary = stream.summary()
        all_rows = summary["all_emissions"]
        self.assertEqual(all_rows["total_processing_rtf"], 1.5)
        self.assertEqual(all_rows["model_forward_passes_total"], 3)
        self.assertEqual(summary["regular_emissions"]["emission_count"], 3)
        self.assertEqual(summary["full_real_context_steady_state_emissions"]["emission_count"], 2)
        self.assertFalse(summary["hardware_end_to_end_latency_measured"])

    def test_empty_and_very_short_streams_flush_exact_length(self):
        for count in (0, 1, 2, 5):
            with self.subTest(count=count):
                stream = self.make_stream()
                audio = np.ones((count, 2), dtype=np.float32)
                emissions = []
                for start in range(0, count, 4):
                    emissions.extend(stream.push(audio[start:start+4]))
                emissions.extend(stream.finish())
                self.assertEqual(sum(len(item.audio) for item in emissions), count)
                self.assertEqual(stream.summary()["output_samples"], count)
                if count:
                    np.testing.assert_array_equal(np.concatenate([item.audio for item in emissions]), audio)
                with self.assertRaises(RuntimeError):
                    stream.finish()
                with self.assertRaises(RuntimeError):
                    stream.push(np.ones((1, 2)))

    def test_invalid_packets_and_schedules_rejected(self):
        for packet in (np.empty((0, 2)), np.zeros((5, 2)), np.zeros(4),
                       np.zeros((1, 1)), np.ones((1, 2))*np.nan,
                       np.ones((1, 2), dtype=np.complex64)):
            with self.subTest(shape=packet.shape):
                with self.assertRaises(ValueError):
                    self.make_stream().push(packet)
        for options in ({"emit_seconds": 0}, {"lookahead_seconds": -1},
                        {"context_seconds": .001}, {"sample_rate": True},
                        {"sample_rate": 1000.5}, {"lookahead_seconds": float("nan")}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.make_stream(**options)

    def test_input_buffer_remains_bounded_for_long_stream(self):
        stream = self.make_stream()
        for _ in range(300):
            stream.push(np.ones((4, 2), dtype=np.float32))
        stream.finish()
        self.assertLessEqual(stream.summary()["max_retained_input_buffer_samples"],
                             stream.context_samples + stream.emit_samples)


if __name__ == "__main__":
    unittest.main()

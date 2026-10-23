"""Checks for measurement preservation and timestamp-based video resampling."""
import csv
from contextlib import ExitStack
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from extract_rosbag import VideoSink, decode_depth, decode_rgb, flatten, topic_key


class ImageDecodingTests(unittest.TestCase):
    def test_compressed_depth_preserves_uint16_and_invalid_pixels(self):
        native = np.array([[0, 1000], [1234, 65535]], dtype=np.uint16)
        ok, encoded = cv2.imencode('.png', native)
        self.assertTrue(ok)
        msg = SimpleNamespace(format='16UC1; compressedDepth',
                              data=np.frombuffer(struct.pack('<iff', 0, 0, 0) + encoded.tobytes(), dtype=np.uint8))
        metres, recovered, encoding = decode_depth(msg, .001)
        np.testing.assert_array_equal(recovered, native)
        self.assertTrue(np.isnan(metres[0, 0]))
        self.assertAlmostEqual(float(metres[1, 0]), 1.234, places=5)
        self.assertEqual(encoding, '16UC1')

    def test_inverse_depth_decodes_to_metres(self):
        inverse = np.array([[0, 110], [210, 60]], dtype=np.uint16)
        _, png = cv2.imencode('.png', inverse)
        payload = struct.pack('<iff', 0, 100, 10) + png.tobytes()
        msg = SimpleNamespace(format='32FC1; compressedDepth png', data=np.frombuffer(payload, dtype=np.uint8))
        metres, native, _ = decode_depth(msg, .001)
        np.testing.assert_allclose(metres, [[np.nan, 1], [.5, 2]], equal_nan=True)
        self.assertIsNone(native)

    def test_raw_depth_respects_big_endian_and_row_padding(self):
        payload = b'\x03\xe8\x07\xd0xx\x00\x00\x0b\xb8yy'
        msg = SimpleNamespace(encoding='16UC1', width=2, height=2, step=6, is_bigendian=True,
                              data=np.frombuffer(payload, dtype=np.uint8))
        metres, native, _ = decode_depth(msg, .001)
        np.testing.assert_array_equal(native, [[1000, 2000], [0, 3000]])
        np.testing.assert_allclose(metres, [[1, 2], [np.nan, 3]], equal_nan=True)

    def test_raw_float_depth_invalid_and_units(self):
        image = np.array([[1.5, 0, np.inf, -1]], dtype='<f4')
        msg = SimpleNamespace(encoding='32FC1', width=4, height=1, step=16, is_bigendian=False,
                              data=np.frombuffer(image.tobytes(), dtype=np.uint8))
        metres, native, _ = decode_depth(msg, .001)
        np.testing.assert_allclose(metres, [[1.5, np.nan, np.nan, np.nan]], equal_nan=True)
        self.assertIsNone(native)

    def test_raw_rgb_channel_order_and_padding(self):
        msg = SimpleNamespace(encoding='rgb8', width=1, height=1, step=4, is_bigendian=False,
                              data=np.array([255, 0, 0, 99], dtype=np.uint8))
        np.testing.assert_array_equal(decode_rgb(msg), [[[0, 0, 255]]])

    def test_rvl_rejected_instead_of_silent_wrong_depth(self):
        with self.assertRaisesRegex(ValueError, 'RVL'):
            decode_depth(SimpleNamespace(format='16UC1; compressedDepth rvl'), .001)


class TimelineTests(unittest.TestCase):
    def test_video_gap_preserves_elapsed_time_and_frame_provenance(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            sink = VideoSink(Path(folder), 2, stack)
            # Seed an encoder stub so this unit test checks timeline logic only.
            sink.proc = Mock()
            sink.shape = (2, 2, 3)
            sink.first_stamp = 10_000_000_000
            first = np.zeros(sink.shape, dtype=np.uint8)
            sink.previous = (first, sink.first_stamp, 7)
            sink.add(np.ones_like(first), 11_500_000_000, 8)
            sink.proc.wait.return_value = 0
            sink.close()
            stack.close()
            with (Path(folder) / 'frames.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([float(r['video_time_s']) for r in rows], [0, .5, 1, 1.5])
            self.assertEqual([int(r['source_bag_timestamp_ns']) for r in rows],
                             [10_000_000_000] * 3 + [11_500_000_000])

    def test_arrays_preserved_in_single_csv_cells(self):
        row = flatten({'transforms': [{'child_frame_id': 'one'}, {'child_frame_id': 'two'}]})
        self.assertIn('two', row['transforms'])
        self.assertNotEqual(topic_key('/a_b/c'), topic_key('/a/b_c'))


if __name__ == '__main__':
    unittest.main()

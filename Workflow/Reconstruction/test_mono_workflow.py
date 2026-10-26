"""Compatibility/error checks for the external COLMAP boundary."""
import unittest
import json
from pathlib import Path
import tempfile

from mono_workflow import calibration_parameters, export_calibration, gpu_option, match_graph, new_dataset_name, registration_count, training_layout


class ColmapBoundaryTests(unittest.TestCase):
    def test_training_layout_preserves_calibration_and_resizes_undistorted_images(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'sparse').mkdir()
            for name in ['cameras', 'images', 'points3D']:
                (root / 'sparse' / f'{name}.bin').write_bytes(b'original model')
            (root / 'images').mkdir()
            Image.new('RGB', (17, 11)).save(root / 'images/frame.png')
            training_layout(root, 2)
            self.assertEqual((root / 'sparse/0/cameras.bin').read_bytes(), b'original model')
            with Image.open(root / 'images_2/frame.png') as image:
                self.assertEqual(image.size, (8, 5))
            with Image.open(root / 'images/frame.png') as image:
                self.assertEqual(image.size, (17, 11))
            training_layout(root, 2)  # Reuse only a matching, complete pyramid.
            (root / 'images_2/frame.png').unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete or stale'):
                training_layout(root, 2)

    def test_training_layout_rejects_missing_model(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'sparse model missing'):
                training_layout(Path(folder), 1)

    def test_calibration_namespace_override_for_full_resolution_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder)
            export = dataset / 'source_export'
            export.mkdir()
            topic = '/camera/camera/color/camera_info'
            (export / 'manifest.json').write_text(json.dumps({
                'rgb_topic': '/camera_full/color/image_raw/compressed',
                'files': {topic: {'jsonl': 'full_color.jsonl'}}}))
            message = {'width': 1280, 'height': 720, 'k': [640., 0., 640., 0., 640., 360., 0., 0., 1.],
                       'd': [0.] * 5, 'distortion_model': 'plumb_bob'}
            (export / 'full_color.jsonl').write_text(json.dumps({'message': message}) + '\n')
            with self.assertRaisesRegex(ValueError, 'camera-info-topic'):
                export_calibration(dataset)
            result = export_calibration(dataset, topic)
            self.assertEqual((result['width'], result['height']), (1280, 720))
            self.assertEqual(result['source_topic'], topic)
            self.assertEqual(result['params'][2:4], [640.5, 360.5])

    def test_calibrated_plumb_bob_preserves_k3_and_pixel_convention(self):
        message = {'width': 448, 'height': 448, 'k': [386., 0., 226., 0., 385., 225., 0., 0., 1.],
                   'd': [-.05, .06, -.0005, .0012, -.02], 'distortion_model': 'plumb_bob'}
        calibration = calibration_parameters(message)
        self.assertEqual(calibration['model'], 'FULL_OPENCV')
        self.assertEqual(calibration['params'], [386., 385., 226.5, 225.5, -.05, .06, -.0005, .0012, -.02, 0., 0., 0.])
        self.assertTrue(calibration['fixed'])

    def test_roi_and_unknown_distortion_not_silently_misinterpreted(self):
        message = {'width': 448, 'height': 448, 'k': [386., 0., 226., 0., 385., 225., 0., 0., 1.],
                   'd': [0.] * 5, 'distortion_model': 'plumb_bob', 'roi': {'x_offset': 10}}
        with self.assertRaisesRegex(ValueError, 'ROI'):
            calibration_parameters(message)
        message['roi'] = {}
        message['distortion_model'] = 'equidistant'
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            calibration_parameters(message)

    def test_connectivity_counts_weak_bridges_and_isolated_images(self):
        pairs = [(1 * 2147483647 + 2, 250), (2 * 2147483647 + 3, 20)]
        loose = match_graph([1, 2, 3, 4], pairs, 15)
        strong = match_graph([1, 2, 3, 4], pairs, 100)
        self.assertEqual(loose['component_sizes'], [3, 1])
        self.assertEqual(loose['isolated_images'], 1)
        self.assertEqual(strong['component_sizes'], [2, 1, 1])
        self.assertEqual(strong['isolated_images'], 2)

    def test_older_and_newer_gpu_options(self):
        self.assertEqual(gpu_option('--SiftExtraction.use_gpu arg (=1)', 'extract', False),
                         ['--SiftExtraction.use_gpu', '0'])
        self.assertEqual(gpu_option('--FeatureExtraction.use_gpu arg (=1)', 'extract', True),
                         ['--FeatureExtraction.use_gpu', '1'])
        self.assertEqual(gpu_option('--SiftMatching.use_gpu arg (=1)', 'match', False),
                         ['--SiftMatching.use_gpu', '0'])
        self.assertEqual(gpu_option('--FeatureMatching.use_gpu arg (=1)', 'match', True),
                         ['--FeatureMatching.use_gpu', '1'])

    def test_unknown_options_rejected_instead_of_silent_gpu_default(self):
        with self.assertRaises(ValueError):
            gpu_option('unrecognized help output', 'extract', False)

    def test_registration_count_in_analyzer_logs(self):
        self.assertEqual(registration_count('I123 analyzer.cc] Registered images: 63\nPoints: 1452'), 63)
        self.assertEqual(registration_count('Images: 20\nPoints: 1452'), 20)
        with self.assertRaises(ValueError):
            registration_count('Could not open sparse model')

    def test_dataset_name_cannot_escape_generated_data_directory(self):
        for name in ['../outside', '/tmp/elsewhere', '..', 'two/names']:
            with self.assertRaises(ValueError):
                new_dataset_name(name)
        self.assertEqual(new_dataset_name('flight_33_70').name, 'flight_33_70')


if __name__ == '__main__':
    unittest.main()

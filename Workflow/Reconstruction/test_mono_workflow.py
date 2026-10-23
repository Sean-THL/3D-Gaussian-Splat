"""Compatibility/error checks for the external COLMAP boundary."""
import unittest

from mono_workflow import gpu_option, new_dataset_name, registration_count


class ColmapBoundaryTests(unittest.TestCase):
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

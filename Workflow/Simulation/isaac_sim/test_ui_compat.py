import unittest

from Workflow.Simulation.isaac_sim.ui_compat import install_property_scroll_guard


class UICompatibilityTests(unittest.TestCase):
    def test_missing_frame_is_skipped_and_valid_calls_are_preserved(self):
        class Window:
            properties_frame = None
            def save_scroll_pos(self, reset=False):
                if reset:
                    self.saved = None
                    return None
                self.saved = (self.properties_frame.scroll_y, reset)
                return self.saved
        self.assertTrue(install_property_scroll_guard(Window)['installed'])
        guarded = Window.save_scroll_pos
        install_property_scroll_guard(Window)
        self.assertIs(Window.save_scroll_pos, guarded)
        window = Window()
        self.assertIsNone(window.save_scroll_pos())
        window.saved = 'old value'
        self.assertIsNone(window.save_scroll_pos(reset=True))
        self.assertIsNone(window.saved)
        window.properties_frame = type('Frame', (), {'scroll_y': 3.})()
        self.assertEqual(window.save_scroll_pos(), (3., False))

    def test_unrelated_errors_are_not_suppressed(self):
        class Window:
            properties_frame = object()
            def save_scroll_pos(self, reset=False):
                raise RuntimeError('unrelated UI error')
        install_property_scroll_guard(Window)
        with self.assertRaisesRegex(RuntimeError, 'unrelated UI error'):
            Window().save_scroll_pos()


if __name__ == '__main__':
    unittest.main()

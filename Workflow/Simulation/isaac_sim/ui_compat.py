"""Project-scoped workaround for the Isaac 4.5 property-window lifecycle bug.

Kit 1.11.x can request scroll saving after the property's scrolling frame has
been destroyed/hidden. Never edit installed extension files: guard that single
public method for this process. Other exceptions still propagate unchanged.
"""
from functools import wraps


def install_property_scroll_guard(window_class=None):
    """Skip saving scroll position when there is no scrolling frame to save.

    Call after SimulationApp starts. An optional class allows tests without Kit.
    The wrapper is idempotent across Script Editor imports/reloads.
    """
    if window_class is None:
        try:
            from omni.kit.window.property.window import PropertyWindow
        except ImportError:
            return dict(installed=False, reason='property-window extension is not loaded')
        window_class = PropertyWindow

    original = window_class.save_scroll_pos
    if getattr(original, '_dasc_property_scroll_guard', False):
        return dict(installed=True, reason='already installed')

    @wraps(original)
    def save_scroll_pos_if_frame_exists(self, reset=False):
        # Reset only clears cached state in Kit 1.11.5 and needs no UI frame.
        if not reset and self.properties_frame is None:
            return None
        return original(self, reset=reset)

    save_scroll_pos_if_frame_exists._dasc_property_scroll_guard = True
    window_class.save_scroll_pos = save_scroll_pos_if_frame_exists
    return dict(installed=True, reason='skip scroll save only while properties_frame is None')

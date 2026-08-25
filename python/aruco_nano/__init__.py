"""
Purpose:  Python wrapper for ArUco Nano, a header-only ArUco marker detector
          (https://github.com/rmsalinas/aruco_nano). Thin, idiomatic layer over
          the pybind11 extension module ``_aruco_nano``.
Status:   Active. Wraps the vendored aruco_nano.h (v10) unmodified.
Future:   Add a cv2.aruco.Dictionary passthrough; expose rejected-candidate
          geometry.
"""
from ._aruco_nano import (  # noqa: F401
    ArucoNanoVersion,
    ArucoDetector,
    DetectorParameters,
    Marker,
    __version__,
    detect,
)

# The library's native default dictionary (aruco_nano.h DetectorParameters).
DEFAULT_DICT_ID = 21  # cv2.aruco.DICT_ARUCO_MIP_36h12

__all__ = [
    "ArucoNanoVersion",
    "ArucoDetector",
    "DetectorParameters",
    "Marker",
    "__version__",
    "detect",
    "detect_markers",
    "draw_detected_markers",
    "DEFAULT_DICT_ID",
]


def detect_markers(image, dict_id=None, params=None):
    """Detect markers and return OpenCV-style ``(corners, ids)``.

    ``corners`` is a list of ``(4, 2)`` float32 arrays; ``ids`` is an int32
    array. ``dict_id`` is a predefined dictionary id (e.g. ``cv2.aruco.DICT_6X6_250``);
    it defaults to ``DICT_ARUCO_MIP_36h12`` (the library's native default).
    ``params`` is an optional ``DetectorParameters``.
    """
    detector = ArucoDetector(
        DEFAULT_DICT_ID if dict_id is None else dict_id, params=params
    )
    return detector.detect_markers(image)


def draw_detected_markers(image, corners, ids, border_color=(0, 255, 0)):
    """Draw detected markers onto a copy of ``image`` using OpenCV.

    Convenience mirror of ``cv2.aruco.drawDetectedMarkers``. Accepts corners in
    the wrapper's native ``(4, 2)`` shape and reshapes to the ``(1, 4, 2)``
    shape cv2 expects.
    """
    import cv2
    import numpy as np

    out = image.copy()
    if ids is not None and len(ids) > 0:
        # cv2.aruco.drawDetectedMarkers requires (1, 4, 2) per marker.
        reshaped = [np.asarray(c, dtype=np.float32).reshape(1, 4, 2) for c in corners]
        cv2.aruco.drawDetectedMarkers(out, reshaped, ids, border_color)
    return out

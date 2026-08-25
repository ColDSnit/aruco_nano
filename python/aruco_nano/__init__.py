"""
Purpose:  Python wrapper for ArUco Nano, a header-only ArUco marker detector
          (https://github.com/rmsalinas/aruco_nano). Thin, idiomatic layer over
          the pybind11 extension module ``_aruco_nano``.
Status:   Active. Wraps the vendored aruco_nano.h (v10) unmodified.
Future:   Add a cv2.aruco.Dictionary passthrough; expose rejected-candidate
          geometry; add a draw helper that mirrors cv2.aruco.drawDetectedMarkers.
"""
from ._aruco_nano import (  # noqa: F401
    ArucoNanoVersion,
    ArucoDetector,
    DetectorParameters,
    Marker,
    __version__,
    detect,
)

__all__ = [
    "ArucoNanoVersion",
    "ArucoDetector",
    "DetectorParameters",
    "Marker",
    "__version__",
    "detect",
    "detect_markers",
    "draw_detected_markers",
]


def detect_markers(image, dict_id=None, params=None):
    """Detect markers and return OpenCV-style ``(corners, ids)``.

    ``corners`` is a list of ``(4, 2)`` float32 arrays; ``ids`` is an int32
    array. ``dict_id`` is a predefined dictionary id (e.g. ``cv2.aruco.DICT_6X6_250``).
    """
    detector = ArucoDetector(dict_id if dict_id is not None else 0)
    return detector.detect_markers(image)


def draw_detected_markers(image, corners, ids, border_color=(0, 255, 0)):
    """Draw detected markers onto a copy of ``image`` using OpenCV.

    Convenience mirror of ``cv2.aruco.drawDetectedMarkers`` so callers do not
    need to import cv2 themselves for visualisation.
    """
    import cv2

    out = image.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids, border_color)
    return out

"""
Purpose:  End-to-end tests for the aruco_nano Python wrapper. Generate real
           ArUco markers with OpenCV and verify the wrapper detects them.
Status:   Active.
Future:   Add multi-marker, inverted-marker, and multi-dictionary cases.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import aruco_nano


def _make_marker(dict_id, marker_id, size=200, border=50):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    img = cv2.aruco.generateImageMarker(d, marker_id, size)
    return cv2.copyMakeBorder(img, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=255)


def test_single_marker_detection():
    dict_id = cv2.aruco.DICT_6X6_250
    marker_id = 42
    img = _make_marker(dict_id, marker_id)
    det = aruco_nano.ArucoDetector(dict_id)
    corners, ids = det.detect_markers(img)
    assert len(ids) == 1
    assert int(ids[0]) == marker_id
    assert len(corners) == 1
    assert corners[0].shape == (4, 2)


def test_module_level_detect():
    dict_id = cv2.aruco.DICT_6X6_250
    marker_id = 7
    img = _make_marker(dict_id, marker_id)
    markers = aruco_nano.detect(img, dict_id=dict_id)
    assert len(markers) == 1
    assert markers[0].id == marker_id
    assert markers[0].dict == 0


def test_pose_estimation():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    markers = aruco_nano.detect(img, dict_id=dict_id)
    K = np.array([[500, 0, 150], [0, 500, 150], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    rvec, tvec = markers[0].estimate_pose(K, D, 0.05)
    assert rvec.shape == (3, 1)
    assert tvec.shape == (3, 1)


def test_detector_parameters():
    p = aruco_nano.DetectorParameters()
    assert p.min_size == 10
    p.min_size = 20
    assert p.min_size == 20
    p.set_dicts(cv2.aruco.DICT_6X6_250)


def test_multi_dict():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 11)
    det = aruco_nano.ArucoDetector([dict_id, cv2.aruco.DICT_4X4_50])
    corners, ids, dict_indices = det.detect_markers_multi_dict(img)
    assert len(ids) == 1
    assert int(ids[0]) == 11
    assert int(dict_indices[0]) == 0


if __name__ == "__main__":
    test_single_marker_detection()
    test_module_level_detect()
    test_pose_estimation()
    test_detector_parameters()
    test_multi_dict()
    print("ALL TESTS PASSED")

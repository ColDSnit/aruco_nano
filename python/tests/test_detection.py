"""
Purpose:  End-to-end tests for the aruco_nano Python wrapper. Generate real
           ArUco markers with OpenCV and verify the wrapper detects them.
Status:   Active.
Future:   Add inverted-marker and multi-marker cases.
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


def test_pose_estimation_flat_camera_matrix():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    markers = aruco_nano.detect(img, dict_id=dict_id)
    K = np.array([500, 0, 150, 0, 500, 150, 0, 0, 1], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    rvec, tvec = markers[0].estimate_pose(K, D, 0.05)
    assert rvec.shape == (3, 1)


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


def test_detect_markers_helper():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 5)
    corners, ids = aruco_nano.detect_markers(img, dict_id=dict_id)
    assert len(ids) == 1
    assert int(ids[0]) == 5


def test_detect_markers_helper_default_dict():
    # Default dict is DICT_ARUCO_MIP_36h12 (id 21); a 6x6_250 marker must NOT match.
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 5)
    corners, ids = aruco_nano.detect_markers(img)
    assert len(ids) == 0


def test_detect_markers_helper_params():
    # params tunes detection; dict_id selects the dictionary (and overrides
    # any dicts carried by params, matching the C++ ArucoDetector semantics).
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 5)
    p = aruco_nano.DetectorParameters()
    p.min_size = 5  # tune a real detection parameter
    corners, ids = aruco_nano.detect_markers(img, dict_id=dict_id, params=p)
    assert len(ids) == 1
    assert int(ids[0]) == 5


def test_draw_detected_markers():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 9)
    corners, ids = aruco_nano.detect_markers(img, dict_id=dict_id)
    out = aruco_nano.draw_detected_markers(img, corners, ids)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_invalid_dict_id_raises():
    import pytest
    with pytest.raises(Exception):
        aruco_nano.ArucoDetector(9999)


def test_float_image_rejected():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 1)
    fimg = img.astype(np.float32) / 255.0
    try:
        aruco_nano.detect_markers(fimg, dict_id=dict_id)
        raise AssertionError("float image should have been rejected")
    except TypeError:
        pass  # expected: pybind11 rejects non-uint8 at the boundary


def test_return_rejected():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 1)
    markers, rejected = aruco_nano.detect(img, dict_id=dict_id, return_rejected=True)
    assert len(markers) == 1
    assert isinstance(rejected, list)


if __name__ == "__main__":
    test_single_marker_detection()
    test_module_level_detect()
    test_pose_estimation()
    test_pose_estimation_flat_camera_matrix()
    test_detector_parameters()
    test_multi_dict()
    test_detect_markers_helper()
    test_detect_markers_helper_default_dict()
    test_detect_markers_helper_params()
    test_draw_detected_markers()
    test_invalid_dict_id_raises()
    test_float_image_rejected()
    test_return_rejected()
    print("ALL TESTS PASSED")

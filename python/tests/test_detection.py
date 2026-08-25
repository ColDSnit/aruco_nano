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
    # column 9-vector must also reshape correctly
    rvec2, tvec2 = markers[0].estimate_pose(K.reshape(9, 1), D, 0.05)
    assert rvec2.shape == (3, 1)


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


def test_bool_dict_id_rejected():
    try:
        aruco_nano.ArucoDetector(True)
        raise AssertionError("bool dict id should have been rejected")
    except RuntimeError:
        pass


def test_empty_image_rejected():
    try:
        aruco_nano.ArucoDetector(10).detect_markers(np.zeros((0, 0), dtype=np.uint8))
        raise AssertionError("empty image should have been rejected")
    except ValueError:
        pass


def test_zero_dim_image_rejected():
    # 0-d and 1-d arrays must raise cleanly, not segfault (round-4 P0 regression)
    for arr in [np.array(5, np.uint8), np.zeros(5, np.uint8)]:
        try:
            aruco_nano.detect(arr)
            raise AssertionError("0-d/1-d image should have been rejected")
        except ValueError:
            pass


def test_numpy_bool_dict_id_rejected():
    for v in [np.bool_(True), np.bool_(False), [True]]:
        try:
            aruco_nano.ArucoDetector(v)
            raise AssertionError("bool dict id should have been rejected")
        except RuntimeError:
            pass


def test_numpy_int_color_accepted():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    m = aruco_nano.detect(img, dict_id=dict_id)[0]
    out = m.draw(img, (np.int64(0), np.int64(255), np.int64(0)))
    assert out.shape == img.shape


def test_malformed_camera_matrix_rejected():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    m = aruco_nano.detect(img, dict_id=dict_id)[0]
    K = np.array([[500, 0], [0, 500]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    try:
        m.estimate_pose(K, D, 0.05)
        raise AssertionError("malformed camera matrix should have been rejected")
    except ValueError:
        pass


def test_marker_border_bits_must_be_one():
    p = aruco_nano.DetectorParameters()
    for bad in [2, 1.5, 4, -1, 8, 3, 0]:
        try:
            p.marker_border_bits = bad
            raise AssertionError("marker_border_bits should reject non-1 values")
        except ValueError:
            pass
    p.marker_border_bits = 1  # valid


def test_dist_coeffs_shape_validated():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    m = aruco_nano.detect(img, dict_id=dict_id)[0]
    K = np.array([[500, 0, 150], [0, 500, 150], [0, 0, 1]], dtype=np.float64)
    for bad in [np.zeros((3, 3)), np.zeros((2, 2)), np.zeros(3), np.zeros(6)]:
        try:
            m.estimate_pose(K, bad, 0.05)
            raise AssertionError("bad dist_coeffs should be rejected")
        except ValueError:
            pass
    r, t = m.estimate_pose(K, np.zeros(5), 0.05)
    assert r.shape == (3, 1)


def test_marker_size_finite_and_positive():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    m = aruco_nano.detect(img, dict_id=dict_id)[0]
    K = np.array([[500, 0, 150], [0, 500, 150], [0, 0, 1]], dtype=np.float64)
    for bad in [0.0, -1.0, float("nan"), float("inf"), 1e-9]:
        try:
            m.estimate_pose(K, np.zeros(5), bad)
            raise AssertionError("bad marker_size should be rejected")
        except ValueError:
            pass


def test_degenerate_camera_matrix_rejected():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    m = aruco_nano.detect(img, dict_id=dict_id)[0]
    D = np.zeros(5, dtype=np.float64)
    for badK in [np.zeros((3, 3)), np.full((3, 3), np.nan)]:
        try:
            m.estimate_pose(badK, D, 0.05)
            raise AssertionError("degenerate camera matrix should be rejected")
        except ValueError:
            pass


def test_marker_size_positive():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    m = aruco_nano.detect(img, dict_id=dict_id)[0]
    K = np.array([[500, 0, 150], [0, 500, 150], [0, 0, 1]], dtype=np.float64)
    try:
        m.estimate_pose(K, np.zeros(5), 0.0)
        raise AssertionError("marker_size=0 should be rejected")
    except ValueError:
        pass


def test_tiny_marker_size_clean_error():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    m = aruco_nano.detect(img, dict_id=dict_id)[0]
    K = np.array([[500, 0, 150], [0, 500, 150], [0, 0, 1]], dtype=np.float64)
    for ms in [1e-6, 5e-5, 1e-4]:
        try:
            m.estimate_pose(K, np.zeros(5), ms)
            raise AssertionError("tiny marker_size should raise")
        except ValueError:
            pass


def test_box_filter_size_validated():
    p = aruco_nano.DetectorParameters()
    for bad in [0, -3, 2, 4]:
        try:
            p.box_filter_size = bad
            raise AssertionError("bad box_filter_size should be rejected")
        except ValueError:
            pass
    p.box_filter_size = 15


def test_draw_length_mismatch():
    dict_id = cv2.aruco.DICT_6X6_250
    img = _make_marker(dict_id, 3)
    corners, ids = aruco_nano.detect_markers(img, dict_id=dict_id)
    try:
        aruco_nano.draw_detected_markers(img, corners[:0], ids)
        raise AssertionError("mismatched corners/ids should raise")
    except ValueError:
        pass


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
    test_bool_dict_id_rejected()
    test_numpy_bool_dict_id_rejected()
    test_empty_image_rejected()
    test_zero_dim_image_rejected()
    test_numpy_int_color_accepted()
    test_malformed_camera_matrix_rejected()
    test_marker_border_bits_must_be_one()
    test_dist_coeffs_shape_validated()
    test_marker_size_positive()
    test_marker_size_finite_and_positive()
    test_degenerate_camera_matrix_rejected()
    test_tiny_marker_size_clean_error()
    test_box_filter_size_validated()
    test_draw_length_mismatch()
    test_float_image_rejected()
    test_return_rejected()
    print("ALL TESTS PASSED")

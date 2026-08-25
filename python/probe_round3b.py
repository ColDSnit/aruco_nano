"""Round-3 follow-up probes: reshape semantics, huge ints, empty ndarray, cv2 parity."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import aruco_nano
from aruco_nano import ArucoDetector, DetectorParameters, detect_markers

def _make_marker(dict_id, marker_id, size=200, border=50):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    img = cv2.aruco.generateImageMarker(d, marker_id, size)
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

img6 = _make_marker(cv2.aruco.DICT_6X6_250, 5)
m = aruco_nano.detect(img6, dict_id=cv2.aruco.DICT_6X6_250)[0]
D = np.zeros(5, dtype=np.float64)

# 1) row case (1x9) — should work
K_row = np.array([500,0,150,0,500,150,0,0,1], dtype=np.float64)
try:
    r, t = m.estimate_pose(K_row, D, 0.05)
    print("ROW 1x9:", "OK", r.shape, t.shape)
except Exception as e:
    print("ROW 1x9: FAIL", type(e).__name__, str(e)[:80])

# 2) column case (9x1) — the claimed fix
K_col = K_row.reshape(9, 1)
try:
    r, t = m.estimate_pose(K_col, D, 0.05)
    print("COL 9x1:", "OK", r.shape, t.shape)
except Exception as e:
    print("COL 9x1: FAIL", type(e).__name__, str(e)[:100])

# 3) 3x1 input (a genuinely wrong shape) — should fail
K_31 = np.array([[500],[0],[150]], dtype=np.float64)
try:
    r, t = m.estimate_pose(K_31, D, 0.05)
    print("3x1 input:", "OK (unexpected!)", r.shape)
except Exception as e:
    print("3x1 input: FAIL (expected)", type(e).__name__, str(e)[:60])

# 4) huge int scalar
try:
    ArucoDetector(2**40)
    print("huge int scalar: ACCEPTED (bad)")
except Exception as e:
    print("huge int scalar:", type(e).__name__, str(e)[:80])

# 5) huge int in sequence
try:
    ArucoDetector([cv2.aruco.DICT_6X6_250, 2**40])
    print("huge int in seq: ACCEPTED (bad)")
except Exception as e:
    print("huge int in seq:", type(e).__name__, str(e)[:80])

# 6) empty ndarray
try:
    ArucoDetector(np.array([], dtype=np.int64))
    print("empty ndarray: ACCEPTED (bad)")
except Exception as e:
    print("empty ndarray:", type(e).__name__, str(e)[:80])

# 7) negative scalar
try:
    ArucoDetector(-1)
    print("negative scalar: ACCEPTED (bad)")
except Exception as e:
    print("negative scalar:", type(e).__name__, str(e)[:80])

# 8) cv2 parity: empty image
try:
    cv2.aruco.detectMarkers(np.zeros((0,0), dtype=np.uint8), cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
    print("cv2 empty image: accepted")
except Exception as e:
    print("cv2 empty image:", type(e).__name__, str(e)[:60])

# 9) cv2 parity: float32 K
try:
    K32 = np.array([[500,0,150],[0,500,150],[0,0,1]], dtype=np.float32)
    r, t = m.estimate_pose(K32, D, 0.05)
    print("float32 K: accepted, rvec", r.shape)
except Exception as e:
    print("float32 K:", type(e).__name__, str(e)[:60])

# 10) bool in sequence
try:
    det = ArucoDetector([True])
    print("bool in seq: accepted as dict", "1")
except Exception as e:
    print("bool in seq:", type(e).__name__, str(e)[:60])

# 11) detect_markers helper with np.int64 dict_id
try:
    c, ids = detect_markers(img6, dict_id=np.int64(cv2.aruco.DICT_6X6_250))
    print("helper np.int64 dict_id:", "OK" if len(ids) == 1 else "no detect", ids)
except Exception as e:
    print("helper np.int64 dict_id:", type(e).__name__, str(e)[:60])

# 12) Marker.draw with numpy color
try:
    m.draw(img6, np.array([0, 255, 0]))
    print("numpy color: OK")
except Exception as e:
    print("numpy color:", type(e).__name__, str(e)[:60])

# 13) color with floats
try:
    m.draw(img6, (0.5, 255.0, 0.0))
    print("float color: OK")
except Exception as e:
    print("float color:", type(e).__name__, str(e)[:60])

# 14) color with out-of-range int (e.g. 300) — cv::Scalar clamps? 
try:
    m.draw(img6, (300, 0, 0))
    print("color 300: OK (clamped by cv)")
except Exception as e:
    print("color 300:", type(e).__name__, str(e)[:60])

# 15) dict_id as float 5.0 — py::cast<int>(5.0)? pybind11 cast<int> on float: does it convert?
try:
    det = ArucoDetector(5.0)
    print("float 5.0 dict_id: ACCEPTED as dict 5")
except Exception as e:
    print("float 5.0 dict_id:", type(e).__name__, str(e)[:80])

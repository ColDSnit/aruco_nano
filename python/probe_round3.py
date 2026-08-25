"""Round-3 adversarial probes for aruco_nano round-2 fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import aruco_nano
from aruco_nano import ArucoDetector, DetectorParameters, detect_markers

PASS = []
FAIL = []

def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"PASS  {name} {detail}")
    else:
        FAIL.append(name)
        print(f"FAIL  {name} {detail}")

def _make_marker(dict_id, marker_id, size=200, border=50):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    img = cv2.aruco.generateImageMarker(d, marker_id, size)
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

# --- Fix 1: detect_markers() honours params.dicts when dict_id is None ---
img6 = _make_marker(cv2.aruco.DICT_6X6_250, 5)
p = DetectorParameters()
p.set_dicts(cv2.aruco.DICT_6X6_250)
corners, ids = detect_markers(img6, params=p)   # dict_id=None, params.dicts=6x6_250
check("P1-fix: params.dicts honoured when dict_id=None", len(ids) == 1 and int(ids[0]) == 5,
      f"ids={ids.tolist() if hasattr(ids,'tolist') else ids}")

# params.dicts with MULTIPLE dicts, dict_id=None
p2 = DetectorParameters()
p2.set_dicts([cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_6X6_250])
corners, ids = detect_markers(img6, params=p2)
check("P1-fix: multi-dict params.dicts honoured", len(ids) == 1 and int(ids[0]) == 5)

# ArucoDetector(dicts=None, params=None) -> C++ default dict (MIP_36h12)
det = ArucoDetector()
corners, ids = det.detect_markers(img6)  # 6x6_250 marker, default MIP_36h12 -> no match
check("P1-fix: ctor dicts=None+params=None uses C++ default", len(ids) == 0)

# ArucoDetector(dicts=None, params=p) -> params.dicts
det = ArucoDetector(None, params=p)
corners, ids = det.detect_markers(img6)
check("P1-fix: ctor dicts=None+params keeps params.dicts", len(ids) == 1 and int(ids[0]) == 5)

# ArucoDetector(dicts=id, params=p) -> dicts overrides params.dicts
det = ArucoDetector(cv2.aruco.DICT_4X4_50, params=p)
corners, ids = det.detect_markers(img6)
check("P1-fix: explicit dicts overrides params.dicts", len(ids) == 0)

# --- Fix 4: bytes rejected as dicts ---
try:
    ArucoDetector(b"abc")
    check("P3-fix: bytes rejected", False, "no exception")
except Exception as e:
    check("P3-fix: bytes rejected", True, f"({type(e).__name__})")

# --- Fix 5: len-2 color throws ---
img = _make_marker(cv2.aruco.DICT_6X6_250, 1)
m = aruco_nano.detect(img, dict_id=cv2.aruco.DICT_6X6_250)[0]
try:
    m.draw(img, (0, 255))
    check("P3-fix: len-2 color throws", False, "no exception")
except Exception as e:
    check("P3-fix: len-2 color throws", True, f"({type(e).__name__})")
# len-3 and len-4 still work
try:
    m.draw(img, (0, 255, 0))
    m.draw(img, (0, 255, 0, 255))
    check("P3-fix: len-3/len-4 color still works", True)
except Exception as e:
    check("P3-fix: len-3/len-4 color still works", False, f"raised {e}")

# --- Fix 6: (9,1) column camera matrix reshaped ---
K_col = np.array([[500],[0],[150],[0],[500],[150],[0],[0],[1]], dtype=np.float64)
D = np.zeros(5, dtype=np.float64)
try:
    rvec, tvec = m.estimate_pose(K_col, D, 0.05)
    check("P3-fix: (9,1) column K reshaped", rvec.shape == (3,1) and tvec.shape == (3,1),
          f"rvec={rvec.shape}")
except Exception as e:
    check("P3-fix: (9,1) column K reshaped", False, f"raised {type(e).__name__}: {e}")

# --- Fix 7: empty dicts list throws ---
try:
    ArucoDetector([])
    check("P3-fix: empty dicts list throws", False, "no exception")
except Exception as e:
    check("P3-fix: empty dicts list throws", True, f"({type(e).__name__})")
try:
    p3 = DetectorParameters(); p3.set_dicts([])
    check("P3-fix: set_dicts([]) throws", False, "no exception")
except Exception as e:
    check("P3-fix: set_dicts([]) throws", True, f"({type(e).__name__})")

# --- Fix 8: np.int64 dict_id accepted ---
try:
    det = ArucoDetector(np.int64(cv2.aruco.DICT_6X6_250))
    corners, ids = det.detect_markers(img6)
    check("P3-fix: np.int64 dict_id accepted", len(ids) == 1 and int(ids[0]) == 5, f"ids={ids}")
except Exception as e:
    check("P3-fix: np.int64 dict_id accepted", False, f"raised {type(e).__name__}: {e}")

# --- NEW-BUG probes: make_dicts try/catch swallowing real errors ---
# 1) float scalar: py::cast<int> on 3.5 -> cast_error? then not a sequence -> generic error
try:
    ArucoDetector(3.5)
    check("new: float scalar rejected", False, "no exception")
except Exception as e:
    check("new: float scalar rejected", True, f"({type(e).__name__}: {str(e)[:60]})")

# 2) bool: py::cast<int>(True) == 1 -> silently accepted as dict 1!?
try:
    det = ArucoDetector(True)
    check("new: bool scalar -> dict 1 (DICT_4X4_50)", True, "accepted as int 1")
except Exception as e:
    check("new: bool scalar -> dict 1", False, f"raised {type(e).__name__}: {e}")

# 3) out-of-range int in a sequence: does the error propagate or get swallowed?
try:
    ArucoDetector([cv2.aruco.DICT_6X6_250, 9999])
    check("new: out-of-range id in sequence propagates", False, "no exception")
except Exception as e:
    check("new: out-of-range id in sequence propagates", True, f"({type(e).__name__}: {str(e)[:60]})")

# 4) non-int items in sequence: e.g. [1, "x"]
try:
    ArucoDetector([1, "x"])
    check("new: non-int item in sequence propagates", False, "no exception")
except Exception as e:
    check("new: non-int item in sequence propagates", True, f"({type(e).__name__}: {str(e)[:60]})")

# 5) numpy float scalar
try:
    ArucoDetector(np.float64(3.0))
    check("new: np.float64 scalar rejected", False, "no exception")
except Exception as e:
    check("new: np.float64 scalar rejected", True, f"({type(e).__name__}: {str(e)[:60]})")

# 6) dict (mapping) as dicts
try:
    ArucoDetector({1: 2})
    check("new: dict mapping rejected", False, "no exception")
except Exception as e:
    check("new: dict mapping rejected", True, f"({type(e).__name__}: {str(e)[:60]})")

# 7) np.array([1,2]) as dicts (ndarray is not a py::sequence? it is via buffer...)
try:
    det = ArucoDetector(np.array([cv2.aruco.DICT_6X6_250], dtype=np.int64))
    corners, ids = det.detect_markers(img6)
    check("new: np.ndarray of ints accepted", len(ids) == 1, f"ids={ids}")
except Exception as e:
    check("new: np.ndarray of ints accepted", False, f"raised {type(e).__name__}: {str(e)[:60]}")

# 8) np.int32 scalar
try:
    det = ArucoDetector(np.int32(cv2.aruco.DICT_6X6_250))
    corners, ids = det.detect_markers(img6)
    check("new: np.int32 scalar accepted", len(ids) == 1, f"ids={ids}")
except Exception as e:
    check("new: np.int32 scalar accepted", False, f"raised {type(e).__name__}: {str(e)[:60]}")

# 9) tuple of ints
try:
    det = ArucoDetector((cv2.aruco.DICT_6X6_250,))
    corners, ids = det.detect_markers(img6)
    check("new: tuple of ints accepted", len(ids) == 1, f"ids={ids}")
except Exception as e:
    check("new: tuple of ints accepted", False, f"raised {type(e).__name__}: {str(e)[:60]}")

# 10) generator as dicts (iterable but not sequence)
try:
    ArucoDetector(x for x in [1])
    check("new: generator rejected cleanly", False, "no exception")
except Exception as e:
    check("new: generator rejected cleanly", True, f"({type(e).__name__}: {str(e)[:60]})")

# 11) float32 camera matrix still silently accepted? (round-2 P3 #3, unchanged)
K32 = np.array([[500,0,150],[0,500,150],[0,0,1]], dtype=np.float32)
try:
    rvec, tvec = m.estimate_pose(K32, D, 0.05)
    check("P3: float32 K silently accepted (unchanged)", True, "accepted")
except Exception as e:
    check("P3: float32 K silently accepted (unchanged)", False, f"raised {type(e).__name__}")

# 12) detect() with dict_id AND dict_ids both set
try:
    aruco_nano.detect(img6, dict_id=1, dict_ids=[2])
    check("detect: both dict_id+dict_ids rejected", False, "no exception")
except Exception as e:
    check("detect: both dict_id+dict_ids rejected", True, f"({type(e).__name__})")

# 13) detect() dict_ids path
try:
    ms = aruco_nano.detect(img6, dict_ids=[cv2.aruco.DICT_6X6_250])
    check("detect: dict_ids works", len(ms) == 1, f"n={len(ms)}")
except Exception as e:
    check("detect: dict_ids works", False, f"raised {type(e).__name__}: {str(e)[:60]}")

# 14) Marker.dict semantics: detect() with multi-dict params
p4 = DetectorParameters()
p4.set_dicts([cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_6X6_250])
ms = aruco_nano.detect(img6, params=p4)
check("detect: multi-dict params, marker.dict index", len(ms) == 1 and ms[0].dict == 1,
      f"dict={ms[0].dict if ms else None}")

# 15) detect_markers_multi_dict dict_indices dtype/shape
det = ArucoDetector([cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_6X6_250])
c, ids, di = det.detect_markers_multi_dict(img6)
check("multi_dict: dict_indices correct", len(ids) == 1 and int(di[0]) == 1, f"di={di}")

# 16) empty image / zero-size
try:
    c, ids = ArucoDetector(cv2.aruco.DICT_6X6_250).detect_markers(np.zeros((0,0), dtype=np.uint8))
    check("empty image handled", True, f"ids={ids}")
except Exception as e:
    check("empty image handled", False, f"raised {type(e).__name__}: {str(e)[:60]}")

# 17) HxWx1 image accepted
img3 = img6[:, :, None] if img6.ndim == 2 else img6
if img6.ndim == 2:
    img1 = img6[:, :, None]
    c, ids = ArucoDetector(cv2.aruco.DICT_6X6_250).detect_markers(img1)
    check("HxWx1 image accepted", len(ids) == 1, f"ids={ids}")

# 18) non-contiguous image
img_nc = np.asfortranarray(img6)
try:
    c, ids = ArucoDetector(cv2.aruco.DICT_6X6_250).detect_markers(img_nc)
    check("non-contiguous image rejected (c_style)", False, "accepted")
except Exception as e:
    check("non-contiguous image rejected (c_style)", True, f"({type(e).__name__})")

print()
print(f"== {len(PASS)} passed, {len(FAIL)} failed ==")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)

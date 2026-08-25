"""Probe: non-contiguous / F-contiguous image handling — silent misread check."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import aruco_nano
from aruco_nano import ArucoDetector

def _make_marker(dict_id, marker_id, size=200, border=50):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    img = cv2.aruco.generateImageMarker(d, marker_id, size)
    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

img6 = _make_marker(cv2.aruco.DICT_6X6_250, 5)
det = ArucoDetector(cv2.aruco.DICT_6X6_250)

# baseline C-contiguous
c, ids = det.detect_markers(img6)
print("C-contig baseline:", ids.tolist())

# F-contiguous
img_f = np.asfortranarray(img6)
print("F-contig flags: C=", img_f.flags['C_CONTIGUOUS'], "F=", img_f.flags['F_CONTIGUOUS'])
try:
    c, ids = det.detect_markers(img_f)
    print("F-contig detect:", ids.tolist())
except Exception as e:
    print("F-contig detect: raised", type(e).__name__, str(e)[:80])

# transposed view (non-contiguous both ways)
img_t = img6.T
print("T view flags: C=", img_t.flags['C_CONTIGUOUS'], "F=", img_t.flags['F_CONTIGUOUS'])
try:
    c, ids = det.detect_markers(img_t)
    print("T view detect:", ids.tolist())
except Exception as e:
    print("T view detect: raised", type(e).__name__, str(e)[:80])

# strided slice (non-contiguous)
img_s = img6[::2, ::2]
print("slice flags: C=", img_s.flags['C_CONTIGUOUS'], "F=", img_s.flags['F_CONTIGUOUS'])
try:
    c, ids = det.detect_markers(img_s)
    print("slice detect:", ids.tolist())
except Exception as e:
    print("slice detect: raised", type(e).__name__, str(e)[:80])

# 3D HxWx1 F-contiguous
img1 = img6[:, :, None]
img1f = np.asfortranarray(img1)
print("HxWx1 F flags: C=", img1f.flags['C_CONTIGUOUS'], "F=", img1f.flags['F_CONTIGUOUS'])
try:
    c, ids = det.detect_markers(img1f)
    print("HxWx1 F detect:", ids.tolist())
except Exception as e:
    print("HxWx1 F detect: raised", type(e).__name__, str(e)[:80])

# 3D HxWx3 F-contiguous (BGR)
img3 = cv2.cvtColor(img6, cv2.COLOR_GRAY2BGR)
img3f = np.asfortranarray(img3)
print("HxWx3 F flags: C=", img3f.flags['C_CONTIGUOUS'], "F=", img3f.flags['F_CONTIGUOUS'])
try:
    c, ids = det.detect_markers(img3f)
    print("HxWx3 F detect:", ids.tolist())
except Exception as e:
    print("HxWx3 F detect: raised", type(e).__name__, str(e)[:80])

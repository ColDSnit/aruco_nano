# aruco-nano (Python wrapper)

Python bindings for [ArUco Nano](https://github.com/rmsalinas/aruco_nano), a
minimalist, header-only C++ ArUco marker detector that is up to 6.5x faster
than the standard OpenCV implementation (author-reported; see the paper's
Table 2 for the full benchmark).

This package wraps the vendored `aruco_nano.h` (v10) **unmodified** via
pybind11, exposing both the native `MarkerDetector` API and the
OpenCV-compatible `ArucoDetector` wrapper.

## Installation

The extension links against the official OpenCV 4.11.0 Windows prebuilt
(`opencv_world4110.dll`, vc16/MSVC 2019). Two paths are supported:

### A. `pip install` (builds the extension via scikit-build-core)

```bash
# Requires: the OpenCV 4.11.0 prebuilt extracted somewhere, and a Python with
# dev headers (the calibration conda env has them). Use cmake.define.* (NOT
# cmake.args) so the generator pin in pyproject.toml is preserved.
pip install . \
  --config-settings=cmake.define.OpenCV_DIR="<path>/opencv/build" \
  --config-settings=cmake.define.Python_EXECUTABLE="<calibration-env>/python.exe"
```

The wheel bundles `_aruco_nano.cp311-win_amd64.pyd` and `opencv_world4110.dll`
inside the `aruco_nano` package, so no separate DLL copy is needed.

### B. Manual CMake build

```bash
# 1. Extract the official OpenCV 4.11.0 Windows prebuilt (headers + libs)
#    https://github.com/opencv/opencv/releases/download/4.11.0/opencv-4.11.0-windows.exe
# 2. Configure + build (VS16 2019 matches the prebuilt's vc16 runtime)
cmake -S . -B build -G "Visual Studio 16 2019" -A x64 \
  -DOpenCV_DIR="<path>/opencv/build" \
  -Dpybind11_DIR="<calibration-env>/Lib/site-packages/pybind11/share/cmake/pybind11" \
  -DPython_EXECUTABLE="<calibration-env>/python.exe" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# 3. Copy the built .pyd and opencv_world4110.dll into aruco_nano/
```

## Usage

```python
import cv2
import aruco_nano

# OpenCV-compatible API
detector = aruco_nano.ArucoDetector(cv2.aruco.DICT_6X6_250)
corners, ids = detector.detect_markers(image)   # corners: list of (4,2) float32

# Native MarkerDetector API (returns Marker objects)
markers = aruco_nano.detect(image, dict_id=cv2.aruco.DICT_6X6_250)
for m in markers:
    print(m.id, m.corners)
    rvec, tvec = m.estimate_pose(camera_matrix, dist_coeffs, marker_size=0.05)

# Convenience helpers
corners, ids = aruco_nano.detect_markers(image, dict_id=cv2.aruco.DICT_6X6_250)
out = aruco_nano.draw_detected_markers(image, corners, ids)
```

## API

- `aruco_nano.detect(image, params=None, dict_id=None, dict_ids=None, return_rejected=False)`
- `aruco_nano.detect_markers(image, dict_id=None, params=None)` — defaults to `DICT_ARUCO_MIP_36h12`
- `aruco_nano.draw_detected_markers(image, corners, ids, border_color=(0,255,0))`
- `aruco_nano.ArucoDetector(dicts, params=None)` — `.detect_markers(image)`, `.detect_markers_multi_dict(image)`
- `aruco_nano.DetectorParameters` — mirrors the C++ `DetectorParameters` struct
- `aruco_nano.Marker` — `.id`, `.dict` (dict index), `.corners`, `.estimate_pose(...)`, `.draw(...)`

## License

MIT. The vendored `aruco_nano.h` is Copyright (c) 2026 University of Cordoba,
MIT licensed. See `LICENSE`.

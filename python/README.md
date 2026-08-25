# aruco-nano (Python wrapper)

Python bindings for [ArUco Nano](https://github.com/rmsalinas/aruco_nano), a
minimalist, header-only C++ ArUco marker detector that is up to 6.5x faster
than the standard OpenCV implementation.

This package wraps the vendored `aruco_nano.h` (v10) **unmodified** via
pybind11, exposing both the native `MarkerDetector` API and the
OpenCV-compatible `ArucoDetector` wrapper.

## Installation

The extension is built against the calibration conda environment
(`calibration_07032026`, Python 3.11) and OpenCV 4.11.0. To build:

```bash
# 1. Extract the official OpenCV 4.11.0 Windows prebuilt (headers + libs)
#    https://github.com/opencv/opencv/releases/download/4.11.0/opencv-4.11.0-windows.exe
# 2. Configure + build
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
```

## API

- `aruco_nano.detect(image, params=None, dict_id=None, dict_ids=None, return_rejected=False)`
- `aruco_nano.ArucoDetector(dicts)` — `.detect_markers(image)`, `.detect_markers_multi_dict(image)`
- `aruco_nano.DetectorParameters` — mirrors the C++ `DetectorParameters` struct
- `aruco_nano.Marker` — `.id`, `.dict`, `.corners`, `.estimate_pose(...)`, `.draw(...)`

## License

MIT. The vendored `aruco_nano.h` is Copyright (c) 2026 University of Cordoba,
MIT licensed. See `LICENSE`.

/*
 * Purpose:  Python bindings for ArUco Nano (header-only ArUco marker detector).
 *           Exposes aruco_nano::MarkerDetector, aruco_nano::DetectorParameters,
 *           aruco_nano::Marker and aruco_nano::ArucoDetector to Python via pybind11.
 * Status:   Active. Wraps the vendored aruco_nano.h (v10) unmodified.
 * Future:   Add GIL release for long detection runs; expose rejected-candidate
 *           geometry; add a cv2.aruco.Dictionary passthrough once a type caster
 *           for cv::aruco::Dictionary is available.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include "aruco_nano.h"

namespace py = pybind11;

// Convert a uint8 NumPy array (HxW grayscale or HxWx3 BGR) into a cv::Mat.
static cv::Mat numpy_to_mat_u8(const py::array_t<uint8_t, py::array::c_style> &a) {
    // The binding declares uint8 without forcecast, so pybind11 rejects
    // non-uint8 at the boundary (TypeError) before this body runs.
    py::buffer_info info = a.request();
    if (info.ndim == 2) {
        cv::Mat m((int)info.shape[0], (int)info.shape[1], CV_8UC1, info.ptr);
        return m.clone();
    }
    if (info.ndim == 3 && info.shape[2] == 3) {
        cv::Mat m((int)info.shape[0], (int)info.shape[1], CV_8UC3, info.ptr);
        return m.clone();
    }
    if (info.ndim == 3 && info.shape[2] == 1) {
        cv::Mat m((int)info.shape[0], (int)info.shape[1], CV_8UC1, info.ptr);
        return m.clone();
    }
    throw std::runtime_error("image must be HxW (grayscale) or HxWx3 (BGR) uint8");
}

// Convert a float64 NumPy array (1D or 2D) into a single-channel cv::Mat.
static cv::Mat numpy_to_mat_f64(const py::array_t<double, py::array::c_style> &a) {
    py::buffer_info info = a.request();
    if (info.ndim == 1) {
        cv::Mat m(1, (int)info.shape[0], CV_64FC1, info.ptr);
        return m.clone();
    }
    if (info.ndim == 2) {
        cv::Mat m((int)info.shape[0], (int)info.shape[1], CV_64FC1, info.ptr);
        return m.clone();
    }
    throw std::runtime_error("matrix must be 1D or 2D float64");
}

// Convert a cv::Mat (any supported depth/channels) into an owning NumPy array.
static py::array mat_to_numpy(const cv::Mat &m) {
    const int depth = m.depth();
    const int ch = m.channels();
    std::vector<py::ssize_t> shape;
    if (ch == 1) shape = {m.rows, m.cols};
    else shape = {m.rows, m.cols, ch};

    py::dtype dt;
    size_t elem_size = 0;
    if (depth == CV_8U) { dt = py::dtype::of<uint8_t>(); elem_size = 1; }
    else if (depth == CV_32F) { dt = py::dtype::of<float>(); elem_size = 4; }
    else if (depth == CV_64F) { dt = py::dtype::of<double>(); elem_size = 8; }
    else throw std::runtime_error("unsupported cv::Mat depth for NumPy conversion");

    py::array out(dt, shape);
    py::buffer_info info = out.request();
    const size_t row_bytes = (size_t)m.cols * (size_t)ch * elem_size;
    uint8_t *dst = static_cast<uint8_t *>(info.ptr);
    for (int r = 0; r < m.rows; ++r)
        std::memcpy(dst + r * row_bytes, m.ptr(r), row_bytes);
    return out;
}

// Convert a vector of cv::Point2f into an (N,2) float32 NumPy array.
static py::array corners_to_numpy(const std::vector<cv::Point2f> &pts) {
    std::vector<py::ssize_t> shape = {(py::ssize_t)pts.size(), 2};
    py::array_t<float> out(shape);
    py::buffer_info info = out.request();
    float *d = static_cast<float *>(info.ptr);
    for (size_t i = 0; i < pts.size(); ++i) {
        d[2 * i] = pts[i].x;
        d[2 * i + 1] = pts[i].y;
    }
    return out;
}

// Convert a vector of int into a 1D int32 NumPy array.
static py::array ids_to_numpy(const std::vector<int> &ids) {
    std::vector<py::ssize_t> shape = {(py::ssize_t)ids.size()};
    py::array_t<int> out(shape);
    py::buffer_info info = out.request();
    if (!ids.empty())
        std::memcpy(info.ptr, ids.data(), ids.size() * sizeof(int));
    return out;
}

// Convert a vector<vector<Point2f>> into a Python list of (4,2) float32 arrays.
static py::list corners_list_to_python(const std::vector<std::vector<cv::Point2f>> &corners) {
    py::list l;
    for (const auto &c : corners) l.append(corners_to_numpy(c));
    return l;
}

// Build a single cv::aruco::Dictionary from a predefined-dictionary integer id.
// OpenCV's getPredefinedDictionary has no default case, so an out-of-range id
// silently falls through to DICT_4X4_50; reject it explicitly instead.
static cv::aruco::Dictionary make_dict(int id) {
    if (id < 0 || id > 21)
        throw std::invalid_argument("invalid predefined dictionary id " + std::to_string(id) +
                                    " (valid range 0..21, matching cv2.aruco.DICT_*)");
    return cv::aruco::getPredefinedDictionary(static_cast<cv::aruco::PredefinedDictionaryType>(id));
}

// Build a vector of dictionaries from an int or a sequence of ints.
static std::vector<cv::aruco::Dictionary> make_dicts(py::object o) {
    std::vector<cv::aruco::Dictionary> out;
    if (py::isinstance<py::str>(o) || py::isinstance<py::bytes>(o)) {
        throw std::runtime_error("dicts must be an int or a sequence of ints, not a string/bytes");
    }
    // Try a scalar int first: py::cast<int> handles Python ints AND numpy
    // integer scalars (via __index__), which py::isinstance<py::int_> misses.
    try {
        out.push_back(make_dict(py::cast<int>(o)));
        return out;
    } catch (const py::cast_error &) {
        // not a scalar int; fall through to sequence handling
    }
    if (py::isinstance<py::sequence>(o)) {
        for (auto item : o) out.push_back(make_dict(py::cast<int>(item)));
        if (out.empty())
            throw std::runtime_error("dicts must not be empty");
        return out;
    }
    throw std::runtime_error("dicts must be an int (predefined dictionary id) or a sequence of ints");
}

PYBIND11_MODULE(_aruco_nano, m) {
    m.doc() = "Python bindings for ArUco Nano, a header-only ArUco marker detector "
              "(https://github.com/rmsalinas/aruco_nano).";
    m.attr("__version__") = "0.1.0";
    m.attr("ArucoNanoVersion") = ArucoNanoVersion;

    // --- Marker ------------------------------------------------------------
    py::class_<aruco_nano::Marker>(m, "Marker")
        .def_property_readonly("corners", [](const aruco_nano::Marker &mk) {
            return corners_to_numpy(mk);
        })
        .def_readonly("id", &aruco_nano::Marker::id)
        .def_readonly("dict", &aruco_nano::Marker::dict)
        .def("estimate_pose",
             [](const aruco_nano::Marker &mk,
                const py::array_t<double, py::array::c_style> &camera_matrix,
                const py::array_t<double, py::array::c_style> &dist_coeffs,
                double marker_size) {
                 cv::Mat K = numpy_to_mat_f64(camera_matrix);
                 // Accept a flat 9-element camera matrix (row or column) and
                 // reshape to 3x3 single-channel. reshape(1, 3) = 1 channel,
                 // 3 rows -> 3x3.
                 if (K.rows == 1 && K.cols == 9) K = K.reshape(1, 3);
                 else if (K.rows == 9 && K.cols == 1) K = K.reshape(1, 3);
                 cv::Mat D = numpy_to_mat_f64(dist_coeffs);
                 auto pose = mk.estimatePose(K, D, marker_size);
                 return py::make_tuple(mat_to_numpy(pose.first), mat_to_numpy(pose.second));
             },
             py::arg("camera_matrix"), py::arg("dist_coeffs"), py::arg("marker_size") = 1.0,
             "Estimate marker pose; returns (rvec, tvec) as float64 NumPy arrays.")
        .def("draw",
             [](const aruco_nano::Marker &mk,
                const py::array_t<uint8_t, py::array::c_style> &image,
                py::object color) {
                 cv::Mat img = numpy_to_mat_u8(image);  // already a clone
                 cv::Scalar c(0, 0, 255);
                 if (!color.is_none()) {
                     py::sequence t = py::cast<py::sequence>(color);
                     if (py::len(t) == 3)
                         c = cv::Scalar(py::cast<int>(t[0]), py::cast<int>(t[1]), py::cast<int>(t[2]));
                     else if (py::len(t) == 4)
                         c = cv::Scalar(py::cast<int>(t[0]), py::cast<int>(t[1]), py::cast<int>(t[2]), py::cast<int>(t[3]));
                     else
                         throw std::invalid_argument("color must be a 3- or 4-element sequence");
                 }
                 mk.draw(img, c);
                 return mat_to_numpy(img);
             },
             py::arg("image"), py::arg("color") = py::none(),
             "Draw the marker onto a copy of the image and return it.");

    // --- DetectorParameters -------------------------------------------------
    py::class_<aruco_nano::DetectorParameters>(m, "DetectorParameters")
        .def(py::init<>())
        .def_readwrite("box_filter_size", &aruco_nano::DetectorParameters::boxFilterSize)
        .def_readwrite("thres", &aruco_nano::DetectorParameters::thres)
        .def_readwrite("min_size", &aruco_nano::DetectorParameters::minSize)
        .def_readwrite("max_attempts_per_candidate", &aruco_nano::DetectorParameters::maxAttemptsPerCandidate)
        .def_readwrite("max_times_revisited", &aruco_nano::DetectorParameters::maxTimesRevisited)
        .def_readwrite("marker_border_bits", &aruco_nano::DetectorParameters::markerBorderBits)
        .def_readwrite("error_correction_rate", &aruco_nano::DetectorParameters::errorCorrectionRate)
        .def_readwrite("max_erroneous_bits_in_border_rate", &aruco_nano::DetectorParameters::maxErroneousBitsInBorderRate)
        .def_readwrite("detect_inverted_marker", &aruco_nano::DetectorParameters::detectInvertedMarker)
        .def("set_dicts", [](aruco_nano::DetectorParameters &p, py::object o) {
            p.dicts = make_dicts(o);
        }, py::arg("dicts"),
        "Set the detection dictionaries from an int (predefined dictionary id) "
        "or a sequence of ints.");

    // --- ArucoDetector (OpenCV-compatible wrapper) --------------------------
    py::class_<aruco_nano::ArucoDetector>(m, "ArucoDetector")
        .def(py::init([](py::object dicts, py::object params) {
            aruco_nano::DetectorParameters p;
            if (!params.is_none())
                p = py::cast<aruco_nano::DetectorParameters>(params);
            if (!dicts.is_none())
                p.dicts = make_dicts(dicts);
            // dicts is None -> keep p.dicts (params' dicts, or the C++ default
            // DICT_ARUCO_MIP_36h12 when params is also None). This makes the
            // helper's dict_id=None path honour params.dicts instead of
            // silently clobbering it with the default.
            return new aruco_nano::ArucoDetector(p.dicts, p);
        }), py::arg("dicts") = py::none(), py::arg("params") = py::none())
        .def("detect_markers",
             [](aruco_nano::ArucoDetector &self,
                const py::array_t<uint8_t, py::array::c_style> &image) {
                 cv::Mat img = numpy_to_mat_u8(image);
                 std::vector<std::vector<cv::Point2f>> corners;
                 std::vector<int> ids;
                 {
                     py::gil_scoped_release release;  // detection is pure C++, no Python state
                     self.detectMarkers(img, corners, ids);
                 }
                 return py::make_tuple(corners_list_to_python(corners), ids_to_numpy(ids));
             },
             py::arg("image"),
             "Detect markers; returns (corners, ids) where corners is a list of "
             "(4,2) float32 arrays and ids is an int32 array.")
        .def("detect_markers_multi_dict",
             [](aruco_nano::ArucoDetector &self,
                const py::array_t<uint8_t, py::array::c_style> &image) {
                 cv::Mat img = numpy_to_mat_u8(image);
                 std::vector<std::vector<cv::Point2f>> corners;
                 std::vector<int> ids, dict_indices;
                 {
                     py::gil_scoped_release release;
                     self.detectMarkersMultiDict(img, corners, ids, cv::noArray(), dict_indices);
                 }
                 return py::make_tuple(corners_list_to_python(corners), ids_to_numpy(ids), ids_to_numpy(dict_indices));
             },
             py::arg("image"),
             "Detect markers across multiple dictionaries; returns "
             "(corners, ids, dict_indices).");

    // --- Module-level detect() ---------------------------------------------
    m.def("detect",
          [](const py::array_t<uint8_t, py::array::c_style> &image,
             py::object params, py::object dict_id, py::object dict_ids, bool return_rejected) -> py::object {
              cv::Mat img = numpy_to_mat_u8(image);
              aruco_nano::DetectorParameters p;
              if (!params.is_none())
                  p = py::cast<aruco_nano::DetectorParameters>(params);
              if (!dict_id.is_none() && !dict_ids.is_none())
                  throw std::runtime_error("pass either dict_id or dict_ids, not both");
              if (!dict_id.is_none())
                  p.dicts = make_dicts(dict_id);
              if (!dict_ids.is_none())
                  p.dicts = make_dicts(dict_ids);

              std::vector<aruco_nano::Marker> rejected;
              std::vector<aruco_nano::Marker> markers;
              {
                  py::gil_scoped_release release;
                  markers = aruco_nano::MarkerDetector::detect(img, p, return_rejected ? &rejected : nullptr);
              }

              py::list out;
              for (const auto &mk : markers) out.append(mk);
              if (return_rejected) {
                  py::list rej;
                  for (const auto &mk : rejected) rej.append(mk);
                  return py::make_tuple(out, rej);
              }
              return out;
          },
          py::arg("image"),
          py::arg("params") = py::none(),
          py::arg("dict_id") = py::none(),
          py::arg("dict_ids") = py::none(),
          py::arg("return_rejected") = false,
          "Detect ArUco markers in an image. Returns a list of Marker objects "
          "(or (markers, rejected) when return_rejected=True).");
}

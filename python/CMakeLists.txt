cmake_minimum_required(VERSION 3.15)
project(aruco_nano_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate OpenCV (headers + libs). Point OpenCV_DIR at the extracted
# opencv-4.11.0-windows build directory if it is not auto-discovered.
find_package(OpenCV REQUIRED)

# Force pybind11 to use the modern FindPython (respects Python_EXECUTABLE /
# Python_ROOT_DIR) instead of the legacy FindPythonLibsNew, which otherwise
# picks up the base conda interpreter and links the wrong python3xx.lib.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Locate the target Python interpreter and its development headers.
find_package(Python COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(_aruco_nano src/bindings.cpp)

# The vendored aruco_nano.h is copied into this directory so the package is
# self-contained (the sdist must build without the repo root).
target_include_directories(_aruco_nano PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(_aruco_nano PRIVATE ${OpenCV_LIBS})

# On Windows, ensure the OpenCV DLLs are discoverable at import time by
# copying them next to the built extension. The official prebuilt ships a
# single opencv_world<ver>.dll; copy the release DLL only (exclude the
# debug-suffixed opencv_world<ver>d.dll, which needs the debug CRT).
if(WIN32)
    file(GLOB _opencv_dlls "${OpenCV_DIR}/x64/vc16/bin/opencv_world[0-9]*.dll")
    list(FILTER _opencv_dlls EXCLUDE REGEX "opencv_world[0-9]*d\.dll$")
    if(NOT _opencv_dlls)
        message(FATAL_ERROR
            "No opencv_world*.dll found under ${OpenCV_DIR}/x64/vc16/bin; "
            "the built extension will fail to import. Check OpenCV_DIR.")
    endif()
    foreach(_dll ${_opencv_dlls})
        add_custom_command(TARGET _aruco_nano POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}"
                $<TARGET_FILE_DIR:_aruco_nano>)
    endforeach()
endif()

# Install rules: used by scikit-build-core to assemble a wheel. The extension
# and the OpenCV DLL land inside the aruco_nano package directory.
install(TARGETS _aruco_nano
        LIBRARY DESTINATION aruco_nano
        RUNTIME DESTINATION aruco_nano)
if(WIN32)
    install(FILES ${_opencv_dlls} DESTINATION aruco_nano)
endif()

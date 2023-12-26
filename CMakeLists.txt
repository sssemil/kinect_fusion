cmake_minimum_required(VERSION 3.27)
project(3dmsc_kinect_fusion_project)

set(CMAKE_CXX_STANDARD 23)

## OpenCV
# Optional: Set OpenCV_DIR if you want to use a custom version of OpenCV
# SET("OpenCV_DIR" "/opt/opencv/usr/local/share/OpenCV")
find_package(OpenCV 3.0 REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

## Eigen3
find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
endif (EIGEN3_INCLUDE_DIR)


# Targets
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories

add_executable(main main.cpp)

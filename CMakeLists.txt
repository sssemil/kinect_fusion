cmake_minimum_required(VERSION 3.22)
project(kinect_fusion)

set(CMAKE_CXX_STANDARD 14)

# Enable CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(CXXOPTS_INCLUDE_DIR ${LIBRARY_DIR}/cxxopts/include CACHE PATH "Path to cxxopts include directory")
set(glog_DIR ${LIBRARY_DIR}/glog-lib/lib/cmake/glog CACHE PATH "Path to installed glog")
# I'm using ceres-solver 2.2.0-3 from
# https://gitlab.archlinux.org/archlinux/packaging/packages/ceres-solver
# with added '-DUSE_CUDA=ON' to the PKGBUILD
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/ CACHE PATH "Path to Flann source folder")

# Set USE_CUDA for Ceres
set(USE_CUDA ON CACHE BOOL "Use CUDA in Ceres")

## OpenCV Integration
# Optional: Set OpenCV_DIR if you want to use a custom version of OpenCV
# SET("OpenCV_DIR" "/opt/opencv/usr/local/share/OpenCV")
find_package(OpenCV 3.0 REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
endif ()

## Eigen3 Integration
find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
endif ()

find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

# Define the source and include directories
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

file(GLOB_RECURSE HEADER_FILES "${PROJECT_INCLUDE_DIR}/*.h")
file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cpp")
list(REMOVE_ITEM SOURCE_FILES "${PROJECT_SOURCE_DIR}/main.cpp")
file(GLOB_RECURSE CUDA_SOURCE_FILES "${PROJECT_SOURCE_DIR}/cuda/*.cu")
file(GLOB_RECURSE CUDA_HEADER_FILES "${PROJECT_INCLUDE_DIR}/*.cuh")

set_source_files_properties(${CUDA_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CUDA_HEADER_FILES} PROPERTIES LANGUAGE CUDA)

link_directories(${FreeImage_LIBRARY_DIR})

# Executable and Linking
add_executable(main "${PROJECT_SOURCE_DIR}/main.cpp" ${SOURCE_FILES} ${HEADER_FILES} ${CUDA_SOURCE_FILES} ${CUDA_HEADER_FILES})
target_include_directories(main PUBLIC
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_INCLUDE_DIR}
        ${FreeImage_INCLUDE_DIR}
        ${CXXOPTS_INCLUDE_DIR}
        ${Flann_INCLUDE_DIR}
)
target_link_libraries(main general Eigen3::Eigen ${OpenCV_LIBS} freeimage ceres ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})

add_custom_target(
        format
        COMMAND find ${CMAKE_SOURCE_DIR} -iname *.h -o -iname *.cpp | xargs clang-format -i
)

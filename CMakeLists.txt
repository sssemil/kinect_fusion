cmake_minimum_required(VERSION 3.27)
project(kinect_fusion)

set(CMAKE_CXX_STANDARD 14)

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

# Define the source and include directories
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# Automatically find all .cpp files in the src directory
file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/*.cpp")

# Executable and Linking
add_executable(main ${SOURCES})
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
target_link_libraries(main general Eigen3::Eigen ${OpenCV_LIBS})

add_custom_target(
        format
        COMMAND find ${CMAKE_SOURCE_DIR} -iname *.h -o -iname *.cpp | xargs clang-format -i
)

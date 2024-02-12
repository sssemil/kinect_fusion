#!/bin/bash

cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
OpenCV_DIR=/opt/opencv3 make -j24
cd ..
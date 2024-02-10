#pragma once

#include <opencv2/opencv.hpp>

#include "Eigen.h"
#include "TSDFVolume.h"

PointCloud ray_marching(const TSDFVolume &tsdf, VirtualSensor &sensor,
                        const Matrix4f &current_pose_estimate);

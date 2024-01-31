#pragma once

#include "Eigen.h"
#include <opencv2/opencv.hpp>
#include "TSDFVolume.h"

PointCloud ray_marching(const TSDFVolume &tsdf, VirtualSensor &sensor,
                        const Matrix4f& current_pose_estimate);

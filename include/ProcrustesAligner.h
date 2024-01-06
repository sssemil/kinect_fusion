#pragma once

#include <iostream>
#include <vector>

#include "Eigen.h"
#include "SimpleMesh.h"

class ProcrustesAligner {
   public:
    Matrix4f estimatePose(const std::vector<Vector3f> &sourcePoints,
                          const std::vector<Vector3f> &targetPoints);

   private:
    Vector3f computeMean(const std::vector<Vector3f> &points);
    Matrix3f estimateRotation(const std::vector<Vector3f> &sourcePoints,
                              const Vector3f &sourceMean,
                              const std::vector<Vector3f> &targetPoints,
                              const Vector3f &targetMean);
    Vector3f computeTranslation(const Vector3f &sourceMean,
                                const Vector3f &targetMean,
                                const Matrix3f &rotation);
};

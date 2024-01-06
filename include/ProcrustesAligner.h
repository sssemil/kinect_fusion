#pragma once

#include "SimpleMesh.h"

class ProcrustesAligner {
   public:
    Matrix4f estimatePose(const std::vector<Vector3f> &sourcePoints,
                          const std::vector<Vector3f> &targetPoints) {
        ASSERT(
            sourcePoints.size() == targetPoints.size() &&
            "The number of source and target points should be the same, since "
            "every source point is matched with corresponding target point.");

        // We estimate the pose between source and target points using
        // Procrustes algorithm. Our shapes have the same scale, therefore we
        // don't estimate scale. We estimated rotation and translation from
        // source points to target points.

        auto sourceMean = computeMean(sourcePoints);
        auto targetMean = computeMean(targetPoints);

        Matrix3f rotation = estimateRotation(sourcePoints, sourceMean,
                                             targetPoints, targetMean);
        Vector3f translation =
            computeTranslation(sourceMean, targetMean, rotation);

        // You can access parts of the matrix with .block(start_row, start_col,
        // num_rows, num_cols) = elements

        std::cout << "Rotation: " << std::endl << rotation << std::endl;
        std::cout << "Translation: " << std::endl << translation << std::endl;

        Matrix4f estimatedPose = Matrix4f::Identity();
        estimatedPose.block<3, 3>(0, 0) = rotation;
        estimatedPose.block<3, 1>(0, 3) = translation;
        return estimatedPose;
    }

   private:
    Vector3f computeMean(const std::vector<Vector3f> &points) {
        // TODO: Compute the mean of input points.
        // Hint: You can use the .size() method to get the length of a vector.

        Vector3f mean = Vector3f::Zero();
        for (const auto &point : points) {
            mean += point;
        }
        mean /= points.size();
        return mean;
    }

    Matrix3f estimateRotation(const std::vector<Vector3f> &sourcePoints,
                              const Vector3f &sourceMean,
                              const std::vector<Vector3f> &targetPoints,
                              const Vector3f &targetMean) {
        // To compute the singular value decomposition you can use JacobiSVD()
        // from Eigen. Hint: You can initialize an Eigen matrix with "MatrixXf
        // m(num_rows,num_cols);" and access/modify parts of it using the
        // .block() method (see above).

        MatrixXf source(sourcePoints.size(), 3);
        MatrixXf target(targetPoints.size(), 3);
        for (long i = 0; i < sourcePoints.size(); ++i) {
            source.row(i) = sourcePoints[i] - sourceMean;
            target.row(i) = targetPoints[i] - targetMean;
        }
        auto t = target.transpose() * source;
        JacobiSVD<MatrixXf> svd(t, ComputeFullU | ComputeFullV);
        Matrix3f rotation = svd.matrixU() * svd.matrixV().transpose();
        if (rotation.determinant() == -1) {
            std::cout << "Determinant is -1" << std::endl;
            Matrix3f check = Matrix3f::Identity(3, 3);
            check(2, 2) = -1;
            std::cout << "Check: " << std::endl << check << std::endl;
            rotation = svd.matrixU() * check * svd.matrixV().transpose();
        }

        return rotation;
    }

    Vector3f computeTranslation(const Vector3f &sourceMean,
                                const Vector3f &targetMean,
                                const Matrix3f &rotation) {
        Vector3f translation = -rotation * sourceMean + targetMean;
        return translation;
    }
};

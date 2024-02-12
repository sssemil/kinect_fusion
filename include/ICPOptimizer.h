#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows
// defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <flann/flann.hpp>

#include "NearestNeighbor.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "SimpleMesh.h"

/**
 * Helper methods for writing Ceres cost functions.
 */
template <typename T>
static inline void fillVector(const Vector3f& input, T* output) {
    output[0] = T(input[0]);
    output[1] = T(input[1]);
    output[2] = T(input[2]);
}

/**
 * Pose increment is only an interface to the underlying array (in constructor,
 * no copy of the input array is made). Important: Input array needs to have a
 * size of at least 6.
 */
template <typename T>
class PoseIncrement {
   public:
    explicit PoseIncrement(T* const array) : m_array{array} {}

    void setZero() {
        for (int i = 0; i < 6; ++i) m_array[i] = T(0);
    }

    T* getData() const { return m_array; }

    /**
     * Applies the pose increment onto the input point and produces transformed
     * output point. Important: The memory for both 3D points (input and output)
     * needs to be reserved (i.e. on the stack) beforehand).
     */
    void apply(T* inputPoint, T* outputPoint) const {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T* rotation = m_array;
        const T* translation = m_array + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation
     * as 3D vector into transformation 4x4 matrix.
     */
    static Matrix4f convertToMatrix(
        const PoseIncrement<double>& poseIncrement) {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double* pose = poseIncrement.getData();
        double* rotation = pose;
        double* translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major
        // storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4f matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);
        matrix(0, 1) = float(rotationMatrix[3]);
        matrix(0, 2) = float(rotationMatrix[6]);
        matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);
        matrix(1, 1) = float(rotationMatrix[4]);
        matrix(1, 2) = float(rotationMatrix[7]);
        matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);
        matrix(2, 1) = float(rotationMatrix[5]);
        matrix(2, 2) = float(rotationMatrix[8]);
        matrix(2, 3) = float(translation[2]);

        return matrix;
    }

   private:
    T* m_array;
};

/**
 * Optimization constraints.
 */
class PointToPointConstraint {
   public:
    PointToPointConstraint(const Vector3f& sourcePoint,
                           const Vector3f& targetPoint, const float weight)
        : m_sourcePoint{sourcePoint},
          m_targetPoint{targetPoint},
          m_weight{weight} {}

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Implemented the point-to-point cost function.
        T pt[3] = {T(m_sourcePoint[0]), T(m_sourcePoint[1]),
                   T(m_sourcePoint[2])};

        const T* rotation = pose;
        const T* translation = pose + 3;

        T tmp[3];
        ceres::AngleAxisRotatePoint(rotation, pt, tmp);

        tmp[0] = tmp[0] + translation[0];
        tmp[1] = tmp[1] + translation[1];
        tmp[2] = tmp[2] + translation[2];

        const T w = T(sqrt(m_weight));

        residuals[0] = w * (-T(m_targetPoint(0)) + tmp[0]);
        residuals[1] = w * (-T(m_targetPoint(1)) + tmp[1]);
        residuals[2] = w * (-T(m_targetPoint(2)) + tmp[2]);

        return true;
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint,
                                       const Vector3f& targetPoint,
                                       const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
            new PointToPointConstraint(sourcePoint, targetPoint, weight));
    }

   protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const float m_weight;
    const float LAMBDA = 0.1f;
};

class PointToPlaneConstraint {
   public:
    PointToPlaneConstraint(const Vector3f& sourcePoint,
                           const Vector3f& targetPoint,
                           const Vector3f& targetNormal, const float weight)
        : m_sourcePoint{sourcePoint},
          m_targetPoint{targetPoint},
          m_targetNormal{targetNormal},
          m_weight{weight} {}

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Implemented the point-to-plane cost function.

        T pt[3] = {T(m_sourcePoint[0]), T(m_sourcePoint[1]),
                   T(m_sourcePoint[2])};
        T normal[3] = {T(m_targetNormal[0]), T(m_targetNormal[1]),
                       T(m_targetNormal[2])};

        const T* rotation = pose;
        const T* translation = pose + 3;

        T srcPt[3];
        ceres::AngleAxisRotatePoint(rotation, pt, srcPt);

        srcPt[0] = srcPt[0] + translation[0];
        srcPt[1] = srcPt[1] + translation[1];
        srcPt[2] = srcPt[2] + translation[2];

        const T weight = T(sqrt(m_weight));

        T x, y, z;
        x = weight * (srcPt[0] - T(m_targetPoint[0])) * normal[0];
        y = weight * (srcPt[1] - T(m_targetPoint[1])) * normal[1];
        z = weight * (srcPt[2] - T(m_targetPoint[2])) * normal[2];

        residuals[0] = x + y + z;

        return true;
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint,
                                       const Vector3f& targetPoint,
                                       const Vector3f& targetNormal,
                                       const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
            new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal,
                                       weight));
    }

   protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const Vector3f m_targetNormal;
    const float m_weight;
    const float LAMBDA = 1.0f;
};

/**
 * ICP optimizer - Abstract Base Class
 */
class ICPOptimizer {
   public:
    ICPOptimizer()
        : m_bUsePointToPlaneConstraints{false},
          m_nIterations{20},
          m_nearestNeighborSearch{
              std::make_unique<NearestNeighborSearchFlann>()} {}

    void setMatchingMaxDistance(float maxDistance) {
        m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
    }

    void usePointToPlaneConstraints(bool bUsePointToPlaneConstraints) {
        m_bUsePointToPlaneConstraints = bUsePointToPlaneConstraints;
    }

    void setNbOfIterations(unsigned nIterations) {
        m_nIterations = nIterations;
    }

    virtual void estimatePose(const PointCloud& source,
                              const PointCloud& target,
                              Matrix4f& initialPose) = 0;

   protected:
    bool m_bUsePointToPlaneConstraints;
    unsigned m_nIterations;
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

//TODO Use CUDA version
    std::vector<Vector3f> transformPoints(
        const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) {
        std::vector<Vector3f> transformedPoints;
        transformedPoints.reserve(sourcePoints.size());

        const auto rotation = pose.block(0, 0, 3, 3);
        const auto translation = pose.block(0, 3, 3, 1);

        for (const auto& point : sourcePoints) {
            transformedPoints.push_back(rotation * point + translation);
        }

        return transformedPoints;
    }

//TODO Use CUDA version
    std::vector<Vector3f> transformNormals(
        const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose) {
        std::vector<Vector3f> transformedNormals;
        transformedNormals.reserve(sourceNormals.size());

        const auto rotation = pose.block(0, 0, 3, 3);

        for (const auto& normal : sourceNormals) {
            transformedNormals.push_back(rotation.inverse().transpose() *
                                         normal);
        }

        return transformedNormals;
    }


//TODO Use CUDA version
    void pruneCorrespondences(const std::vector<Vector3f>& sourceNormals,
                              const std::vector<Vector3f>& targetNormals,
                              std::vector<Match>& matches) {
        const unsigned nPoints = sourceNormals.size();

        for (unsigned i = 0; i < nPoints; i++) {
            Match& match = matches[i];
            if (match.idx >= 0) {
                const auto& sourceNormal = sourceNormals[i];
                const auto& targetNormal = targetNormals[match.idx];

                // Invalidate the match (set it to -1) if the angle between the
                // normals is greater than 60 Calculate the angle between
                // normals in degrees
                float angle =
                    std::acos(sourceNormal.dot(targetNormal) /
                              (sourceNormal.norm() * targetNormal.norm())) *
                    (180.0f / M_PI);

                // Invalidate the match if the angle is greater than 60 degrees
                if (angle > 60.0f) {
                    match.idx = -1;
                }
            }
        }
    }
};

/**
 * ICP optimizer - using Ceres for optimization.
 */
class CeresICPOptimizer : public ICPOptimizer {
   public:
    CeresICPOptimizer() {}

    virtual void estimatePose(const PointCloud& source,
                              const PointCloud& target,
                              Matrix4f& initialPose) override {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        m_nearestNeighborSearch->buildIndex(target.getPoints());

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        // We optimize on the transformation in SE3 notation: 3 parameters for
        // the axis-angle vector of the rotation (its length presents the
        // rotation angle) and 3 parameters for the translation vector.
        double incrementArray[6];
        auto poseIncrement = PoseIncrement<double>(incrementArray);
        poseIncrement.setZero();

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();

            auto transformedPoints =
                transformPoints(source.getPoints(), estimatedPose);
            auto transformedNormals =
                transformNormals(source.getNormals(), estimatedPose);

            auto matches =
                m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(),
                                 matches);

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds."
                      << std::endl;

            // Prepare point-to-point and point-to-plane constraints.
            ceres::Problem problem;
            prepareConstraints(transformedPoints, target.getPoints(),
                               target.getNormals(), matches, poseIncrement,
                               problem);

            // Configure options for the solver.
            ceres::Solver::Options options;
            configureSolver(options);

            // Run the solver (for one iteration).
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            // std::cout << summary.FullReport() << std::endl;

            // Update the current pose estimate (we always update the pose from
            // the left, using left-increment notation).
            Matrix4f matrix =
                PoseIncrement<double>::convertToMatrix(poseIncrement);
            estimatedPose =
                PoseIncrement<double>::convertToMatrix(poseIncrement) *
                estimatedPose;
            poseIncrement.setZero();

            std::cout << "Optimization iteration done." << std::endl;
        }

        // Store result
        initialPose = estimatedPose;
    }

   private:
    void configureSolver(ceres::Solver::Options& options) {
        // Ceres options.
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 1;
        options.num_threads = 8;
        options.dense_linear_algebra_library_type = ceres::CUDA;
    }

    void prepareConstraints(const std::vector<Vector3f>& sourcePoints,
                            const std::vector<Vector3f>& targetPoints,
                            const std::vector<Vector3f>& targetNormals,
                            const std::vector<Match> matches,
                            const PoseIncrement<double>& poseIncrement,
                            ceres::Problem& problem) const {
        const unsigned nPoints = sourcePoints.size();

        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto& sourcePoint = sourcePoints[i];
                const auto& targetPoint = targetPoints[match.idx];

                if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                    continue;

                // Create a new point-to-point cost function and add it as
                // constraint (i.e. residual block) to the Ceres problem.
                ceres::CostFunction* pointToPointCost =
                    PointToPointConstraint::create(sourcePoint, targetPoint,
                                                   1.0f);
                problem.AddResidualBlock(pointToPointCost, nullptr,
                                         poseIncrement.getData());

                if (m_bUsePointToPlaneConstraints) {
                    const auto& targetNormal = targetNormals[match.idx];

                    if (!targetNormal.allFinite()) continue;

                    // Create a new point-to-plane cost function and add it as
                    // constraint (i.e. residual block) to the Ceres problem.
                    ceres::CostFunction* pointToPlaneCost =
                        PointToPlaneConstraint::create(sourcePoint, targetPoint,
                                                       targetNormal, 1.0f);
                    problem.AddResidualBlock(pointToPlaneCost, nullptr,
                                             poseIncrement.getData());
                }
            }
        }
    }
};

#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense> // Assuming you're using Eigen for Vector3f and Matrix4f
#include <iostream>
#include <cmath>
#include <PointCloud.h>
#include <TSDFVolume.h>
// Define Vector3f and Matrix4f types if not already defined
using Vector3f = Eigen::Vector3f;
using Matrix4f = Eigen::Matrix4f;
using Vector3i = Eigen::Vector3i;

__global__ void transformPointsKernel(const Vector3f* sourcePoints, const Matrix4f pose, Vector3f* transformedPoints, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        const auto rotation = pose.block(0, 0, 3, 3);
        const auto translation = pose.block(0, 3, 3, 1);
        transformedPoints[idx] = rotation * sourcePoints[idx] + translation;
    }
}

std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) {
    int numPoints = sourcePoints.size();
    std::vector<Vector3f> transformedPoints(numPoints);

    // Allocate memory on the GPU
    Vector3f* d_sourcePoints;
    Vector3f* d_transformedPoints;
    cudaMalloc((void**)&d_sourcePoints, numPoints * sizeof(Vector3f));
    cudaMalloc((void**)&d_transformedPoints, numPoints * sizeof(Vector3f));

    // Copy sourcePoints from host to device
    cudaMemcpy(d_sourcePoints, sourcePoints.data(), numPoints * sizeof(Vector3f), cudaMemcpyHostToDevice);

    // Calculate block and grid dimensions
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    // Launch kernel
    transformPointsKernel<<<numBlocks, blockSize>>>(d_sourcePoints, pose, d_transformedPoints, numPoints);

    // Copy result from device to host
    cudaMemcpy(transformedPoints.data(), d_transformedPoints, numPoints * sizeof(Vector3f), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_sourcePoints);
    cudaFree(d_transformedPoints);

    return transformedPoints;
}



__global__ void transformNormalsKernel(const Vector3f* sourceNormals, const Matrix4f pose, Vector3f* transformedNormals, int numNormals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNormals) {
        const auto rotation = pose.block(0, 0, 3, 3);
        transformedNormals[idx] = rotation.inverse().transpose() * sourceNormals[idx];
    }
}

std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose) {
    int numNormals = sourceNormals.size();
    std::vector<Vector3f> transformedNormals(numNormals);

    // Allocate memory on the GPU
    Vector3f* d_sourceNormals;
    Vector3f* d_transformedNormals;
    cudaMalloc((void**)&d_sourceNormals, numNormals * sizeof(Vector3f));
    cudaMalloc((void**)&d_transformedNormals, numNormals * sizeof(Vector3f));

    // Copy sourceNormals from host to device
    cudaMemcpy(d_sourceNormals, sourceNormals.data(), numNormals * sizeof(Vector3f), cudaMemcpyHostToDevice);

    // Calculate block and grid dimensions
    int blockSize = 256;
    int numBlocks = (numNormals + blockSize - 1) / blockSize;

    // Launch kernel
    transformNormalsKernel<<<numBlocks, blockSize>>>(d_sourceNormals, pose, d_transformedNormals, numNormals);

    // Copy result from device to host
    cudaMemcpy(transformedNormals.data(), d_transformedNormals, numNormals * sizeof(Vector3f), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_sourceNormals);
    cudaFree(d_transformedNormals);

    return transformedNormals;
}




struct Match {
    int idx;
    float weight;
    // Add other members if needed
};

__global__ void pruneCorrespondencesKernel(const Vector3f* sourceNormals, const Vector3f* targetNormals, Match* matches, unsigned nPoints) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPoints) {
        Match& match = matches[idx];
        if (match.idx >= 0) {
            const auto& sourceNormal = sourceNormals[idx];
            const auto& targetNormal = targetNormals[match.idx];

            // Calculate the angle between normals in degrees
            float angle = acosf(sourceNormal.dot(targetNormal) / (sourceNormal.norm() * targetNormal.norm())) * (180.0f / M_PI);

            // Invalidate the match if the angle is greater than 60 degrees
            if (angle > 60.0f) {
                match.idx = -1;
            }
        }
    }
}

void pruneCorrespondences(const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches) {
    const unsigned nPoints = sourceNormals.size();

    // Allocate memory on the GPU
    Vector3f* d_sourceNormals;
    Vector3f* d_targetNormals;
    Match* d_matches;
    cudaMalloc((void**)&d_sourceNormals, nPoints * sizeof(Vector3f));
    cudaMalloc((void**)&d_targetNormals, targetNormals.size() * sizeof(Vector3f));
    cudaMalloc((void**)&d_matches, matches.size() * sizeof(Match));

    // Copy sourceNormals, targetNormals, and matches from host to device
    cudaMemcpy(d_sourceNormals, sourceNormals.data(), nPoints * sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targetNormals, targetNormals.data(), targetNormals.size() * sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matches, matches.data(), matches.size() * sizeof(Match), cudaMemcpyHostToDevice);

    // Calculate block and grid dimensions
    int blockSize = 256;
    int numBlocks = (nPoints + blockSize - 1) / blockSize;

    // Launch kernel
    pruneCorrespondencesKernel<<<numBlocks, blockSize>>>(d_sourceNormals, d_targetNormals, d_matches, nPoints);

    // Copy result from device to host
    cudaMemcpy(matches.data(), d_matches, matches.size() * sizeof(Match), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_sourceNormals);
    cudaFree(d_targetNormals);
    cudaFree(d_matches);
}



// __global__ void integrateKernel(const Vector3f* points, const Vector3f* normals, const Eigen::Matrix4f pose, const int width, const int height, const int depth, const float truncationDistance, TSDFVolume::Voxel* voxels) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < points.size()) {
//         Eigen::Vector3f point = pose * points[idx];
//         Eigen::Vector3f normal = pose.rotation() * normals[idx];

//         // Transform point to TSDF grid coordinates
//         Vector3i voxelCoord = getVoxelCoordinatesForWorldCoordinates(point);

//         // Update voxel if within TSDF volume bounds
//         if (voxelCoord[0] >= 0 && voxelCoord[0] < width && voxelCoord[1] >= 0 &&
//             voxelCoord[1] < height && voxelCoord[2] >= 0 &&
//             voxelCoord[2] < depth) {
//             int index = toLinearIndex(voxelCoord[0], voxelCoord[1], voxelCoord[2]);
//             TSDFVolume::Voxel& voxel = voxels[index];

//             // Compute signed distance and update voxel
//             float sdf = normal.dot(point);
//             sdf = min(max(sdf, -truncationDistance), truncationDistance);

//             // Weighted average update
//             float wNew = 1.0f;  // Example: constant weight
//             atomicAdd(&(voxel.distance), (voxel.distance * voxel.weight + sdf * wNew) / (voxel.weight + wNew));
//             atomicAdd(&(voxel.weight), wNew);
//         }
//     }
// }

// void integrate(const PointCloud& pointCloud, const Eigen::Matrix4f& pose, float truncationDistance) {
//     // Camera transformation
//     const Eigen::Affine3f transform(pose);

//     // Iterate over each point in the PointCloud
//     const auto& points = pointCloud.getPoints();
//     const auto& normals = pointCloud.getNormals();

//     // Allocate memory on the GPU
//     Vector3f* d_points;
//     Vector3f* d_normals;
//     cudaMalloc((void**)&d_points, points.size() * sizeof(Vector3f));
//     cudaMalloc((void**)&d_normals, normals.size() * sizeof(Vector3f));
//     cudaMemcpy(d_points, points.data(), points.size() * sizeof(Vector3f), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_normals, normals.data(), normals.size() * sizeof(Vector3f), cudaMemcpyHostToDevice);

//     TSDFVolume::Voxel* d_voxels;
//     cudaMalloc((void**)&d_voxels, width * height * depth * sizeof(TSDFVolume::Voxel));
//     cudaMemcpy(d_voxels, voxels, width * height * depth * sizeof(TSDFVolume::Voxel), cudaMemcpyHostToDevice);

//     // Calculate block and grid dimensions
//     int blockSize = 256;
//     int numBlocks = (points.size() + blockSize - 1) / blockSize;

//     // Launch kernel
//     integrateKernel<<<numBlocks, blockSize>>>(d_points, d_normals, pose, width, height, depth, truncationDistance, d_voxels);

//     // Copy result from device to host
//     cudaMemcpy(voxels, d_voxels, width * height * depth * sizeof(TSDFVolume::Voxel), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_points);
//     cudaFree(d_normals);
//     cudaFree(d_voxels);
// }
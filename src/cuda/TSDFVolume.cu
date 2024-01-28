#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
using blockDim = struct { int x; int y; };
using threadIdx = struct { int x; int y; int z; };
using blockIdx = struct { int x; int y; int z; };

#include <PointCloud.h>


__global__ void integrate(const float* points,
                          const float* normals,
                          size_t pointCloudSize,
                          float truncationDistance,
                          Voxel* voxels,
                          int width, int height, int depth,
                          float voxelSize) {
     int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < pointCloudSize) {
        const float* point = &points[index * 3];
        const float* normal = &normals[index * 3];

        // Transform point to TSDF grid coordinates
        int voxelCoordX = static_cast<int>((point[0] + 2.0f) / 4.0f * width);
        int voxelCoordY = static_cast<int>((point[1] + 2.0f) / 4.0f * height);
        int voxelCoordZ = static_cast<int>((point[2] + 2.0f) / 4.0f * depth);

        // Update voxel if within TSDF volume bounds
        if (voxelCoordX >= 0 && voxelCoordX < width &&
            voxelCoordY >= 0 && voxelCoordY < height &&
            voxelCoordZ >= 0 && voxelCoordZ < depth) {
            int voxelIndex = voxelCoordX + voxelCoordY * width + voxelCoordZ * width * height;
            Voxel& voxel = voxels[voxelIndex];

            // Compute signed distance and update voxel
            float sdf = normal[0] * (point[0] - voxelCoordX * voxelSize) +
                        normal[1] * (point[1] - voxelCoordY * voxelSize) +
                        normal[2] * (point[2] - voxelCoordZ * voxelSize);
            sdf = fminf(fmaxf(sdf, -truncationDistance), truncationDistance);

            // Weighted average update
            float wNew = 1.0f;  // Example: constant weight
            voxel.distance = (voxel.distance * voxel.weight + sdf * wNew) /
                             (voxel.weight + wNew);
            voxel.weight += wNew;
        }
    }
}
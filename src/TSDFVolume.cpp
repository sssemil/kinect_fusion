#include "TSDFVolume.h"

#include <chrono>
#include <stdexcept>

#include "MarchingCubes.h"
#include "Volume.h"

TSDFVolume::TSDFVolume(int width, int height, int depth, float voxelSize)
    : width(width), height(height), depth(depth), voxelSize(voxelSize) {
    voxels.resize(width * height * depth);
}

TSDFVolume::Voxel& TSDFVolume::getVoxel(int x, int y, int z) {
    if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
        throw std::out_of_range("Voxel coordinates are out of bounds");
    }
    return voxels[toLinearIndex(x, y, z)];
}

const TSDFVolume::Voxel& TSDFVolume::getVoxel(int x, int y, int z) const {
    if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
        throw std::out_of_range("Voxel coordinates are out of bounds");
    }
    return voxels[toLinearIndex(x, y, z)];
}

void TSDFVolume::integrate(const PointCloud& pointCloud,
                           float truncationDistance) {
    // Iterate over each point in the PointCloud
    const auto& points = pointCloud.getPoints();
    const auto& normals = pointCloud.getNormals();

    for (size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3f& point = points[i];
        const Eigen::Vector3f& normal = normals[i];

        // Transform point to TSDF grid coordinates
        // Eigen::Vector3i voxelCoord = (point / voxelSize).cast<int>();
        Eigen::Vector3i voxelCoord =
            ((point + Eigen::Vector3f(2, 2, 2)) / 4.0 * width).cast<int>();

        // Update voxel if within TSDF volume bounds
        if (voxelCoord[0] >= 0 && voxelCoord[0] < width && voxelCoord[1] >= 0 &&
            voxelCoord[1] < height && voxelCoord[2] >= 0 &&
            voxelCoord[2] < depth) {
            int index =
                toLinearIndex(voxelCoord[0], voxelCoord[1], voxelCoord[2]);
            Voxel& voxel = voxels[index];

            // Compute signed distance and update voxel
            float sdf =
                normal.dot(point - voxelCoord.cast<float>() * voxelSize);
            sdf = std::min(std::max(sdf, -truncationDistance),
                           truncationDistance);

            // Weighted average update
            float wNew = 1.0;  // Example: constant weight
            voxel.distance = (voxel.distance * voxel.weight + sdf * wNew) /
                             (voxel.weight + wNew);
            voxel.weight += wNew;
        }
    }
}

void TSDFVolume::storeAsOff(const std::string& filenameBaseOut) {
    std::stringstream ss;
    ss << filenameBaseOut << "tsdf_volume.off";
    std::cout << "Storing TSDFVolume as OFF at: " << ss.str() << "..."
              << std::endl;

    // convert our TSDF to Volume
    Volume vol(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), width,
               height, depth, 1);
    for (unsigned int x = 0; x < vol.getDimX(); x++) {
        for (unsigned int y = 0; y < vol.getDimY(); y++) {
            for (unsigned int z = 0; z < vol.getDimZ(); z++) {
                Voxel vox = getVoxel(x, y, z);
                vol.set(x, y, z, vox.distance);
            }
        }
    }

    // extract the zero iso-surface using marching cubes

    SimpleMesh mesh;

    // Count duration of marching cubes
    auto start = std::chrono::high_resolution_clock::now();

    for (unsigned int x = 0; x < vol.getDimX() - 1; x++) {
        auto intermittent_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = intermittent_end - start;
        auto it_per_sec = x / elapsed.count();
        printf(
            "\rMarching Cubes on slice %d of %d (%0.2f seconds) [%0.2f it/s]",
            x, vol.getDimX(), elapsed.count(), it_per_sec);
        fflush(stdout);

#pragma omp parallel for
        for (unsigned int y = 0; y < vol.getDimY() - 1; y++) {
            for (unsigned int z = 0; z < vol.getDimZ() - 1; z++) {
                ProcessVolumeCell(&vol, x, y, z, 0.00f, &mesh);
            }
        }
    }
    printf("\n");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Marching Cubes took " << elapsed.count() << " seconds."
              << std::endl;

    // write mesh to file
    if (!mesh.writeMesh(ss.str())) {
        std::cout << "ERROR: unable to write output file!" << std::endl;
    }
}

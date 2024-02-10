#include "TSDFVolume.h"

#include <chrono>
#include <stdexcept>

#include "MarchingCubes.h"
#include "Volume.h"

TSDFVolume::TSDFVolume(int width, int height, int depth, float voxelSize,
                       Vector3f offset)
    : width(width),
      height(height),
      depth(depth),
      size(voxelSize * width),
      voxelSize(voxelSize),
      offset(offset) {
    voxels.resize(width * height * depth);
    std::fill(voxels.begin(), voxels.end(), Voxel());
}

TSDFVolume::TSDFVolume(float size, int resolution, Vector3f offset)
    : TSDFVolume(resolution, resolution, resolution, size / resolution,
                 offset) {}

TSDFVolume TSDFVolume::buildSphere() {
    float radius = 4.f;
    float size = 10;
    int resolution = 256;
    TSDFVolume tsdf(size, resolution, Vector3f(0, 0, 0));

    float vsize = size / resolution;

    for (int x = 0; x < tsdf.width; x++) {
        for (int y = 0; y < tsdf.height; y++) {
            for (int z = 0; z < tsdf.depth; z++) {
                auto s = sqrt(pow((x - tsdf.width / 2.f) * vsize, 2) +
                              pow((y - tsdf.height / 2.f) * vsize, 2) +
                              pow((z - tsdf.depth / 2.f) * vsize, 2));
                auto d = s - radius;
                auto val = fmin(TRUNCATION, fmax(-TRUNCATION, d));
                tsdf.getVoxel(x, y, z).distance = val;
            }
        }
    }

    return tsdf;
}

void TSDFVolume::printSdf(const Vector3i& from, const Vector3i& to,
                          std::ostream& out) {
    std::cout << "[" << std::endl;
    for (int x = from.x(); x < to.x(); x++) {
        std::cout << "\t[" << std::endl;
        for (int y = from.y(); y < to.y(); y++) {
            std::cout << "\t\t[";
            for (int z = from.z(); z < to.z(); z++) {
                std::cout << getVoxel(x, y, z).distance;
                if (z < depth - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (y < height - 1) std::cout << ",";
            std::cout << std::endl;
        }
        std::cout << "\t]";
        if (x < width - 1) std::cout << ",";
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

void TSDFVolume::countNonThreshold() {
    size_t count = 0;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < depth; k++) {
                if (getVoxel(i, j, k).distance != TRUNCATION) {
                    count++;
                }
            }
        }
    }
    std::cout << count << " voxels with non-truncated value" << std::endl
              << "That's " << 100.f * count / (width * height * depth)
              << "% percent" << std::endl;
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

float TSDFVolume::getVoxelDistanceValue(int x, int y, int z) const {
    if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
        return TRUNCATION;
    }
    return voxels[toLinearIndex(x, y, z)].distance;
}

Vector3i TSDFVolume::getVoxelCoordinatesForWorldCoordinates(
    const Vector3f& pos) const {
    Vector3f half(size / 2.f, size / 2.f, size / 2.f);
    return ((pos + half + offset) / voxelSize).cast<int>();
}

void TSDFVolume::integrate(const PointCloud& pointCloud,
                           const Eigen::Matrix4f& pose,
                           float truncationDistance) {
    // Camera transformation
    const Eigen::Affine3f transform(pose);

    // Iterate over each point in the PointCloud
    const auto& points = pointCloud.getPoints();
    const auto& normals = pointCloud.getNormals();

    for (size_t i = 0; i < points.size(); ++i) {
        Eigen::Vector3f point = transform * points[i];
        Eigen::Vector3f normal = transform.rotation() * normals[i];

        // Transform point to TSDF grid coordinates
        // TODO: find the exact relationship between voxelSize and the SDF
        // dimensions
        Vector3i voxelCoord = getVoxelCoordinatesForWorldCoordinates(point);

        // Update voxel if within TSDF volume bounds
        if (voxelCoord[0] >= 0 && voxelCoord[0] < width && voxelCoord[1] >= 0 &&
            voxelCoord[1] < height && voxelCoord[2] >= 0 &&
            voxelCoord[2] < depth) {
            int index =
                toLinearIndex(voxelCoord[0], voxelCoord[1], voxelCoord[2]);
            Voxel& voxel = voxels[index];

            // Compute signed distance and update voxel
            float sdf = normal.dot(point);
            sdf = std::min(std::max(sdf, -TRUNCATION), TRUNCATION);

            // Weighted average update
            float wNew = 1.0;  // Example: constant weight
            voxel.distance = (voxel.distance * voxel.weight + sdf * wNew) /
                             (voxel.weight + wNew);
            voxel.weight += wNew;
        }
    }
}

void TSDFVolume::storeAsOff(const std::string& filenameBaseOut,
                            unsigned int frameNumber) {
    std::stringstream ss;
    ss << filenameBaseOut << "tsdf_volume_" << frameNumber << ".off";
    std::cout << "Storing TSDFVolume as OFF at: " << ss.str() << "..."
              << std::endl;

    // convert our TSDF to Volume
    // Volume vol(Vector3d(-0.5, -0.5, -0.5),
    //            Vector3d(0.5, 0.5, 0.5),
    //            width, height, depth, 1);

    Vector3d half(width / 2.f, height / 2.f, depth / 2.f);
    Volume vol(-half * voxelSize, half * voxelSize, width, height, depth, 1);
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

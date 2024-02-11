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

float TSDFVolume::getVoxelWeightValue(int x, int y, int z) const {
    if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
        return 0.0f;
    }
    return voxels[toLinearIndex(x, y, z)].weight;
}

Vector3i TSDFVolume::getVoxelCoordinatesForWorldCoordinates(
    const Vector3f& pos) const {
    Vector3f half(size / 2.f, size / 2.f, size / 2.f);
    return ((pos + half + offset) / voxelSize).cast<int>();
}

void TSDFVolume::integrate(const PointCloud& pointCloud,
                           const Eigen::Matrix4f& pose) {
    auto start = std::chrono::high_resolution_clock::now();

    const Eigen::Affine3f transform(pose);
    const std::vector<Eigen::Vector3f>& points = pointCloud.getPoints();
    const std::vector<Eigen::Vector3f>& normals = pointCloud.getNormals();

    int range = static_cast<int>(std::ceil(TRUNCATION / voxelSize));
    std::cout << "Integration range: " << range << std::endl;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < points.size(); ++i) {
        if (i % 100 == 0) {
            auto intermittent_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = intermittent_end - start;
            double progress = static_cast<double>(i) / points.size();
            double totalExpectedTime = elapsed.count() / progress;
            double expectedRemainingTime = totalExpectedTime - elapsed.count();
            printf(
                "\rIntegrating point %zu of %zu (%0.2f seconds) [Expected "
                "remaining time: %0.2f seconds]",
                i, points.size(), elapsed.count(), expectedRemainingTime);
            fflush(stdout);
        }

        Eigen::Vector3f globalPoint =
            transform * points[i];  // Apply transformation
        Eigen::Vector3f globalNormal =
            transform.rotation() *
            normals[i];  // Transform normals without translation

        Eigen::Vector3i voxelCoords =
            getVoxelCoordinatesForWorldCoordinates(globalPoint);

        // Iterate over the neighborhood of voxels around the point within the
        // truncation distance
        for (int dz = -range; dz <= range; dz++) {
            for (int dy = -range; dy <= range; dy++) {
                for (int dx = -range; dx <= range; dx++) {
                    int nx = voxelCoords.x() + dx;
                    int ny = voxelCoords.y() + dy;
                    int nz = voxelCoords.z() + dz;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                        nz >= 0 && nz < depth) {
                        Eigen::Vector3f voxelCenter =
                            offset +
                            Eigen::Vector3f(nx + 0.5f, ny + 0.5f, nz + 0.5f) *
                                voxelSize;
                        float distance = (globalPoint - voxelCenter)
                                             .dot(globalNormal.normalized());
                        distance = std::min(std::max(distance, -TRUNCATION),
                                            TRUNCATION);

                        Voxel& voxel = getVoxel(nx, ny, nz);
                        // Adjust the weight according to the distance from the
                        // point
                        float weight =
                            std::exp(-distance * distance / (0.03f * 0.03f));
                        voxel.distance = (voxel.distance * voxel.weight +
                                          distance * weight) /
                                         (voxel.weight + weight);
                        voxel.weight += weight;
                    }
                }
            }
        }
    }
    printf("\n");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Integration took " << elapsed.count() << " seconds."
              << std::endl;
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

#pragma omp parallel for collapse(3)
    for (unsigned int x = 0; x < vol.getDimX(); x++) {
        for (unsigned int y = 0; y < vol.getDimY(); y++) {
            for (unsigned int z = 0; z < vol.getDimZ(); z++) {
                Voxel vox = getVoxel(x, y, z);
                vol.set(x, y, z, vox.distance);
            }
        }
    }

    // extract the zero iso-surface using marching cubes

    std::vector<SimpleMesh> meshes(
        omp_get_max_threads());  // Vector of meshes, one per thread

    // Count duration of marching cubes
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (unsigned int x = 0; x < vol.getDimX() - 1; x++) {
        //        auto intermittent_end =
        //        std::chrono::high_resolution_clock::now();
        //        std::chrono::duration<double> elapsed = intermittent_end -
        //        start; auto it_per_sec = x / elapsed.count(); printf(
        //            "\rMarching Cubes on slice %d of %d (%0.2f seconds) [%0.2f
        //            it/s]", x, vol.getDimX(), elapsed.count(), it_per_sec);
        //        fflush(stdout);

        for (unsigned int y = 0; y < vol.getDimY() - 1; y++) {
            for (unsigned int z = 0; z < vol.getDimZ() - 1; z++) {
                int thread_id =
                    omp_get_thread_num();  // Get the current thread's ID
                ProcessVolumeCell(&vol, x, y, z, 0.00f,
                                  &meshes[thread_id]);  // Use thread-local mesh
            }
        }
    }
    printf("\n");

    // Combine all the meshes into one
    SimpleMesh finalMesh;
    for (const auto& mesh : meshes) {
        finalMesh =
            SimpleMesh::joinMeshes(finalMesh, mesh, Matrix4f::Identity());
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Marching Cubes took " << elapsed.count() << " seconds."
              << std::endl;
    std::cout << "Processed " << vol.getDimX() << " x-slices in total."
              << std::endl;
    auto it_per_sec = vol.getDimX() / elapsed.count();
    std::cout << "That's " << it_per_sec << " it/s" << std::endl;

    // write mesh to file
    if (!finalMesh.writeMesh(ss.str())) {
        std::cout << "ERROR: unable to write output file!" << std::endl;
    }
}

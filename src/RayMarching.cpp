#include "RayMarching.h"

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>

#define MIN_DEPTH 0.4f
#define MAX_DEPTH 10.0f
#define MAX_MARCHING_STEPS 255

// Ray marching in a given direction
float ray_marching_in_direction(const TSDFVolume& tsdf,
                                const Eigen::Vector3f& origin,
                                const Eigen::Vector3f& direction) {
    float depth = MIN_DEPTH;
    float prev_dist = std::numeric_limits<float>::max();

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        Eigen::Vector3f currentPos = origin + depth * direction;
        Eigen::Vector3i target =
            tsdf.getVoxelCoordinatesForWorldCoordinates(currentPos);
        float dist =
            tsdf.getVoxelDistanceValue(target.x(), target.y(), target.z());

        if (prev_dist >= 0 && dist < 0) {
            return depth;  // Surface intersection found
        }

        depth += dist;
        prev_dist = dist;

        if (depth >= MAX_DEPTH) {
            break;  // Max depth reached without finding an intersection
        }
    }

    return -std::numeric_limits<float>::infinity();  // No intersection found
}

// Main ray marching function that uses camera pose
PointCloud ray_marching(const TSDFVolume& tsdf, VirtualSensor& sensor,
                        const Eigen::Matrix4f& camera_pose,
                        const std::string& filenameBaseOut,
                        unsigned int frameNumber, bool screenshot) {
    auto start = std::chrono::high_resolution_clock::now();

    int width = sensor.getDepthImageWidth();
    int height = sensor.getDepthImageHeight();
    std::vector<float> distances(width * height,
                                 -std::numeric_limits<float>::infinity());

    // Iterate over each pixel in the image
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Transform the pixel to camera space direction
            Eigen::Vector3f pixel(x, height - y - 1, 1.0f);
            Eigen::Vector3f direction =
                (sensor.getDepthIntrinsics().inverse() * pixel).normalized();

            // Apply camera pose to get the global direction
            Eigen::Vector3f globalDirection =
                camera_pose.block<3, 3>(0, 0) * direction;

            Eigen::Vector3f globalOrigin = camera_pose.block<3, 1>(0, 3);

            float dist =
                ray_marching_in_direction(tsdf, globalOrigin, globalDirection);
            distances[y * width + x] = dist;
        }
    }

    if (screenshot) {
        auto distances_copy = distances;
        typeof(distances_copy) filtered;

        // copy only finite numbers:
        std::copy_if(distances_copy.begin(), distances_copy.end(),
                     std::back_inserter(filtered), [](auto i) {
                         return abs(i) !=
                                std::numeric_limits<float>::infinity();
                     });

        float maximum =
            (filtered.size() > 0)
                ? *std::max_element(filtered.begin(), filtered.end())
                : MINF;
        float minimum =
            (filtered.size() > 0)
                ? *std::min_element(filtered.begin(), filtered.end())
                : MINF;
        std::cout << "Max distance: " << maximum
                  << ", min distance: " << minimum << std::endl;
        for (auto& i : distances_copy) {
            i = 1 - ((i - minimum) / (maximum - minimum));
        }

        FreeImage img(width, height, 1);
        img.data = distances_copy.data();
        // append frame number to filename
        std::stringstream ss;
        ss << filenameBaseOut << "ray_" << std::setw(4) << std::setfill('0') << frameNumber << ".png";
        std::cout << "Saving depth map to " << ss.str() << std::endl;
        img.SaveImageToFile(ss.str(), true);
        img.data = nullptr;
        std::cout << "Depth map rendered." << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Ray marching took " << elapsed.count() << " seconds."
              << std::endl;

    return PointCloud{distances.data(),
                      sensor.getDepthIntrinsics(),
                      sensor.getDepthExtrinsics(),
                      sensor.getDepthImageWidth(),
                      sensor.getDepthImageHeight(),
                      8};
}

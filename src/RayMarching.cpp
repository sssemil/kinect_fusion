#include "RayMarching.h"

#include <algorithm>
#include <limits>

#include "SurfaceMeasurement.h"

#define MIN_DEPTH 0.4f
#define MAX_DEPTH 100.0f
#define MAX_MARCHING_STEPS 255

#define EPSILON 0.001f

float ray_marching_in_direction(const TSDFVolume& tsdf, const Vector3f& origin,
                                const Vector3f& direction) {
    float depth = MIN_DEPTH;
    float prev_dist = MINF;
    float last_increment = MINF;

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        Eigen::Vector3i target = tsdf.getVoxelCoordinatesForWorldCoordinates(
            origin + depth * direction);
        float dist =
            tsdf.getVoxelDistanceValue(target[0], target[1], target[2]);

        // TODO: handle if backface encountered

        if (dist < EPSILON) {
            // inside the surface
            return depth - (last_increment * prev_dist) / (dist - prev_dist);
        }
        // Move along the view ray
        depth += dist;
        last_increment = dist;
        prev_dist = dist;

        if (depth >= MAX_DEPTH) {
            // Gone too far; give up
            break;
        }
    }
    return MINF;
}

PointCloud ray_marching(const TSDFVolume& tsdf, VirtualSensor& sensor,
                        const Matrix4f& current_pose_estimate) {
    auto width = sensor.getDepthImageWidth();
    auto height = sensor.getDepthImageHeight();

    Vector3f orr(0, 0, -3.f);
    Vector3f hurka = current_pose_estimate.block(0, 0, 3, 3) *
                     sensor.getDepthIntrinsics().inverse() *
                     Vector3f(320, 240, 1.0f);
    float d = ray_marching_in_direction(tsdf, orr, hurka.normalized());
    std::cout << d;

    std::vector<float> distances(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            //            Vector3f origin = current_pose_estimate.block(0, 3, 3,
            //            1);
            Vector3f origin(0, 0, -3.f);
            Vector3f ray_direction = current_pose_estimate.block(0, 0, 3, 3) *
                                     sensor.getDepthIntrinsics().inverse() *
                                     Vector3f(x, y, 1.0f);
            float dist = ray_marching_in_direction(tsdf, origin,
                                                   ray_direction.normalized());
            distances[width * y + x] = dist;
        }
    }

    auto distances_copy = distances;
    float maximum =
        -*std::min_element(distances_copy.begin(), distances_copy.end());
    float minimum =
        -*std::max_element(distances_copy.begin(), distances_copy.end());
    //    std::cout << maximum << " " << minimum << std::endl;
    //    for (auto& i : distances_copy) {
    //        std::cout << i << ", ";
    //    }

    FreeImage img(width, height, 1);
    img.data = distances_copy.data();
    img.SaveImageToFile("output.png");
    img.data = nullptr;

    return PointCloud{distances.data(),
                      sensor.getDepthIntrinsics(),
                      sensor.getDepthExtrinsics(),
                      sensor.getDepthImageWidth(),
                      sensor.getDepthImageHeight(),
                      8};
}

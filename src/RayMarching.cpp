#include "RayMarching.h"

#include <algorithm>
#include <limits>

#include "SurfaceMeasurement.h"

#define MIN_DEPTH 0.4f
#define MAX_DEPTH 10.0f
#define MAX_MARCHING_STEPS 255

#define EPSILON 0.001f

float ray_marching_in_direction(const TSDFVolume& tsdf, const Vector3f& origin,
                                const Vector3f& direction) {
    float depth = MIN_DEPTH;
    float prev_dist = MINF;
    float last_increment = MINF;

    Vector3f half(tsdf.getPhysicalSize() / 2, tsdf.getPhysicalSize() / 2, 0);
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        Vector3f currentPos = origin + half + depth * direction;
        Eigen::Vector3i target =
            tsdf.getVoxelCoordinatesForWorldCoordinates(currentPos);
        float dist =
            tsdf.getVoxelDistanceValue(target[0], target[1], target[2]);

        // TODO: handle if backface encountered

        if (prev_dist >= 0 && dist < 0) {
            // inside the surface
            return depth;  // - (last_increment * prev_dist) / (dist -
                           // prev_dist);
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
    return -MINF;
}

PointCloud ray_marching(const TSDFVolume& tsdf, VirtualSensor& sensor,
                        const Matrix4f& current_pose_estimate) {
    int width = sensor.getDepthImageWidth();
    int height = sensor.getDepthImageHeight();

    Vector3f orr(0, 0, -5.f);
    Vector3f hurka1 =
        sensor.getDepthIntrinsics().inverse() * Vector3f(0, 0, 1.0f);
    Vector3f hurka =
        sensor.getDepthIntrinsics().inverse() * Vector3f(320, 240, 1.0f);
    Vector3f hurka2 =
        sensor.getDepthIntrinsics().inverse() * Vector3f(640, 480, 1.0f);

    float d = ray_marching_in_direction(tsdf, orr, hurka.normalized());
    std::cout << d;

    std::vector<float> distances(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            //            Vector3f origin = current_pose_estimate.block(0, 3, 3,
            //            1);
            Vector3f origin(-2, -2, -2.f);
            Vector3f target =
                /*current_pose_estimate.block(0, 0, 3, 3) **/ sensor
                    .getDepthIntrinsics()
                    .inverse() *
                Vector3f(x, y, 1.0f);
            Vector3f ray_direction = target;

            // orthographic projection
            //            origin = Vector3f((tsdf.getWidth() - width / 2 + x) *
            //            tsdf.getVoxelSize(),
            //                              (tsdf.getHeight() - height / 2 + y)
            //                              * tsdf.getVoxelSize(), -5);

            float dx = tsdf.getPhysicalSize() / width;
            float dy = tsdf.getPhysicalSize() / height;
            origin = Vector3f((x - width) * dx, (y - height) * dy, 5);
            ray_direction = Vector3f(0, 0, -1);

            float dist = ray_marching_in_direction(tsdf, origin,
                                                   ray_direction.normalized());
            distances[width * y + x] = dist;
        }
    }

    auto distances_copy = distances;
    typeof(distances_copy) filtered;

    // copy only finite numbers:
    std::copy_if(distances_copy.begin(), distances_copy.end(),
                 std::back_inserter(filtered), [](auto i) {
                     return abs(i) != std::numeric_limits<float>::infinity();
                 });

    float maximum = (filtered.size() > 0)
                        ? *std::max_element(filtered.begin(), filtered.end())
                        : MINF;
    float minimum = (filtered.size() > 0)
                        ? *std::min_element(filtered.begin(), filtered.end())
                        : MINF;
    std::cout << "Max distance: " << maximum << ", min distance: " << minimum
              << std::endl;
    for (auto& i : distances_copy) {
        i = 1 - ((i - minimum) / (maximum - minimum));
    }

    FreeImage img(width, height, 1);
    img.data = distances_copy.data();
    img.SaveImageToFile("output.png", true);
    img.data = nullptr;
    std::cout << "Depth map rendered." << std::endl;

    return PointCloud{distances.data(),
                      sensor.getDepthIntrinsics(),
                      sensor.getDepthExtrinsics(),
                      sensor.getDepthImageWidth(),
                      sensor.getDepthImageHeight(),
                      8};
}

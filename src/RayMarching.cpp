#include "RayMarching.h"

#include <limits>

#include "SurfaceMeasurement.h"

#define MIN_DEPTH 0.4f
#define MAX_DEPTH 8.0f
#define MAX_MARCHING_STEPS 255

#define EPSILON 0.001f

float ray_marching_in_direction(const TSDFVolume &tsdf, const Vector3f& origin, const Vector3f& direction) {
    float depth = MIN_DEPTH;

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        Eigen::Vector3i target = tsdf.getVoxelCoordinatesForWorldCoordinates(origin + depth * direction);
        float dist = tsdf.getVoxel(target[0], target[1], target[2]).distance;

        // TODO: handle if backface encountered

        if (dist < EPSILON) {
            // inside the surface
            return depth;
        }
        // Move along the view ray
        depth += dist;

        if (depth >= MAX_DEPTH) {
            // Gone too far; give up
            break;
        }
    }
    return MINF;
}

PointCloud ray_marching(const TSDFVolume &tsdf, VirtualSensor &sensor, const Matrix4f& current_pose_estimate) {
    auto width = sensor.getDepthImageHeight();
    auto height = sensor.getDepthImageHeight();

    std::vector<float> distances(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vector3f origin = current_pose_estimate.block(0, 3, 3, 1);
            Vector3f ray_direction = current_pose_estimate.block(0, 0, 3, 3) * sensor.getDepthIntrinsics() * Vector3f(x, y, 1.0f);
            float dist = ray_marching_in_direction(tsdf, origin, ray_direction.normalized());
            distances[width * y + x] = dist;
        }
    }

    return PointCloud{
        distances.data(),
        sensor.getDepthIntrinsics(),
        sensor.getDepthExtrinsics(),
        sensor.getDepthImageWidth(),
        sensor.getDepthImageHeight(),
        8
    };
}

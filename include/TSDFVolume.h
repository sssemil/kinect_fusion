#pragma once

#include <Eigen/Dense>
#include <vector>

#include "PointCloud.h"

struct Coord {
    int x;
    int y;
    int z;
};

class TSDFVolume {
   public:
    TSDFVolume(int width, int height, int depth, float voxelSize);

    struct Voxel {
        float distance;
        float weight;
        Voxel()
            : distance(1.0f), weight(0.0f) {}  // Initialize with default values
    };

    Voxel& getVoxel(int x, int y, int z);
    const Voxel& getVoxel(int x, int y, int z) const;

    void integrate(const PointCloud& pointCloud, const Eigen::Matrix4f& pose, float truncationDistance);

    void storeAsOff(const std::string& filenameBaseOut);

    static TSDFVolume buildSphere();

   private:
    std::vector<Voxel> voxels;
    int width, height, depth;
    float voxelSize;

    inline int toLinearIndex(int x, int y, int z) const {
        return x + width * (y + height * z);
    }

    inline Coord fromLinearIndex(int index) const {
        int x = index % width;
        int y = (index / width) % height;
        int z = index / (width * height);
        return Coord{.x = x, .y = y, .z = z};
    }
};

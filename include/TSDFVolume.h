#pragma once

#include <Eigen/Dense>
#include <vector>

class TSDFVolume {
public:
    TSDFVolume(int width, int height, int depth, float voxelSize);

    struct Voxel {
        float distance;
        float weight;
        Voxel() : distance(1.0f), weight(0.0f) {} // Initialize with default values
    };

    Voxel& getVoxel(int x, int y, int z);
    const Voxel& getVoxel(int x, int y, int z) const;

private:
    std::vector<Voxel> voxels;
    int width, height, depth;
    float voxelSize;

    inline int toLinearIndex(int x, int y, int z) const {
        return x + width * (y + height * z);
    }
};

#endif // TSDFVOLUME_H

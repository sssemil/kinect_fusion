#include "TSDFVolume.h"
#include <stdexcept>

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

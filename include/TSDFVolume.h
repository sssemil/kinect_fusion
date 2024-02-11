#pragma once

#include <Eigen/Dense>
#include <vector>

#include "PointCloud.h"

#define TRUNCATION 0.01f

struct Coord {
    int x;
    int y;
    int z;
};

class TSDFVolume {
   public:
    TSDFVolume(float size, int resolution, Vector3f offset = Vector3f());
    TSDFVolume(int width, int height, int depth, float voxelSize,
               Vector3f offset = Vector3f());

    struct Voxel {
        float distance;
        float weight;
        Voxel()
            : distance(TRUNCATION),
              weight(0.0f) {}  // Initialize with default values
    };

    Voxel& getVoxel(int x, int y, int z);
    const Voxel& getVoxel(int x, int y, int z) const;
    float getVoxelDistanceValue(int x, int y, int z) const;

    //    Voxel& getVoxelCoordinatesForWorldCoordinates(const Vector3f& pos);
    Vector3i getVoxelCoordinatesForWorldCoordinates(const Vector3f& pos) const;

    void integrate(const PointCloud& pointCloud, const Eigen::Matrix4f& pose);

    void storeAsOff(const std::string& filenameBaseOut,
                    unsigned int frameNumber);

    static TSDFVolume buildSphere();

    int getWidth() const { return width; }

    int getHeight() const { return height; }

    int getDepth() const { return depth; }

    float getPhysicalSize() const { return size; }

    float getVoxelSize() const { return voxelSize; }

    void printSdf(const Vector3i& from, const Vector3i& to,
                  std::ostream& out = std::cout);
    void countNonThreshold();

   private:
    std::vector<Voxel> voxels;
    int width, height, depth;
    const Vector3f offset;
    const float size;
    const float voxelSize;

    inline int toLinearIndex(int x, int y, int z) const {
        return x + width * (y + height * z);
    }

    inline Coord fromLinearIndex(int index) const {
        int x = index % width;
        int y = (index / width) % height;
        int z = index / (width * height);
        return Coord{.x = x, .y = y, .z = z};
    }

    float getVoxelWeightValue(int x, int y, int z) const;
};

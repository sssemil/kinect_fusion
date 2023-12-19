#ifndef DEPTHFRAME_H
#define DEPTHFRAME_H

#include <vector>
#include <cstdint>

struct Point3D {
    float x, y;
    float d;
    uint8_t r, g, b;
    float confidence;
};

class DepthFrame {
public:
    DepthFrame(int width, int height);
    ~DepthFrame();

    void setData(int x, int y, float depth, uint8_t r, uint8_t g, uint8_t b, float confidence);

    Point3D getData(int x, int y) const;

    std::vector<Point3D> toPointCloud() const;

private:
    int width, height;
    std::vector<std::vector<Point3D>> data;
};

#endif // DEPTHFRAME_H

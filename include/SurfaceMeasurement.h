#ifndef SURFACEMEASUREMENT_H
#define SURFACEMEASUREMENT_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

struct KinectFusionData {
    std::vector<cv::Mat> depthPyramid;
    std::vector<cv::Mat> filteredDepthPyramid;
    std::vector<cv::Mat> surfacePyramid;
    std::vector<cv::Mat> normalPyramid;
};

struct CameraParams {
    int imageWidth, imageHeight;
    float focalX, focalY;
    float principalX, principalY;

    CameraParams(int imageWidth, int imageHeight, float focalX, float focalY,
                 float principalX, float principalY);
    CameraParams();
    CameraParams cameraParametersByLevel(int level) const;
};

class SurfaceMeasurement {
   public:
    SurfaceMeasurement();

    void compute_normal_map(cv::Mat &vertexMap, cv::Mat &normalMap);
    static void compute_surface_vertex(cv::Mat &depthMap, cv::Mat &vertexMap,
                                       float depthCutOff,
                                       CameraParams &cameraParameters);
    void surface_mesasurement(cv::Mat &inputFrame,
                              CameraParams &cameraParameters, int depthLevel);

   private:
    // TODO: Remove this?
    CameraParams cameraParameters{640, 480, 525.0, 525.0, 319.5, 239.5};
};

#endif  // SURFACEMEASUREMENT_H

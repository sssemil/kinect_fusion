#include "SurfaceMeasurement.h"

CameraParams::CameraParams(int imageWidth, int imageHeight, float focalX,
                           float focalY, float principalX, float principalY)
    : imageWidth(imageWidth),
      imageHeight(imageHeight),
      focalX(focalX),
      focalY(focalY),
      principalX(principalX),
      principalY(principalY) {}

CameraParams::CameraParams()
    : imageWidth(0),
      imageHeight(0),
      focalX(0),
      focalY(0),
      principalX(0),
      principalY(0) {}

CameraParams CameraParams::cameraParametersByLevel(int level) {
    return CameraParams{imageWidth >> level,       imageHeight >> level,
                        focalX / (1 << level),     focalY / (1 << level),
                        principalX / (1 << level), principalY / (1 << level)};
}

SurfaceMeasurement::SurfaceMeasurement() {}

void SurfaceMeasurement::compute_normal_map(cv::Mat &vertexMap,
                                            cv::Mat &normalMap) {
    int imageWidth = vertexMap.cols;
    int imageHeight = vertexMap.rows;

    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            cv::Vec3f normal;
            if (i == 0 || i == imageHeight - 1 || j == 0 ||
                j == imageWidth - 1) {
                normal = cv::Vec3f(0.f, 0.f, 0.f);
            } else {
                cv::Vec3f v1 = vertexMap.at<cv::Vec3f>(i - 1, j);
                cv::Vec3f v2 = vertexMap.at<cv::Vec3f>(i + 1, j);
                cv::Vec3f v3 = vertexMap.at<cv::Vec3f>(i, j - 1);
                cv::Vec3f v4 = vertexMap.at<cv::Vec3f>(i, j + 1);

                if (v1[2] == 0.f || v2[2] == 0.f || v3[2] == 0.f ||
                    v4[2] == 0.f) {
                    normal = cv::Vec3f(0.f, 0.f, 0.f);
                } else {
                    cv::Vec3f dx = v1 - v2;
                    cv::Vec3f dy = v3 - v4;

                    normal = dx.cross(dy);
                    normal = normal / cv::norm(normal);

                    if (normal[2] > 0) {
                        normal = -normal;
                    }
                }
            }
            normalMap.at<cv::Vec3f>(i, j) = normal;
        }
    }
}

void SurfaceMeasurement::compute_surface_vertex(
    cv::Mat &depthMap, cv::Mat &vertexMap, const float depthCutOff,
    CameraParams &cameraParameters) {
    int imageWidth = depthMap.cols;
    int imageHeight = depthMap.rows;

    float focalX = cameraParameters.focalX;
    float focalY = cameraParameters.focalY;
    float principalX = cameraParameters.principalX;
    float principalY = cameraParameters.principalY;

    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            float depth = depthMap.at<float>(i, j);
            if (depth > depthCutOff) {
                depth = 0.f;
            }
            float x = (j - principalX) * depth / focalX;
            float y = (i - principalY) * depth / focalY;
            float z = depth;
            vertexMap.at<cv::Vec3f>(i, j) = cv::Vec3f(x, y, z);
        }
    }
}

void SurfaceMeasurement::surface_mesasurement(cv::Mat &inputFrame,
                                              CameraParams &cameraParameters,
                                              int depthLevel) {
    KinectFusionData data;

    for (size_t level = 0; level < depthLevel; ++level) {
        cv::Mat depthMap(cameraParameters.imageHeight,
                         cameraParameters.imageWidth, CV_32FC1);
        cv::Mat filteredDepthMap(cameraParameters.imageHeight,
                                 cameraParameters.imageWidth, CV_32FC1);
        cv::Mat surfaceMap(cameraParameters.imageHeight,
                           cameraParameters.imageWidth, CV_32FC3);
        cv::Mat normalMap(cameraParameters.imageHeight,
                          cameraParameters.imageWidth, CV_32FC3);

        data.depthPyramid.push_back(depthMap);
        data.filteredDepthPyramid.push_back(filteredDepthMap);
        data.surfacePyramid.push_back(surfaceMap);
        data.normalPyramid.push_back(normalMap);

        CameraParams cameraParametersTemp =
            cameraParameters.cameraParametersByLevel(level);
    }

    // downsample the input frame to the lowest level
    data.depthPyramid[0] = inputFrame;

    for (int level = 1; level < depthLevel; ++level) {
        cv::pyrDown(data.depthPyramid[level - 1], data.depthPyramid[level],
                    cv::Size(data.depthPyramid[level - 1].cols / 2,
                             data.depthPyramid[level - 1].rows / 2));
    }
    for (int level = 0; level < depthLevel; ++level) {
        cv::bilateralFilter(data.depthPyramid[level],
                            data.filteredDepthPyramid[level], 5, 1.0, 1.0,
                            cv::BORDER_DEFAULT);
    }

    for (int level = depthLevel - 1; level >= 0; --level) {
        compute_surface_vertex(data.filteredDepthPyramid[level],
                               data.surfacePyramid[level], 1000.0f,
                               cameraParameters);
        compute_normal_map(data.surfacePyramid[level],
                           data.normalPyramid[level]);
    }
}

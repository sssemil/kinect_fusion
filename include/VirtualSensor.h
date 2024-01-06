#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Eigen.h"
#include "FreeImageHelper.h"

typedef unsigned char BYTE;

class VirtualSensor {
   public:
    VirtualSensor();
    ~VirtualSensor();

    bool Init(const std::string& datasetDir);
    bool ProcessNextFrame();
    unsigned int GetCurrentFrameCnt();
    BYTE* GetColorRGBX();
    float* GetDepth();
    Eigen::Matrix3f GetColorIntrinsics();
    Eigen::Matrix4f GetColorExtrinsics();
    unsigned int GetColorImageWidth();
    unsigned int GetColorImageHeight();
    Eigen::Matrix3f GetDepthIntrinsics();
    Eigen::Matrix4f GetDepthExtrinsics();
    unsigned int GetDepthImageWidth();
    unsigned int GetDepthImageHeight();
    Eigen::Matrix4f GetTrajectory();

   private:
    bool ReadFileList(const std::string& filename,
                      std::vector<std::string>& result,
                      std::vector<double>& timestamps);
    bool ReadTrajectoryFile(const std::string& filename,
                            std::vector<Eigen::Matrix4f>& result,
                            std::vector<double>& timestamps);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int m_currentIdx;
    int m_increment;
    float* m_depthFrame;
    BYTE* m_colorFrame;
    Eigen::Matrix4f m_currentTrajectory;
    Eigen::Matrix3f m_colorIntrinsics;
    Eigen::Matrix4f m_colorExtrinsics;
    unsigned int m_colorImageWidth;
    unsigned int m_colorImageHeight;
    Eigen::Matrix3f m_depthIntrinsics;
    Eigen::Matrix4f m_depthExtrinsics;
    unsigned int m_depthImageWidth;
    unsigned int m_depthImageHeight;
    std::string m_baseDir;
    std::vector<std::string> m_filenameDepthImages;
    std::vector<double> m_depthImagesTimeStamps;
    std::vector<std::string> m_filenameColorImages;
    std::vector<double> m_colorImagesTimeStamps;
    std::vector<Eigen::Matrix4f> m_trajectory;
    std::vector<double> m_trajectoryTimeStamps;
};

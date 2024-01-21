#include <iostream>
#include <random>

#include "Eigen.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "SimpleMesh.h"
#include "VirtualSensor.h"
#include "cxxopts.hpp"

int run(const std::string &datasetPath, const std::string &filenameBaseOut) {
    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.init(datasetPath)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!"
                  << std::endl;
        return -1;
    }

    // We store a first frame as a reference frame. All next frames are tracked
    // relatively to the first frame.
    sensor.processNextFrame();
    PointCloud target{sensor.getDepth(), sensor.getDepthIntrinsics(),
                      sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(),
                      sensor.getDepthImageHeight()};

    // Setup the optimizer.
    auto optimizer = new CeresICPOptimizer();

    optimizer->setMatchingMaxDistance(0.1f);
    optimizer->usePointToPlaneConstraints(false);
    optimizer->setNbOfIterations(20);

    // We store the estimated camera poses.
    std::vector<Matrix4f> estimatedPoses;
    Matrix4f currentCameraToWorld = Matrix4f::Identity();
    estimatedPoses.push_back(currentCameraToWorld.inverse());

    int i = 0;
    const int iMax = 50;
    while (sensor.processNextFrame() && i <= iMax) {
        float *depthMap = sensor.getDepth();
        Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
        Matrix4f depthExtrinsics = sensor.getDepthExtrinsics();

        // Estimate the current camera pose from source to target mesh with ICP
        // optimization. We downsample the source image to speed up the
        // correspondence matching.
        PointCloud source{sensor.getDepth(),
                          sensor.getDepthIntrinsics(),
                          sensor.getDepthExtrinsics(),
                          sensor.getDepthImageWidth(),
                          sensor.getDepthImageHeight(),
                          8};
        optimizer->estimatePose(source, target, currentCameraToWorld);

        // Invert the transformation matrix to get the current camera pose.
        Matrix4f currentCameraPose = currentCameraToWorld.inverse();
        std::cout << "Current camera pose: " << std::endl
                  << currentCameraPose << std::endl;
        estimatedPoses.push_back(currentCameraPose);

        if (i % 5 == 0) {
            // We write out the mesh to file for debugging.
            SimpleMesh currentDepthMesh{sensor, currentCameraPose, 0.1f};
            SimpleMesh currentCameraMesh =
                SimpleMesh::camera(currentCameraPose, 0.0015f);
            SimpleMesh resultingMesh = SimpleMesh::joinMeshes(
                currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

            std::stringstream ss;
            ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
            std::cout << filenameBaseOut << sensor.getCurrentFrameCnt()
                      << ".off" << std::endl;
            if (!resultingMesh.writeMesh(ss.str())) {
                std::cout << "Failed to write mesh!\nCheck file path!"
                          << std::endl;
                return -1;
            }
        }

        i++;
    }

    delete optimizer;

    return 0;
}

int main(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], " - command line options");
        options.allow_unrecognised_options().add_options()(
            "d,dataset", "Path to the dataset", cxxopts::value<std::string>())(
            "o,output", "Base output filename", cxxopts::value<std::string>())(
            "h,help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        std::string datasetPath = "../Data/rgbd_dataset_freiburg1_xyz/";
        if (result.count("dataset")) {
            datasetPath = result["dataset"].as<std::string>();
        }

        std::string filenameBaseOut = "mesh_";
        if (result.count("output")) {
            filenameBaseOut = result["output"].as<std::string>();
        }

        std::cout << "Dataset Path: " << datasetPath << std::endl;
        std::cout << "Base Output Filename: " << filenameBaseOut << std::endl;

        return run(datasetPath, filenameBaseOut);
    } catch (cxxopts::exceptions::option_has_no_value &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
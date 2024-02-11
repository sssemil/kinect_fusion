#include <iostream>
#include <random>

#include "Eigen.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "RayMarching.h"
#include "SimpleMesh.h"
#include "TSDFVolume.h"
#include "VirtualSensor.h"
#include "cxxopts.hpp"

int logMesh(VirtualSensor& sensor, const Matrix4f& currentCameraPose,
            const std::string& filenameBaseOut) {
    // We write out the mesh to file for debugging.
    SimpleMesh currentDepthMesh{sensor, currentCameraPose, 0.1f};
    SimpleMesh currentCameraMesh =
        SimpleMesh::camera(currentCameraPose, 0.0015f);
    SimpleMesh resultingMesh = SimpleMesh::joinMeshes(
        currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

    std::stringstream ss;
    ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
    std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off"
              << std::endl;
    if (!resultingMesh.writeMesh(ss.str())) {
        std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
        return -1;
    }

    return 0;
}

int run(const std::string& datasetPath, const std::string& filenameBaseOut,
        float size, int resolution, Vector3f offset,
        bool relativeToPreviousFrame, unsigned int stopAfterFrame,
        bool applyBilateralEnabled) {
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
    sensor.processNextFrame(applyBilateralEnabled);
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
    Matrix4f cumulativeCameraToWorld = Matrix4f::Identity();
    Matrix4f currentCameraToWorld = Matrix4f::Identity();
    estimatedPoses.emplace_back(currentCameraToWorld.inverse());

    // Define the dimensions and resolution of the TSDF volume
    TSDFVolume tsdfVolume(size, resolution, offset);
    // TSDFVolume tsdfVolume(resolution, resolution, resolution, voxelSize);

    // Build TSDF using the first frame
    tsdfVolume.integrate(target, currentCameraToWorld);

    int i = 1;
    while (true) {
        bool stop_reached = !(sensor.processNextFrame(applyBilateralEnabled) &&
                              i < stopAfterFrame);
        // Log every 20th frame or if we are about to stop
        if (i % 20 == 0 || stop_reached) {
            auto ray_target = ray_marching(
                tsdfVolume, sensor, cumulativeCameraToWorld, filenameBaseOut, i);
            tsdfVolume.storeAsOff(filenameBaseOut, i);
        }
        if (stop_reached) {
            break;
        }
        i++;

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

        // TODO: Track camera pose and then get the target image from the TSDF
        // from that pose.
        // TODO: Replace target with a raycasted image from the TSDF volume.

        currentCameraToWorld.setIdentity();
        optimizer->estimatePose(source, target, currentCameraToWorld);
//        optimizer->estimatePose(source, target, cumulativeCameraPose);

        // Invert the transformation matrix to get the current camera pose.
        Matrix4f currentCameraPose = currentCameraToWorld.inverse();
        std::cout << "Current camera pose: " << std::endl
                  << currentCameraPose << std::endl;
        estimatedPoses.push_back(currentCameraPose);

        cumulativeCameraToWorld = cumulativeCameraToWorld * currentCameraToWorld;
        tsdfVolume.integrate(source, cumulativeCameraToWorld);

        // Replace target (reference frame) with source (current) frame
        target = source;
//        ray_marching(tsdfVolume, sensor, cumulativeCameraToWorld);

        // if (i % 10 == 0) {
        //     if (logMesh(sensor, currentCameraPose, filenameBaseOut) !=
        //     0) {
        //         return -1;
        //     }
        // }

        i++;
    }

    // Building an SDF of a sphere manually
    // TSDFVolume tsdfVolume = TSDFVolume::buildSphere();

    delete optimizer;

    return 0;
}

int main(int argc, char* argv[]) {
    try {
        cxxopts::Options options(argv[0], " - command line options");
        options.allow_unrecognised_options().add_options()(
            "d,dataset", "Path to the dataset", cxxopts::value<std::string>())(
            "o,output", "Base output filename", cxxopts::value<std::string>())(
            "r,resolution", "TSDF resolution", cxxopts::value<int>())(
            "s,stopAfterFrame", "Stop after this number of frames",
            cxxopts::value<unsigned int>())("b,applyBilateral",
                                            "Apply bilateral filter",
                                            cxxopts::value<bool>())(
            "x,dx", "X-offset", cxxopts::value<float>())(
            "y,dy", "Y-offset", cxxopts::value<float>())(
            "z,dz", "Z-offset", cxxopts::value<float>())(
            "f,relativeToPreviousFrame", "Use relative frame as first frame",
            cxxopts::value<bool>())("h,help", "Print help");

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

        float size = 4.0f;  // meters
        if (result.count("size")) {
            size = result["size"].as<float>();
        }

        int resolution = 512;
        if (result.count("resolution")) {
            resolution = result["resolution"].as<int>();
        }

        unsigned int stopAfterFrame = 21;
        if (result.count("stopAfterFrame")) {
            stopAfterFrame = result["stopAfterFrame"].as<unsigned int>();
        }

        bool applyBilateralEnabled = false;
        if (result.count("applyBilateral")) {
            applyBilateralEnabled = result["applyBilateral"].as<bool>();
        }

        float dx = 2.f;  // meters
        if (result.count("dx")) {
            dx = result["dx"].as<float>();
        }

        float dy = 2.f;  // meters
        if (result.count("dy")) {
            dy = result["dy"].as<float>();
        }

        float dz = -0.5f;  // meters
        if (result.count("dz")) {
            dz = result["dz"].as<float>();
        }

        bool relativeToPreviousFrame = false;
        if (result.count("relativeToPreviousFrame")) {
            relativeToPreviousFrame =
                result["relativeToPreviousFrame"].as<bool>();
        }

        std::cout << "Dataset Path: " << datasetPath << std::endl;
        std::cout << "Base Output Filename: " << filenameBaseOut << std::endl;
        std::cout << "Size: " << size << std::endl;
        std::cout << "Resolution: " << resolution << std::endl;
        std::cout << "Offsets: " << dx << ", " << dy << ", " << dz << std::endl;
        std::cout << "Relative to Previous Frame: " << relativeToPreviousFrame
                  << std::endl;
        std::cout << "Stop After Frame: " << stopAfterFrame << std::endl;
        std::cout << "Apply Bilateral Filter: " << applyBilateralEnabled
                  << std::endl;

        return run(datasetPath, filenameBaseOut, size, resolution,
                   Vector3f(dx, dy, dz), relativeToPreviousFrame,
                   stopAfterFrame, applyBilateralEnabled);
    } catch (cxxopts::exceptions::option_has_no_value& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}

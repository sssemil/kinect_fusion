#include <iostream>
#include <random>

#include "Eigen.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "SimpleMesh.h"
#include "TSDFVolume.h"
#include "VirtualSensor.h"

std::vector<int> iterations{2,5};

#define LEVEL 2

std::vector<PointCloud> createPointClouds(VirtualSensor& sensor, int level) {
    std::vector<PointCloud> current;
    std::vector<float> depthMap = sensor.getDepth();
    CameraParams cameraParameters{
        static_cast<int>(sensor.getDepthImageWidth()),
        static_cast<int>(sensor.getDepthImageHeight()),
        sensor.getDepthIntrinsics()(0, 0),
        sensor.getDepthIntrinsics()(1, 1),
        sensor.getDepthIntrinsics()(0, 2),
        sensor.getDepthIntrinsics()(1, 2)};

    std::cout << "Create PointClouds..." << std::endl;
    for (int i = 0; i < level; i++) {
        CameraParams tmp = cameraParameters.cameraParametersByLevel(i);
        Eigen::Matrix3f depthInstrinsic = Eigen::Matrix3f::Identity();
        depthInstrinsic(0, 0) = tmp.focal_x;
        depthInstrinsic(1, 1) = tmp.focal_y;
        depthInstrinsic(0, 2) = tmp.principal_x;
        depthInstrinsic(1, 2) = tmp.principal_y;
        depthInstrinsic(2, 2) = 1;

        PointCloud tmpPointCloud{depthMap,
                                depthInstrinsic,
                                sensor.getDepthExtrinsics(),
                                static_cast<unsigned int>(tmp.image_width),
                                static_cast<unsigned int>(tmp.image_height)};
        current.push_back(tmpPointCloud);
    }
    std::cout << "PointClouds created..." << std::endl;
    return current;
}


int logMesh(VirtualSensor &sensor, const Matrix4f &currentCameraPose,
            const std::string &filenameBaseOut) {
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



int run(const std::string &datasetPath, const std::string &filenameBaseOut,
        int resolution, float voxelSize) {
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
    std::vector<PointCloud> target = createPointClouds(sensor, LEVEL);

    // Setup the optimizer.
    auto optimizer = new CeresICPOptimizer();

    optimizer->setMatchingMaxDistance(0.1f);
    optimizer->usePointToPlaneConstraints(false);
    optimizer->setNbOfIterations(20);

    // We store the estimated camera poses.
    std::vector<Matrix4f> estimatedPoses;
    Matrix4f currentCameraToWorld = Matrix4f::Identity();
    estimatedPoses.emplace_back(currentCameraToWorld.inverse());

    // Define the dimensions and resolution of the TSDF volume
    TSDFVolume tsdfVolume(resolution, resolution, resolution, voxelSize);
    std::cout << "Integrate tsdfVolume..." << std::endl;
    // Build TSDF using the first frame
    tsdfVolume.integrate(target[0], 0.1f);    
    std::cout << "tsdfVolume integrated..." << std::endl;


    int i = 0;
    const int iMax = 10;
    while (sensor.processNextFrame() && i < iMax) {
        // Estimate the current camera pose from source to target mesh with ICP
        // optimization. We downsample the source image to speed up the
        // correspondence matching.
        std::vector<PointCloud> source = createPointClouds(sensor, LEVEL);
      

        // TODO: Track camera pose and then get the target image from the TSDF
        // from that pose.
        // TODO: Replace target with a raycasted image from the TSDF volume.
        // PointCloud target = ray_marching(tsdfVolume, sensor,
        // estimatedPoses.back());

        for (int it = LEVEL-1; it >= 0; it--) {
            optimizer->setNbOfIterations(iterations[it]);
            optimizer->estimatePose(source[it], target[it],
                                    currentCameraToWorld);
        }

        // Invert the transformation matrix to get the current camera pose.
        Matrix4f currentCameraPose = currentCameraToWorld.inverse();
        std::cout << "Current camera pose: " << std::endl
                  << currentCameraPose << std::endl;
        estimatedPoses.push_back(currentCameraPose);

        Matrix4f cameraToWorld = currentCameraPose.inverse();
        tsdfVolume.integrate(source[0], 0.1f);
        // Replace target (reference frame) with source (current) frame

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

    tsdfVolume.storeAsOff(filenameBaseOut);

    delete optimizer;

    return 0;
}

int main(int argc, char *argv[]) {
   try {
        cxxopts::Options options(argv[0], " - command line options");
        options.allow_unrecognised_options().add_options()(
            "d,dataset", "Path to the dataset", cxxopts::value<std::string>())(
            "o,output", "Base output filename", cxxopts::value<std::string>())(
            "r,resolution", "TSDF resolution", cxxopts::value<int>())(
            "v,voxel", "TSDF voxel size", cxxopts::value<float>())(
            "h,help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        std::string datasetPath = "../data/rgbd_dataset_freiburg1_xyz/";
   
        std::string filenameBaseOut = "mesh_";
   

        int resolution = 512;
     

        float voxelSize = 0.001f;  // meters
     
        std::cout << "Dataset Path: " << datasetPath << std::endl;
        std::cout << "Base Output Filename: " << filenameBaseOut << std::endl;
       return run(datasetPath, filenameBaseOut, resolution, voxelSize);
    } catch (cxxopts::exceptions::option_has_no_value &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
#include <iostream>
#include <random>

#include "Eigen.h"
#include "MeshWriter.h"
#include "ProcrustesAligner.h"
#include "SimpleMesh.h"
#include "Vertex.h"
#include "VirtualSensor.h"
#include "cxxopts.hpp"

#define NUMBER_OF_POINTS 10

int processFrame(VirtualSensor &sensor, const std::string &filenameBaseOut, SimpleMesh &mainMesh, int frameCnt) {
    std::cout << "Processing frame " << frameCnt << std::endl;
    std::cout << "Main mesh size at start: " << mainMesh.getVertices().size() << std::endl;

    // get ptr to the current depth frame
    // depth is stored in row major (get dimensions via
    // sensor.GetDepthImageWidth() / GetDepthImageHeight())
    float *depthMap = sensor.GetDepth();
    // get ptr to the current color frame
    // color is stored as RGBX in row major (4 byte values per pixel, get
    // dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
    BYTE *colorMap = sensor.GetColorRGBX();

    // get depth intrinsics
    Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
    Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();

    float fX = depthIntrinsics(0, 0);
    float fY = depthIntrinsics(1, 1);
    float cX = depthIntrinsics(0, 2);
    float cY = depthIntrinsics(1, 2);

    // compute inverse depth extrinsics
    Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();

    Matrix4f trajectory = sensor.GetTrajectory();
    Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();

    // back-projection
    // write result to the vertices array below, keep pixel ordering!
    // if the depth value at idx is invalid (MINF) write the following
    // values to the vertices array vertices[idx].position = Vector4f(MINF,
    // MINF, MINF, MINF); vertices[idx].color = Vector4uc(0,0,0,0);
    // otherwise apply back-projection and transform the vertex to world
    // space, use the corresponding color from the colormap
    std::vector<Vertex> vertices;
    for (unsigned int y = 0; y < sensor.GetDepthImageHeight(); ++y) {
        for (unsigned int x = 0; x < sensor.GetDepthImageWidth(); ++x) {
            unsigned int idx = y * sensor.GetDepthImageWidth() + x;
            float depth = depthMap[idx];
            if (depth != MINF) {
                Vector4f p = Vector4f(depth * (x - cX) / fX,
                                      depth * (y - cY) / fY, depth, 1.0f);
                Vector4f p_world = trajectoryInv * depthExtrinsicsInv * p;
                Vertex v;
                v.position = p_world;
                v.color =
                    Vector4uc(colorMap[idx * 4], colorMap[idx * 4 + 1],
                              colorMap[idx * 4 + 2], colorMap[idx * 4 + 3]);
                vertices.push_back(v);
            }
        }
    }

    // merge meshes
    SimpleMesh currentMesh;
    for (const Vertex & v : vertices) {
        currentMesh.addVertex(v);
    }

    bool firstFrame = mainMesh.getVertices().empty();
    if (firstFrame) {
        std::cout << "First frame" << std::endl;
        mainMesh = currentMesh;
        return 0;
    }

    // Fill in the matched points: sourcePoints[i] is matched with
    // targetPoints[i].
    std::vector<Vector3f> sourcePoints;
    std::vector<Vector3f> targetPoints;

    std::default_random_engine generator; // NOLINT(*-msc51-cpp)
    std::uniform_int_distribution<unsigned int> distributionMain(
        0, mainMesh.getVertices().size() - 1);
    std::uniform_int_distribution<unsigned int> distributionCurrent(
        0, currentMesh.getVertices().size() - 1);

    for (int i = 0; i < NUMBER_OF_POINTS; ++i) {
        auto randomMainIndex = distributionMain(generator);
        auto randomCurrentIndex = distributionCurrent(generator);
        auto src = mainMesh.getVertexPosition3f(randomMainIndex);
        auto tgt = currentMesh.getVertexPosition3f(randomCurrentIndex);
        sourcePoints.push_back(src);
        targetPoints.push_back(tgt);
    }

    // Estimate the pose from source to target mesh with Procrustes alignment.
    ProcrustesAligner aligner;
    Matrix4f estimatedPose = aligner.estimatePose(sourcePoints, targetPoints);

    // Visualize the resulting joined mesh. We add triangulated spheres for
    // point matches.
    // TODO: change to join into mainMesh obj instead of creating new one
    SimpleMesh resultingMesh =
        SimpleMesh::joinMeshes(mainMesh, currentMesh, estimatedPose);
    mainMesh = resultingMesh;

    std::cout << "Main mesh size at end: " << mainMesh.getVertices().size() << std::endl;
    return 0;
}

int run(const std::string &datasetPath, const std::string &filenameBaseOut) {
    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.Init(datasetPath)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!"
                  << std::endl;
        return -1;
    }

    // create main mesh
    SimpleMesh mainMesh;

    // convert video to meshes
    int frameCnt = 0;
    while (sensor.ProcessNextFrame()) {
        int r = processFrame(sensor, filenameBaseOut, mainMesh, frameCnt);
        if (r != 0) {
            return r;
        }
        frameCnt++;

        if (frameCnt > 1) {
            break;
        }
    }

    // TODO: write mesh file
    std::stringstream ss;
    ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
    if (!WriteMesh(mainMesh.getVertices(), sensor.GetDepthImageWidth(),
                   sensor.GetDepthImageHeight(), ss.str())) {
        std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
        return -1;
    }

    std::cout << "Finished!" << std::endl;

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
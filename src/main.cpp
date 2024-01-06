#include <iostream>

#include "Eigen.h"
#include "MeshWriter.h"
#include "ProcrustesAligner.h"
#include "SimpleMesh.h"
#include "Vertex.h"
#include "VirtualSensor.h"
#include "cxxopts.hpp"

int processFrame(VirtualSensor &sensor, const std::string &filenameBaseOut,
                 SimpleMesh &mainMesh) {
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
    auto *vertices =
        new Vertex[sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight()];
    for (unsigned int y = 0; y < sensor.GetDepthImageHeight(); ++y) {
        for (unsigned int x = 0; x < sensor.GetDepthImageWidth(); ++x) {
            unsigned int idx = y * sensor.GetDepthImageWidth() + x;
            float depth = depthMap[idx];
            if (depth == MINF) {
                vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
                vertices[idx].color = Vector4uc(0, 0, 0, 0);
            } else {
                Vector4f p = Vector4f(depth * (x - cX) / fX,
                                      depth * (y - cY) / fY, depth, 1.0f);
                Vector4f p_world = trajectoryInv * depthExtrinsicsInv * p;
                vertices[idx].position = p_world;
                vertices[idx].color =
                    Vector4uc(colorMap[idx * 4], colorMap[idx * 4 + 1],
                              colorMap[idx * 4 + 2], colorMap[idx * 4 + 3]);
            }
        }
    }

    // write mesh file
    std::stringstream ss;
    ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
    if (!WriteMesh(vertices, sensor.GetDepthImageWidth(),
                   sensor.GetDepthImageHeight(), ss.str())) {
        std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
        return -1;
    }

    // merge meshes
    SimpleMesh currentMesh;
    // TODO: load vertices into a SimpleMesh object

    // Fill in the matched points: sourcePoints[i] is matched with
    // targetPoints[i].
    // TODO: select random points from the meshes and fill in the source and
    // target points.
    std::vector<Vector3f> sourcePoints;
    sourcePoints.push_back(
        Vector3f(-0.02744f, 0.179958f, 0.00980739f));  // left ear
    sourcePoints.push_back(
        Vector3f(-0.0847672f, 0.180632f, -0.0148538f));  // right ear
    sourcePoints.push_back(
        Vector3f(0.0544159f, 0.0715162f, 0.0231181f));  // tail
    sourcePoints.push_back(
        Vector3f(-0.0854079f, 0.10966f, 0.0842135f));  // mouth

    std::vector<Vector3f> targetPoints;
    targetPoints.push_back(
        Vector3f(-0.0106867f, 0.179756f, -0.0283248f));  // left ear
    targetPoints.push_back(
        Vector3f(-0.0639191f, 0.179114f, -0.0588715f));  // right ear
    targetPoints.push_back(
        Vector3f(0.0590575f, 0.066407f, 0.00686641f));  // tail
    targetPoints.push_back(
        Vector3f(-0.0789843f, 0.13256f, 0.0519517f));  // mouth

    // Estimate the pose from source to target mesh with Procrustes alignment.
    ProcrustesAligner aligner;
    Matrix4f estimatedPose = aligner.estimatePose(sourcePoints, targetPoints);

    // Visualize the resulting joined mesh. We add triangulated spheres for
    // point matches.
    // TODO: change to join into mainMesh obj instead of creating new one
    SimpleMesh resultingMesh =
        SimpleMesh::joinMeshes(mainMesh, currentMesh, estimatedPose);
    for (const auto &sourcePoint : sourcePoints) {
        resultingMesh =
            SimpleMesh::joinMeshes(SimpleMesh::sphere(sourcePoint, 0.002f),
                                   resultingMesh, estimatedPose);
    }
    for (const auto &targetPoint : targetPoints) {
        resultingMesh =
            SimpleMesh::joinMeshes(SimpleMesh::sphere(targetPoint, 0.002f),
                                   resultingMesh, Matrix4f::Identity());
    }

    // free mem
    delete[] vertices;

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
    while (sensor.ProcessNextFrame()) {
        int r = processFrame(sensor, filenameBaseOut, mainMesh);
        if (r != 0) {
            return r;
        }
    }

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
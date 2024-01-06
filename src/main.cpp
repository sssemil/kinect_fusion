#include <iostream>

#include "Eigen.h"
#include "MeshWriter.h"
#include "Vertex.h"
#include "VirtualSensor.h"
#include "cxxopts.hpp"

int run(const std::string& datasetPath, const std::string& filenameBaseOut) {
    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.Init(datasetPath)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!"
                  << std::endl;
        return -1;
    }

    // convert video to meshes
    while (sensor.ProcessNextFrame()) {
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
        auto *vertices = new Vertex[sensor.GetDepthImageWidth() *
                                    sensor.GetDepthImageHeight()];
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

        // free mem
        delete[] vertices;
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

        std::string datasetPath;
        if (result.count("dataset")) {
            datasetPath = result["dataset"].as<std::string>();
        } else {
            std::cerr
                << "Dataset path not provided. Use -d or --dataset option."
                << std::endl;
            return -1;
        }

        std::string filenameBaseOut = "mesh_";  // Default value
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
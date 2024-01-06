#include <iostream>

#include "Eigen.h"
#include "VirtualSensor.h"

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position;
    // color stored as 4 unsigned char
    Vector4uc color;
};

using VertexAction = std::function<void(const Vertex &, const Vertex &,
                                        const Vertex &, const Vertex &)>;

bool areAllEdgesValid(const Vertex *vs[4], float edgeThreshold) {
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if ((vs[i]->position - vs[j]->position).norm() >= edgeThreshold) {
                return false;
            }
        }
    }
    return true;
}

void processVertex(unsigned int idx, unsigned int width, Vertex *vertices,
                   float edgeThreshold, const VertexAction &action) {
    const Vertex &v0 = vertices[idx];
    const Vertex &v1 = vertices[idx + 1];
    const Vertex &v2 = vertices[idx + width];
    const Vertex &v3 = vertices[idx + width + 1];

    if (v0.position.x() != MINF && v1.position.x() != MINF &&
        v2.position.x() != MINF && v3.position.x() != MINF) {
        const Vertex *vs[4] = {&v0, &v1, &v2, &v3};

        if (areAllEdgesValid(vs, edgeThreshold)) {
            // Call the action function
            action(v0, v1, v2, v3);
        }
    }
}

bool WriteMesh(Vertex *vertices, unsigned int width, unsigned int height,
               const std::string &filename) {
    float edgeThreshold = 0.01f;  // 1cm

    // use the OFF file format to save the vertices grid
    // (http://www.geomview.org/docs/html/OFF.html)
    // - have a look at the "off_sample.off" file to see how to store the
    // vertices and triangles
    // - for debugging we recommend to first only write out the vertices (set
    // the number of faces to zero)
    // - for simplicity write every vertex to file, even if it is not valid
    // (position.x() == MINF) (note that all vertices in the off file have to be
    // valid, thus, if a point is not valid write out a dummy point like
    // (0,0,0))
    // - use a simple triangulation exploiting the grid structure (neighboring
    // vertices build a triangle, two triangles per grid cell)
    // - you can use an arbitrary triangulation of the cells, but make sure that
    // the triangles are consistently oriented
    // - only write triangles with valid vertices and an edge length smaller
    // then edgeThreshold

    // Get number of vertices
    unsigned int nVertices = width * height;

    // Determine number of valid faces
    unsigned nFaces = 0;
    VertexAction countFaces = [&nFaces](const Vertex &, const Vertex &,
                                        const Vertex &,
                                        const Vertex &) { nFaces += 2; };
    for (unsigned int y = 0; y < height - 1; ++y) {
        for (unsigned int x = 0; x < width - 1; ++x) {
            unsigned int idx = y * width + x;
            processVertex(idx, width, vertices, edgeThreshold, countFaces);
        }
    }

    // Write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) return false;

    // write header
    outFile << "COFF" << std::endl;

    outFile << "# numVertices numFaces numEdges" << std::endl;

    outFile << nVertices << " " << nFaces << " 0" << std::endl;

    // Example off file:
    // COFF
    // # numVertices numFaces numEdges
    // 4 2 0
    // # list of vertices
    // # X Y Z R G B A
    // 0.0 1.0 0.0 255 255 255 255
    // 0.0 0.0 0.0 255 255 255 255
    // 1.0 0.0 0.0 255 255 255 255
    // 1.0 1.0 0.0 255 255 255 255
    // # list of faces
    // # nVerticesPerFace idx0 idx1 idx2 ...
    // 3 0 1 2
    // 3 0 2 3

    // save vertices
    outFile << "# list of vertices" << std::endl;
    outFile << "# X Y Z R G B A" << std::endl;
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            unsigned int idx = y * width + x;
            Vertex v = vertices[idx];
            if (v.position.x() != MINF) {
                outFile << v.position.x() << " " << v.position.y() << " "
                        << v.position.z() << " " << (int)v.color.x() << " "
                        << (int)v.color.y() << " " << (int)v.color.z() << " "
                        << (int)v.color.w() << std::endl;
            } else {
                outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
            }
        }
    }

    // save valid faces
    std::cout << "# list of faces" << std::endl;
    std::cout << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;

    VertexAction writeFaces = [&outFile, &vertices](
                                  const Vertex &v0, const Vertex &v1,
                                  const Vertex &v2, const Vertex &v3) {
        unsigned int idx0 = &v0 - &vertices[0];
        unsigned int idx1 = &v1 - &vertices[0];
        unsigned int idx2 = &v2 - &vertices[0];
        unsigned int idx3 = &v3 - &vertices[0];
        outFile << "3 " << idx0 << " " << idx1 << " " << idx2 << std::endl;
        outFile << "3 " << idx1 << " " << idx3 << " " << idx2 << std::endl;
    };

    for (unsigned int y = 0; y < height - 1; ++y) {
        for (unsigned int x = 0; x < width - 1; ++x) {
            unsigned int idx = y * width + x;
            processVertex(idx, width, vertices, edgeThreshold, writeFaces);
        }
    }

    // close file
    outFile.close();

    return true;
}

int main() {
    // Make sure this path points to the data folder
    std::string filenameIn = "../../Data/rgbd_dataset_freiburg1_xyz/";
    std::string filenameBaseOut = "mesh_";

    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.Init(filenameIn)) {
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
        Vertex *vertices = new Vertex[sensor.GetDepthImageWidth() *
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
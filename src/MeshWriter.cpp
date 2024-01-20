#include "MeshWriter.h"

#include <fstream>
#include <iostream>

#include "MeshUtils.h"

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
            ProcessVertex(idx, width, vertices, edgeThreshold, countFaces);
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
            ProcessVertex(idx, width, vertices, edgeThreshold, writeFaces);
        }
    }

    // close file
    outFile.close();

    return true;
}

bool WriteMesh(const std::vector<Vertex>& vertices, unsigned int width, unsigned int height, const std::string &filename) {
    float edgeThreshold = 0.01f;  // 1cm

    // Get number of vertices
    unsigned int nVertices = vertices.size();

    // Determine number of valid faces
    unsigned nFaces = 0;
    for (unsigned int y = 0; y < height - 1; ++y) {
        for (unsigned int x = 0; x < width - 1; ++x) {
            unsigned int idx = y * width + x;
            if (idx < nVertices - width - 1) {
                // Process each quad to count valid faces
                // Assuming ProcessVertex function is adapted to work with vector
                ProcessVertex(idx, width, vertices, edgeThreshold, [&nFaces](const Vertex &, const Vertex &, const Vertex &, const Vertex &) { nFaces += 2; });
            }
        }
    }

    // Write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) return false;

    // write header
    outFile << "COFF" << std::endl;
    outFile << nVertices << " " << nFaces << " 0" << std::endl;

    // save vertices
    for (const Vertex& v : vertices) {
        if (v.position.x() != MINF) {
            outFile << v.position.x() << " " << v.position.y() << " " << v.position.z() << " " << (int)v.color.x() << " " << (int)v.color.y() << " " << (int)v.color.z() << " " << (int)v.color.w() << std::endl;
        } else {
            outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
        }
    }

    // save valid faces
    for (unsigned int y = 0; y < height - 1; ++y) {
        for (unsigned int x = 0; x < width - 1; ++x) {
            unsigned int idx = y * width + x;
            if (idx < nVertices - width - 1) {
                // Process each quad to write valid faces
                // Assuming ProcessVertex function is adapted to work with vector
                ProcessVertex(idx, width, vertices, edgeThreshold, [&outFile, &vertices](const Vertex &v0, const Vertex &v1, const Vertex &v2, const Vertex &v3) {
                    unsigned int idx0 = &v0 - &vertices[0];
                    unsigned int idx1 = &v1 - &vertices[0];
                    unsigned int idx2 = &v2 - &vertices[0];
                    unsigned int idx3 = &v3 - &vertices[0];
                    outFile << "3 " << idx0 << " " << idx1 << " " << idx2 << std::endl;
                    outFile << "3 " << idx1 << " " << idx3 << " " << idx2 << std::endl;
                });
            }
        }
    }

    // close file
    outFile.close();

    return true;
}

#pragma once

#include <fstream>
#include <iostream>

#include "Eigen.h"
#include "Vertex.h"

struct Triangle {
    unsigned int idx0;
    unsigned int idx1;
    unsigned int idx2;

    Triangle() : idx0{0}, idx1{0}, idx2{0} {}

    Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2)
        : idx0(_idx0), idx1(_idx1), idx2(_idx2) {}
};

class SimpleMesh {
   public:
    SimpleMesh() {}

    void clear() {
        m_vertices.clear();
        m_triangles.clear();
    }

    unsigned int addVertex(Vertex& vertex) {
        unsigned int vId = (unsigned int)m_vertices.size();
        m_vertices.push_back(vertex);
        return vId;
    }

    unsigned int addFace(unsigned int idx0, unsigned int idx1,
                         unsigned int idx2) {
        unsigned int fId = (unsigned int)m_triangles.size();
        Triangle triangle(idx0, idx1, idx2);
        m_triangles.push_back(triangle);
        return fId;
    }

    std::vector<Vertex>& getVertices() { return m_vertices; }

    const std::vector<Vertex>& getVertices() const { return m_vertices; }

    std::vector<Triangle>& getTriangles() { return m_triangles; }

    const std::vector<Triangle>& getTriangles() const { return m_triangles; }

    void transform(const Matrix4f& transformation) {
        for (Vertex& v : m_vertices) {
            v.position = transformation * v.position;
        }
    }

    bool loadMesh(const std::string& filename) {
        // Read off file (Important: Only .off files are supported).
        m_vertices.clear();
        m_triangles.clear();

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "Mesh file wasn't read successfully." << std::endl;
            return false;
        }

        // First line should say 'COFF'.
        char string1[5];
        file >> string1;

        // Read header.
        unsigned int numV = 0;
        unsigned int numP = 0;
        unsigned int numE = 0;
        file >> numV >> numP >> numE;

        m_vertices.reserve(numV);
        m_triangles.reserve(numP);

        // Read vertices.
        if (std::string(string1).compare("COFF") == 0) {
            // We have color information.
            for (unsigned int i = 0; i < numV; i++) {
                Vertex v;
                file >> v.position.x() >> v.position.y() >> v.position.z();
                v.position.w() = 1.f;
                // Colors are stored as integers. We need to convert them.
                Vector4i colorInt;
                file >> colorInt.x() >> colorInt.y() >> colorInt.z() >>
                    colorInt.w();
                v.color = Vector4uc(
                    (unsigned char)colorInt.x(), (unsigned char)colorInt.y(),
                    (unsigned char)colorInt.z(), (unsigned char)colorInt.w());
                m_vertices.push_back(v);
            }
        } else if (std::string(string1).compare("OFF") == 0) {
            // We only have vertex information.
            for (unsigned int i = 0; i < numV; i++) {
                Vertex v;
                file >> v.position.x() >> v.position.y() >> v.position.z();
                v.position.w() = 1.f;
                v.color.x() = 0;
                v.color.y() = 0;
                v.color.z() = 0;
                v.color.w() = 255;
                m_vertices.push_back(v);
            }
        } else {
            std::cout << "Incorrect mesh file type." << std::endl;
            return false;
        }

        // Read faces (i.e. triangles).
        for (unsigned int i = 0; i < numP; i++) {
            unsigned int num_vs;
            file >> num_vs;
            ASSERT(num_vs == 3 && "We can only read triangular mesh.");

            Triangle t;
            file >> t.idx0 >> t.idx1 >> t.idx2;
            m_triangles.push_back(t);
        }

        return true;
    }

    bool writeMesh(const std::string& filename) {
        // Write off file.
        std::ofstream outFile(filename);
        if (!outFile.is_open()) return false;

        // Write header.
        outFile << "COFF" << std::endl;
        outFile << m_vertices.size() << " " << m_triangles.size() << " 0"
                << std::endl;

        // Save vertices.
        for (unsigned int i = 0; i < m_vertices.size(); i++) {
            const auto& vertex = m_vertices[i];
            if (vertex.position.allFinite())
                outFile << vertex.position.x() << " " << vertex.position.y()
                        << " " << vertex.position.z() << " "
                        << int(vertex.color.x()) << " " << int(vertex.color.y())
                        << " " << int(vertex.color.z()) << " "
                        << int(vertex.color.w()) << std::endl;
            else
                outFile << "0.0 0.0 0.0 0 0 0 0" << std::endl;
        }

        // Save faces.
        for (unsigned int i = 0; i < m_triangles.size(); i++) {
            outFile << "3 " << m_triangles[i].idx0 << " " << m_triangles[i].idx1
                    << " " << m_triangles[i].idx2 << std::endl;
        }

        // Close file.
        outFile.close();

        return true;
    }

    /**
     * Joins two meshes together by putting them into the common mesh and
     * transforming the vertex positions of mesh1 with transformation
     * 'pose1to2'.
     */
    static SimpleMesh joinMeshes(const SimpleMesh& mesh1,
                                 const SimpleMesh& mesh2,
                                 Matrix4f pose1to2 = Matrix4f::Identity()) {
        SimpleMesh joinedMesh;
        const auto& vertices1 = mesh1.getVertices();
        const auto& triangles1 = mesh1.getTriangles();
        const auto& vertices2 = mesh2.getVertices();
        const auto& triangles2 = mesh2.getTriangles();

        auto& joinedVertices = joinedMesh.getVertices();
        auto& joinedTriangles = joinedMesh.getTriangles();

        const unsigned nVertices1 = static_cast<unsigned>(vertices1.size());
        const unsigned nVertices2 = static_cast<unsigned>(vertices2.size());
        joinedVertices.reserve(nVertices1 + nVertices2);

        const unsigned nTriangles1 = static_cast<unsigned>(triangles1.size());
        const unsigned nTriangles2 = static_cast<unsigned>(triangles2.size());
        joinedTriangles.reserve(nVertices1 + nVertices2);

        // Add all vertices (we need to transform vertices of mesh 1).
        for (auto i = 0u; i < nVertices1; ++i) {
            const auto& v1 = vertices1[i];
            Vertex v;
            v.position = pose1to2 * v1.position;
            v.color = v1.color;
            joinedVertices.push_back(v);
        }
        for (auto i = 0u; i < nVertices2; ++i)
            joinedVertices.push_back(vertices2[i]);

        // Add all faces (the indices of the second mesh need to be added an
        // offset).
        for (auto i = 0u; i < nTriangles1; ++i)
            joinedTriangles.push_back(triangles1[i]);
        for (auto i = 0u; i < nTriangles2; ++i) {
            const auto& t2 = triangles2[i];
            Triangle t{t2.idx0 + nVertices1, t2.idx1 + nVertices1,
                       t2.idx2 + nVertices1};
            joinedTriangles.push_back(t);
        }

        return joinedMesh;
    }

    /**
     * Generates a sphere around the given center point.
     */
    static SimpleMesh sphere(Vector3f center, float scale = 1.f,
                             Vector4uc color = {0, 0, 255, 255}) {
        SimpleMesh mesh;
        Vector4f centerHomogenous =
            Vector4f{center.x(), center.y(), center.z(), 1.f};

        // These are precomputed values for sphere aproximation.
        const std::vector<double> vertexComponents = {
            -0.525731, 0,         0.850651,  0.525731,  0,         0.850651,
            -0.525731, 0,         -0.850651, 0.525731,  0,         -0.850651,
            0,         0.850651,  0.525731,  0,         0.850651,  -0.525731,
            0,         -0.850651, 0.525731,  0,         -0.850651, -0.525731,
            0.850651,  0.525731,  0,         -0.850651, 0.525731,  0,
            0.850651,  -0.525731, 0,         -0.850651, -0.525731, 0};
        const std::vector<unsigned> faceIndices = {
            0, 4,  1,  0, 9, 4,  9, 5,  4, 4,  5, 8, 4, 8, 1,
            8, 10, 1,  8, 3, 10, 5, 3,  8, 5,  2, 3, 2, 7, 3,
            7, 10, 3,  7, 6, 10, 7, 11, 6, 11, 0, 6, 0, 1, 6,
            6, 1,  10, 9, 0, 11, 9, 11, 2, 9,  2, 5, 7, 2, 11};

        // Add vertices.
        for (int i = 0; i < 12; ++i) {
            Vertex v;
            v.position =
                centerHomogenous +
                scale * Vector4f{float(vertexComponents[3 * i + 0]),
                                 float(vertexComponents[3 * i + 1]),
                                 float(vertexComponents[3 * i + 2]), 0.f};
            v.color = color;
            mesh.addVertex(v);
        }

        // Add faces.
        for (int i = 0; i < 20; ++i) {
            mesh.addFace(faceIndices[3 * i + 0], faceIndices[3 * i + 1],
                         faceIndices[3 * i + 2]);
        }

        return mesh;
    }

    /**
     * Generates a camera object with a given pose.
     */
    static SimpleMesh camera(const Matrix4f& cameraPose, float scale = 1.f,
                             Vector4uc color = {255, 0, 0, 255}) {
        SimpleMesh mesh;
        Matrix4f cameraToWorld = cameraPose.inverse();

        // These are precomputed values for sphere aproximation.
        std::vector<double> vertexComponents = {
            25,       25,        0,        -50,       50,        100,
            49.99986, 49.9922,   99.99993, -24.99998, 25.00426,  0.005185,
            25.00261, -25.00023, 0.004757, 49.99226,  -49.99986, 99.99997,
            -50,      -50,       100,      -25.00449, -25.00492, 0.019877};
        const std::vector<unsigned> faceIndices = {
            1, 2, 3, 2, 0, 3, 2, 5, 4, 4, 0, 2, 5, 6, 7, 7, 4, 5,
            6, 1, 7, 1, 3, 7, 3, 0, 4, 7, 3, 4, 5, 2, 1, 5, 1, 6};

        // Add vertices.
        for (int i = 0; i < 8; ++i) {
            Vertex v;
            v.position =
                cameraToWorld *
                Vector4f{scale * float(vertexComponents[3 * i + 0]),
                         scale * float(vertexComponents[3 * i + 1]),
                         scale * float(vertexComponents[3 * i + 2]), 1.f};
            v.color = color;
            mesh.addVertex(v);
        }

        // Add faces.
        for (int i = 0; i < 12; ++i) {
            mesh.addFace(faceIndices[3 * i + 0], faceIndices[3 * i + 1],
                         faceIndices[3 * i + 2]);
        }

        return mesh;
    }

    /**
     * Generates a cylinder, ranging from point p0 to point p1.
     */
    static SimpleMesh cylinder(const Vector3f& p0, const Vector3f& p1,
                               float radius, unsigned stacks, unsigned slices,
                               const Vector4uc color = Vector4uc{0, 0, 255,
                                                                 255}) {
        SimpleMesh mesh;
        auto& vertices = mesh.getVertices();
        auto& triangles = mesh.getTriangles();

        vertices.resize((stacks + 1) * slices);
        triangles.resize(stacks * slices * 2);

        float height = (p1 - p0).norm();

        unsigned vIndex = 0;
        for (unsigned i = 0; i <= stacks; i++)
            for (unsigned i2 = 0; i2 < slices; i2++) {
                auto& v = vertices[vIndex++];
                float theta = float(i2) * 2.0f * float(M_PI) / float(slices);
                v.position =
                    Vector4f{p0.x() + radius * cosf(theta),
                             p0.y() + radius * sinf(theta),
                             p0.z() + height * float(i) / float(stacks), 1.f};
                v.color = color;
            }

        unsigned iIndex = 0;
        for (unsigned i = 0; i < stacks; i++)
            for (unsigned i2 = 0; i2 < slices; i2++) {
                int i2p1 = (i2 + 1) % slices;

                triangles[iIndex].idx0 = (i + 1) * slices + i2;
                triangles[iIndex].idx1 = i * slices + i2;
                triangles[iIndex].idx2 = i * slices + i2p1;

                triangles[iIndex + 1].idx0 = (i + 1) * slices + i2;
                triangles[iIndex + 1].idx1 = i * slices + i2p1;
                triangles[iIndex + 1].idx2 = (i + 1) * slices + i2p1;

                iIndex += 2;
            }

        Matrix4f transformation = Matrix4f::Identity();
        transformation.block(0, 0, 3, 3) = face(Vector3f{0, 0, 1}, p1 - p0);
        transformation.block(0, 3, 3, 1) = p0;
        mesh.transform(transformation);

        return mesh;
    }

   private:
    std::vector<Vertex> m_vertices;
    std::vector<Triangle> m_triangles;

    /**
     * Returns a rotation that transforms vector vA into vector vB.
     */
    static Matrix3f face(const Vector3f& vA, const Vector3f& vB) {
        auto a = vA.normalized();
        auto b = vB.normalized();
        auto axis = b.cross(a);
        float angle = acosf(a.dot(b));

        if (angle == 0.0f) {  // No rotation
            return Matrix3f::Identity();
        }

        // Convert the rotation from SO3 to matrix notation.
        // First we create a skew symetric matrix from the axis vector.
        Matrix3f skewSymetricMatrix;
        skewSymetricMatrix.setIdentity();
        skewSymetricMatrix(0, 0) = 0;
        skewSymetricMatrix(0, 1) = -axis.z();
        skewSymetricMatrix(0, 2) = axis.y();
        skewSymetricMatrix(1, 0) = axis.z();
        skewSymetricMatrix(1, 1) = 0;
        skewSymetricMatrix(1, 2) = -axis.x();
        skewSymetricMatrix(2, 0) = -axis.y();
        skewSymetricMatrix(2, 1) = axis.x();
        skewSymetricMatrix(2, 2) = 0;

        // We compute a rotation matrix using Rodrigues formula.
        Matrix3f rotation =
            Matrix3f::Identity() + sinf(angle) * skewSymetricMatrix +
            (1 - cos(angle)) * skewSymetricMatrix * skewSymetricMatrix;

        return rotation;
    }
};

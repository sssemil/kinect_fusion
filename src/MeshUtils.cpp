#include "MeshUtils.h"

bool AreAllEdgesValid(const Vertex *vs[4], float edgeThreshold) {
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if ((vs[i]->position - vs[j]->position).norm() >= edgeThreshold) {
                return false;
            }
        }
    }
    return true;
}

void ProcessVertex(unsigned int idx, unsigned int width, Vertex *vertices,
                   float edgeThreshold, const VertexAction &action) {
    const Vertex &v0 = vertices[idx];
    const Vertex &v1 = vertices[idx + 1];
    const Vertex &v2 = vertices[idx + width];
    const Vertex &v3 = vertices[idx + width + 1];

    if (v0.position.x() != MINF && v1.position.x() != MINF &&
        v2.position.x() != MINF && v3.position.x() != MINF) {
        const Vertex *vs[4] = {&v0, &v1, &v2, &v3};

        if (AreAllEdgesValid(vs, edgeThreshold)) {
            // Call the action function
            action(v0, v1, v2, v3);
        }
    }
}

void ProcessVertex(unsigned int idx, unsigned int width, const std::vector<Vertex>& vertices,
                   float edgeThreshold, const VertexAction &action) {
    // Ensure index is within bounds to prevent out-of-range access
    if (idx + width + 1 >= vertices.size()) return;

    const Vertex &v0 = vertices.at(idx);
    const Vertex &v1 = vertices.at(idx + 1);
    const Vertex &v2 = vertices.at(idx + width);
    const Vertex &v3 = vertices.at(idx + width + 1);

    if (v0.position.x() != MINF && v1.position.x() != MINF &&
        v2.position.x() != MINF && v3.position.x() != MINF) {
        const Vertex *vs[4] = {&v0, &v1, &v2, &v3};

        if (AreAllEdgesValid(vs, edgeThreshold)) {
            // Call the action function
            action(v0, v1, v2, v3);
        }
    }
}

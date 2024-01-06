#pragma once

#include "Eigen.h"

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vector4f position;  // position stored as 4 floats (4th component is 1.0)
    Vector4uc color;    // color stored as 4 unsigned char
};

using VertexAction = std::function<void(const Vertex&, const Vertex&,
                                        const Vertex&, const Vertex&)>;

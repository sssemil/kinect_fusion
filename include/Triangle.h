#pragma once

#include <string>
#include <vector>

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

#pragma once

#include <string>
#include <vector>

#include "Eigen.h"
#include "Vertex.h"

struct Triangle {
    unsigned int idx0;
    unsigned int idx1;
    unsigned int idx2;

    Triangle();
    Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2);
};

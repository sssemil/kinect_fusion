#ifndef FUSIONALGORITHM_H
#define FUSIONALGORITHM_H

#include "DepthFrame.h"

class FusionAlgorithm {
public:
    FusionAlgorithm();
    ~FusionAlgorithm();

    void processDepthData(const DepthFrame& depthData);
};

#endif // FUSIONALGORITHM_H

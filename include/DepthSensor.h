#ifndef DEPTHSENSOR_H
#define DEPTHSENSOR_H

#include "DepthFrame.h"

class DepthSensor {
public:
    DepthSensor();
    ~DepthSensor();

    DepthFrame getNextFrame();
};

#endif // DEPTHSENSOR_H

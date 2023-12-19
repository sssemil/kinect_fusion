#include "DepthSensor.h"
#include "FusionAlgorithm.h"

int main() {
    DepthSensor sensor;
    FusionAlgorithm fusion;

    while (true) {
        auto depthData = sensor.getNextFrame();
        fusion.processDepthData(depthData);
    }

    return 0;
}

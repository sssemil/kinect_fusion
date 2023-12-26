#include "DepthSensor.h"
#include "FusionAlgorithm.h"
#include "SurfaceMeasurement.h"

int main() {
    DepthSensor sensor;
    FusionAlgorithm fusion;

    while (true) {
        auto depthData = sensor.getNextFrame();
        fusion.processDepthData(depthData);
    }

    return 0;
}

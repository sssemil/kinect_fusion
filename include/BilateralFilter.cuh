#pragma once

#include <vector>
#include <cuda_runtime.h>



void applyBilateral(const std::vector<float>& depthmap, int width, int height, std::vector<float>& output);

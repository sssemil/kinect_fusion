#include <BilateralFilter.cuh>

__device__ float bilateral(const float* depthmap, int width, int height, float x, float y) {
    // Perform bilinear interpolation
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height)
        return 0.0f; // Outside the depth map, return default value

    float dx = x - x0;
    float dy = y - y0;

    float q11 = depthmap[y0 * width + x0];
    float q21 = depthmap[y0 * width + x1];
    float q12 = depthmap[y1 * width + x0];
    float q22 = depthmap[y1 * width + x1];

    float result = q11 * (1 - dx) * (1 - dy) + q21 * dx * (1 - dy) + q12 * (1 - dx) * dy + q22 * dx * dy;

    return result;
}

__global__ void applyBilateralKernel(const float* depthmap, int width, int height, float* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[y * width + x] = bilateral(depthmap, width, height, x, y);
    }
}

void applyBilateral(const std::vector<float>&depthmap, int width, int height, std::vector<float>&output) {
    float *dev_depthmap, *dev_output;
    cudaMalloc((void **)&dev_depthmap, width * height * sizeof(float));
    cudaMalloc((void **)&dev_output, width * height * sizeof(float));

    cudaMemcpy(dev_depthmap, depthmap.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    applyBilateralKernel<<<gridSize, blockSize>>>(dev_depthmap, width, height, dev_output);

    cudaMemcpy(output.data(), dev_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_depthmap);
    cudaFree(dev_output);
}


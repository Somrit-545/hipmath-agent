#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", \
                hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void saxpy_kernel(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

__global__ void dot_kernel(const float* x, const float* y, float* result, int n) {
    __shared__ float sdata[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) sdata[threadIdx.x] = x[i] * y[i];
    else    sdata[threadIdx.x] = 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0) result[blockIdx.x] = sdata[threadIdx.x];
}


// Host wrapper
void saxpy_gpu(float a, const float* h_x, float* h_y, int n) {
    float *dev_x, *dev_y;
    size_t size = n * sizeof(float);

    HIP_CHECK(hipMalloc(&dev_x, size));
    HIP_CHECK(hipMalloc(&dev_y, size));

    HIP_CHECK(hipMemcpy(dev_x, h_x, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_y, h_y, size, hipMemcpyHostToDevice));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1)/threadsPerBlock;

    hipLaunchKernelGGL(saxpy_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, 
    a, dev_x, dev_y, n);

    HIP_CHECK(hipMemcpy(h_y, dev_y, size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dev_x)); 
    HIP_CHECK(hipFree(dev_y));
}
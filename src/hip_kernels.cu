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

__global__ void dot_kernel(const float* x, const float* y, float *result, int n) {
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

__global__ void gemv_kernel(float alpha, const float* A, const float* x,
                            float beta, float* y, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
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

float dot_gpu(const float* h_x, const float* h_y, int n) {
    float *dev_x, *dev_y, *result;
    size_t size = n * sizeof(float);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    HIP_CHECK(hipMalloc(&dev_x, size));
    HIP_CHECK(hipMalloc(&dev_y, size));
    HIP_CHECK(hipMalloc(&result, blocksPerGrid * sizeof(float)));

    HIP_CHECK(hipMemcpy(dev_x, h_x, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_y, h_y, size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(dot_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, dev_x, dev_y, result, n);

    // allocate host array and copy device→host
    float* h_partials = (float*)malloc(blocksPerGrid * sizeof(float));
    HIP_CHECK(hipMemcpy(h_partials, result, blocksPerGrid * sizeof(float),
                        hipMemcpyDeviceToHost));

    // NOW sum on host
    float sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) sum += h_partials[i];

    free(h_partials);
    HIP_CHECK(hipFree(dev_x));
    HIP_CHECK(hipFree(dev_y));
    HIP_CHECK(hipFree(result));                                      

    return sum;
}

void gemv_gpu(float alpha, const float* h_A, const float* h_x,
              float beta, float* h_y, int rows, int cols) {
    float *d_A, *d_x, *d_y;
    size_t size_A = (size_t)rows * cols * sizeof(float);
    size_t size_x = (size_t)cols * sizeof(float);
    size_t size_y = (size_t)rows * sizeof(float);

    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_x, size_x));
    HIP_CHECK(hipMalloc(&d_y, size_y));

    HIP_CHECK(hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x, size_x, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y, size_y, hipMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    hipLaunchKernelGGL(gemv_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, alpha, d_A, d_x, beta, d_y, rows, cols);

    HIP_CHECK(hipMemcpy(h_y, d_y, size_y, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
}
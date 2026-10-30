#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <hip/hip_runtime.h>

// Declarations of the GPU wrappers (definitions in hip_kernels.cu)
void saxpy_gpu(float a, const float* h_x, float* h_y, int n);
float dot_gpu(const float* h_x, const float* h_y, int n);
void gemv_gpu(float alpha, const float* h_A, const float* h_x,
              float beta, float* h_y, int rows, int cols);

// Declarations for the kernels (definitions in hip_kernels.cu)
__global__ void saxpy_kernel(float a, const float* x, float* y, int n);
__global__ void dot_kernel(const float* x, const float* y, float* result, int n);

namespace py = pybind11;

// Helper: validate dtype is float32 and array is contiguous
static void check_float32(const py::array_t<float>& arr, const char* name) {
    if (arr.dtype() != py::dtype::of<float>()) {
        throw std::runtime_error(std::string(name) + " must be float32");
    }
}

PYBIND11_MODULE(_hipmath, m) {
    m.doc() = "HIP math library: saxpy, dot, gemv";

    // --- STANDARD API ---

    m.def("saxpy", [](float a, py::array_t<float> x, py::array_t<float> y) {
        check_float32(x, "x");
        check_float32(y, "y");
        if (x.size() != y.size())
            throw std::runtime_error("x and y must have the same size");

        py::array_t<float> out = py::array_t<float>(y.size());
        std::memcpy(out.mutable_data(), y.data(), y.size() * sizeof(float));

        saxpy_gpu(a, x.data(), out.mutable_data(), (int)x.size());
        return out;
    }, py::arg("a"), py::arg("x"), py::arg("y"),
       "Compute y = a*x + y, returns new array");

    m.def("dot", [](py::array_t<float> x, py::array_t<float> y) {
        check_float32(x, "x");
        check_float32(y, "y");
        if (x.size() != y.size())
            throw std::runtime_error("x and y must have the same size");
        return dot_gpu(x.data(), y.data(), (int)x.size());
    }, py::arg("x"), py::arg("y"),
       "Dot product of two 1-D float32 vectors");

    m.def("gemv", [](float alpha, py::array_t<float> A, py::array_t<float> x,
                     float beta, py::array_t<float> y) {
        check_float32(A, "A");
        check_float32(x, "x");
        check_float32(y, "y");

        if (A.ndim() != 2)
            throw std::runtime_error("A must be 2-D");
        int rows = (int)A.shape(0);
        int cols = (int)A.shape(1);

        if (x.size() != cols)
            throw std::runtime_error("x length must equal A.columns");
        if (y.size() != rows)
            throw std::runtime_error("y length must equal A.rows");

        py::array_t<float> out = py::array_t<float>(rows);
        std::memcpy(out.mutable_data(), y.data(), rows * sizeof(float));

        gemv_gpu(alpha, A.data(), x.data(), beta, out.mutable_data(), rows, cols);
        return out;
    }, py::arg("alpha"), py::arg("A"), py::arg("x"),
       py::arg("beta"), py::arg("y"),
       "Compute y = alpha*A*x + beta*y, returns new y");

    // --- BENCHMARK VARIANTS (No WSL2 memcpy overhead) ---

    m.def("saxpy_bench", [](float a, py::array_t<float> x, py::array_t<float> y, int iters) {
        float *d_x, *d_y;
        size_t size = x.size() * sizeof(float);
        
        hipMalloc(&d_x, size); hipMalloc(&d_y, size);
        hipMemcpy(d_x, x.data(), size, hipMemcpyHostToDevice);
        hipMemcpy(d_y, y.data(), size, hipMemcpyHostToDevice);
        
        int threads = 256;
        int blocks = (x.size() + threads - 1) / threads;
        
        // Warmup
        hipLaunchKernelGGL(saxpy_kernel, dim3(blocks), dim3(threads), 0, 0, a, d_x, d_y, (int)x.size());
        hipDeviceSynchronize();
        
        // Timed loop
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            hipLaunchKernelGGL(saxpy_kernel, dim3(blocks), dim3(threads), 0, 0, a, d_x, d_y, (int)x.size());
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        hipFree(d_x); hipFree(d_y);
        
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / iters;
    }, "Benchmark saxpy kernel time (ms)");
    
    m.def("dot_bench", [](py::array_t<float> x, py::array_t<float> y, int iters) {
        float *d_x, *d_y, *d_res;
        size_t size = x.size() * sizeof(float);
        int threads = 256;
        int blocks = (x.size() + threads - 1) / threads;
        
        hipMalloc(&d_x, size); hipMalloc(&d_y, size); hipMalloc(&d_res, blocks * sizeof(float));
        hipMemcpy(d_x, x.data(), size, hipMemcpyHostToDevice);
        hipMemcpy(d_y, y.data(), size, hipMemcpyHostToDevice);
        
        // Warmup
        hipLaunchKernelGGL(dot_kernel, dim3(blocks), dim3(threads), 0, 0, d_x, d_y, d_res, (int)x.size());
        hipDeviceSynchronize();
        
        // Timed loop
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            hipLaunchKernelGGL(dot_kernel, dim3(blocks), dim3(threads), 0, 0, d_x, d_y, d_res, (int)x.size());
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        hipFree(d_x); hipFree(d_y); hipFree(d_res);
        
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / iters;
    }, "Benchmark dot kernel time (ms)");
}
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

// Declarations of the GPU wrappers (definitions in hip_kernels.cu)
void saxpy_gpu(float a, const float* h_x, float* h_y, int n);
float dot_gpu(const float* h_x, const float* h_y, int n);
void gemv_gpu(float alpha, const float* h_A, const float* h_x,
              float beta, float* h_y, int rows, int cols);

namespace py = pybind11;

// Helper: validate dtype is float32 and array is contiguous
static void check_float32(const py::array_t<float>& arr, const char* name) {
    if (arr.dtype() != py::dtype::of<float>()) {
        throw std::runtime_error(std::string(name) + " must be float32");
    }
}

PYBIND11_MODULE(_hipmath, m) {
    m.doc() = "HIP math library: saxpy, dot, gemv";

    // saxpy: returns a new array (copy of y, then modified)
    m.def("saxpy", [](float a, py::array_t<float> x, py::array_t<float> y) {
        check_float32(x, "x");
        check_float32(y, "y");
        if (x.size() != y.size())
            throw std::runtime_error("x and y must have the same size");

        // Copy y so caller's array is untouched
        py::array_t<float> out = py::array_t<float>(y.size());
        std::memcpy(out.mutable_data(), y.data(), y.size() * sizeof(float));

        saxpy_gpu(a, x.data(), out.mutable_data(), (int)x.size());
        return out;
    }, py::arg("a"), py::arg("x"), py::arg("y"),
       "Compute y = a*x + y, returns new array");

    // dot: returns a scalar
    m.def("dot", [](py::array_t<float> x, py::array_t<float> y) {
        check_float32(x, "x");
        check_float32(y, "y");
        if (x.size() != y.size())
            throw std::runtime_error("x and y must have the same size");
        return dot_gpu(x.data(), y.data(), (int)x.size());
    }, py::arg("x"), py::arg("y"),
       "Dot product of two 1-D float32 vectors");

    // gemv: returns a new y array
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
}
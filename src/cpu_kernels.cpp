#include "cpu_kernels.h"

void saxpy_cpu(float a, const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) y[i] = a * x[i] + y[i];
}

float dot_cpu(const float* x, const float* y, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * y[i];
    return sum;
}

void gemv_cpu(float alpha, const float* A, const float*x, float beta, float* y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) sum += A[i * cols + j] * x[j];
        y[i] = alpha * sum + beta * y[i];
    }
}
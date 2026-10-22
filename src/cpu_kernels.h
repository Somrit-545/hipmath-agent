#pragma once

void saxpy_cpu(float a, const float* x, float* y, int n);
float dot_cpu(const float* x, const float* y, int n);
void gemv_cpu(float alpha, const float* A, const float*x,
                float beta, float* y, int rows, int cols);

// if time allows
// void matmul_tiled(float *C, const float* A, const float* B,
//                      int M, int N, int K);
# CUDA to HIP Translation Notes

Although this project was tested on an NVIDIA RTX 3050 (using `HIP_PLATFORM=nvidia`), the source code is written in AMD's HIP language. Translating it to run on AMD hardware is trivial because HIP is a thin wrapper over CUDA.

If you are porting this code to an AMD GPU, here are the API translations:

- `cudaMalloc` becomes `hipMalloc`
- `cudaMemcpy` becomes `hipMemcpy`
- `cudaFree` becomes `hipFree`
- `cudaDeviceSynchronize` becomes `hipDeviceSynchronize`
- The `<<<blocks, threads>>>` launch syntax becomes `hipLaunchKernelGGL(...)`
- The compiler changes from `nvcc` to `hipcc`

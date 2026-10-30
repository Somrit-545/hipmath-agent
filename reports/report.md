# GPU Math Library Benchmark Report

Generated: 2026-10-30 20:11:53

## Performance Summary

| Op | N | CPU (ms) | GPU (ms) | Speedup | GPU Bandwidth (GB/s) |
|---|---|---|---|---|---|
| dot | 1024 | 0.001 | 0.006 | 0.1x | 1.5 |
| dot | 1048576 | 0.215 | 0.059 | 3.6x | 141.3 |
| dot | 16384 | 0.002 | 0.015 | 0.1x | 8.7 |
| dot | 16777216 | 5.149 | 0.877 | 5.9x | 153.1 |
| dot | 262144 | 0.029 | 0.018 | 1.6x | 116.8 |
| dot | 4194304 | 1.340 | 0.222 | 6.0x | 151.3 |
| dot | 67108864 | 21.283 | 3.673 | 5.8x | 146.2 |
| saxpy | 1024 | 0.002 | 0.013 | 0.2x | 0.9 |
| saxpy | 1048576 | 0.519 | 0.084 | 6.2x | 150.6 |
| saxpy | 16384 | 0.004 | 0.007 | 0.5x | 27.5 |
| saxpy | 16777216 | 17.484 | 1.274 | 13.7x | 158.1 |
| saxpy | 262144 | 0.105 | 0.018 | 5.8x | 174.4 |
| saxpy | 4194304 | 3.100 | 0.323 | 9.6x | 156.0 |
| saxpy | 67108864 | 82.421 | 5.066 | 16.3x | 159.0 |

## AI Agent Notes

The AI agent was queried 1 times. Results (tests and optimizations) are cached in `docs/ai_cache.json`.

## Methodology
- CPU timings measured using `time.perf_counter()` with 5 warmups and 20 repeats.
- GPU timings measured using C++ `std::chrono` inside the kernel wrapper, executing 100 iterations per call to bypass WSL2 host-device transfer overhead.

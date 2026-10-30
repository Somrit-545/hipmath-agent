# hipmath-agent

A GPU math library built with HIP/C++, exposed to Python via pybind11. It implements SAXPY, dot product, and GEMV kernels, and includes a Python AI agent that automatically generates edge-case tests.

## Performance & The WSL2 Bottleneck

On my RTX 3050 running WSL2, my initial benchmarks were terrible. The GPU was running at 5 GB/s, which is slower than the CPU. I quickly realized I was measuring the overhead of `hipMalloc` and `hipMemcpy` crossing the WSL2 VM boundary, not the actual kernel.

To fix this, I added C++ benchmark wrappers (`saxpy_bench`, `dot_bench`) to `bindings.cpp`. These wrappers allocate memory once and launch the kernel 100 times in a tight loop. By bypassing the WSL2 memory transfer, the true hardware limit was revealed. 

At large sizes (N=64M), the SAXPY kernel achieves **159 GB/s** and is **16.3x faster** than the CPU implementation. The dot product kernel reaches **153 GB/s** and is **5.9x faster**. (At tiny sizes like N=1024, the GPU is actually slower due to launch overhead — the crossover point is around N=262,144).

## AI Agent Integration

The project includes a Python AI agent (`hipmath.testgen`) that asks an LLM to write edge-case tests for the kernels. 

- **Model:** Google Gemini (`gemini-3.5-flash-lite`).
- **Caching:** Prompts are hashed and cached locally to avoid hitting API rate limits.
- **Human Oversight:** The AI successfully found critical edge cases (like passing `n=0` and mismatched array dimensions), but it generated code that forgot to prefix calls with `hipmath.`. I manually corrected the tests before committing them. You can read the full prompt log in `docs/ai_prompts.md`.

## Setup & Usage

Requires ROCm/CUDA and Python 3.10+.

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Build the HIP extensions
cd python
pip install -e . --no-build-isolation
cd ..

# 3. Run the benchmark suite
python -c "from hipmath.bench import run_sweep; run_sweep(['saxpy','dot'], [1024, 1048576], ['cpu','gpu'], 'benchmarks/results.csv')"

# 4. Generate the markdown report
python -c "from hipmath.report import generate_report; generate_report()"

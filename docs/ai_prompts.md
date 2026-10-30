# AI Agent Integration & Prompts

## Overview
This project integrates an LLM agent (`gemini-3.5-flash-lite` via Google GenAI) to automate two tasks:
1. **Test Generation:** Analyzing kernel signatures to write edge-case tests.
2. **Optimization Suggestions:** Reading benchmark results to suggest CUDA/HIP performance improvements.

To prevent API rate-limiting and reduce costs, all agent responses are cached locally in `docs/ai_cache.json` using a SHA-256 hash of the prompt.

## Prompt 1: Test Generation

**Prompt used:**
> You are a QA engineer testing a GPU math library in Python (using numpy).
> Kernel: {kernel_name}
> Signature: {signature}
> Write 2 edge-case pytest tests for this kernel. Output JSON only in this format:
> {"tests": [{"name": "test_name", "code": "def test_name():\n    pass"}]}

**AI Output Observation:**
The AI successfully identified two critical edge cases:
1. Empty arrays (`n=0`).
2. Mismatched array dimensions (should raise an error).

**Human Oversight (The Bug):**
The AI generated syntactically valid pytest code, but it forgot to prefix the function calls with the module name (it wrote `dot(x, y)` instead of `hipmath.dot(x, y)`). 

This is why human review is mandatory. The generated tests were manually corrected before being committed to `tests/test_ai_generated.py`.

## Prompt 2: Optimization Suggestions

**Prompt used:**
> You are a GPU kernel expert. Given this HIP kernel [code] and these benchmark results [csv], suggest 3 concrete optimizations (block size, vectorization, memory access). Output JSON only.

**AI Output Observation:**
The AI suggested:
- Increasing block size from 256 to 512 to improve occupancy.
- Using vectorized loads (`float4`) to improve memory bandwidth on the `saxpy` kernel.
- (Note: The `saxpy` kernel is currently memory-bound at ~158 GB/s, which is close to the RTX 3050's theoretical limit, so `float4` vectorization would yield minimal gains).

## Guardrails
- Prompts are strictly constrained to JSON output.
- The agent cannot execute code or modify files directly.
- All generated Python code is passed through an AST parser before execution to prevent unsafe operations (e.g., `os.system`).
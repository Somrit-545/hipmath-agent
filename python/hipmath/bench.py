import time
import statistics
import numpy as np
import hipmath
import csv

def time_call(fn, warmup=5, repeats=20):
    """Return (median_ms, stddev_ms) of `fn` over `repeats` runs."""
    # 1. run fn warmup times, discard
    for _ in range(warmup):
        fn()

    # 2. run fn repeats times, collect elapsed in ms
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)   # seconds → ms

    # 3. return median, stddev
    median_ms = statistics.median(times)
    stddev_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    return median_ms, stddev_ms

def bench_saxpy(device, n):
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    if device == "cpu":
        fn = lambda: 2.0 * x + y
        median, sd = time_call(fn)
        ms = median
    elif device == "gpu":
        ms = hipmath.saxpy_bench(2.0, x, y, 100)  # runs 100 iters in C++
        sd = 0.0
    else:
        raise ValueError(f"unknown device: {device}")
        
    bytes_ = 3 * n * 4
    flops = 2 * n
    return {
        "op": "saxpy", "device": device, "n": n,
        "median_ms": ms, "stddev_ms": sd,
        "gbs": bytes_ / (ms * 1e6),
        "gflops": flops / (ms * 1e6),
    }

def bench_dot(device, n):
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    if device == "cpu":
        fn = lambda: np.dot(x, y)
        median, sd = time_call(fn)
        ms = median
    elif device == "gpu":
        ms = hipmath.dot_bench(x, y, 100)
        sd = 0.0
    else:
        raise ValueError(f"unknown device: {device}")
        
    bytes_ = 2 * n * 4
    flops = 2 * n
    return {
        "op": "dot", "device": device, "n": n,
        "median_ms": ms, "stddev_ms": sd,
        "gbs": bytes_ / (ms * 1e6),
        "gflops": flops / (ms * 1e6),
    }

def bench_gemv(device, rows, cols):
    A = np.random.rand(rows, cols).astype(np.float32)
    x = np.random.rand(cols).astype(np.float32)
    y = np.random.rand(rows).astype(np.float32)
    if device == "cpu":
        fn = lambda: A @ x
    elif device == "gpu":
        fn = lambda: hipmath.gemv(1.0, A, x, 0.0, y)
    else:
        raise ValueError(f"unknown device: {device}")
    median, sd = time_call(fn)
    bytes_ = (rows * cols + cols + 2 * rows) * 4
    flops = 2 * rows * cols
    return {
        "op": "gemv", "device": device, "n": f"{rows}x{cols}",
        "median_ms": median, "stddev_ms": sd,
        "gbs": bytes_ / (median * 1e6),
        "gflops": flops / (median * 1e6),
    }

FIELDS = ["op", "device", "n", "median_ms", "stddev_ms", "gbs", "gflops"]

def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})

def bench_one(op, device, size):
    if op == "gemv":
        rows_, cols_ = size
        return bench_gemv(device, rows_, cols_)
    elif op == "saxpy":
        return bench_saxpy(device, size)
    elif op == "dot":
        return bench_dot(device, size)
    else:
        raise ValueError(f"unknown op: {op}")

def run_sweep(ops, sizes, devices, out_path):
    rows = []
    for op in ops:
        for size in sizes:
            for device in devices:
                print(f"  {op} {device} n={size} ...", flush=True)
                rows.append(bench_one(op, device, size))
    write_csv(rows, out_path)
    return rows
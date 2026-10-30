import csv
import json
import os
from datetime import datetime

CSV_PATH = "benchmarks/results.csv"
REPORT_PATH = "reports/report.md"

def generate_report():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run a benchmark sweep first.")
        return

    rows = []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["median_ms"] = float(r["median_ms"])
            r["gbs"] = float(r["gbs"])
            rows.append(r)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    
    with open(REPORT_PATH, "w") as f:
        f.write("# GPU Math Library Benchmark Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Op | N | CPU (ms) | GPU (ms) | Speedup | GPU Bandwidth (GB/s) |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        # Group by op and n
        data = {}
        for r in rows:
            key = (r["op"], r["n"])
            if key not in data: data[key] = {}
            data[key][r["device"]] = r
            
        for (op, n) in sorted(data.keys()):
            if "cpu" in data[(op, n)] and "gpu" in data[(op, n)]:
                cpu = data[(op, n)]["cpu"]
                gpu = data[(op, n)]["gpu"]
                speedup = cpu["median_ms"] / gpu["median_ms"]
                f.write(f"| {op} | {n} | {cpu['median_ms']:.3f} | {gpu['median_ms']:.3f} | {speedup:.1f}x | {gpu['gbs']:.1f} |\n")
                
        f.write("\n## AI Agent Notes\n\n")
        cache_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "ai_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as c:
                cache = json.load(c)
                f.write(f"The AI agent was queried {len(cache)} times. Results (tests and optimizations) are cached in `docs/ai_cache.json`.\n")
        else:
            f.write("No AI agent queries cached yet.\n")
            
        f.write("\n## Methodology\n")
        f.write("- CPU timings measured using `time.perf_counter()` with 5 warmups and 20 repeats.\n")
        f.write("- GPU timings measured using C++ `std::chrono` inside the kernel wrapper, executing 100 iterations per call to bypass WSL2 host-device transfer overhead.\n")

    print(f"Report generated at {REPORT_PATH}")
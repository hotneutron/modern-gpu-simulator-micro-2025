#!/usr/bin/env python3
"""Wall-time speedup measurement: baseline vs pilot sampling.

For each workload, runs accel-sim.out in two modes (baseline / pilot)
N_TRIALS times and records wall time. Writes a CSV in the same schema
as history/speedup_results.csv so successive measurements can be
diffed.

Usage (from simulator-remodeled/):
  source ./gpu-simulator/setup_environment_no_git.sh
  ./util/cta_sampling/measure_speedup.py --out history/speedup_results_<date>.csv
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_WORKLOADS = {
    "hotspot":    "hotspot-rodinia-2.0-ft/30_6_40___data_result_30_6_40_txt/traces/dynamic_trace.pb",
    "backprop":   "backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/dynamic_trace.pb",
    "pathfinder": "pathfinder-rodinia-2.0-ft/1000_20_5___data_result_1000_20_5_txt/traces/dynamic_trace.pb",
    "bfs":        "bfs-rodinia-2.0-ft/__data_graph4096_txt___data_graph4096_result_txt/traces/dynamic_trace.pb",
    "srad_v2":    "srad_v2-rodinia-2.0-ft/__data_matrix128x128_txt_0_127_0_127__5_2___data_result_matrix128x128_1_150_1_100__5_2_txt/traces/dynamic_trace.pb",
    "lud":        "lud-rodinia-2.0-ft/_v__b__i___data_64_dat/traces/dynamic_trace.pb",
    "heartwall":  "heartwall-rodinia-2.0-ft/__data_test_avi_1___data_result_1_txt/traces/dynamic_trace.pb",
    "nn":         "nn-rodinia-2.0-ft/__data_filelist_4_3_30_90___data_filelist_4_3_30_90_result_txt/traces/dynamic_trace.pb",
    "nw":         "nw-rodinia-2.0-ft/128_10___data_result_128_10_txt/traces/dynamic_trace.pb",
}

REPO = Path(__file__).resolve().parents[2]
SIM  = REPO / "gpu-simulator/bin/release/accel-sim.out"
GCFG = REPO / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM75_RTX2070_S/gpgpusim.config"
TCFG = REPO / "gpu-simulator/configs/tested-cfgs/SM75_RTX2070_S/trace.config"

MODES = {
    "baseline": [],
    "pilot":    ["-cta_sampling_mode", "1", "-cta_sampling_pilot_max_doublings", "2"],
}


def run_once(trace, extra, env, log_path):
    cmd = [str(SIM), "-trace", str(trace),
           "-config", str(GCFG), "-config", str(TCFG)] + extra
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=1800)
    return time.time() - t0, proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads",
                    default="hotspot,backprop,pathfinder,bfs,srad_v2,lud,heartwall,nn,nw")
    ap.add_argument("--trace-root", default="/tmp/traces_extracted/rodinia2/12.8")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default="/tmp/cta_sampling_speedup_logs")
    args = ap.parse_args()

    if not SIM.exists():
        sys.exit(f"error: simulator not built at {SIM}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    rows = []
    workloads = args.workloads.split(",")
    trace_root = Path(args.trace_root)

    for wl in workloads:
        rel = DEFAULT_WORKLOADS.get(wl)
        if not rel:
            print(f"warning: unknown workload {wl}", file=sys.stderr); continue
        trace = trace_root / rel
        if not trace.exists():
            print(f"warning: missing trace {trace}", file=sys.stderr); continue
        print(f"=== {wl} ===")
        for mode, extra in MODES.items():
            trials = []
            for t in range(args.trials):
                log_path = log_dir / f"{wl}_{mode}_t{t}.log"
                wall, exit_code = run_once(trace, extra, env, log_path)
                trials.append(wall)
                print(f"  {mode} trial {t+1}/{args.trials}: {wall:.2f}s (exit={exit_code})")
            mean = statistics.fmean(trials)
            stdev = statistics.stdev(trials) if len(trials) > 1 else 0.0
            row = {
                "workload": wl, "mode": mode,
                "mean_s": mean, "stdev_s": stdev,
            }
            for i, t in enumerate(trials, 1):
                row[f"trial{i}"] = t
            rows.append(row)
            print(f"  {mode} mean = {mean:.3f}s (stdev {stdev:.3f})")

    # Write CSV in the same schema as history/speedup_results.csv.
    fieldnames = ["workload", "mode", "mean_s", "stdev_s"] + \
                 [f"trial{i+1}" for i in range(args.trials)]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out}")

    # Pretty per-workload speedup summary.
    print("\nWorkload         baseline       pilot         speedup")
    print("-" * 56)
    by_wl = {}
    for r in rows:
        by_wl.setdefault(r["workload"], {})[r["mode"]] = r["mean_s"]
    cum_b, cum_p = 0.0, 0.0
    for wl in workloads:
        d = by_wl.get(wl, {})
        b = d.get("baseline"); p = d.get("pilot")
        if b is None or p is None: continue
        cum_b += b; cum_p += p
        s = b / p if p else float("nan")
        print(f"{wl:<14} {b:>10.2f}s {p:>10.2f}s   {s:>5.2f}x")
    if cum_p:
        print("-" * 56)
        print(f"{'cumulative':<14} {cum_b:>10.2f}s {cum_p:>10.2f}s   {cum_b/cum_p:>5.2f}x")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CTA-sampling: compare concurrency-throughput models for the cycle estimator.

For each workload, runs baseline (no sampling) plus one pilot run per
concurrency_model in {logfit, sat_exp, roofline_clamp, roofline_exp} and
prints a wide cycle_err% comparison + per-mode p50/p90 summary.

Usage (from simulator-remodeled/):
  source ./gpu-simulator/setup_environment_no_git.sh
  ./util/cta_sampling/validate_models.py
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

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
SIM = REPO / "gpu-simulator/bin/release/accel-sim.out"
DEFAULT_GCFG = REPO / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM75_RTX2070_S/gpgpusim.config"
DEFAULT_TCFG = REPO / "gpu-simulator/configs/tested-cfgs/SM75_RTX2070_S/trace.config"
DEFAULT_TRACE_ROOT = Path("/tmp/traces_extracted/rodinia2/12.8")

MODELS = ["logfit", "sat_exp", "roofline_clamp", "roofline_exp"]

CYCLE_EST_PAT = re.compile(r"^gpu_tot_sim_cycle_estimated\s*=\s*(\d+)", re.M)
CYCLE_RAW_PAT = re.compile(r"^gpu_tot_sim_cycle\s*=\s*(\d+)", re.M)
EST_MODE_PAT  = re.compile(r"^gpu_tot_sim_cycle_estimation_mode\s*=\s*(\S+)", re.M)


def run(label, extra_args, trace, gcfg, tcfg, out_dir, env):
    log = out_dir / f"{label}.log"
    cmd = [str(SIM), "-trace", str(trace),
           "-config", str(gcfg), "-config", str(tcfg)] + extra_args
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=1800)
    wall = time.time() - t0
    text = log.read_text()
    cycles_est = [int(x) for x in CYCLE_EST_PAT.findall(text)]
    cycles_raw = [int(x) for x in CYCLE_RAW_PAT.findall(text)]
    em         = EST_MODE_PAT.findall(text)
    return {
        "cycles_est": cycles_est[-1] if cycles_est else None,
        "cycles_raw": cycles_raw[-1] if cycles_raw else None,
        "est_mode":   em[-1] if em else None,
        "wall":       wall,
        "exit":       proc.returncode,
    }


def err_pct(s, b):
    if s is None or b in (None, 0): return None
    return 100.0 * (s - b) / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads",
                    default="hotspot,backprop,pathfinder,bfs,srad_v2,lud,heartwall,nn,nw")
    ap.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT))
    ap.add_argument("--gcfg", default=str(DEFAULT_GCFG))
    ap.add_argument("--tcfg", default=str(DEFAULT_TCFG))
    ap.add_argument("--out-dir", default="/tmp/cta_sampling_models")
    ap.add_argument("--pilot-doublings", type=int, default=2)
    args = ap.parse_args()

    if not SIM.exists():
        print(f"error: simulator not built at {SIM}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    env = os.environ.copy()
    workloads = args.workloads.split(",")
    trace_root = Path(args.trace_root)
    rows = []

    for wl in workloads:
        rel = DEFAULT_WORKLOADS.get(wl)
        if not rel:
            print(f"warning: unknown workload {wl}", file=sys.stderr); continue
        trace = trace_root / rel
        if not trace.exists():
            print(f"warning: missing trace {trace}", file=sys.stderr); continue
        print(f"=== {wl} ===")
        wl_dir = out_dir / wl
        wl_dir.mkdir()

        # Baseline
        print("  baseline...", flush=True)
        base = run(f"baseline", [], trace, Path(args.gcfg), Path(args.tcfg), wl_dir, env)
        print(f"    cycles={base['cycles_raw']} wall={base['wall']:.1f}s")

        per_model = {}
        for m in MODELS:
            print(f"  {m}...", flush=True)
            r = run(m, ["-cta_sampling_mode", "1",
                        "-cta_sampling_pilot_max_doublings", str(args.pilot_doublings),
                        "-cta_sampling_concurrency_model", m],
                    trace, Path(args.gcfg), Path(args.tcfg), wl_dir, env)
            r["err_pct"] = err_pct(r["cycles_est"], base["cycles_raw"])
            print(f"    cycles_est={r['cycles_est']} err={r['err_pct']:+.1f}%  mode={r['est_mode']}  wall={r['wall']:.1f}s")
            per_model[m] = r

        rows.append((wl, base, per_model))

    # Wide comparison table
    print()
    print("=" * 96)
    print(f"{'workload':<12}  {'baseline':>10}  " + "  ".join(f"{m:>16}" for m in MODELS))
    print("-" * 96)
    for wl, base, per_model in rows:
        line = f"{wl:<12}  {base['cycles_raw']:>10}  "
        for m in MODELS:
            r = per_model[m]
            est = r["cycles_est"] if r["cycles_est"] is not None else 0
            err = r["err_pct"]
            tag = f"{est:>7} ({err:+6.1f}%)"
            line += f"  {tag:>16}"
        print(line)
    print("-" * 96)

    # Per-model p50 / p90 |err%|
    print()
    print("Per-model summary (over |err%|):")
    print(f"{'model':<18}  {'p50':>8}  {'p90':>8}  {'max':>8}  {'mean':>8}")
    for m in MODELS:
        errs = sorted([abs(per_model[m]["err_pct"])
                       for _, _, per_model in rows
                       if per_model[m]["err_pct"] is not None])
        if not errs:
            print(f"{m:<18}  -")
            continue
        p50 = median(errs)
        idx90 = max(0, int(round(0.9 * (len(errs) - 1))))
        p90 = errs[idx90]
        print(f"{m:<18}  {p50:>7.1f}%  {p90:>7.1f}%  {max(errs):>7.1f}%  {sum(errs)/len(errs):>7.1f}%")

    # Per-workload sign + model details
    print()
    print("Per-workload err% (signed):")
    print(f"{'workload':<12}  " + "  ".join(f"{m:>16}" for m in MODELS))
    for wl, _, per_model in rows:
        cells = []
        for m in MODELS:
            r = per_model[m]
            cells.append(f"{r['err_pct']:+15.1f}%" if r['err_pct'] is not None else f"{'-':>16}")
        print(f"{wl:<12}  " + "  ".join(cells))

    print()
    print(f"raw logs: {out_dir}")


if __name__ == "__main__":
    main()

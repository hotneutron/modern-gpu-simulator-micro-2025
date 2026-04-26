#!/usr/bin/env python3
"""CTA-sampling validation harness.

Runs accel-sim.out on a set of trace+config tuples in several modes
(baseline, K-rep sampling, classifier-driven sampling, adaptive pilot loop)
and emits a comparison table of cumulative simulator stats and per-mode
relative error vs. baseline. Intended to be run after a build.

Usage:
  validate.py [--workloads hotspot,backprop,pathfinder] [--config SM75_RTX2070_S]

Example end-to-end (from simulator-remodeled/):
  source ./gpu-simulator/setup_environment_no_git.sh
  ./util/cta_sampling/validate.py
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Default trace lookup. Each entry is (workload key -> (trace.pb, label)).
# Paths are relative to TRACE_ROOT (defaults to /tmp/traces_extracted/rodinia2/12.8).
DEFAULT_WORKLOADS = {
    "hotspot":    ("hotspot-rodinia-2.0-ft/30_6_40___data_result_30_6_40_txt/traces/dynamic_trace.pb",
                   "stencil (compute-bound expected)"),
    "backprop":   ("backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/dynamic_trace.pb",
                   "neural net (mixed)"),
    "pathfinder": ("pathfinder-rodinia-2.0-ft/1000_20_5___data_result_1000_20_5_txt/traces/dynamic_trace.pb",
                   "stencil (small grid)"),
}

REPO = Path(__file__).resolve().parents[2]
SIM = REPO / "gpu-simulator/bin/release/accel-sim.out"
DEFAULT_GCFG = REPO / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM75_RTX2070_S/gpgpusim.config"
DEFAULT_TCFG = REPO / "gpu-simulator/configs/tested-cfgs/SM75_RTX2070_S/trace.config"
DEFAULT_TRACE_ROOT = Path("/tmp/traces_extracted/rodinia2/12.8")

MODES = [
    ("baseline",  []),
    ("K-rep",     ["-cta_sampling_mode", "1"]),
    ("expanded40", ["-cta_sampling_mode", "1", "-cta_sampling_target_ctas", "40"]),
    ("pilot",     ["-cta_sampling_mode", "1", "-cta_sampling_pilot_max_doublings", "2"]),
]

STAT_PATTERNS = {
    "gpu_tot_sim_cycle":  re.compile(r"^gpu_tot_sim_cycle\s*=\s*(\d+)", re.M),
    "gpu_tot_sim_insn":   re.compile(r"^gpu_tot_sim_insn\s*=\s*(\d+)", re.M),
    "gpu_tot_issued_cta": re.compile(r"^gpu_tot_issued_cta\s*=\s*(\d+)", re.M),
    "gpu_tot_ipc":        re.compile(r"^gpu_tot_ipc\s*=\s*([0-9.]+)", re.M),
}


def run_one(label, workload, trace_path, gcfg, tcfg, extra_args, out_dir, env):
    """Run accel-sim.out once, capture log, return parsed stats + wall time."""
    log = out_dir / f"{workload}_{label}.log"
    cmd = [str(SIM), "-trace", str(trace_path),
           "-config", str(gcfg), "-config", str(tcfg)] + extra_args
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=1800)
    wall = time.time() - t0
    text = log.read_text()
    stats = {}
    for k, pat in STAT_PATTERNS.items():
        # Last match is the post-final-kernel cumulative.
        matches = pat.findall(text)
        stats[k] = float(matches[-1]) if matches else None
    stats["wall_sec"] = wall
    stats["exit"] = proc.returncode
    return stats


def fmt(v, fmt_spec="{:>14}"):
    if v is None: return fmt_spec.format("-")
    if isinstance(v, float):
        return fmt_spec.format(f"{v:.4g}")
    return fmt_spec.format(v)


def err_pct(sampled, baseline):
    if baseline in (None, 0) or sampled is None: return None
    return 100.0 * (sampled - baseline) / baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="hotspot,backprop,pathfinder",
                    help="Comma-separated workload keys")
    ap.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT))
    ap.add_argument("--gcfg", default=str(DEFAULT_GCFG))
    ap.add_argument("--tcfg", default=str(DEFAULT_TCFG))
    ap.add_argument("--out-dir", default="/tmp/cta_sampling_validate")
    args = ap.parse_args()

    if not SIM.exists():
        print(f"error: simulator not built at {SIM}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Inherit env (so LD_LIBRARY_PATH etc. set by setup_environment carries)
    env = os.environ.copy()

    workloads = args.workloads.split(",")
    trace_root = Path(args.trace_root)
    rows = []
    for wl in workloads:
        if wl not in DEFAULT_WORKLOADS:
            print(f"warning: skip unknown workload {wl}")
            continue
        rel, descr = DEFAULT_WORKLOADS[wl]
        trace = trace_root / rel
        if not trace.exists():
            print(f"warning: missing trace {trace}, skipping")
            continue
        print(f"=== {wl} ({descr}) ===")
        per_mode = {}
        for label, extra in MODES:
            print(f"  running {label}...", flush=True)
            stats = run_one(label, wl, trace, Path(args.gcfg), Path(args.tcfg),
                            extra, out_dir, env)
            print(f"    cycles={stats['gpu_tot_sim_cycle']} "
                  f"insns={stats['gpu_tot_sim_insn']} "
                  f"ipc={stats['gpu_tot_ipc']} "
                  f"ctas={stats['gpu_tot_issued_cta']} "
                  f"wall={stats['wall_sec']:.1f}s "
                  f"exit={stats['exit']}")
            per_mode[label] = stats
        rows.append((wl, descr, per_mode))

    # Comparison table
    header = ["workload", "mode", "cycles", "insn", "ipc", "ctas",
              "cycle_err%", "insn_err%", "ipc_err%", "wall_s"]
    widths = [12, 11, 12, 14, 8, 8, 11, 11, 11, 7]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    sep = "-" * len(line)
    print()
    print(sep)
    print(line)
    print(sep)
    for wl, descr, per_mode in rows:
        base = per_mode.get("baseline", {})
        for label, _ in MODES:
            stats = per_mode.get(label, {})
            cycles = stats.get("gpu_tot_sim_cycle")
            insns  = stats.get("gpu_tot_sim_insn")
            ipc    = stats.get("gpu_tot_ipc")
            ctas   = stats.get("gpu_tot_issued_cta")
            cell = [wl, label,
                    fmt(cycles, "{:>12}"),
                    fmt(insns, "{:>14}"),
                    fmt(ipc, "{:>8}"),
                    fmt(ctas, "{:>8}")]
            if label != "baseline":
                cell.append(fmt(err_pct(cycles, base.get("gpu_tot_sim_cycle")), "{:>11}"))
                cell.append(fmt(err_pct(insns,  base.get("gpu_tot_sim_insn")),  "{:>11}"))
                cell.append(fmt(err_pct(ipc,    base.get("gpu_tot_ipc")),       "{:>11}"))
            else:
                cell += ["{:>11}".format("-")] * 3
            cell.append(fmt(stats.get("wall_sec"), "{:>7}"))
            print("  ".join(c.ljust(w) for c, w in zip([cell[0].ljust(12),
                                                         cell[1].ljust(11)] +
                                                        cell[2:], widths)))
        print(sep)
    print()
    print(f"raw logs: {out_dir}")


if __name__ == "__main__":
    main()

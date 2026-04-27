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
# bfs/srad_v2/lud added per the misclassification-prone-kernels acceptance
# criterion: with the original AI proxy these were routing as COMPUTE despite
# being memory-heavy. The refined classifier (Phase A) should pull them into
# MEMORY/MIXED so the pilot loop can expand.
DEFAULT_WORKLOADS = {
    "hotspot":    ("hotspot-rodinia-2.0-ft/30_6_40___data_result_30_6_40_txt/traces/dynamic_trace.pb",
                   "stencil (compute-bound expected)"),
    "backprop":   ("backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/dynamic_trace.pb",
                   "neural net (mixed)"),
    "pathfinder": ("pathfinder-rodinia-2.0-ft/1000_20_5___data_result_1000_20_5_txt/traces/dynamic_trace.pb",
                   "stencil (small grid)"),
    "bfs":        ("bfs-rodinia-2.0-ft/__data_graph4096_txt___data_graph4096_result_txt/traces/dynamic_trace.pb",
                   "graph BFS (memory-bound, irregular)"),
    "srad_v2":    ("srad_v2-rodinia-2.0-ft/__data_matrix128x128_txt_0_127_0_127__5_2___data_result_matrix128x128_1_150_1_100__5_2_txt/traces/dynamic_trace.pb",
                   "memory-bound stencil"),
    "lud":        ("lud-rodinia-2.0-ft/_v__b__i___data_64_dat/traces/dynamic_trace.pb",
                   "LU decomposition (GEMM-like dense)"),
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
    # pilot+refined: same mechanics as `pilot` but explicit AI weights and the
    # mem-stall-fraction memory-pressure threshold, which the refined Phase A
    # classifier consumes. With the build's defaults these match the `pilot`
    # mode's effective behavior; the explicit knobs make the classifier-input
    # contract visible in the table and let the harness sweep weights later.
    ("pilot+refined", [
        "-cta_sampling_mode", "1",
        "-cta_sampling_pilot_max_doublings", "2",
        "-cta_sampling_ai_w_dp",  "2.0",
        "-cta_sampling_ai_w_tc",  "8.0",
        "-cta_sampling_ai_w_sfu", "4.0",
        "-cta_sampling_pressure_mstall", "0.4",
    ]),
]

STAT_PATTERNS = {
    "gpu_tot_sim_cycle":  re.compile(r"^gpu_tot_sim_cycle\s*=\s*(\d+)", re.M),
    "gpu_tot_sim_insn":   re.compile(r"^gpu_tot_sim_insn\s*=\s*(\d+)", re.M),
    "gpu_tot_issued_cta": re.compile(r"^gpu_tot_issued_cta\s*=\s*(\d+)", re.M),
    "gpu_tot_ipc":        re.compile(r"^gpu_tot_ipc\s*=\s*([0-9.]+)", re.M),
    # Phase B: whole-kernel projected cycles + IPC. Distinct from the raw
    # gpu_tot_sim_cycle, which is the sampled-wave wall-clock.
    "gpu_tot_sim_cycle_estimated": re.compile(r"^gpu_tot_sim_cycle_estimated\s*=\s*(\d+)", re.M),
    "gpu_tot_ipc_estimated":       re.compile(r"^gpu_tot_ipc_estimated\s*=\s*([0-9.]+)", re.M),
}
# Captured separately because the value is a string (per_cta | steady_state | none).
ESTIMATION_MODE_PAT = re.compile(r"^gpu_tot_sim_cycle_estimation_mode\s*=\s*(\S+)", re.M)

# Per-kernel pressure-signal fields parsed out of the CTA_PRESSURE_SIGNALS log
# line. We keep the *last* observed value per field (i.e. the final accepted
# pilot iteration of the last kernel) for the table. These mirror the
# pressure_signals_t struct fields produced by the refined Phase A classifier.
PRESSURE_FIELDS = [
    "kernel_ai", "ridge_ratio", "achieved_bw_ratio", "dram_queue_occupancy_avg",
    "mem_stall_frac", "total_stall_frac",
    "compute_ops", "mem_ops",
    "n_fp_decoded", "n_int_decoded", "n_dp_acc", "n_tc_acc", "n_sfu_acc",
    "n_load_insn", "n_store_insn",
    "stall_l1c_cycles", "issue_eval_cycles",
    "class", "pilot_iter", "pilot_accepted",
]
PRESSURE_PATTERNS = {f: re.compile(rf"\b{f}=([^\s]+)") for f in PRESSURE_FIELDS}


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
    em = ESTIMATION_MODE_PAT.findall(text)
    stats["estimation_mode"] = em[-1] if em else None
    # Pressure-signal fields from the last accepted CTA_PRESSURE_SIGNALS line.
    # We restrict to lines marked pilot_accepted=1 so we get the final-iteration
    # signals rather than the K-rep iteration of an expanded pilot run.
    accepted_lines = [ln for ln in text.splitlines()
                      if ln.startswith("CTA_PRESSURE_SIGNALS:") and "pilot_accepted=1" in ln]
    target_line = accepted_lines[-1] if accepted_lines else None
    for k, pat in PRESSURE_PATTERNS.items():
        if not target_line:
            stats[k] = None
            continue
        m = pat.search(target_line)
        if not m:
            stats[k] = None
            continue
        v = m.group(1)
        if k in ("class",):
            stats[k] = v
        else:
            try:
                stats[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except ValueError:
                stats[k] = v
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
    ap.add_argument("--workloads",
                    default="hotspot,backprop,pathfinder,bfs,srad_v2,lud",
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

    # Comparison table. cycle_err% compares the *estimated* whole-kernel cycle
    # count (gpu_tot_sim_cycle_estimated) against the baseline's raw cycle
    # count -- the baseline didn't sample, so its raw equals its estimate.
    # cycle_err_raw% compares the un-estimated sampled-wave cycle for context.
    header = ["workload", "mode", "cycles_est", "cycles_raw", "insn", "ipc_est",
              "ctas", "cycle_err%", "cycle_err_raw%", "insn_err%", "wall_s"]
    widths = [12, 14, 12, 12, 14, 9, 8, 11, 14, 11, 7]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    sep = "-" * len(line)
    print()
    print(sep)
    print(line)
    print(sep)
    for wl, descr, per_mode in rows:
        base = per_mode.get("baseline", {})
        # Compare against baseline's raw cycle count (the baseline did not
        # sample, so its raw == its estimate).
        base_cycle_ref = base.get("gpu_tot_sim_cycle")
        for label, _ in MODES:
            stats = per_mode.get(label, {})
            cycles_raw = stats.get("gpu_tot_sim_cycle")
            cycles_est = stats.get("gpu_tot_sim_cycle_estimated") or cycles_raw
            insns      = stats.get("gpu_tot_sim_insn")
            ipc_est    = stats.get("gpu_tot_ipc_estimated") or stats.get("gpu_tot_ipc")
            ctas       = stats.get("gpu_tot_issued_cta")
            cell = [wl, label,
                    fmt(cycles_est, "{:>12}"),
                    fmt(cycles_raw, "{:>12}"),
                    fmt(insns, "{:>14}"),
                    fmt(ipc_est, "{:>9}"),
                    fmt(ctas, "{:>8}")]
            if label != "baseline":
                cell.append(fmt(err_pct(cycles_est, base_cycle_ref), "{:>11}"))
                cell.append(fmt(err_pct(cycles_raw, base_cycle_ref), "{:>14}"))
                cell.append(fmt(err_pct(insns,  base.get("gpu_tot_sim_insn")), "{:>11}"))
            else:
                cell += ["{:>11}".format("-"), "{:>14}".format("-"), "{:>11}".format("-")]
            cell.append(fmt(stats.get("wall_sec"), "{:>7}"))
            print("  ".join(c.ljust(w) for c, w in zip([cell[0].ljust(12),
                                                         cell[1].ljust(14)] +
                                                        cell[2:], widths)))
        print(sep)
    # Pressure-signal table: class transitions and the new Phase A fields per
    # mode. Helps see "did the refined classifier route this kernel through
    # MEMORY/MIXED instead of COMPUTE iter-0?" at a glance.
    psig_header = ["workload", "mode", "class", "p_iter", "kernel_ai",
                   "ridge_r", "ach_bw", "mem_st%", "compute_ops", "mem_ops"]
    psig_widths = [12, 14, 7, 6, 10, 9, 7, 8, 12, 10]
    pline = "  ".join(h.ljust(w) for h, w in zip(psig_header, psig_widths))
    psep = "-" * len(pline)
    print()
    print(psep)
    print(pline)
    print(psep)
    for wl, descr, per_mode in rows:
        for label, _ in MODES:
            stats = per_mode.get(label, {})
            if not stats:
                continue
            cells = [
                wl, label,
                str(stats.get("class") or "-"),
                str(stats.get("pilot_iter") if stats.get("pilot_iter") is not None else "-"),
                fmt(stats.get("kernel_ai"), "{:>10}"),
                fmt(stats.get("ridge_ratio"), "{:>9}"),
                fmt(stats.get("achieved_bw_ratio"), "{:>7}"),
                fmt(stats.get("mem_stall_frac"), "{:>8}"),
                fmt(stats.get("compute_ops"), "{:>12}"),
                fmt(stats.get("mem_ops"), "{:>10}"),
            ]
            print("  ".join(c.ljust(w) for c, w in zip(cells, psig_widths)))
        print(psep)

    print()
    print(f"raw logs: {out_dir}")


if __name__ == "__main__":
    main()

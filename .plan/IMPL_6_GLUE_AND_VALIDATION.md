# Implementation Plan: [6] Glue Script + Validation

**Files:** `tools/trace_selector.py`, `tools/sim_compare.py`

---

## trace_selector.py

Orchestrates the full pipeline. Each step is skippable if its output already exists.

```
python tools/trace_selector.py \
  --workload "./app arg1 arg2" \
  --gpu-config tools/configs/sm86_rtx3080.json \
  --output-dir ./output/ \
  --hotspot-top-k 5 \
  [--skip-profiling]       # reuse existing kernel_report.csv
  [--skip-clustering]      # reuse existing representatives.json
  [--skip-tracing]         # reuse existing traces/
  [--skip-simulation]
  [--validate]             # run full simulation too; emit accuracy report
  [--sim-config tools/configs/sm86_rtx3080_sim.cfg]
```

### Step sequencing

```python
def run(args):
    out = Path(args.output_dir)

    # Step 1: Kernel profiler Pass A
    if not (out / "kernel_report.csv").exists() or not args.skip_profiling:
        KernelProfiler().run_pass_a(args.workload, out)

    # Step 2: Classify + rank
    report = KernelClassifier(args.gpu_config).run(out / "kernel_report.csv", out)
    hot_kernels = report[report.rank <= args.hotspot_top_k]

    # Step 3: Kernel profiler Pass B (hot kernels only, if clustering needed)
    cluster_needed = hot_kernels[hot_kernels.needs_clustering].kernel_id.tolist()
    if cluster_needed and not (out / "cta_features_done").exists():
        KernelProfiler().run_pass_b(args.workload, cluster_needed, out)

    # Step 4: CTA clustering
    if not (out / "representatives.json").exists() or not args.skip_clustering:
        CTAClusterer().run(out, hot_kernels)

    # Step 5: Selective tracing
    if not (out / "traces").exists() or not args.skip_tracing:
        run_selective_tracer(args.workload, out / "representatives.json",
                             out / "traces")

    # Step 6: Simulate (sampled)
    run_simulator(out / "traces", args.sim_config,
                  cta_weights=out / "traces/cta_weights.json",
                  output=out / "sim_sampled.txt")

    # Step 7 (optional): Validate
    if args.validate:
        run_simulator(out / "traces_full", args.sim_config,
                      output=out / "sim_full.txt")
        SimCompare().run(out / "sim_sampled.txt", out / "sim_full.txt",
                         out / "accuracy_report.csv")
```

---

## sim_compare.py

Compares sampled vs. full simulation output. Parses `gpu_print_stat` output.

```python
class SimCompare:
    METRICS = [
        "gpu_tot_sim_cycle", "gpu_tot_sim_cycle_estimated",
        "gpu_tot_sim_insn", "gpu_tot_ipc", "gpu_tot_ipc_estimated",
        "gpu_tot_issued_cta", "l2_hit_rate", "dram_bw_utilization",
    ]

    def run(self, sampled_log: Path, full_log: Path, output_csv: Path):
        sampled = self._parse_stats(sampled_log)
        full    = self._parse_stats(full_log)

        rows = []
        for kernel_name in full:
            s = sampled.get(kernel_name, {})
            f = full[kernel_name]
            row = {"kernel": kernel_name}
            for m in self.METRICS:
                fv = f.get(m, None)
                sv = s.get(m, None)
                if fv and sv and fv != 0:
                    row[f"{m}_err_pct"] = 100 * (sv - fv) / fv
                row[f"{m}_full"]    = fv
                row[f"{m}_sampled"] = sv
            row["speedup"] = f.get("sim_time_s", 1) / max(s.get("sim_time_s", 1), 1e-9)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        self._print_summary(df)

    def _print_summary(self, df):
        err_col = "gpu_tot_sim_cycle_estimated_err_pct"
        if err_col not in df.columns:
            err_col = "gpu_tot_sim_cycle_err_pct"
        errs = df[err_col].abs().dropna()
        print(f"\nCycle error: median={errs.median():.1f}%  "
              f"p90={errs.quantile(0.9):.1f}%  max={errs.max():.1f}%")
        print(f"Speedup: median={df['speedup'].median():.1f}x")
        failures = df[errs > 20]
        if len(failures):
            print(f"\nFAILURES (>20% error):")
            print(failures[["kernel", err_col, "speedup"]].to_string(index=False))
```

---

## Validation Benchmark Script

`tools/validate_suite.sh` — runs the full suite against known-good full-simulation
baselines for the 10-kernel validation set:

```bash
#!/bin/bash
KERNELS="hotspot backprop pathfinder bfs srad_v2 lud heartwall nn nw streamcluster"
for k in $KERNELS; do
    python tools/trace_selector.py \
      --workload "$RODINIA_DIR/$k/$k" \
      --gpu-config tools/configs/sm86_rtx3080.json \
      --output-dir ./validation/$k/ \
      --validate
done
python tools/sim_compare.py --suite-dir ./validation/ --summary
```

---

## Acceptance Gate

`sim_compare.py --suite-dir` emits a PASS/FAIL gate:

| Target | Threshold | Status |
|---|---|---|
| Median cycle error | < 8% | gate |
| p90 cycle error | < 15% | gate |
| Max cycle error | < 30% | warning |
| Median speedup | > 20× (large kernels) | gate |
| Pipeline end-to-end runtime | < 30min (full suite) | advisory |

---

## Notes

- `trace_selector.py` uses subprocess for all external tool invocations (ncu,
  tracer_nvbit, accel-sim.out). Stderr is captured and logged to
  `output/logs/{step}.log`.
- Idempotency: each step checks for output existence before running. Re-running
  with `--skip-*` flags allows partial reruns after failures.
- The full simulation for `--validate` mode uses the same GPU config as the
  sampled run. It does NOT use CTA sampling (`-cta_sampling_mode 0`).

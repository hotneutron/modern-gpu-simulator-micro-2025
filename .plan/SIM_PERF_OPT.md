# Simulator Wall-Clock Performance Optimization (2026-07-21)

Goal: reduce the **wall-clock time** of a single Accel-Sim run (currently ~12 h for the
smallest FA3 forward kernel, kernel 5). This is a *simulator-speed* problem, **not** a
sim-vs-HW *accuracy* problem — none of the work here is allowed to change the reported
`sim_cycle` or any timing metric. All instrumentation must be timing-neutral.

## 1. Context / what we already know (verified from source)

- Accel-Sim is a **single-process, cycle-driven** simulator. Wall-clock is roughly
  `(total modeled sim-cycles) x (per-cycle model cost)`. FA3 forward fills all 132 H100
  SMs for hundreds of thousands of cycles, so the run is long by construction.
- The main loop `gpgpu_sim::cycle()`
  (`gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc:4357`) is split by clock domain
  (CORE / ICNT / L2 / DRAM). Only **two** sections are OpenMP-parallelized:
  - **CORE cluster loop** over 132 clusters — `gpu-sim.cc:4533`
    (`#pragma omp parallel for schedule(runtime) reduction(...)`). This is the bulk of
    the work (SM pipeline, L1, compute).
  - **DRAM partition loop** — `gpu-sim.cc:4445`.
- Sections that remain **serial** (candidate Amdahl ceiling):
  - `icnt_cycle()` per-cluster loop — `gpu-sim.cc:4362-4364`
  - L2 sub-partition `cache_cycle` loop — `gpu-sim.cc:4473-4519`
    (touches the global interconnect via `icnt_pop`/`icnt_push`)
  - `icnt_transfer(0)` (booksim / intersim2 crossbar) — `gpu-sim.cc:4528`
    (global state; hard to parallelize by design)
- A custom OpenMP scheduler already exists — `gpu-sim.cc:4565-4576`: it flips
  `static <-> dynamic` when the active-SM ratio drops below
  `custom_omp_scheduler_ratio_to_dynamic` (config default 0.3), chunk size 1.
- **Thread count is NOT the low-hanging fruit — it is already maxed out.**
  - The launcher path resolves as: `run_simulations.py -c 192` sets the **procman
    `jobLimit`** (number of concurrent *jobs*), **not** the per-run thread count
    (`util/job_launching/procman.py:443-465`, `run_simulations.py:445`).
  - Per-run threads come from `openmp-cpus` in the app YAML
    (`util/job_launching/apps/define-flashattn.yml:36` = `4`), templated into
    `slurm.sim:4` as `--cpus-per-task`.
  - But with `--launcher local`, `slurm.sim:56-60` sees **no** `SLURM_CPUS_PER_TASK`,
    so it **unsets `OMP_NUM_THREADS`**. With `OMP_NUM_THREADS` unset, GNU OpenMP defaults
    to **all logical cores = 192**. That is why observed CPU utilization is ~5000%+
    (many threads) at kernel start.
  - Machine: `nproc = 192` logical = **2 sockets x 48 cores x 2 HT = 96 physical cores**.
    So the run is currently **oversubscribed 2:1** (192 threads on 96 cores).
  - "Utilization drops later" is expected: as the FA3 kernel tail drains, fewer SMs are
    active, the parallel CORE loop has less real work, and the **serial** sections
    (icnt / L2 / booksim) start to dominate — i.e. Amdahl's serial fraction becomes
    visible in the tail.

Conclusion: the real levers are (a) shrinking the **serial fraction**, (b) removing
**oversubscription** overhead, and (c) removing **fork/join** overhead — not adding
threads. We must **profile first** to quantify which of these dominates before changing
any model code.

## 2. Tooling status (verified)

- `perf` present (`/usr/bin/perf`); `gprof` present (`/usr/bin/gprof`); `valgrind` not
  found.
- The built `libcudart.so` (release) is compiled `-O3 -g3` and ships **debug_info, not
  stripped** — perf can attribute samples to functions/lines with **no rebuild**.
- No built-in wall-clock instrumentation exists in `gpu-sim.cc` (no `chrono` /
  `clock_gettime`), so section-level timing must be added if we want the most trustworthy
  serial-vs-parallel split (Step D).

## 3. Profiling plan (measure before changing model code)

### 3.0 How long do I actually have to run? (do NOT run the full 12 h)

Short answer: **almost nothing needs a full run.** Different questions need different run
lengths, and only ONE of them needs the very end of the kernel:

| Step | Question it answers | How long to run | Needs the kernel tail? |
|---|---|---|---|
| A (perf record) | Which functions eat CPU time? | **first ~3-5 min**, sampled | No |
| B (perf stat) | Is 192-thread oversubscription hurting? | **~1 min**, sampled | No |
| C (scaling curve) | What is the serial fraction `f` / best thread count? | **fixed short window**, ~1-3 min per thread setting (6 settings) | No (same window each time) |
| D (section timers) | Where does the wall-clock go, head vs tail? | **head: first few min is enough**; **tail: must reach kernel end** | Tail only |

Why the hotspot / scaling steps (A, B, C) do **not** need a full run:
- The per-cycle model cost is dominated by the same code every cycle. Sampling a few
  million cycles at the start gives the same function-level distribution as the whole run.
- For the scaling curve (C), the ONLY requirement is that **every thread setting runs the
  exact same amount of work** (same sim-cycle window). Absolute length is irrelevant to the
  *shape* of the speedup curve, so we pick the shortest window that is still stable
  (a few million cycles, ~1-3 min at 96 threads).

Why the "tail" question (part of D) is the one exception:
- The utilization drop the user observed happens **near the end** of the kernel, when only
  a few CTAs remain and the parallel CORE loop starves while the serial icnt/L2/booksim
  sections keep running. That behavior literally does not exist in the first few minutes.
- But we do **not** need to sit through 12 h of wall-clock to see it. Two cheaper options:
  1. **Cheapest: use an already-finished run.** A full kernel-5 run already exists under
     `sim_run_12.8/.../H100_80GB-OnlyKernel5/`. If we add the Step-D section timers and
     they print periodically (e.g. every 100k sim-cycles) plus once at kernel end, a single
     future full run captures head AND tail for free — we never profile more than once.
  2. **Faster proxy: shrink the workload just for the tail measurement.** Run a much
     smaller FA3 config (shorter sequence length, e.g. s512 instead of s2048) so the whole
     kernel finishes in minutes. The tail *shape* (parallel starvation as CTAs drain) is a
     structural property of the loop, not of the exact problem size, so a small config
     reproduces it quickly. Use this only to *characterize* the tail; the head numbers
     should come from the real config.

Practical recommendation: run A, B, C on the **real** kernel-5 config but stop after a few
minutes (sampling / fixed window). For D, add periodic timers and read them from the
**next** full run you launch anyway, OR use a shrunk config as a fast proxy. In no case do
we launch a *new* 12 h run solely to profile.

Fallback if `perf` is blocked by kernel `perf_event_paranoid`: use `gprof` (rebuild with
`-pg`) or the in-code timers of Step D.

### Step A — System-hotspot (perf record, no code change)
Launch kernel-5 directly (bypass the job launcher so we control threads and can grab the
PID), let it warm up ~20-30 s, then sample for ~3-5 min.
```bash
# 1. Launch the real kernel-5 run in the background.
cd sim_run_12.8/flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24/\
flashattn_fa3_bf16_bwd_causal_b1_s2048_hd64_nh24___warmup_0___iters_1/H100_80GB-OnlyKernel5
bash justrun.sh > /tmp/k5_profile.log 2>&1 &          # or run accel-sim.out directly
PID=$(pgrep -f "accel-sim.out -config")

# 2. Let it reach steady state, then sample the call graph for 5 min.
sleep 30
perf record -F 199 -g --call-graph dwarf -o /tmp/k5.perf.data -p "$PID" -- sleep 300

# 3. Read the top hotspots.
perf report -i /tmp/k5.perf.data --stdio | head -80

# 4. Stop the run (we do NOT need it to finish for Step A).
kill "$PID"
```
Notes:
- If `perf record -p` fails with a permission error, check
  `cat /proc/sys/kernel/perf_event_paranoid` (needs <= 2, ideally 1). If it cannot be
  lowered, fall back to `gprof` or Step-D timers.
- `-F 199` (199 Hz) is a light sampling rate; raise to 999 for finer detail if needed.

Interpretation:
- CORE / cluster `core_cycle` / `simt_core::cycle` dominate -> parallel path is healthy;
  remaining cost is thread efficiency + serial tail.
- `icnt_transfer` / booksim (`intersim`) / L2 `cache_cycle` large -> Amdahl ceiling is
  here -> serial-section parallelization is the payoff.
- GNU OpenMP runtime large (`gomp_*`, spin/barrier) -> **fork/join + oversubscription**
  overhead is the problem.

### Step B — Oversubscription check (perf stat)
Compare the default 192-thread run against a 96-thread (physical-core) run over the same
short window. Same launch-and-attach pattern as Step A.
```bash
# Default (OMP_NUM_THREADS unset -> 192 threads under --launcher local):
perf stat -p "$PID" -- sleep 60      # watch context-switches, cpu-migrations, IPC

# Then relaunch with an explicit cap and repeat:
OMP_NUM_THREADS=96 bash justrun.sh > /tmp/k5_96.log 2>&1 &
PID=$(pgrep -f "accel-sim.out -config"); sleep 30
perf stat -p "$PID" -- sleep 60
```
- High context-switches / cpu-migrations at 192 threads confirm the 2:1 HT penalty.
- Also compare throughput: grep the periodic `gpu_sim_cycle` progress prints in each log
  and compute **sim-cycles per wall-second** for 192 vs 96. If 96 is not slower (or is
  faster), oversubscription is real and capping threads is a free win.

### Step C — Parallel-efficiency scaling curve (decisive for Amdahl fraction)
Run the **same fixed, short** sim-cycle window at each thread count and record wall-clock.
The absolute window length does not matter as long as it is identical across settings; a
few million cycles (~1-3 min at 96 threads) is plenty.
```bash
for T in 1 4 16 48 96 192; do
  /usr/bin/time -v env OMP_NUM_THREADS=$T timeout 180 bash justrun.sh \
      > /tmp/scale_$T.log 2>&1
  echo "threads=$T"; grep -m1 "Elapsed (wall clock)" /tmp/scale_$T.log
done
```
- To make the window *exactly* equal instead of a wall-clock `timeout` (cleaner), add a
  "stop after N sim-cycles and exit(0)" hook. `g_single_step` already fires `SIGTRAP` at a
  target cycle (`gpu-sim.cc:4583`); repurpose it (or add a `-gpgpu_max_sim_cycle N` knob)
  to exit cleanly at N, then run each thread setting to the same N.
- Plot wall-clock (or 1/wall-clock = speedup) vs threads. The knee where speedup flattens
  gives the serial fraction `f` via Amdahl: `speedup(T) = 1 / (f + (1-f)/T)`.
- Expected: speedup climbs to ~48-96 then flattens (HT + serial icnt/booksim). The flatten
  point tells us whether serial-section work (Candidate #3/#4) is worth it.

### Step D — Section timers to explain the tail drop (most trustworthy split)
Add timing-neutral `std::chrono::steady_clock` accumulators inside `gpgpu_sim::cycle()`
around the five sections, print them periodically (e.g. every 100k sim-cycles) AND once at
kernel end:
- `t_icnt_cycle` (`gpu-sim.cc:4362`), `t_dram` (`4444`), `t_l2` (`4471`),
  `t_icnt_transfer` (`4528`), `t_core` (`4531`).

Implementation sketch (guard behind a config flag so normal runs pay nothing):
```cpp
// pseudo-code, inside gpgpu_sim::cycle()
auto t0 = steady_clock::now();
/* ... CORE section ... */
t_core += steady_clock::now() - t0;   // repeat per section
// every 100k cycles: print the 5 accumulators + reset, so head vs tail is visible
```
This yields the exact **parallel (CORE) vs serial (rest)** wall-clock ratio and how it
shifts from head (all SMs active) to tail (few CTAs left). Timers do NOT change any
simulated timing (verify `sim_cycle` unchanged vs a reference `.oNN`).

**Run length for D:** the periodic prints make a single normal run self-profiling — read
head numbers from the first prints, tail numbers from the last. To get the tail *fast*
without a 12 h run, launch a shrunk FA3 config (e.g. s512) that finishes in minutes; the
tail shape (CORE starvation while icnt/L2/booksim keep ticking) reproduces structurally.

## 4. Deliverables
- perf top-30 functions, classified into {parallel CORE, serial icnt/L2/booksim, OpenMP
  runtime}.
- Speedup-vs-threads curve -> serial fraction `f`, optimal thread count (expected near 96,
  not 192).
- Per-section wall-clock share from cycle() timers, head vs tail.
- A decision, evidence-based, among the candidate fixes in Section 5.

## 5. Candidate optimizations (to be prioritized by the profile — not yet started)
1. **Pin thread count to physical cores (96)** — remove 2:1 HT oversubscription.
   Zero model change; likely immediate. (Also: fix the `--launcher local` path so
   `OMP_NUM_THREADS` is set deliberately instead of defaulting to 192.)
2. **Remove per-cycle fork/join** — the CORE parallel region is opened/closed every
   sim-cycle (hundreds of thousands of times). Hoist to a persistent thread pool
   (`#pragma omp parallel` outside the loop + `#pragma omp for` inside). Source change +
   rebuild.
3. **Parallelize the serial L2 sub-partition loop** (`gpu-sim.cc:4473`). Sub-partitions
   are independent for `cache_cycle`, but the loop calls `icnt_pop`/`icnt_push` on global
   interconnect state — must split the interconnect access into a separate serial stage
   (or lock it) to preserve ordering. Source change + careful validation that `sim_cycle`
   is unchanged.
4. **Simplify the interconnect model** (booksim `icnt_transfer`) — serial and a hard
   Amdahl ceiling. Only if accuracy methodology allows; affects timing, so out of scope
   unless explicitly approved.

## 6. Constraints / invariants
- Every step above is **timing-neutral**: reported `sim_cycle` and all stall/wait metrics
  must be byte-for-byte identical before and after (verify against a reference `.oNN`).
- No config-only shortcuts (trace_enabled off, `-g3` removal, workload shrink) are counted
  here — the user already handles those and wants a *simulator-source* speedup.
- Profile before editing model code; land no speed claim without a measured before/after.

#!/usr/bin/env python3
"""Produce a "spin-trimmed" copy of a GPU dynamic-instruction trace.

An NVIDIA Nsight Compute (NCU) report is used as ground truth for how many
times each producer spin-loop should iterate on real hardware.  For every
FlashAttn-style device kernel (any kernel whose NCU Source block contains
``NANOSLEEP`` lines) we read the exact ``Instructions Executed`` hardware
counter of each NANOSLEEP line.  That number is the number of real spin
iterations the corresponding producer loop executed, summed over every CTA of
that kernel.

The trace's producer warp replays each spin iteration as a small recurring
block that starts at the loop-head TRYWAIT instruction:

    TRYWAIT_pc  (SYNCS.PHASECHK.TRANS64.TRYWAIT, sync.valid)
    TRYWAIT_pc + 0x10  (NANOSLEEP.SYNCS)          # exactly one per iteration
    TRYWAIT_pc + 0x20  (SYNCS.PHASECHK.TRANS64)
    ... BRA back to TRYWAIT_pc

We keep at most the hardware-derived number of iterations per loop (distributed
proportionally across CTAs) and delete the instructions of the excess
iterations.  Every non-spin instruction is preserved byte-identically.

IMPORTANT: this script must run INSIDE the ``gpu-sim`` container, which is the
only place with protobuf-python + ncu.  It also requires the environment
variable ``PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`` to be set before the
generated pb2 modules are imported (the container's protobuf runtime is newer
than its protoc), and the pb2 modules compiled into ``--pb2-dir`` (default
/tmp/pb2).

STRUCTURAL ASSUMPTION (documented + asserted, the only hardcoded constant):
    The loop-head TRYWAIT sits exactly 0x10 bytes before its NANOSLEEP, i.e.
    ``trywait_offset = nanosleep_offset - 0x10``.  We assert both that the
    trace producer stream contains an instruction at trywait_offset that
    recurs, and that ``trywait_offset + 0x10`` also appears in the stream.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path

# The only structural constant we are allowed to hardcode: the fixed distance
# (in bytes) from a loop-head TRYWAIT to its NANOSLEEP.
NANOSLEEP_DELTA = 0x10

DEFAULT_NCU_BIN = "/usr/local/cuda/bin/ncu"

# Stable structural signatures shared by BOTH the trace's *mangled* kernel_name
# (from enhanced_execution_info.json) and NCU's *demangled* "Kernel Name"
# string.  Each of these FlashAttn class tokens appears verbatim as a substring
# on both sides, so the presence of a UNIQUE common token gives an exact,
# order-independent 1:1 kernel identity.  The longer/more specific tokens are
# listed first so that, when extracting the token from a name, we do not stop at
# a shorter prefix that is also a substring (e.g. do not match the common
# prefix "FlashAttnBwd" when the real class is "FlashAttnBwdPreprocess").
FLASHATTN_CLASS_TOKENS = (
    "FlashAttnBwdPostprocessConvertdQ",
    "FlashAttnBwdPreprocess",
    "FlashAttnFwdSm90",
    "FlashAttnBwdSm90",
)


# --------------------------------------------------------------------------- #
# NCU report parsing
# --------------------------------------------------------------------------- #
def dump_ncu_source_csv(ncu_rep, ncu_bin):
    """Run ``ncu --page source --csv`` and return the CSV text."""
    cmd = [ncu_bin, "--import", str(ncu_rep), "--page", "source", "--csv"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, (
        "ncu failed (exit %d): %s" % (proc.returncode, proc.stderr.decode("utf-8", "replace"))
    )
    text = proc.stdout.decode("utf-8", "replace")
    assert text.strip(), "ncu produced empty source-page CSV for %s" % ncu_rep
    return text


def parse_ncu_kernels(csv_text):
    """Parse the NCU source-page CSV into per-kernel spin-loop targets.

    Returns an ``OrderedDict`` keyed by ``(kernel_name, base_va)`` mapping to a
    dict::

        {
          "name": <kernel name>,
          "base": <base virtual address (int)>,
          "targets": { trywait_offset : hw_iterations, ... },   # nonzero only
          "all_nanosleep": [ (nanosleep_offset, count), ... ],
        }

    Only kernels whose Source block contains NANOSLEEP lines are returned.
    Real spin loops are the NANOSLEEP lines with a non-zero ``Instructions
    Executed`` counter; zero-count NANOSLEEP lines are recorded for reporting
    but do not contribute trim targets.  Duplicate blocks (identical
    kernel-name + base VA) are de-duplicated.
    """
    rows = list(csv.reader(csv_text.splitlines()))

    kernels = OrderedDict()
    n = len(rows)
    i = 0
    current_name = None
    while i < n:
        row = rows[i]
        if not row:
            i += 1
            continue
        cell0 = row[0]
        if cell0 == "Kernel Name":
            assert len(row) >= 2, "Malformed 'Kernel Name' row at CSV line %d" % (i + 1)
            current_name = row[1]
            i += 1
            continue
        if cell0 == "Address":
            # This is an Address-table header.  Locate the exact column of
            # "Instructions Executed" BY NAME because its index varies between
            # kernel blocks (some blocks lack the "# Samples" column).
            assert "Instructions Executed" in row, (
                "Address header without 'Instructions Executed' column at CSV line %d" % (i + 1)
            )
            ie_col = row.index("Instructions Executed")
            assert current_name is not None, (
                "Address block at CSV line %d has no preceding 'Kernel Name' row" % (i + 1)
            )
            # Consume the data rows of this block.
            j = i + 1
            block_rows = []
            while j < n and rows[j] and rows[j][0] not in ("Kernel Name", "Address"):
                block_rows.append(rows[j])
                j += 1
            i = j

            if not block_rows:
                continue
            base = int(block_rows[0][0], 16)

            all_nanosleep = []
            targets = {}
            for r in block_rows:
                src = r[1] if len(r) > 1 else ""
                if "NANOSLEEP" not in src:
                    continue
                va = int(r[0], 16)
                off = va - base
                raw = r[ie_col].strip() if ie_col < len(r) else ""
                cnt = int(raw) if raw not in ("", "-") else 0
                all_nanosleep.append((off, cnt))
                if cnt > 0:
                    trywait_off = off - NANOSLEEP_DELTA
                    # A single NANOSLEEP offset must be unique inside a block;
                    # if two spin loops collided that would be a data error.
                    assert trywait_off not in targets, (
                        "Duplicate trywait offset 0x%x in kernel %r" % (trywait_off, current_name)
                    )
                    targets[trywait_off] = cnt

            if not all_nanosleep:
                continue  # not a FlashAttn-style spin kernel

            key = (current_name, base)
            if key in kernels:
                # duplicated (identical) block -> dedupe.  Sanity-check equal.
                prev = kernels[key]
                assert prev["targets"] == targets, (
                    "Duplicate block for %r base 0x%x disagrees on targets" % (current_name, base)
                )
                continue
            kernels[key] = {
                "name": current_name,
                "base": base,
                "targets": targets,
                "all_nanosleep": all_nanosleep,
            }
            continue
        i += 1

    return kernels


# --------------------------------------------------------------------------- #
# Exact 1:1 mapping by mangled kernel name
# --------------------------------------------------------------------------- #
def load_enhanced_execution_info(trace_root):
    """Return {function_unique_id (int): kernel_name (full mangled str)}.

    Reads ``<trace_root>/extra_info/enhanced_execution_info.json`` whose
    ``"kernels"`` list carries, per entry, ``"unique_function_id"`` and the
    FULL mangled ``"kernel_name"``.
    """
    path = Path(trace_root) / "extra_info" / "enhanced_execution_info.json"
    assert path.is_file(), "enhanced_execution_info.json not found: %s" % path
    with open(path) as fh:
        data = json.load(fh)
    assert isinstance(data, dict) and "kernels" in data, (
        "Malformed %s: expected a dict with a 'kernels' key" % path
    )
    out = {}
    for entry in data["kernels"]:
        fid = entry["unique_function_id"]
        name = entry["kernel_name"]
        assert fid not in out, "Duplicate unique_function_id %d in %s" % (fid, path)
        out[int(fid)] = name
    assert out, "No kernels listed in %s" % path
    return out


def cta_function_unique_id(tb):
    """Return the single ``function_unique_id`` carried by a CTA's instructions.

    Every instruction of a given trace kernel belongs to the same function, so
    we read the id from any instruction and assert it is unique across the CTA.
    """
    ids = set()
    for _, warp in iter_warp_entries(tb):
        for ins in warp.instructions:
            ids.add(ins.function_unique_id)
    assert len(ids) == 1, (
        "Expected exactly one function_unique_id per CTA, got %s" % sorted(ids)
    )
    return next(iter(ids))


def flashattn_tokens(name):
    """Return the set of FlashAttn class tokens present in ``name``.

    Longer/more-specific tokens are tested first and, once matched, their
    span is blanked out so that a shorter token which is a prefix substring
    (e.g. "FlashAttnBwdSm90" inside nothing, but "FlashAttnBwd*" families) does
    not spuriously also match.  Returns a set (usually of size 1 for the kernels
    we care about).
    """
    found = set()
    haystack = name
    for tok in FLASHATTN_CLASS_TOKENS:
        if tok in haystack:
            found.add(tok)
            haystack = haystack.replace(tok, " " * len(tok))
    return found


def match_ncu_kernel_by_name(trace_kernel_id, func_id, trace_name, ncu_kernels):
    """Map a trace kernel to its NCU block by EXACT kernel-name identity.

    First try byte-identical string equality between the trace's mangled
    ``kernel_name`` and each NCU block's "Kernel Name" string.  If that fails
    (NCU typically demangles), fall back to the unique FlashAttn class-token
    signature shared by both the mangled and demangled forms.

    The match MUST be unique and 1:1: exactly one NCU block (after the caller's
    base-VA dedup) may match.  Zero or multiple candidates -> ``assert False``
    with a descriptive message naming the trace kernel, its function_unique_id,
    its kernel_name and every candidate NCU name.
    """
    # 1) exact string equality
    exact = [(key, kd) for key, kd in ncu_kernels.items() if kd["name"] == trace_name]
    if len(exact) == 1:
        return exact[0]

    # 2) unique FlashAttn class-token signature
    trace_toks = flashattn_tokens(trace_name)
    candidates = []
    for key, kd in ncu_kernels.items():
        ncu_toks = flashattn_tokens(kd["name"])
        if trace_toks and (trace_toks & ncu_toks):
            candidates.append((key, kd))

    all_ncu_names = [kd["name"] for kd in ncu_kernels.values()]
    assert len(candidates) >= 1, (
        "No NCU block matches trace kernel_%d (function_unique_id=%d, "
        "kernel_name=%r, tokens=%s). Candidate NCU names: %s"
        % (trace_kernel_id, func_id, trace_name, sorted(trace_toks), all_ncu_names)
    )
    assert len(candidates) == 1, (
        "Ambiguous NCU mapping for trace kernel_%d (function_unique_id=%d, "
        "kernel_name=%r, tokens=%s): matched %d NCU blocks: %s"
        % (
            trace_kernel_id,
            func_id,
            trace_name,
            sorted(trace_toks),
            len(candidates),
            [kd["name"] for _, kd in candidates],
        )
    )
    return candidates[0]


# --------------------------------------------------------------------------- #
# Trace protobuf handling
# --------------------------------------------------------------------------- #
def load_pb2(pb2_dir):
    if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") != "python":
        # We cannot safely re-import after protobuf loaded a different impl, so
        # fail loudly with a clear instruction.
        assert False, (
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python must be set in the "
            "environment before running this script (the container's protobuf "
            "runtime is newer than its protoc)."
        )
    sys.path.insert(0, str(pb2_dir))
    try:
        import threadblock_pb2  # noqa: E402
    except Exception as exc:  # pragma: no cover - environment dependent
        assert False, (
            "Failed to import threadblock_pb2 from %s (%s). Compile the schema "
            "first: protoc --python_out=%s *.proto" % (pb2_dir, exc, pb2_dir)
        )
    return threadblock_pb2


def iter_warp_entries(tb):
    """Yield (warp_id, warp) pairs.

    Under the pure-python protobuf impl a ``map<int32,warp>`` iterates as a
    repeated container of MapEntry messages exposing ``.key`` / ``.value``.
    Each yielded ``value`` is mutable in place, and re-serializing ``tb``
    preserves the trimmed instruction list.
    """
    for entry in tb.warps:
        yield entry.key, entry.value


def cta_files(kernel_dir):
    return sorted(p for p in Path(kernel_dir).iterdir() if p.suffix == ".pb")


def kernel_dirs(trace_root):
    base = Path(trace_root) / "threadblocks" / "device_0" / "stream_0"
    assert base.is_dir(), "Trace threadblocks dir not found: %s" % base
    out = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("kernel_"):
            try:
                k = int(d.name.split("_", 1)[1])
            except ValueError:
                continue
            out[k] = d
    return out


# --------------------------------------------------------------------------- #
# Spin-iteration analysis / trimming on a single warp
# --------------------------------------------------------------------------- #
def find_loop_runs(insts, trywait_pc):
    """Find maximal runs of consecutive spin iterations for one loop.

    An iteration is the block ``[TRYWAIT_pc .. next TRYWAIT_pc)`` where the two
    TRYWAIT occurrences are contiguous (no other TRYWAIT_pc of this loop and no
    intervening exit).  We treat as one "run" a maximal sequence of
    back-to-back iterations, i.e. occurrences of ``trywait_pc`` at index
    positions p0 < p1 < ... where each iteration's body ends exactly where the
    next begins.

    Returns a list of runs; each run is a list of (start_idx, end_idx) tuples
    (end exclusive) — one tuple per iteration.  Only iterations whose body ends
    at the next trywait occurrence are grouped; the final iteration in a run
    (the ready-exit one) ends where the loop body diverges to non-spin code.
    """
    n = len(insts)
    occ = [idx for idx in range(n) if insts[idx].pc == trywait_pc]
    if not occ:
        return []

    runs = []
    cur = []  # list of start indices forming a contiguous run
    for pos, start in enumerate(occ):
        if pos + 1 < len(occ):
            nxt = occ[pos + 1]
        else:
            nxt = None
        if not cur:
            cur = [(start, nxt)]
            continue
        prev_start, prev_nxt = cur[-1]
        # contiguous iff the previous iteration's body ends exactly at this
        # iteration's start (i.e. previous trywait's "next" == this start).
        if prev_nxt == start:
            cur.append((start, nxt))
        else:
            runs.append(cur)
            cur = [(start, nxt)]
    if cur:
        runs.append(cur)
    return runs


def count_loop_iters(insts, trywait_pc):
    return sum(len(run) for run in find_loop_runs(insts, trywait_pc))


def trim_warp_loop(insts, trywait_pc, cap):
    """Trim the spin loop at ``trywait_pc`` so at most ``cap`` iterations remain
    across the whole warp stream.  Returns (new_insts, removed_count).

    Within each run of N contiguous iterations we keep the first ``keep`` and
    drop the middle (N - keep), preserving the loop entry and its ready-exit
    final iteration.  ``cap`` is distributed across runs proportionally to run
    size (largest-remainder), never dropping a whole run below 1 iteration.
    """
    runs = find_loop_runs(insts, trywait_pc)
    total = sum(len(r) for r in runs)
    if total == 0 or cap >= total:
        return list(insts), 0

    # Distribute cap across runs proportionally (largest remainder), min 1.
    keeps = []
    for r in runs:
        raw = cap * len(r) / total
        keeps.append([len(r), max(1, int(raw)), raw])
    # fix rounding so sum(keep) == cap (as close as possible, capped by run len)
    def cur_sum():
        return sum(k[1] for k in keeps)
    # increase toward cap by largest fractional remainder while below run size
    while cur_sum() < cap:
        cand = [i for i, k in enumerate(keeps) if k[1] < k[0]]
        if not cand:
            break
        i = max(cand, key=lambda i: keeps[i][2] - keeps[i][1])
        keeps[i][1] += 1
    while cur_sum() > cap:
        cand = [i for i, k in enumerate(keeps) if k[1] > 1]
        if not cand:
            break
        i = min(cand, key=lambda i: keeps[i][2] - keeps[i][1])
        keeps[i][1] -= 1

    # Build set of instruction indices to drop.
    drop = set()
    removed = 0
    for run, (rlen, keep, _) in zip(runs, keeps):
        if keep >= rlen:
            continue
        # keep the first `keep` iterations, drop the middle, but always keep the
        # final iteration of the run (ready-exit) as well.
        # -> keep indices [0 .. keep-2] and the last one (rlen-1).
        keep_first = keep - 1 if keep >= 1 else 0
        keep_idx = set(range(keep_first)) | {rlen - 1}
        for it_i, (start, end) in enumerate(run):
            if it_i in keep_idx:
                continue
            # end is exclusive; for the last iteration end is None but the last
            # iteration is always kept, so end is always a valid index here.
            assert end is not None
            for x in range(start, end):
                drop.add(x)
            removed += 1

    if not drop:
        return list(insts), 0
    new_insts = [ins for idx, ins in enumerate(insts) if idx not in drop]
    return new_insts, removed


# --------------------------------------------------------------------------- #
# Per-CTA producer analysis (reused for planning and trimming)
# --------------------------------------------------------------------------- #
def analyze_cta_producer(tb, target_trywaits):
    """Return (producer_warp, per_loop_iter_counts) for a CTA.

    ``per_loop_iter_counts`` maps trywait_pc -> number of trace iterations in
    the producer warp.  The producer warp is the one containing the most of
    ``target_trywaits``.
    """
    best = None
    for wid, warp in iter_warp_entries(tb):
        pcs = set(i.pc for i in warp.instructions)
        score = len(pcs & target_trywaits)
        if best is None or score > best[0]:
            best = (score, wid, warp)
    assert best is not None and best[0] > 0, (
        "No producer warp found for target trywaits %s"
        % sorted(hex(x) for x in target_trywaits)
    )
    warp = best[2]
    counts = {}
    for tw in target_trywaits:
        c = count_loop_iters(warp.instructions, tw)
        if c > 0:
            counts[tw] = c
    return best[1], counts


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-root", required=True, help="Trace directory (contains threadblocks/)")
    ap.add_argument("--ncu-rep", required=True, help="NCU report (.ncu-rep)")
    ap.add_argument("--out-root", required=True, help="Output trace directory (created/overwritten)")
    ap.add_argument("--kernels", default=None,
                    help="Comma-separated trace kernel ids to trim (default: all mapped)")
    ap.add_argument("--pb2-dir", default="/tmp/pb2", help="Directory with compiled *_pb2.py")
    ap.add_argument("--ncu-bin", default=DEFAULT_NCU_BIN, help="Path to ncu binary")
    ap.add_argument("--map-only", action="store_true",
                    help="Only extract NCU targets + auto-map kernels, do not trim/copy")
    ap.add_argument("--only-cta", default=None,
                    help="For validation: trim only this single CTA 'x,y,z' of the selected "
                         "kernel(s); still copies the whole tree unmodified otherwise")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pb2 = load_pb2(args.pb2_dir)
    threadblock_pb2 = pb2

    # ---- 1. Extract NCU targets ------------------------------------------- #
    csv_text = dump_ncu_source_csv(args.ncu_rep, args.ncu_bin)
    ncu_kernels = parse_ncu_kernels(csv_text)
    assert ncu_kernels, "No FlashAttn-style (NANOSLEEP-bearing) kernels found in NCU report"

    print("=" * 70)
    print("NCU spin-loop targets (kernels with NANOSLEEP lines):")
    for (name, base), kd in ncu_kernels.items():
        tot = sum(kd["targets"].values())
        short = name.split("<")[0]
        # try to surface a human tag from the mangled name
        for tag in FLASHATTN_CLASS_TOKENS:
            if tag in name:
                short = tag
                break
        print("  %-32s base=0x%x loops=%d hw_total=%d"
              % (short, base, len(kd["targets"]), tot))
        if args.verbose:
            for off in sorted(kd["targets"]):
                print("      trywait 0x%-6x hw_iters=%d" % (off, kd["targets"][off]))

    # ---- 2. Map each trace kernel to an NCU kernel BY NAME --------------- #
    # Direct, exact 1:1 mapping: trace kernel_<K> -> read one CTA -> its
    # instructions' function_unique_id -> kernel_name (mangled) from
    # enhanced_execution_info.json -> match to the NCU block's "Kernel Name".
    kdirs = kernel_dirs(args.trace_root)
    assert kdirs, "No kernel_<K> dirs under trace root %s" % args.trace_root

    funcid_to_name = load_enhanced_execution_info(args.trace_root)

    want = None
    if args.kernels:
        want = set(int(x) for x in args.kernels.split(",") if x.strip() != "")

    mapping = {}  # trace_kernel_id -> (ncu_key, ncu_kd, producer_warp_id)
    print("=" * 70)
    print("Mapping trace kernels -> NCU kernels by mangled kernel name "
          "(1 representative CTA each):")
    for kid in sorted(kdirs):
        if want is not None and kid not in want:
            continue
        files = cta_files(kdirs[kid])
        assert files, "kernel_%d has no CTA .pb files" % kid
        tb = threadblock_pb2.threadblock()
        tb.ParseFromString(files[0].read_bytes())

        func_id = cta_function_unique_id(tb)
        assert func_id in funcid_to_name, (
            "kernel_%d function_unique_id=%d not in enhanced_execution_info.json"
            % (kid, func_id)
        )
        trace_name = funcid_to_name[func_id]

        # A trace kernel is a spin (FlashAttn) kernel only if it carries a
        # FlashAttn class token AND some NANOSLEEP-bearing NCU block shares that
        # token.  Kernels without such a token (reduce/elementwise/...) or whose
        # matched NCU block has no NANOSLEEP loops (e.g. Preprocess) are left
        # untrimmed (copied through), which is not an error.
        trace_toks = flashattn_tokens(trace_name)
        has_candidate = any(
            trace_toks & flashattn_tokens(kd["name"]) for kd in ncu_kernels.values()
        )
        if not (trace_name in {kd["name"] for kd in ncu_kernels.values()} or
                (trace_toks and has_candidate)):
            short = next(iter(trace_toks)) if trace_toks else trace_name.split("(")[0][:32]
            print("  kernel_%-3d (funcid=%d) : no NANOSLEEP-bearing NCU block for %r "
                  "-> untrimmed" % (kid, func_id, short))
            continue

        ncu_key, ncu_kd = match_ncu_kernel_by_name(kid, func_id, trace_name, ncu_kernels)

        # verify the +0x10 assumption on the trace side for the loops that
        # actually execute in this representative CTA: locate the producer warp
        # (the one carrying the trywait offsets) and require, for every trywait
        # offset present in its stream, that its NANOSLEEP pc (= trywait + 0x10)
        # is present too.  Not every NCU loop need appear in a single CTA, so we
        # only assert for the trywait offsets that are actually present here.
        prod_wid, _ = analyze_cta_producer(tb, set(ncu_kd["targets"].keys()))
        prod_warp = [w for k, w in iter_warp_entries(tb) if k == prod_wid][0]
        prod_pcs = set(i.pc for i in prod_warp.instructions)
        present = [tw for tw in ncu_kd["targets"] if tw in prod_pcs]
        assert present, (
            "kernel_%d: producer warp %d carries none of the NCU trywait "
            "offsets %s" % (kid, prod_wid, sorted(hex(x) for x in ncu_kd["targets"]))
        )
        for tw in present:
            assert (tw + NANOSLEEP_DELTA) in prod_pcs, (
                "kernel_%d: +0x10 assumption violated for trywait 0x%x "
                "(NANOSLEEP pc 0x%x absent from producer stream)"
                % (kid, tw, tw + NANOSLEEP_DELTA)
            )
        mapping[kid] = (ncu_key, ncu_kd, prod_wid)

        short = ncu_kd["name"].split("<")[0]
        for tag in FLASHATTN_CLASS_TOKENS:
            if tag in ncu_kd["name"]:
                short = tag
                break
        tot = sum(ncu_kd["targets"].values())
        print("  kernel_%-3d (funcid=%d) -> %-32s block (hw spin total %d, %d loops, "
              "base=0x%x, producer warp %d) via NAME match"
              % (kid, func_id, short, tot, len(ncu_kd["targets"]), ncu_kd["base"], prod_wid))

    assert mapping, "No trace kernels could be mapped to NCU kernels"

    if args.map_only:
        print("=" * 70)
        print("--map-only: stopping after extraction + mapping.")
        return

    # ---- 3. Copy tree + trim ---------------------------------------------- #
    out_root = Path(args.out_root)
    trace_root = Path(args.trace_root)
    if out_root.exists():
        print("Output root exists, removing: %s" % out_root)
        shutil.rmtree(out_root)
    print("Copying trace tree %s -> %s ..." % (trace_root, out_root))
    shutil.copytree(trace_root, out_root)

    only_cta = None
    if args.only_cta:
        only_cta = tuple(int(x) for x in args.only_cta.split(","))

    print("=" * 70)
    for kid in sorted(mapping):
        ncu_key, ncu_kd, _ = mapping[kid]
        targets = ncu_kd["targets"]  # trywait_pc -> hw_total
        src_dir = kdirs[kid]
        files = cta_files(src_dir)

        # Pass 1: compute per-CTA trace iteration counts for each loop to build
        # the proportional per-CTA caps.  When only-cta is given we only need
        # that CTA, but the kernel_total must still sum over all CTAs; the NCU
        # target is a whole-kernel sum.  For --only-cta validation we scope the
        # kernel_total to the analysed CTAs so the single-CTA cap is meaningful.
        per_cta_counts = {}  # cta_name -> {trywait_pc: trace_iters}
        analyse = files
        if only_cta is not None:
            analyse = [f for f in files
                       if _cta_tuple_from_name(f.name) == only_cta]
            assert analyse, "kernel_%d has no CTA %s" % (kid, only_cta)

        for f in analyse:
            tb = threadblock_pb2.threadblock()
            tb.ParseFromString(f.read_bytes())
            _, counts = analyze_cta_producer(tb, set(targets.keys()))
            per_cta_counts[f.name] = counts

        # kernel_total per loop (over analysed CTAs)
        kernel_total = Counter()
        for counts in per_cta_counts.values():
            for tw, c in counts.items():
                kernel_total[tw] += c

        # report accumulators
        orig_iter_total = Counter()
        trim_iter_total = Counter()
        orig_inst_total = 0
        trim_inst_total = 0

        # Pass 2: trim + write
        for f in files:
            do_trim = True
            if only_cta is not None and _cta_tuple_from_name(f.name) != only_cta:
                do_trim = False

            out_path = out_root / f.relative_to(trace_root)

            if not do_trim:
                # already copied byte-identically by copytree; nothing to do
                continue

            tb = threadblock_pb2.threadblock()
            tb.ParseFromString(f.read_bytes())

            prod_wid, counts = analyze_cta_producer(tb, set(targets.keys()))
            # count original totals for this CTA
            for _, warp in iter_warp_entries(tb):
                orig_inst_total += len(warp.instructions)

            # find the producer warp entry (mutable)
            prod_entry = None
            for entry in tb.warps:
                if entry.key == prod_wid:
                    prod_entry = entry
                    break
            assert prod_entry is not None

            insts = list(prod_entry.value.instructions)
            for tw, hw_total in targets.items():
                trace_iters = counts.get(tw, 0)
                orig_iter_total[tw] += trace_iters
                if trace_iters == 0:
                    continue
                kt = kernel_total[tw]
                if kt <= 0:
                    cap = trace_iters
                else:
                    cap = round(hw_total * trace_iters / kt)
                # never below 1 if the loop executes, never above this CTA's own count
                cap = max(1, min(cap, trace_iters))
                new_insts, removed = trim_warp_loop(insts, tw, cap)
                insts = new_insts
                trim_iter_total[tw] += (trace_iters - removed)

            # rewrite producer warp instructions
            del prod_entry.value.instructions[:]
            prod_entry.value.instructions.extend(insts)

            # trimmed inst total
            for _, warp in iter_warp_entries(tb):
                trim_inst_total += len(warp.instructions)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(tb.SerializeToString())

        # ---- report for this kernel --------------------------------------- #
        short = ncu_kd["name"].split("<")[0]
        for tag in FLASHATTN_CLASS_TOKENS:
            if tag in ncu_kd["name"]:
                short = tag
                break
        scope = ("CTA %s only" % (only_cta,)) if only_cta is not None else "all CTAs"
        print("kernel_%d -> %s  [%s]" % (kid, short, scope))
        print("  %-10s %12s %14s %14s" % ("trywaitPC", "HW_target", "orig_trace", "trim_trace"))
        for tw in sorted(targets):
            print("  0x%-8x %12d %14d %14d"
                  % (tw, targets[tw], orig_iter_total.get(tw, 0), trim_iter_total.get(tw, 0)))
        print("  total instructions: before=%d after=%d (removed %d)"
              % (orig_inst_total, trim_inst_total, orig_inst_total - trim_inst_total))
        print("-" * 70)

    print("Done. Trimmed trace written to %s" % out_root)


def _cta_tuple_from_name(name):
    # d_0_s_0_k_10_0,0,0.pb  ->  (0,0,0)
    stem = name
    if stem.endswith(".pb"):
        stem = stem[:-3]
    coord = stem.rsplit("_", 1)[-1]
    return tuple(int(x) for x in coord.split(","))


if __name__ == "__main__":
    main()

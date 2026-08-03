#!/usr/bin/env python3
"""trim_spin_from_ncu.py — produce a spin-trimmed copy of an FA3 dynamic trace.

Reusable, server-independent tool. Given a FULL-run trace directory and the
matching NCU report, it caps every producer mbarrier spin loop to the real
hardware NANOSLEEP-exec count (§24 of HW_VS_SIM_SFU.md), by deleting whole spin
iterations from each CTA's producer warp. All non-spin instructions (HGMMA /
MUFU / TMA / LDGSTS / ...) and every other warp are preserved byte-for-byte, and
every extra_info / dynamic_trace.pb sidecar is copied unchanged.

Ground truth: NCU Source page per-PC `Instructions Executed` on each NANOSLEEP
line = the real per-loop spin count on HW (summed over CTAs). Mapping:
  trace loop-head TRYWAIT pc  ==  (NCU NANOSLEEP VA - kernel base VA) - 0x10
i.e. every spin loop is keyed by its static offset (§24 Link 3, verified).

Iteration model (§19, verified against bwd/fwd SASS): one spin iteration is the
fixed 4-instruction body
    lp -> lp+0x10 (NANOSLEEP) -> lp+0x20 (PHASECHK) -> lp+0x30 (BRA)
appearing as 4 CONSECUTIVE instructions. Occurrences that do not match this exact
consecutive pattern (a non-spin op interleaved) are never deleted, so no compute
op is ever removed even when spin loops are interleaved.

Per-CTA cap = round(hw_total * cta_iters / kernel_trace_total), clamped
[1, cta_iters]; sum over CTAs ~= hw_total.

protobuf backend: prefers the fast upb/cpp impl via pre-generated *_pb2 modules
(auto-generated with grpc_tools.protoc if absent); falls back to a dependency-free
varint codec so the tool runs on any server.

Usage:
  trim_spin_from_ncu.py --full-run <trace_dir> --ncu-rep <report.ncu-rep>
                        --out <trimmed_trace_dir>
                        [--ncu-bin ncu] [--kernels 5,10] [--dry-run] [--verify N]

<trace_dir> / <out> point at the `.../traces` directory (the one containing
threadblocks/ and extra_info/).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

NANOSLEEP_DELTA = 0x10           # TRYWAIT sits 0x10 before its NANOSLEEP
SPIN_BODY = (0x0, 0x10, 0x20, 0x30)   # lp, NANOSLEEP, PHASECHK, BRA
DEFAULT_NCU_BIN = "ncu"

FLASHATTN_CLASS_TOKENS = (
    "FlashAttnBwdPostprocessConvertdQ",
    "FlashAttnBwdPreprocess",
    "FlashAttnFwdSm90",
    "FlashAttnBwdSm90",
)


# --------------------------------------------------------------------------- #
# protobuf backend (upb preferred, varint fallback)
# --------------------------------------------------------------------------- #
class PB:
    """Threadblock (de)serialization abstraction."""

    def __init__(self, proto_dir):
        self.mode = None
        self.mod = None
        self._try_load_pb2(proto_dir)

    def _try_load_pb2(self, proto_dir):
        pb2_dir = self._ensure_pb2(proto_dir)
        if pb2_dir is None:
            self.mode = "varint"
            return
        sys.path.insert(0, str(pb2_dir))
        try:
            import threadblock_pb2  # noqa
            self.mod = threadblock_pb2
            self.mode = "pb2"
        except Exception:
            self.mode = "varint"

    def _ensure_pb2(self, proto_dir):
        """Return a dir containing threadblock_pb2.py, generating it if needed."""
        # 1) already-compiled next to us?
        here = Path(__file__).resolve().parent
        for cand in (here / "_pb2_cache", Path(tempfile.gettempdir()) / "trim_pb2"):
            if (cand / "threadblock_pb2.py").is_file():
                return cand
        # 2) generate from .proto with grpc_tools.protoc
        if proto_dir is None or not Path(proto_dir).is_dir():
            return None
        out = Path(tempfile.gettempdir()) / "trim_pb2"
        out.mkdir(parents=True, exist_ok=True)
        protos = [str(p) for p in Path(proto_dir).glob("*.proto")]
        if not protos:
            return None
        try:
            from grpc_tools import protoc
        except Exception:
            return None
        args = ["protoc", "-I%s" % proto_dir, "--python_out=%s" % out] + protos
        if protoc.main(args) != 0:
            return None
        return out if (out / "threadblock_pb2.py").is_file() else None

    # --- unified access: return (list_of_(warp_key, pcs_list), raw) ---------- #
    def load(self, path):
        data = Path(path).read_bytes()
        if self.mode == "pb2":
            tb = self.mod.threadblock()
            tb.ParseFromString(data)
            warps = []
            for k, w in tb.warps.items():
                warps.append((k, [i.pc for i in w.instructions]))
            return tb, warps
        else:
            warps = _vpb_parse(data)
            return data, [(k, [pc for pc, _s in insts]) for k, insts in warps]

    def rewrite_producer(self, handle, prod_key, drop_idx, out_path):
        """Delete instruction indices drop_idx from producer warp prod_key and
        write to out_path. Non-producer warps untouched."""
        if self.mode == "pb2":
            tb = handle
            w = tb.warps[prod_key]
            kept = [ins for j, ins in enumerate(w.instructions) if j not in drop_idx]
            del w.instructions[:]
            w.instructions.extend(kept)
            Path(out_path).write_bytes(tb.SerializeToString())
        else:
            new = _vpb_rewrite(handle, prod_key, drop_idx)
            Path(out_path).write_bytes(new)


# ---- dependency-free varint protobuf codec (fallback) --------------------- #
def _rv(buf, i):
    shift = 0; r = 0
    while True:
        b = buf[i]; i += 1
        r |= (b & 0x7F) << shift
        if not (b & 0x80):
            return r, i
        shift += 7


def _ev(v):
    o = bytearray()
    while True:
        b = v & 0x7F; v >>= 7
        o.append(b | 0x80 if v else b)
        if not v:
            return bytes(o)


def _skip(buf, i, wire):
    if wire == 0:
        _, i = _rv(buf, i)
    elif wire == 1:
        i += 8
    elif wire == 2:
        ln, i = _rv(buf, i); i += ln
    elif wire == 5:
        i += 4
    else:
        raise ValueError("wire %d" % wire)
    return i


def _inst_pc_sync(buf, s, e):
    i = s; pc = 0; sv = False   # proto3: absent uint32 pc defaults to 0
    while i < e:
        tag, i = _rv(buf, i); f, w = tag >> 3, tag & 7
        if f == 1 and w == 0:
            pc, i = _rv(buf, i)
        elif f == 6 and w == 2:
            ln, j = _rv(buf, i); end = j + ln; k = j
            while k < end:
                t2, k = _rv(buf, k); f2, w2 = t2 >> 3, t2 & 7
                if f2 == 1 and w2 == 0:
                    v, k = _rv(buf, k); sv = bool(v)
                else:
                    k = _skip(buf, k, w2)
            i = end
        else:
            i = _skip(buf, i, w)
    return pc, sv


def _vpb_parse(buf):
    """Return list of (warp_key, [(pc, sync_valid), ...])."""
    i = 0; n = len(buf); warps = []
    while i < n:
        tag, i = _rv(buf, i); f, w = tag >> 3, tag & 7
        if f == 2 and w == 2:
            ln, i = _rv(buf, i); entry_end = i + ln; j = i
            key = None; insts = []
            while j < entry_end:
                t2, j = _rv(buf, j); f2, w2 = t2 >> 3, t2 & 7
                if f2 == 1 and w2 == 0:
                    key, j = _rv(buf, j)
                elif f2 == 2 and w2 == 2:
                    l2, j = _rv(buf, j); ve = j + l2; p = j
                    while p < ve:
                        t3, p = _rv(buf, p); f3, w3 = t3 >> 3, t3 & 7
                        if f3 == 2 and w3 == 2:
                            l3, pp = _rv(buf, p); se = pp + l3
                            pc, sv = _inst_pc_sync(buf, pp, se)
                            insts.append((pc, sv)); p = se
                        else:
                            p = _skip(buf, p, w3)
                    j = ve
                else:
                    j = _skip(buf, j, w2)
            warps.append((key, insts))
            i = entry_end
        else:
            i = _skip(buf, i, w)
    return warps


def _vpb_rewrite(buf, prod_key, drop_idx):
    """Re-emit threadblock with drop_idx instructions removed from prod_key warp."""
    out = bytearray(); i = 0; n = len(buf)
    while i < n:
        ts = i
        tag, j = _rv(buf, i); f, w = tag >> 3, tag & 7
        if f == 2 and w == 2:
            ln, k = _rv(buf, j); entry_end = k + ln
            # peek key
            p = k; key = None
            while p < entry_end:
                t2, q = _rv(buf, p); f2, w2 = t2 >> 3, t2 & 7
                if f2 == 1 and w2 == 0:
                    key, q = _rv(buf, q); p = q
                elif f2 == 2 and w2 == 2:
                    l2, _ = _rv(buf, q); p = q + (_ - q) + l2 if False else q
                    # skip value
                    l2b, vs = _rv(buf, q); p = vs + l2b
                else:
                    p = _skip(buf, q, w2)
            if key != prod_key:
                out += buf[ts:entry_end]
            else:
                out += _vpb_emit_entry(buf, ts, entry_end, drop_idx)
            i = entry_end
        else:
            end = _skip(buf, j, w)
            out += buf[ts:end]; i = end
    return bytes(out)


def _vpb_emit_entry(buf, entry_start, entry_end, drop_idx):
    tag, j = _rv(buf, entry_start)          # map-entry tag
    ln, bstart = _rv(buf, j); bend = bstart + ln
    body = bytearray(); p = bstart
    while p < bend:
        ts = p
        t2, q = _rv(buf, p); f2, w2 = t2 >> 3, t2 & 7
        if f2 == 2 and w2 == 2:             # value = warp -> rebuild insts
            l2, vs = _rv(buf, q); ve = vs + l2
            body += _ev((2 << 3) | 2)
            v_new = _vpb_emit_warp(buf, vs, ve, drop_idx)
            body += _ev(len(v_new)); body += v_new
            p = ve
        else:
            end = _skip(buf, q, w2); body += buf[ts:end]; p = end
    return _ev(tag) + _ev(len(body)) + bytes(body)


def _vpb_emit_warp(buf, s, e, drop_idx):
    out = bytearray(); i = s; idx = 0
    while i < e:
        ts = i
        tag, j = _rv(buf, i); f, w = tag >> 3, tag & 7
        if f == 2 and w == 2:               # instruction
            ln, ps = _rv(buf, j); se = ps + ln
            if idx not in drop_idx:
                out += buf[ts:se]
            idx += 1; i = se
        else:
            end = _skip(buf, j, w); out += buf[ts:end]; i = end
    return bytes(out)


# --------------------------------------------------------------------------- #
# NCU report -> per-kernel per-loop HW spin targets
# --------------------------------------------------------------------------- #
def dump_ncu_source_csv(ncu_rep, ncu_bin):
    cmd = [ncu_bin, "--import", str(ncu_rep), "--page", "source", "--csv"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, "ncu failed: %s" % proc.stderr.decode("utf-8", "replace")
    txt = proc.stdout.decode("utf-8", "replace")
    assert txt.strip(), "empty ncu source CSV"
    return txt


def parse_ncu_kernels(csv_text):
    rows = list(csv.reader(csv_text.splitlines()))
    kernels = {}
    n = len(rows); i = 0; cur = None
    while i < n:
        row = rows[i]
        if not row:
            i += 1; continue
        if row[0] == "Kernel Name":
            cur = row[1]; i += 1; continue
        if row[0] == "Address":
            assert "Instructions Executed" in row, "no IE column"
            ie = row.index("Instructions Executed")
            j = i + 1; blk = []
            while j < n and rows[j] and rows[j][0] not in ("Kernel Name", "Address"):
                blk.append(rows[j]); j += 1
            i = j
            if not blk:
                continue
            base = int(blk[0][0], 16)
            targets = {}
            all_ns = []
            for r in blk:
                src = r[1] if len(r) > 1 else ""
                if "NANOSLEEP" not in src:
                    continue
                off = int(r[0], 16) - base
                raw = r[ie].strip() if ie < len(r) else ""
                cnt = int(raw) if raw not in ("", "-") else 0
                all_ns.append((off, cnt))
                if cnt > 0:
                    tw = off - NANOSLEEP_DELTA
                    assert tw not in targets, "dup trywait 0x%x in %s" % (tw, cur)
                    targets[tw] = cnt
            if not all_ns:
                continue
            key = (cur, base)
            if key in kernels:
                assert kernels[key]["targets"] == targets, "dup block disagree %s" % cur
                continue
            kernels[key] = {"name": cur, "base": base, "targets": targets}
            continue
        i += 1
    return kernels


def flashattn_tokens(name):
    found = set(); hay = name
    for tok in FLASHATTN_CLASS_TOKENS:
        if tok in hay:
            found.add(tok); hay = hay.replace(tok, " " * len(tok))
    return found


def load_funcid_to_name(trace_root):
    p = Path(trace_root) / "extra_info" / "enhanced_execution_info.json"
    assert p.is_file(), "missing %s" % p
    data = json.loads(p.read_text())
    out = {}
    for e in data["kernels"]:
        out[int(e["unique_function_id"])] = e["kernel_name"]
    return out


def match_ncu(trace_name, ncu_kernels):
    exact = [(k, kd) for k, kd in ncu_kernels.items() if kd["name"] == trace_name]
    if len(exact) == 1:
        return exact[0]
    toks = flashattn_tokens(trace_name)
    cands = [(k, kd) for k, kd in ncu_kernels.items()
             if toks and (toks & flashattn_tokens(kd["name"]))]
    assert len(cands) == 1, "NCU match for %r not unique: %d" % (trace_name, len(cands))
    return cands[0]


# --------------------------------------------------------------------------- #
# spin iteration detection (fixed 4-pc body)
# --------------------------------------------------------------------------- #
def find_iters(pcs, lp):
    want = [lp + d for d in SPIN_BODY]
    iters = []; n = len(pcs); k = 0
    while k < n:
        if pcs[k] == lp and k + 4 <= n and pcs[k:k + 4] == want:
            iters.append((k, k + 4)); k += 4
        else:
            k += 1
    return iters


def pick_producer(warps, targets):
    """warps: list of (key, pcs). Producer = warp with most matched spin iters."""
    best = None
    for key, pcs in warps:
        tot = sum(len(find_iters(pcs, lp)) for lp in targets)
        if best is None or tot > best[0]:
            best = (tot, key, pcs)
    return best[1], best[2]


# --------------------------------------------------------------------------- #
def kernel_dirs(trace_root):
    base = Path(trace_root) / "threadblocks" / "device_0" / "stream_0"
    out = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("kernel_"):
            try:
                out[int(d.name.split("_", 1)[1])] = d
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-run", required=True, help=".../traces dir of the FULL run")
    ap.add_argument("--ncu-rep", required=True)
    ap.add_argument("--out", required=True, help="output .../traces dir (trimmed)")
    ap.add_argument("--ncu-bin", default=DEFAULT_NCU_BIN)
    ap.add_argument("--kernels", default=None, help="comma list of trace kernel ids (default: all mapped)")
    ap.add_argument("--proto-dir", default=None,
                    help="dir with *.proto (to auto-generate pb2); default: repo dynamic_trace")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", type=int, default=4, help="verify N CTAs per kernel after trim")
    args = ap.parse_args()

    full = Path(args.full_run)
    assert (full / "threadblocks").is_dir(), "not a traces dir: %s" % full

    proto_dir = args.proto_dir
    if proto_dir is None:
        # repo default: util/traces_enhanced/dynamic_trace relative to this file
        here = Path(__file__).resolve().parent
        cand = here.parent / "traces_enhanced" / "dynamic_trace"
        proto_dir = str(cand) if cand.is_dir() else None
    pb = PB(proto_dir)
    print("[pb backend] %s" % pb.mode)

    # 1) NCU targets
    ncu_kernels = parse_ncu_kernels(dump_ncu_source_csv(args.ncu_rep, args.ncu_bin))
    assert ncu_kernels, "no NANOSLEEP-bearing kernels in NCU report"
    funcid_to_name = load_funcid_to_name(full)
    kdirs = kernel_dirs(full)
    want = set(int(x) for x in args.kernels.split(",")) if args.kernels else None

    # map trace kernel -> targets{trywait_pc: hw_total}
    kernel_targets = {}
    for kid, kdir in kdirs.items():
        if want and kid not in want:
            continue
        files = sorted(p for p in kdir.iterdir() if p.suffix == ".pb")
        if not files:
            continue
        _, warps = pb.load(files[0])
        # function_unique_id of this kernel (from any inst) via name lookup:
        # enhanced json is keyed by funcid; we need this kernel's funcid. The
        # per-CTA pb doesn't expose it here, so read it from the pb2 path if
        # available; otherwise fall back to matching by kernel dir order is
        # unsafe -> require pb2 for funcid. Simpler: read funcid from json by
        # scanning which name maps to an NCU block AND appears; but robust route:
        # get funcid from the trace instruction (field 4) via a raw peek.
        funcid = _peek_funcid(files[0])
        assert funcid in funcid_to_name, "funcid %s not in enhanced json" % funcid
        tname = funcid_to_name[funcid]
        toks = flashattn_tokens(tname)
        if not toks:
            print("  kernel_%d (%s): non-flash, skipped" % (kid, tname.split("(")[0][:30]))
            continue
        try:
            _, kd = match_ncu(tname, ncu_kernels)
        except AssertionError as ex:
            print("  kernel_%d: %s -> no NCU match, skipped" % (kid, list(toks)))
            continue
        if not kd["targets"]:
            print("  kernel_%d (%s): NCU block has no spin loops, skipped" % (kid, list(toks)))
            continue
        kernel_targets[kid] = kd["targets"]
        print("  kernel_%d -> %s  loops=%d hw_spin=%d"
              % (kid, list(toks)[0], len(kd["targets"]), sum(kd["targets"].values())))
    assert kernel_targets, "no trimmable kernels mapped"

    # 2) copy tree once (unless dry-run)
    out = Path(args.out)
    if not args.dry_run:
        if out.exists():
            shutil.rmtree(out)
        print("copy %s -> %s" % (full, out))
        shutil.copytree(full, out)

    # 3) trim each kernel
    for kid, targets in kernel_targets.items():
        _trim_kernel(pb, full, out, kdirs[kid], targets, args.dry_run)

    # 4) verify
    if not args.dry_run and args.verify > 0:
        ok = True
        for kid, targets in kernel_targets.items():
            ok &= _verify_kernel(pb, full, out, kid, targets, args.verify)
        print("VERIFY:", "OK" if ok else "FAILED")
    print("done.", "(dry-run)" if args.dry_run else out)


def _peek_funcid(path):
    """Read function_unique_id (instruction field 4) of the first instruction,
    via a raw varint scan (works regardless of the protobuf backend)."""
    return _vpb_parse_raw_funcid(Path(path).read_bytes())


def _vpb_parse_raw_funcid(buf):
    """Return the function_unique_id of the first instruction found."""
    i = 0; n = len(buf)
    while i < n:
        tag, i = _rv(buf, i); f, w = tag >> 3, tag & 7
        if f == 2 and w == 2:
            ln, i = _rv(buf, i); entry_end = i + ln; j = i
            while j < entry_end:
                t2, j = _rv(buf, j); f2, w2 = t2 >> 3, t2 & 7
                if f2 == 2 and w2 == 2:
                    l2, j = _rv(buf, j); ve = j + l2; p = j
                    while p < ve:
                        t3, p = _rv(buf, p); f3, w3 = t3 >> 3, t3 & 7
                        if f3 == 2 and w3 == 2:
                            l3, pp = _rv(buf, p); se = pp + l3
                            # scan instruction for field 4 (function_unique_id)
                            q = pp
                            while q < se:
                                t4, q = _rv(buf, q); f4, w4 = t4 >> 3, t4 & 7
                                if f4 == 4 and w4 == 0:
                                    v, q = _rv(buf, q); return v
                                else:
                                    q = _skip(buf, q, w4)
                            p = se
                        else:
                            p = _skip(buf, p, w3)
                    return None
                else:
                    j = _skip(buf, j, w2)
            i = entry_end
        else:
            i = _skip(buf, i, w)
    return None


def _producer_and_counts(pb, path, targets):
    _, warps = pb.load(path)
    # producer = warp with the most matched spin iterations across target loops
    best = None
    for key, pcs in warps:
        tot = sum(len(find_iters(pcs, lp)) for lp in targets)
        if best is None or tot > best[0]:
            best = (tot, key, pcs)
    return best[1], best[2]


def _trim_kernel(pb, full, out, kdir, targets, dry):
    files = sorted(p for p in kdir.iterdir() if p.suffix == ".pb")
    # pass 1: kernel totals per loop
    kernel_total = Counter(); per_cta = {}
    for f in files:
        prod, pcs = _producer_and_counts(pb, f, targets)
        counts = {}
        for lp in targets:
            n = len(find_iters(pcs, lp))
            if n:
                counts[lp] = n; kernel_total[lp] += n
        per_cta[f.name] = (prod, counts)
    orig = Counter(); trim = Counter()
    for f in files:
        prod, counts = per_cta[f.name]
        handle, warps = pb.load(f)
        pcs = dict(warps)[prod]
        drop = set()
        for lp, n_it in counts.items():
            orig[lp] += n_it
            kt = kernel_total[lp]
            cap = round(targets[lp] * n_it / kt) if kt else n_it
            cap = max(1, min(cap, n_it))
            trim[lp] += cap
            if cap >= n_it:
                continue
            its = find_iters(pcs, lp)
            keep = set(range(cap - 1)) | {n_it - 1}
            for it_i, (a, b) in enumerate(its):
                if it_i in keep:
                    continue
                for x in range(a, b):
                    drop.add(x)
        if drop and not dry:
            op = out / f.relative_to(full)
            pb.rewrite_producer(handle, prod, drop, op)
    kid = kdir.name
    print("[%s] per-loop trim:" % kid)
    for lp in sorted(targets):
        print("   0x%-8x hw=%d orig=%d trim=%d"
              % (lp, targets[lp], orig.get(lp, 0), trim.get(lp, 0)))
    print("   TOTAL hw=%d orig=%d trim=%d"
          % (sum(targets.values()), sum(orig.values()), sum(trim.values())))


def _verify_kernel(pb, full, out, kid, targets, nverify):
    kb = full / "threadblocks/device_0/stream_0" / ("kernel_%d" % kid)
    files = sorted(p.name for p in kb.iterdir() if p.suffix == ".pb")[:nverify]
    known = set(lp + d for lp in targets for d in SPIN_BODY)
    ok = True
    for fn in files:
        _, wb = pb.load(kb / fn)
        _, wt = pb.load(out / "threadblocks/device_0/stream_0" / ("kernel_%d" % kid) / fn)
        db = dict(wb); dt = dict(wt)
        if set(db) != set(dt):
            print("   [FAIL] %s warp keys differ" % fn); ok = False; continue
        prod, _ = _producer_and_counts(pb, kb / fn, targets)
        for k in db:
            if k == prod:
                # subsequence + only spin-body pcs deleted
                bpc, tpc = db[k], dt[k]
                j = 0; deleted = []
                for pc in bpc:
                    if j < len(tpc) and tpc[j] == pc:
                        j += 1
                    else:
                        deleted.append(pc)
                if j != len(tpc):
                    print("   [FAIL] %s producer not a subsequence" % fn); ok = False
                bad = [pc for pc in deleted if pc not in known]
                if bad:
                    print("   [FAIL] %s deleted non-spin pcs: %s"
                          % (fn, [hex(x) for x in bad[:5]])); ok = False
            else:
                if db[k] != dt[k]:
                    print("   [FAIL] %s non-producer warp %d changed" % (fn, k)); ok = False
    if ok:
        print("   [kernel_%d] verify OK (%d CTAs): only spin-body deletions, other warps intact"
              % (kid, len(files)))
    return ok


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


# SYNCS.ARRIVE.TRANS64 variants whose operand/semantics have been validated
# against real runtime traces (see .plan/SYNC_ISA.md). Any other arrive variant
# that actually executes is rejected by classify_sync_opcode().
#
# Classification is driven at runtime by the captured semantic_raw, so each
# validated variant only needs a confirmed operand layout (operand2 = barrier's
# semantic input), not a fixed static kind:
#   - semantic_raw == 0  -> simple ARRIVE
#   - semantic_raw != 0  -> ARRIVE_EXPECT_TX
VALIDATED_ARRIVE_OPCODES = {
    # CUTLASS / FA3 (observed executing)
    "SYNCS.ARRIVE.TRANS64",            # suffix-less, operand2 = register (tx bytes)
    "SYNCS.ARRIVE.TRANS64.RED.A1T0",  # operand2 = RZ (plain arrive)
    # nvcc microbenches (observed executing)
    "SYNCS.ARRIVE.TRANS64.RED.A0TR",  # operand2 = register (tx bytes)
    "SYNCS.ARRIVE.TRANS64.ART0",      # operand2 = register, runtime value 0 (plain arrive)
    "SYNCS.ARRIVE.TRANS64.A1T0",      # operand2 = RZ (plain arrive)
}


def classify_sync_opcode(opcode: str):
    if opcode.startswith("SYNCS.EXCH"):
        return {
            "sync_kind": "EXCH",
            "barrier_operand_index": 1,
            "semantic_operand_index": 2,
            "semantic_operand_role": "EXCH_ARRIVE_COUNT_ENCODED",
        }
    # All SYNCS.ARRIVE.TRANS64 variants are recorded with operand 2 as the
    # semantic input and labelled ARRIVE_EXPECT_TX. The actual ARRIVE vs
    # ARRIVE_EXPECT_TX decision is made at runtime in the simulator from the
    # captured semantic_raw (see remodeling/sm.cc):
    #   - semantic_raw == 0  -> simple ARRIVE (no tx bytes)
    #   - semantic_raw != 0  -> ARRIVE_EXPECT_TX
    #
    # Only two arrive variants have been validated end-to-end against real
    # runtime traces (see .plan/SYNC_ISA.md):
    #   - "SYNCS.ARRIVE.TRANS64"          (suffix-less, operand2 = register, tx bytes)
    #   - "SYNCS.ARRIVE.TRANS64.RED.A1T0" (operand2 = RZ, plain arrive)
    # Other statically-present variants (e.g. ".RED.A0T1", ".A1T0", nvcc ".ART0")
    # were never observed executing, so their semantics are unverified. If any
    # such variant actually executes it must be validated before trusting it;
    # we flag it here so the resolver build fails loudly instead of silently
    # mislabelling it.
    if opcode.startswith("SYNCS.ARRIVE.TRANS64"):
        if opcode not in VALIDATED_ARRIVE_OPCODES:
            raise SystemExit(
                "unvalidated SYNCS.ARRIVE.TRANS64 variant executed: "
                f"'{opcode}'. Only {sorted(VALIDATED_ARRIVE_OPCODES)} are "
                "validated. Validate its operand/semantics (see "
                ".plan/SYNC_ISA.md) before adding it here."
            )
        return {
            "sync_kind": "ARRIVE_EXPECT_TX",
            "barrier_operand_index": 1,
            "semantic_operand_index": 2,
            "semantic_operand_role": "EXPECT_TX_BYTES",
        }
    if opcode.startswith("SYNCS.PHASECHK"):
        return {
            "sync_kind": "TRYWAIT" if "TRYWAIT" in opcode else "PHASECHK",
            "barrier_operand_index": 1,
            "semantic_operand_index": 2,
            "semantic_operand_role": "WAIT_STATE",
        }
    return None


def get_operand_text(operand_strings, operand_index):
    if operand_index is None:
        return None
    if operand_index < 0 or operand_index >= len(operand_strings):
        return None
    return operand_strings[operand_index]


def build_resolver(extra_info_dir: Path):
    info_path = extra_info_dir / "enhanced_execution_info.json"
    if not info_path.exists():
        raise SystemExit(f"missing: {info_path}")

    data = json.loads(info_path.read_text())
    resolver = []
    for kernel in data.get("kernels", []):
        unique_function_id = kernel.get("unique_function_id")
        kernel_name = kernel.get("kernel_name")
        for instruction in kernel.get("instructions", []):
            opcode = instruction.get("op_code", "")
            classification = classify_sync_opcode(opcode)
            if classification is None:
                continue

            operand_strings = [
                operand.get("operand_string", "")
                for operand in instruction.get("operands", [])
            ]
            barrier_operand_index = classification["barrier_operand_index"]
            semantic_operand_index = classification["semantic_operand_index"]
            resolver.append({
                "unique_function_id": unique_function_id,
                "kernel_name": kernel_name,
                "pc": int(instruction.get("pc_num_dec", 0)),
                "pc_hex": f"0x{int(instruction.get('pc_num_dec', 0)):x}",
                "opcode": opcode,
                "sync_kind": classification["sync_kind"],
                "barrier_operand_index": barrier_operand_index,
                "semantic_operand_index": semantic_operand_index,
                "semantic_operand_role": classification["semantic_operand_role"],
                "operand_strings": operand_strings,
                "barrier_operand_text": get_operand_text(
                    operand_strings, barrier_operand_index
                ),
                "semantic_operand_text": get_operand_text(
                    operand_strings, semantic_operand_index
                ),
            })

    return {
        "version": 1,
        "benchmark_name": data.get("benchmark_name"),
        "resolver": resolver,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("extra_info_dir", type=Path)
    parser.add_argument("--resolver-out", type=Path)
    args = parser.parse_args()

    payload = build_resolver(args.extra_info_dir)
    resolver_out = args.resolver_out or (args.extra_info_dir / "sync_operand_resolver.json")
    resolver_out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

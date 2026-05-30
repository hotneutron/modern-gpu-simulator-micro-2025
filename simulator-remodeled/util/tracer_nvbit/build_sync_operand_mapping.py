#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def classify_sync_opcode(opcode: str):
    if opcode.startswith("SYNCS.EXCH"):
        return {
            "sync_kind": "EXCH",
            "barrier_operand_index": 1,
            "semantic_operand_index": 2,
            "semantic_operand_role": "EXCH_ARRIVE_COUNT_ENCODED",
        }
    if opcode.startswith("SYNCS.ARRIVE.TRANS64.RED.A0TR"):
        return {
            "sync_kind": "ARRIVE_EXPECT_TX",
            "barrier_operand_index": 1,
            "semantic_operand_index": 2,
            "semantic_operand_role": "EXPECT_TX_BYTES",
        }
    if opcode.startswith("SYNCS.ARRIVE.TRANS64.ART0"):
        return {
            "sync_kind": "ARRIVE_COUNTED",
            "barrier_operand_index": 1,
            "semantic_operand_index": 2,
            "semantic_operand_role": "ARRIVE_COUNT",
        }
    if opcode.startswith("SYNCS.ARRIVE.TRANS64.A1T0"):
        return {
            "sync_kind": "ARRIVE",
            "barrier_operand_index": 1,
            "semantic_operand_index": None,
            "semantic_operand_role": "NONE",
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

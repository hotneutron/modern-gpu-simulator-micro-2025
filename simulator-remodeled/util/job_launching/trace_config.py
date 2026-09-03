#!/usr/bin/env python3

import json
import re
from pathlib import Path


TMA_OPCODE_PREFIXES = (
    "UBLKCP",
    "UBLKPF",
    "UBLKRED",
    "UTMACCTL",
    "UTMACMDFLUSH",
    "UTMALDG",
    "UTMAPF",
    "UTMAREDG",
    "UTMASTG",
)


class TraceFeatureDetectionError(RuntimeError):
    pass


def effective_bool_option(config_text, option_name):
    option_re = re.compile(
        r"(?<!\S)" + re.escape(option_name) + r"\s+([01])(?=\s|$)"
    )
    values = []
    for line in config_text.splitlines():
        config_part = line.split("#", 1)[0]
        values.extend(match.group(1) for match in option_re.finditer(config_part))
    if not values:
        return None
    return values[-1] == "1"


def trace_uses_tma(trace_dir):
    metadata_path = (
        Path(trace_dir) / "extra_info" / "enhanced_execution_info.json"
    )
    try:
        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as error:
        raise TraceFeatureDetectionError(
            f"cannot classify TMA usage from {metadata_path}: {error}"
        ) from error

    kernels = metadata.get("kernels") if isinstance(metadata, dict) else None
    if not isinstance(kernels, list) or not kernels:
        raise TraceFeatureDetectionError(
            f"cannot classify TMA usage: {metadata_path} has no kernel list"
        )

    for kernel_index, kernel in enumerate(kernels):
        instructions = kernel.get("instructions") if isinstance(kernel, dict) else None
        if not isinstance(instructions, list):
            raise TraceFeatureDetectionError(
                "cannot classify TMA usage: "
                f"{metadata_path} kernel {kernel_index} has no instruction list"
            )
        for instruction_index, instruction in enumerate(instructions):
            opcode = (
                instruction.get("op_code")
                if isinstance(instruction, dict)
                else None
            )
            if not isinstance(opcode, str):
                raise TraceFeatureDetectionError(
                    "cannot classify TMA usage: "
                    f"{metadata_path} kernel {kernel_index} instruction "
                    f"{instruction_index} has no op_code"
                )
            if opcode.startswith(TMA_OPCODE_PREFIXES):
                return True
    return False


def non_tma_config_override(config_text, trace_dir):
    if effective_bool_option(
        config_text, "-tma_real_base_addr_enable"
    ) is not True:
        return ""
    if trace_uses_tma(trace_dir):
        return ""

    lines = [
        "# Auto-selected: no traced kernel contains a TMA-family instruction.",
        "-tma_real_base_addr_enable 0",
    ]
    dependent_options = (
        "-tma_operand_addr_tiling_enable",
        "-tma_debug_enable",
        "-wait_block_late_arrival_instrument_enable",
        "-wait_block_exact_generation_enable",
        "-wait_block_enable",
        "-wait_block_debug",
    )
    for option_name in dependent_options:
        if effective_bool_option(config_text, option_name) is True:
            lines.append(f"{option_name} 0")
    return "\n".join(lines) + "\n"

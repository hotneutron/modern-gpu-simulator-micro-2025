#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SSH_USER="${SSH_USER:-jihyun}"
KERNEL4_HOST="${KERNEL4_HOST:-10.73.164.139}"
KERNEL5_HOST="${KERNEL5_HOST:-10.73.164.139}"
KERNEL9_HOST="${KERNEL9_HOST:-10.73.164.173}"
KERNEL10_HOST="${KERNEL10_HOST:-10.73.164.173}"
REMOTE_BASE="${REMOTE_BASE:-/home/jihyun/modern-gpu-simulator-micro-2025/simulator-remodeled/sim_run_12.8}"
LOCAL_BASE="${LOCAL_BASE:-$SCRIPT_DIR/sim_run_12.8}"

default_benchmarks=(
    "flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh24"
    "flashattn-fa3-bf16-bwd-causal-b1-s1024-hd64-nh24"
    "flashattn-fa3-bf16-bwd-causal-b1-s2048-hd64-nh8"
    "flashattn-fa3-bf16-bwd-causal-b1-s2048-hd128-nh24"
    "flashattn-fa3-bf16-bwd-noncausal-b1-s2048-hd64-nh24"
)

if (( $# > 0 )); then
    benchmarks=("$@")
else
    benchmarks=("${default_benchmarks[@]}")
fi

summary_entries=()

fetch_latest_pair() {
    local benchmark="$1"
    local kernel="$2"
    local host_ip="$3"
    local host="${SSH_USER}@${host_ip}"
    local run="${benchmark//-/_}___warmup_0___iters_1"
    local remote_dir="${REMOTE_BASE}/${benchmark}/${run}/H100_80GB-OnlyKernel${kernel}"
    local local_dir="${LOCAL_BASE}/${benchmark}/${run}/H100_80GB-OnlyKernel${kernel}"
    local remote_files
    local filename
    local job_id
    local latest_id=-1
    local latest_e=""
    local latest_o
    local remote_sizes
    local remote_e_size
    local remote_o_size
    local local_e
    local local_o
    local local_e_size
    local local_o_size

    echo "Checking ${host}: ${benchmark} Kernel${kernel}"

    if ! remote_files="$(
        ssh "$host" \
            "find '$remote_dir' -maxdepth 1 -type f -printf '%f\n'"
    )"; then
        echo "ERROR: could not list $remote_dir on $host" >&2
        return 1
    fi

    while IFS= read -r filename; do
        if [[ "$filename" =~ \.e([0-9]+)$ ]]; then
            job_id="${BASH_REMATCH[1]}"
            if (( job_id > latest_id )); then
                latest_id="$job_id"
                latest_e="$filename"
            fi
        fi
    done <<< "$remote_files"

    if [[ -z "$latest_e" ]]; then
        echo "ERROR: no .eN log found in $remote_dir" >&2
        return 1
    fi

    latest_o="${latest_e%.e${latest_id}}.o${latest_id}"
    if ! remote_sizes="$(
        ssh "$host" \
            "stat -c '%s' '$remote_dir/$latest_e' '$remote_dir/$latest_o'"
    )"; then
        echo "ERROR: matching $latest_o was not found on $host" >&2
        return 1
    fi

    remote_e_size="${remote_sizes%%$'\n'*}"
    remote_o_size="${remote_sizes##*$'\n'}"
    local_e="$local_dir/$latest_e"
    local_o="$local_dir/$latest_o"

    if [[ -f "$local_e" && -f "$local_o" ]]; then
        local_e_size="$(wc -c < "$local_e" | tr -d '[:space:]')"
        local_o_size="$(wc -c < "$local_o" | tr -d '[:space:]')"
        if [[ "$local_e_size" == "$remote_e_size" &&
              "$local_o_size" == "$remote_o_size" ]]; then
            echo "No change: Kernel${kernel} job ${latest_id}"
            summary_entries+=(
                "${benchmark} | Kernel${kernel} | e${latest_id}/o${latest_id} (no change, not copied)"
            )
            return 0
        fi
    fi

    mkdir -p "$local_dir"
    if ! scp \
        "$host:$remote_dir/$latest_e" \
        "$host:$remote_dir/$latest_o" \
        "$local_dir/"; then
        echo "ERROR: failed to copy Kernel${kernel} job ${latest_id}" >&2
        return 1
    fi

    echo "Copied Kernel${kernel} job ${latest_id} to $local_dir"
    summary_entries+=("${benchmark} | Kernel${kernel} | e${latest_id}/o${latest_id}")
}

status=0
for requested_benchmark in "${benchmarks[@]}"; do
    benchmark="${requested_benchmark#flashattn:}"
    if [[ "$benchmark" == *-noncausal-* ]]; then
        fetch_latest_pair "$benchmark" 4 "$KERNEL4_HOST" || status=1
        fetch_latest_pair "$benchmark" 9 "$KERNEL9_HOST" || status=1
    else
        fetch_latest_pair "$benchmark" 5 "$KERNEL5_HOST" || status=1
        fetch_latest_pair "$benchmark" 10 "$KERNEL10_HOST" || status=1
    fi
done

echo
echo "Copy summary (${#summary_entries[@]} pairs)"
if (( ${#summary_entries[@]} == 0 )); then
    echo "  No logs found."
else
    for summary in "${summary_entries[@]}"; do
        echo "  - $summary"
    done
fi

exit "$status"

#!/usr/bin/env python3

from optparse import OptionParser
import hashlib
import json
import os
import subprocess
import os
import shlex
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
import sys
sys.path.insert(0,os.path.join(this_directory,"..","job_launching"))
import common
import re
import shutil
import glob
import datetime
import yaml
import common
import re
import datetime


LIGHTER_TRACER_FEATURE_ORDER = (
    "compact_wire",
    "prune_operands",
    "host_count",
    "metadata_record",
    "sparse_reconstruct",
    "warp_batch_1k",
    "warp_batch_2k",
    "warp_batch_4k",
    "spin_fusion",
)

LIGHTER_TRACER_IMPLEMENTED_FEATURES = frozenset(
    ("compact_wire", "prune_operands", "host_count")
)

LIGHTER_TRACER_PRODUCTION_FEATURES = frozenset(
    ("compact_wire", "prune_operands")
)

LIGHTER_TRACER_FEATURE_DEPENDENCIES = {
    "prune_operands": frozenset(("compact_wire",)),
    "host_count": frozenset(("compact_wire",)),
    "metadata_record": frozenset(("compact_wire",)),
    "sparse_reconstruct": frozenset(
        ("compact_wire", "prune_operands")
    ),
    "warp_batch_1k": frozenset(("compact_wire",)),
    "warp_batch_2k": frozenset(("compact_wire",)),
    "warp_batch_4k": frozenset(("compact_wire",)),
    "spin_fusion": frozenset(
        ("compact_wire", "prune_operands", "sparse_reconstruct")
    ),
}

LIGHTER_TRACER_WARP_BATCH_FEATURES = frozenset(
    ("warp_batch_1k", "warp_batch_2k", "warp_batch_4k")
)


class LighterTracerConfig:
    def __init__(self, requested, features, source, publish_default=False):
        self.requested = requested
        self.features = tuple(features)
        self.source = source
        self.publish_default = publish_default
        self.canonical = ",".join(self.features) if self.features else "off"
        self.config_hash = hashlib.sha256(
            self.canonical.encode("ascii")
        ).hexdigest()[:8]

    def enabled(self, feature):
        return feature in self.features

    @property
    def compact_wire_enabled(self):
        return self.enabled("compact_wire")

    @property
    def artifact_suffix(self):
        if not self.features or self.publish_default:
            return ""
        readable = "+".join(self.features)
        return "_lighter-{0}".format(readable)


def parse_lighter_tracer_feature_set(raw_value):
    if raw_value is None:
        return ()

    stripped = raw_value.strip()
    if not stripped:
        raise ValueError("LIGHTER_TRACER must not be empty; use 'off'")
    if stripped == "off":
        return ()
    if stripped == "all":
        return tuple(
            feature
            for feature in LIGHTER_TRACER_FEATURE_ORDER
            if feature in LIGHTER_TRACER_PRODUCTION_FEATURES
        )

    requested_tokens = stripped.split(",")
    tokens = []
    seen = set()
    for requested_token in requested_tokens:
        token = requested_token.strip()
        if not token:
            raise ValueError("LIGHTER_TRACER contains an empty feature token")
        if token in ("off", "all"):
            raise ValueError(
                "LIGHTER_TRACER token '{0}' cannot be combined".format(token)
            )
        if token not in LIGHTER_TRACER_FEATURE_ORDER:
            raise ValueError(
                "unknown LIGHTER_TRACER feature '{0}'".format(token)
            )
        if token in seen:
            raise ValueError(
                "duplicate LIGHTER_TRACER feature '{0}'".format(token)
            )
        seen.add(token)
        tokens.append(token)

    selected = frozenset(tokens)
    selected_batches = selected & LIGHTER_TRACER_WARP_BATCH_FEATURES
    if len(selected_batches) > 1:
        raise ValueError(
            "LIGHTER_TRACER warp_batch_1k/2k/4k features are mutually exclusive"
        )

    for feature in tokens:
        missing = (
            LIGHTER_TRACER_FEATURE_DEPENDENCIES.get(
                feature, frozenset()
            )
            - selected
        )
        if missing:
            raise ValueError(
                "LIGHTER_TRACER feature '{0}' requires: {1}".format(
                    feature, ",".join(sorted(missing))
                )
            )

    unimplemented = selected - LIGHTER_TRACER_IMPLEMENTED_FEATURES
    if unimplemented:
        raise ValueError(
            "LIGHTER_TRACER features are recognized but not implemented: "
            "{0}".format(",".join(sorted(unimplemented)))
        )

    return tuple(
        feature
        for feature in LIGHTER_TRACER_FEATURE_ORDER
        if feature in selected
    )


def resolve_lighter_tracer(lighter_tracer_value):
    if lighter_tracer_value is None:
        return LighterTracerConfig(
            requested="off",
            features=(),
            source="default",
        )

    stripped_value = lighter_tracer_value.strip()
    features = parse_lighter_tracer_feature_set(lighter_tracer_value)
    return LighterTracerConfig(
        requested=lighter_tracer_value,
        features=features,
        source="LIGHTER_TRACER",
        publish_default=stripped_value == "all",
    )

# MOD. Begin. Improved tracer
import psutil
traces_root_dir = os.getenv('TRACES_ROOT_DIR')
if not traces_root_dir:
    traces_root_dir = "../.."
warp_lifetime_only = os.getenv("WARP_LIFETIME_ONLY") == "1"
warp_lifetime_sidecar = os.getenv("WARP_LIFETIME_SIDECAR") == "1"
if warp_lifetime_only and warp_lifetime_sidecar:
    raise RuntimeError(
        "WARP_LIFETIME_ONLY and WARP_LIFETIME_SIDECAR are mutually exclusive"
    )
# Spin-trim post-step: when TRIM_SPIN=1, after the normal full run + trace
# post-processing, move the full trace aside to "<run>.fullrun" and write a
# spin-trimmed copy back to the original "<run>" location (so the simulator
# consumes the trimmed trace). Requires the matching NCU report as ground truth.
trim_spin_enabled = os.getenv("TRIM_SPIN") == "1"
trim_spin_ncu_rep = os.getenv("TRIM_SPIN_NCU_REP")
trim_spin_ncu_bin = os.getenv("TRIM_SPIN_NCU_BIN", "ncu")
try:
    lighter_tracer_config = resolve_lighter_tracer(os.getenv("LIGHTER_TRACER"))
except ValueError as error:
    raise RuntimeError(str(error))
compact_full_capture_enabled = (
    lighter_tracer_config.compact_wire_enabled
)
print(
    "[LIGHTER-TRACER] requested={0} canonical={1} config_hash={2} "
    "artifact_suffix={3}".format(
        lighter_tracer_config.requested,
        lighter_tracer_config.canonical,
        lighter_tracer_config.config_hash,
        lighter_tracer_config.artifact_suffix or "<none>",
    )
)
if trim_spin_enabled and compact_full_capture_enabled:
    raise RuntimeError(
        "TRIM_SPIN=1 and LIGHTER_TRACER compact capture are mutually exclusive: "
        "use compact full capture for Option 1, or trim/reorganization for Option 2"
    )
if trim_spin_enabled and not trim_spin_ncu_rep:
    raise RuntimeError(
        "TRIM_SPIN=1 requires TRIM_SPIN_NCU_REP=<path to matching .ncu-rep>"
    )
if trim_spin_enabled and (warp_lifetime_only or warp_lifetime_sidecar):
    raise RuntimeError(
        "TRIM_SPIN is incompatible with WARP_LIFETIME_ONLY / WARP_LIFETIME_SIDECAR"
    )
# MOD. End. Improved tracer

# We will look for the benchmarks 
parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for " +\
                       "the benchmark suite names.",
                 default="rodinia_2.0-ft")
parser.add_option("-D", "--device_num", dest="device_num",
                 help="CUDA device number",
                 default="0")
parser.add_option("-n", "--norun", dest="norun", action="store_true", 
                 help="Do not actually run the apps, just create the dir structure and launch files")
parser.add_option("-l", "--limit_kernel_number", dest='kernel_number', default=-99, help="Sets a hard limit to the " +\
                        "number of traced limits")
parser.add_option("-t", "--terminate_upon_limit", dest='terminate_upon_limit', action="store_true", help="Once the kernel limit is " +\
                        "reached, terminate the tracing process")
# MOD. Begin. Improved tracer
parser.add_option("-C", "--compressed", dest="compressed", action="store_true", 
                 help="It runs the tracer in compressed mode", default="0")
# MOD. End. Improved tracer

(options, args) = parser.parse_args()

common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cuda_version = common.get_cuda_version( this_directory )
now_time = datetime.datetime.now()
day_string = now_time.strftime("%y.%m.%d-%A")
time_string = now_time.strftime("%H:%M:%S")
logfile = day_string + "--" + time_string + ".csv"

nvbit_tracer_path = os.path.join(this_directory, "tracer_tool")


def build_tma_descriptor_mapping_if_available(trace_folder):
    extra_info_dir = os.path.join(trace_folder, "extra_info")
    discovery_inputs = [
        os.path.join(extra_info_dir, "enhanced_execution_info.json"),
        os.path.join(extra_info_dir, "sass"),
    ]
    if all(os.path.exists(path) for path in discovery_inputs):
        discovery_script = os.path.join(this_directory, "discover_tma_producers.py")
        subprocess.run([sys.executable, discovery_script, "--traces", trace_folder], check=True)
    # Descriptor static-offset extractor (feeds the UTMACCTL prefetch chain in the base
    # map). Needs the disassembly; runs offline.
    if os.path.exists(os.path.join(extra_info_dir, "sass")):
        offsets_script = os.path.join(this_directory, "extract_tma_descriptor_offsets.py")
        subprocess.run([sys.executable, offsets_script, "--traces", trace_folder], check=True)
    mapping_inputs = [
        os.path.join(extra_info_dir, "tensor_map_encode_dump.csv"),
    ]
    if not all(os.path.exists(path) for path in mapping_inputs):
        return
    # configs-only: emit tma_descriptor_configs.json (box_dim/strides/swizzle, still used
    # by the simulator) but NOT the handle_hi->config_id heuristic resolver — the exact
    # (uid,pc) base map supersedes it.
    mapping_script = os.path.join(this_directory, "build_tma_descriptor_mapping.py")
    subprocess.run(
        [sys.executable, mapping_script, extra_info_dir, "--configs-only"],
        check=True,
    )
    # Exact (uid,pc) -> real GMEM base + full descriptor fields. Strict: fail if any
    # descriptor LDG/STG site is unresolved or falls back to the pool.
    base_map_inputs = [
        os.path.join(extra_info_dir, "tma_launch_param_dump.csv"),
        os.path.join(extra_info_dir, "tensor_map_encode_dump.csv"),
        os.path.join(extra_info_dir, "tma_runtime_operand_debug.jsonl"),
    ]
    if all(os.path.exists(path) for path in base_map_inputs):
        base_map_script = os.path.join(this_directory, "build_tma_pc_base_map.py")
        subprocess.run(
            [sys.executable, base_map_script, "--traces", trace_folder, "--strict"],
            check=True,
        )


def build_tma_operand_mapping_if_available(trace_folder):
    extra_info_dir = os.path.join(trace_folder, "extra_info")
    operand_inputs = [
        os.path.join(extra_info_dir, "tma_discovery.json"),
        os.path.join(extra_info_dir, "tma_runtime_operand_debug.jsonl"),
    ]
    if not all(os.path.exists(path) for path in operand_inputs):
        return
    operand_script = os.path.join(this_directory, "build_tma_operand_mapping.py")
    subprocess.run([sys.executable, operand_script, extra_info_dir], check=True)


def build_sync_operand_mapping_if_available(trace_folder):
    extra_info_dir = os.path.join(trace_folder, "extra_info")
    info_path = os.path.join(extra_info_dir, "enhanced_execution_info.json")
    if not os.path.exists(info_path):
        return
    sync_script = os.path.join(this_directory, "build_sync_operand_mapping.py")
    subprocess.run([sys.executable, sync_script, extra_info_dir], check=True)


def spin_trim_run_dir(this_run_dir):
    """TRIM_SPIN post-step. The full run has just completed at this_run_dir.
    Move it aside to '<this_run_dir>.fullrun' (the untouched full trace) and
    write a spin-trimmed copy back to this_run_dir so the simulator consumes the
    trimmed trace. Uses trim_spin_from_ncu.py with the matching NCU report as HW
    ground truth. Idempotent-ish: if '<...>.fullrun' already exists it is reused
    as the source (so re-running does not lose the original full trace)."""
    fullrun_dir = this_run_dir + ".fullrun"
    if not os.path.isdir(fullrun_dir):
        # first time: the full run currently lives at this_run_dir -> move aside
        shutil.move(this_run_dir, fullrun_dir)
    else:
        # a previous full run is preserved; drop any stale trimmed dir
        if os.path.isdir(this_run_dir):
            shutil.rmtree(this_run_dir)

    full_traces = os.path.join(fullrun_dir, "traces")
    trimmed_traces = os.path.join(this_run_dir, "traces")

    # recreate the run_dir shell and copy every non-'traces' artifact
    # (run.sh, data symlinks, etc.) so the trimmed run_dir is self-contained.
    os.makedirs(this_run_dir, exist_ok=True)
    for name in os.listdir(fullrun_dir):
        if name == "traces":
            continue
        src = os.path.join(fullrun_dir, name)
        dst = os.path.join(this_run_dir, name)
        if os.path.islink(src):
            linkto = os.readlink(src)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(linkto, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    trim_tool = os.path.join(this_directory, "trim_spin_from_ncu.py")
    ncu_rep = os.path.abspath(os.path.expandvars(os.path.expanduser(trim_spin_ncu_rep)))
    cmd = [sys.executable, trim_tool,
           "--full-run", full_traces,
           "--ncu-rep", ncu_rep,
           "--out", trimmed_traces,
           "--ncu-bin", trim_spin_ncu_bin]
    print("Spin-trimming trace: %s -> %s (NCU %s)"
          % (full_traces, trimmed_traces, ncu_rep))
    subprocess.run(cmd, check=True)


for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    for argpair in argslist:
        args = argpair["args"]
        extra_env = argpair.get("env", {})
        conflicting_config_keys = {"LIGHTER_TRACER"} & set(extra_env)
        if conflicting_config_keys:
            raise RuntimeError(
                "benchmark environment cannot override tracer selection: {0}".format(
                    ",".join(sorted(conflicting_config_keys))
                )
            )
        benchmark_name = argpair.get("benchmark_name", exe)
        argument_folder = common.get_argfoldername(args)
        if warp_lifetime_only:
            lifetime_kernel = os.getenv("DYNAMIC_KERNEL_LIMIT_START", "all")
            argument_folder += "-warp-lifetime-k" + lifetime_kernel
        elif warp_lifetime_sidecar:
            lifetime_kernel = os.getenv("DYNAMIC_KERNEL_LIMIT_START", "all")
            argument_folder += "-fulltrace-lifetime-k" + lifetime_kernel
        argument_folder += lighter_tracer_config.artifact_suffix
        run_name = os.path.join(benchmark_name, argument_folder)

        # MOD. Begin. Improved tracer
        if(options.compressed == "0"):
            trace_version_root = os.path.join(
                this_directory, "..", "..", "hw_run", "traces",
                "device-" + options.device_num, cuda_version
            )
        else:
            trace_version_root = os.path.join(
                this_directory, traces_root_dir, "hw_run", "traces",
                "device-" + options.device_num, cuda_version
            )
        this_run_dir = os.path.abspath(
            os.path.expandvars(os.path.join(trace_version_root, run_name))
        )
        # MOD. End. Improved tracer

        this_trace_folder = os.path.join(this_run_dir, "traces")
        if os.path.exists(this_run_dir):
            if lighter_tracer_config.features:
                raise RuntimeError(
                    "refusing to overwrite existing lighter trace directory: "
                    "{0}".format(this_run_dir)
                )
        else:
            os.makedirs(this_run_dir)
        if not os.path.exists(this_trace_folder):
            os.makedirs(this_trace_folder)

        try:
            git_sha = subprocess.check_output(
                ["git", "-C", this_directory, "rev-parse", "HEAD"],
                universal_newlines=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_sha = "unknown"
        config_manifest = {
            "requested": lighter_tracer_config.requested,
            "canonical": lighter_tracer_config.canonical,
            "features": list(lighter_tracer_config.features),
            "source": lighter_tracer_config.source,
            "publish_default": lighter_tracer_config.publish_default,
            "config_hash": lighter_tracer_config.config_hash,
            "artifact_suffix": lighter_tracer_config.artifact_suffix,
            "git_sha": git_sha,
        }
        with open(
            os.path.join(this_run_dir, "lighter_tracer_config.json"),
            "w",
        ) as manifest_file:
            json.dump(
                config_manifest,
                manifest_file,
                indent=2,
                sort_keys=True,
            )
            manifest_file.write("\n")

        # link the data directory
        try:
            benchmark_data_dir = common.dir_option_test(os.path.join(ddir,exe,"data"),"",this_directory)
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(benchmark_data_dir, os.path.join(this_run_dir,"data"))
        except common.PathMissing:
            pass

        all_data_link = os.path.join(this_run_dir,"data_dirs")
        if os.path.lexists(all_data_link):
            os.remove(all_data_link)
        top_data_dir_path = common.dir_option_test(ddir, "", this_directory)
        os.symlink(top_data_dir_path, all_data_link)

        if args == None:
            args = ""
        exec_path = common.file_option_test(os.path.join(edir, exe),"",this_directory)
        sh_contents = "set -e\n"

        env_exports = ""
        for key, value in extra_env.items():
            env_exports += "export {0}={1}; ".format(key, shlex.quote(str(value)))
        # Persist the canonical feature set so --norun/manual execution cannot
        # silently select a different tracer configuration.
        env_exports += "export LIGHTER_TRACER={0}; ".format(
            shlex.quote(lighter_tracer_config.canonical)
        )

        if options.terminate_upon_limit:
            sh_contents += "export TERMINATE_UPON_LIMIT=1; "

        options.kernel_number = int(options.kernel_number) # MOD. Fix comparison below for kernel limit
        if('mlperf' in exec_path):
            # For mlperf by default we turn this flag on
            sh_contents += "export TERMINATE_UPON_LIMIT=1; "
            exec_path = '. ' + exec_path

            if(options.kernel_number > 0):
                os.environ['DYNAMIC_KERNEL_LIMIT_END'] = str(options.kernel_number)
            else:
                os.environ['DYNAMIC_KERNEL_LIMIT_END'] = '50'
        else:
            if(options.kernel_number > 0):
                os.environ['DYNAMIC_KERNEL_LIMIT_END'] = str(options.kernel_number)
            else:
                os.environ['DYNAMIC_KERNEL_LIMIT_END'] = '0'

	# first we generate the traces (.trace and kernelslist files)
	# then, we do post-processing for the traces and generate (.traceg and kernelslist.g files)
	# then, we delete the intermediate files ((.trace and kernelslist files files)
        
        # MOD. Begin. Improved tracer
        if(options.compressed == "0"):
            sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num + "\" ; export USER_DEFINED_FOLDERS=1; " +\
                "export TRACES_FOLDER="+ this_trace_folder + "; " + env_exports + "export CUDA_INJECTION64_PATH=" + os.path.join(nvbit_tracer_path, "tracer_tool.so") +\
                " " + "; LD_PRELOAD=" + os.path.join(nvbit_tracer_path, "tracer_tool.so") + " " +\
                exec_path + " " + str(args) + " ;"
        else:
            sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num + "\" ; export USER_DEFINED_FOLDERS=1; " +\
                "export TRACES_FOLDER="+ this_trace_folder + "; " + env_exports + "export CUDA_INJECTION64_PATH=" + os.path.join(nvbit_tracer_path, "tracer_tool.so") +\
                " " + "; LD_PRELOAD=" + os.path.join(nvbit_tracer_path, "tracer_tool.so") + " " +\
                exec_path + " " + str(args) + " ; " + os.path.join(nvbit_tracer_path,"traces-processing", "post-traces-processing-compressed") + " " +\
                os.path.join(this_trace_folder, "kernelslist") + " " + str(psutil.virtual_memory()[1]*.8) + " ; rm -f " + this_trace_folder + "/*.trace ; rm -f " + this_trace_folder + "/kernelslist "
        # MOD. End. Improved tracer

        open(os.path.join(this_run_dir,"run.sh"), "w").write(sh_contents)
        if subprocess.call(['chmod', 'u+x', os.path.join(this_run_dir,"run.sh")]) != 0:
            exit("Error chmod runfile")

        if not options.norun:
            saved_dir = os.getcwd()
            os.chdir(this_run_dir)
            print("Running {0}".format(exe))

            if subprocess.call(["bash","run.sh"]) != 0:
                sys.exit("Error invoking nvbit on {0}".format(this_run_dir))
            os.chdir(saved_dir)
            if not warp_lifetime_only:
                build_tma_descriptor_mapping_if_available(this_trace_folder)
                build_tma_operand_mapping_if_available(this_trace_folder)
                build_sync_operand_mapping_if_available(this_trace_folder)
            if trim_spin_enabled:
                spin_trim_run_dir(this_run_dir)

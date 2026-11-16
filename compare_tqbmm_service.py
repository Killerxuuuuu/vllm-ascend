# SPDX-License-Identifier: Apache-2.0
"""Controlled composition/Triton whole-service comparison, not a kernel timer.

Stdlib-only orchestrator: init a frozen study, serve each implementation, run
identical warmups/benchmarks, then compare validated rounds. Linux /proc is
required for serve/run. No packages are installed and no existing server is
killed. Use the base vLLM Python, in the same container for both terminals.
"""

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
VARIANT_ENV = "VLLM_ASCEND_DSV4_WO_A_TRITON"  # Existing centralized envs.py option; no new variable.
PROFILE_ENV = (
    "SERVICE_PROF_CONFIG_PATH",
    "PROFILING_SYMBOLS_PATH",
    "VLLM_TORCH_PROFILER_DIR",
    "TRITON_KERNEL_DUMP",
    "TRITON_DUMP_DIR",
    "TRITON_ALWAYS_COMPILE",
    "ASCEND_LAUNCH_BLOCKING",
)
ENV_PREFIXES = (
    "VLLM_",
    "ASCEND_",
    "HCCL_",
    "TORCH_",
    "TRITON_",
    "OMP_",
    "MKL_",
    "PYTHON",
    "LD_",
    "CANN_",
    "NPU_",
    "CUDA_",
)
ENV_KEYS = (
    "PATH",
    "HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "XDG_CACHE_HOME",
    "TOKENIZERS_PARALLELISM",
    "TASK_QUEUE_ENABLE",
    "COMBINED_ENABLE",
    "MALLOC_ARENA_MAX",
    "GLOO_SOCKET_IFNAME",
)
CRITICAL_FILES = (
    "vllm_ascend/attention/dsa_v1.py",
    "vllm_ascend/attention/context_parallel/dsa_cp.py",
    "vllm_ascend/ops/triton/transpose_quant_batch_matmul.py",
    "vllm_ascend/envs.py",
)
METRICS = ("output_throughput", "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "duration")
DEFAULT_TIMEOUT = 7200
VARIANTS = ("composition", "triton")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def capture(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=30, check=True)
    return result.stdout.strip()


def package_path(name):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        raise ValueError(f"Missing {name}; use the existing base vLLM Python, not the AISBench/ModelSlim venv")
    return Path(spec.origin).resolve().parent


def tree_fingerprint(repo):
    # Includes tracked working-tree edits; does not read credentials or untracked outputs.
    head = capture(["git", "rev-parse", "HEAD"], repo)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", ".", ":(exclude)*.md"],
        cwd=repo,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout
    return {"head": head, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}


def runtime_identity(model):
    vllm_path = package_path("vllm")
    ascend_path = package_path("vllm_ascend")
    if ascend_path != REPO / "vllm_ascend":
        raise ValueError(f"Wrong Ascend source: {ascend_path}; put {REPO} first in PYTHONPATH")
    metadata = {}
    for path in sorted(Path(model).iterdir()):
        if path.is_file() and path.suffix in (".json", ".jinja", ".model", ".safetensors"):
            stat = path.stat()
            metadata[path.name] = [stat.st_size, stat.st_mtime_ns]
            if path.suffix != ".safetensors":
                metadata[path.name].append(hashlib.sha256(path.read_bytes()).hexdigest())
    if not metadata or not (Path(model) / "config.json").is_file():
        raise ValueError("Expected a local exported checkpoint directory with config.json")
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("vllm", "torch", "torch-npu", "triton-ascend")
        },
        "vllm_source": str(vllm_path),
        "ascend_source": str(ascend_path),
        "vllm_git": tree_fingerprint(vllm_path.parent),
        "ascend_git": tree_fingerprint(REPO),
        "critical_files": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in CRITICAL_FILES},
        "model_metadata_sha256": digest(metadata),
        "weight_check": "safetensors names/size/mtime, NOT a full content hash of weights",
    }


def kernel_config():
    tree = ast.parse((REPO / "vllm_ascend/attention/dsa_v1.py").read_text(encoding="utf-8"))
    names = ("A3_WO_A_TRITON_BLOCK_SHAPE", "A3_WO_A_TRITON_NUM_STAGES")
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = ast.literal_eval(node.value)
    if found != {names[0]: (64, 64, 128), names[1]: 2}:
        raise ValueError(f"Expected best measured (64,64,128), stages=2; source contains {found}")
    return {"block_shape": [64, 64, 128], "num_stages": 2}


def selected_env(environment):
    # Save hashes, not raw values (some framework variables may contain secrets).
    return {
        name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in environment.items()
        if (name.startswith(ENV_PREFIXES) or name in ENV_KEYS) and name != VARIANT_ENV
    }


def child_env(variant):
    environment = dict(os.environ)
    for name in PROFILE_ENV:
        environment.pop(name, None)
    environment[VARIANT_ENV] = "1" if variant == "triton" else "0"
    environment["VLLM_LOGGING_LEVEL"] = "INFO"
    # Keep CANN paths, but make this checkout take precedence for all workers.
    old_paths = environment.get("PYTHONPATH", "").split(os.pathsep)
    environment["PYTHONPATH"] = os.pathsep.join([str(REPO), *[p for p in old_paths if p and p != str(REPO)]])
    environment["no_proxy"] = environment["NO_PROXY"] = "127.0.0.1,localhost"
    return environment


def server_command(plan):
    return [plan["identity"]["python"], "-m", "vllm.entrypoints.cli.main", "serve", plan["model"], *plan["server_args"]]


def make_plan(args):
    model = str(args.model.resolve(strict=True))
    compilation = {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": args.capture_sizes}
    server_args = [
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "dsv4-int8",
        "--trust-remote-code",
        "--quantization",
        "ascend",
        "--tensor-parallel-size",
        str(args.tp),
        "--data-parallel-size",
        str(args.dp),
        "--enable-expert-parallel",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_batched_tokens),
        "--max-num-seqs",
        str(args.max_seqs),
        "--block-size",
        "128",
        "--gpu-memory-utilization",
        str(args.memory_utilization),
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--seed",
        "0",
        "--generation-config",
        "vllm",
        "--compilation-config",
        json.dumps(compilation, separators=(",", ":")),
    ]
    extra = json.loads(args.server_extra_json)
    if not isinstance(extra, list) or any(not isinstance(x, str) for x in extra):
        raise ValueError("--server-extra-json must be a JSON array of argument strings")
    # Only additional-config is accepted, to prevent overriding fixed flags, enabling
    # profiling/speculation, or accidentally changing the benchmark protocol.
    if extra and (len(extra) != 2 or extra[0] != "--additional-config" or not isinstance(json.loads(extra[1]), dict)):
        raise ValueError('Only ["--additional-config", "{...}"] is allowed as server extras')
    plan = {
        "schema_version": 1,
        "model": model,
        "server_args": server_args + extra,
        "base_url": f"http://127.0.0.1:{args.port}",
        "port": args.port,
        "model_name": "dsv4-int8",
        "kernel_config": kernel_config(),
        "identity": runtime_identity(model),
        "environment_hashes": selected_env(child_env("composition")),
        "workload": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "min_prompts": args.num_prompts,
            "waves": args.waves,
            "concurrencies": args.concurrencies,
            "repeats": args.repeats,
            "warmup_min_prompts": args.warmup_prompts,
            "seed": 123,
        },
        "limitations": [
            "Known best microbenchmark tile, NOT proof of globally optimal whole-network settings.",
            "Same seeded random workload and checked token totals; generated content/scheduler timing may differ.",
            "No profiler: reports whole-service latency/throughput, not isolated Wo_a time or accuracy.",
            "Cannot enforce exclusive hardware use, clock/temperature, or prove actual Graph/Prepared hits.",
            "Do not compare total device task counts: fusion legitimately changes the number of kernels.",
        ],
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def load_plan(study):
    plan = read_json(study / "plan.json")
    stored = plan.pop("plan_sha256")
    if digest(plan) != stored:
        raise ValueError("Plan was edited; create a new study rather than changing the existing plan")
    plan["plan_sha256"] = stored
    return plan


def check_identity(plan, environment):
    if runtime_identity(plan["model"]) != plan["identity"]:
        raise ValueError("Code, Python/packages, or checkpoint metadata changed since init; create a new study")
    if selected_env(environment) != plan["environment_hashes"]:
        raise ValueError(
            "Framework environment changed since init; use matching container terminals or create a new study"
        )


def process_token(pid):
    # /proc starttime distinguishes PID reuse, even when vLLM changes its process title.
    text = (Path("/proc") / str(pid) / "stat").read_text()
    return text.rsplit(")", 1)[1].split()[19]


def process_environment(pid):
    entries = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
    return dict(item.decode().split("=", 1) for item in entries if b"=" in item)


def check_server(manifest):
    if process_token(manifest["pid"]) != manifest["process_starttime"]:
        raise ValueError("Server PID was reused; do not benchmark a different server")


def http_json(url):
    # Loopback only; explicitly bypass HTTP proxies inherited from the shell.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=15) as response:
        return json.load(response)


def require_ready(plan, manifest):
    check_server(manifest)
    models = http_json(plan["base_url"] + "/v1/models")
    if plan["model_name"] not in [item["id"] for item in models.get("data", [])]:
        raise ValueError("Expected served-model-name not present; wait for startup and confirm the port")


def ensure_free_port(port):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def serve(args, plan):
    if sys.platform != "linux":
        raise ValueError("serve/run require the Linux NPU container")
    environment = child_env(args.variant)
    check_identity(plan, environment)
    ensure_free_port(plan["port"])
    folder = args.study / "sessions" / args.session
    folder.mkdir(parents=True, exist_ok=False)
    command = server_command(plan)
    print("Starting:", shlex.join(command), flush=True)
    print("Variant:", args.variant, "Wo_a switch:", environment[VARIANT_ENV], flush=True)
    with (folder / "server.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            manifest = {
                "session": args.session,
                "variant": args.variant,
                "plan_sha256": plan["plan_sha256"],
                "pid": process.pid,
                "process_starttime": process_token(process.pid),
                "command": command,
                "environment_hashes": selected_env(environment),
                "wo_a_switch": environment[VARIANT_ENV],
            }
            write_json(folder / "server.json", manifest)
            print(
                f"Server log: {folder / 'server.log'}\nFollow from another terminal: tail -f {folder / 'server.log'}",
                flush=True,
            )
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                # Only the process group started here, never pkill/fuser other services.
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    print(
                        f"Server group {process.pid} still shutting down; inspect it before restarting", file=sys.stderr
                    )
            raise
    if returncode:
        raise ValueError(f"Server exited {returncode}; inspect {folder / 'server.log'}")


def bench_command(plan, concurrency, repeat, output, warmup=False):
    work = plan["workload"]
    count = (
        max(work["warmup_min_prompts"], 2 * concurrency)
        if warmup
        else max(work["min_prompts"], work["waves"] * concurrency)
    )
    return [
        plan["identity"]["python"],
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        plan["base_url"],
        "--endpoint",
        "/v1/completions",
        "--model",
        plan["model_name"],
        "--tokenizer",
        plan["model"],
        "--dataset-name",
        "random",
        "--random-input-len",
        str(work["input_len"]),
        "--random-output-len",
        str(work["output_len"]),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(count),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--ignore-eos",
        "--seed",
        str(work["seed"] + repeat),
        "--num-warmups",
        "0",
        "--save-result",
        "--result-dir",
        str(output.parent),
        "--result-filename",
        output.name,
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--metric-percentiles",
        "50,90,99",
    ], count


def validate_result(result, work, count):
    expected = {
        "completed": count,
        "failed": 0,
        "total_input_tokens": count * work["input_len"],
        "total_output_tokens": count * work["output_len"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Invalid round: {key}={result.get(key)}, expected {value}; retained raw result/log")
    for name in METRICS:
        value = result.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing/nonfinite benchmark metric {name}")
    if result["output_throughput"] <= 0 or result["duration"] <= 0:
        raise ValueError("Throughput and duration must be positive")
    return {**expected, **{name: result[name] for name in METRICS}}


def npu_snapshot(path):
    try:
        text = capture(["npu-smi", "info"])
    except (OSError, subprocess.SubprocessError) as error:
        text = f"Snapshot unavailable: {error}"
    path.write_text(text + "\n", encoding="utf-8")


def run_bench(args, plan):
    folder = args.study / "sessions" / args.session
    manifest = read_json(folder / "server.json")
    if manifest["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Server was launched with another plan")
    environment = child_env(manifest["variant"])
    check_identity(plan, environment)
    require_ready(plan, manifest)
    # Check initial process environment without recording raw secrets.
    process_env = process_environment(manifest["pid"])
    if (
        selected_env(process_env) != plan["environment_hashes"]
        or process_env.get(VARIANT_ENV) != manifest["wo_a_switch"]
    ):
        raise ValueError("Running server environment does not match the frozen launcher")
    results = folder / "bench"
    results.mkdir(exist_ok=False)
    log_text = (folder / "server.log").read_text(encoding="utf-8", errors="replace")
    if not re.search(r"Captur(?:ing|ed).*graphs?", log_text, flags=re.IGNORECASE):
        raise ValueError("No Graph capture evidence in server.log; check startup/config before benchmarking")
    write_json(
        results / "checks.json", {"manifest": manifest, "graph_capture_log_seen": True, "graph_replay_proven": False}
    )
    npu_snapshot(results / "npu-before.txt")
    records = []
    try:
        work = plan["workload"]
        for repeat in range(work["repeats"]):
            # Rotate C order across rounds, identically in both variants.
            levels = work["concurrencies"]
            offset = repeat % len(levels)
            for concurrency in levels[offset:] + levels[:offset]:
                for warmup in (True, False):
                    require_ready(plan, manifest)
                    tag = f"c{concurrency}-r{repeat + 1}" + ("-warmup" if warmup else "")
                    output = results / f"{tag}.json"
                    command, count = bench_command(plan, concurrency, repeat, output, warmup)
                    write_json(results / f"{tag}-command.json", {"argv": command, "measured": not warmup})
                    print(
                        f"[{args.session}] {tag}: {count} requests, {work['input_len']} in / {work['output_len']} out",
                        flush=True,
                    )
                    with (results / f"{tag}.log").open("x", encoding="utf-8") as log:
                        subprocess.run(
                            command,
                            cwd=REPO,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                            check=True,
                        )
                    metrics = validate_result(read_json(output), work, count)
                    require_ready(plan, manifest)
                    if not warmup:
                        records.append({"concurrency": concurrency, "repeat": repeat + 1, **metrics})
                        print(
                            f"  out={metrics['output_throughput']:.2f} tok/s "
                            f"TTFT={metrics['mean_ttft_ms']:.2f} ms TPOT={metrics['mean_tpot_ms']:.2f} ms",
                            flush=True,
                        )
        check_identity(plan, environment)
        write_json(
            folder / "validated.json",
            {
                "status": "completed",
                "session": args.session,
                "variant": manifest["variant"],
                "plan_sha256": plan["plan_sha256"],
                "records": records,
            },
        )
    except BaseException as error:
        write_json(results / "failure.json", {"error": str(error), "validated_rounds": len(records), "rankable": False})
        raise
    finally:
        npu_snapshot(results / "npu-after.txt")


def compare(args, plan):
    reports = [read_json(args.study / "sessions" / session / "validated.json") for session in args.sessions]
    if any(report["status"] != "completed" or report["plan_sha256"] != plan["plan_sha256"] for report in reports):
        raise ValueError("Only completed sessions from this exact plan may be compared")
    if {report["variant"] for report in reports} != set(VARIANTS):
        raise ValueError("Need at least one composition and one Triton session")
    rows = []
    for concurrency in plan["workload"]["concurrencies"]:
        summaries = {}
        for report in reports:
            records = [record for record in report["records"] if record["concurrency"] == concurrency]
            if sorted(record["repeat"] for record in records) != list(range(1, plan["workload"]["repeats"] + 1)):
                raise ValueError("Missing/duplicate rounds in validated session")
            summaries[report["session"]] = {
                "variant": report["variant"],
                **{
                    metric: {
                        "median": statistics.median(r[metric] for r in records),
                        "min": min(r[metric] for r in records),
                        "max": max(r[metric] for r in records),
                    }
                    for metric in METRICS
                },
            }
        comparisons = []
        for candidate_name, candidate in summaries.items():
            if candidate["variant"] != "triton":
                continue
            for baseline_name, baseline in summaries.items():
                if baseline["variant"] != "composition":
                    continue
                changes = {}
                for metric in METRICS:
                    before = baseline[metric]["median"]
                    changes[metric] = 100 * (candidate[metric]["median"] / before - 1) if before > 0 else None
                comparisons.append({"candidate": candidate_name, "baseline": baseline_name, "change_pct": changes})
        rows.append({"concurrency": concurrency, "sessions": summaries, "comparisons": comparisons})
    report = {
        "plan_sha256": plan["plan_sha256"],
        "sessions": args.sessions,
        "by_concurrency": rows,
        "interpretation": "Each number is a median/min/max of round metrics; NOT a pooled per-request percentile.",
        "limitations": plan["limitations"],
    }
    output = args.study / ("comparison-" + "-".join(args.sessions) + ".json")
    write_json(output, report)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["C", "session", "variant", *METRICS])
        for row in rows:
            for session, summary in row["sessions"].items():
                values = [
                    row["concurrency"],
                    session,
                    summary["variant"],
                    *[summary[name]["median"] for name in METRICS],
                ]
                writer.writerow(values)
                print(
                    f"C={row['concurrency']:2} {session:16} "
                    f"out={summary['output_throughput']['median']:.2f} tok/s "
                    f"TTFT={summary['mean_ttft_ms']['median']:.2f} ms "
                    f"TPOT={summary['mean_tpot_ms']['median']:.2f} ms"
                )
    print(f"Share: {output}\nCSV: {csv_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init", help="Freeze common settings once, using the base vLLM Python")
    init.add_argument("--study", type=Path, required=True)
    init.add_argument("--model", type=Path, required=True)
    init.add_argument("--tp", type=int, default=8)
    init.add_argument("--dp", type=int, default=2)
    init.add_argument("--port", type=int, default=8000)
    init.add_argument("--max-model-len", type=int, default=133120)
    init.add_argument("--max-batched-tokens", type=int, default=8192)
    init.add_argument("--max-seqs", type=int, default=32)
    init.add_argument("--memory-utilization", type=float, default=0.9)
    init.add_argument("--capture-sizes", type=int, nargs="+", default=[8, 16, 24, 32])
    init.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 32, 64])
    init.add_argument("--num-prompts", type=int, default=128, help="Minimum prompts per round")
    init.add_argument("--waves", type=int, default=4, help="prompts=max(num-prompts,waves*C)")
    init.add_argument("--repeats", type=int, default=3)
    init.add_argument("--warmup-prompts", type=int, default=8, help="Warmup=max(this,2*C), same token lengths")
    init.add_argument("--input-len", type=int, default=1024)
    init.add_argument("--output-len", type=int, default=256)
    init.add_argument("--server-extra-json", default="[]", help="Optional fixed --additional-config argument pair")
    for action in ("serve", "run", "compare"):
        item = sub.add_parser(action)
        item.add_argument("--study", type=Path, required=True)
        if action == "compare":
            item.add_argument("--sessions", nargs="+", required=True)
        else:
            item.add_argument("--session", required=True)
        if action == "serve":
            item.add_argument("--variant", choices=VARIANTS, required=True)
        if action == "run":
            item.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per benchmark subprocess seconds")
    args = parser.parse_args(argv)
    args.study = args.study.resolve()
    sessions = getattr(args, "sessions", [getattr(args, "session", "unused")])
    if len(sessions) != len(set(sessions)) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name) for name in sessions):
        parser.error("Sessions must be unique safe names (letters/numbers/_/-, <=40 chars)")
    if args.action == "init":
        if (
            min(
                args.tp,
                args.dp,
                args.max_model_len,
                args.max_batched_tokens,
                args.max_seqs,
                args.num_prompts,
                args.waves,
                args.repeats,
                args.warmup_prompts,
                args.input_len,
                args.output_len,
            )
            <= 0
        ):
            parser.error("All sizes/counts must be positive")
        if args.output_len < 2 or args.input_len + args.output_len > args.max_model_len:
            parser.error("Need output-len >=2 and input+output <= max-model-len")
        if not 0 < args.memory_utilization < 1 or not 1 <= args.port <= 65535:
            parser.error("Invalid memory utilization / port")
        if (
            args.capture_sizes != sorted(set(args.capture_sizes))
            or not args.capture_sizes
            or min(args.capture_sizes) <= 0
            or max(args.capture_sizes) != args.max_seqs
        ):
            parser.error("Capture sizes must be unique ascending positive values ending at max-seqs")
        if (
            not args.concurrencies
            or min(args.concurrencies) <= 0
            or len(set(args.concurrencies)) != len(args.concurrencies)
        ):
            parser.error("Concurrencies must be unique positive values")
    elif args.action == "run" and args.timeout <= 0:
        parser.error("Timeout must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.action == "init":
            plan = make_plan(args)
            args.study.mkdir(parents=True, exist_ok=False)
            write_json(args.study / "plan.json", plan)
            print(f"Frozen plan: {args.study / 'plan.json'}\n{shlex.join(server_command(plan))}")
            return 0
        plan = load_plan(args.study)
        if args.action == "serve":
            serve(args, plan)
        elif args.action == "run":
            # Reject overlapping clients in this study; stale locks are retained
            # after a hard kill for explicit investigation, not auto-deleted.
            lock = args.study / "client.lock"
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            try:
                run_bench(args, plan)
            finally:
                lock.unlink()
        else:
            compare(args, plan)
    except KeyboardInterrupt:
        print("Interrupted; incomplete rounds are not eligible for comparison", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

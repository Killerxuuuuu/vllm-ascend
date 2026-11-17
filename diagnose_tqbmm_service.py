# SPDX-License-Identifier: Apache-2.0
"""Short, reproducible whole-service bottleneck capture, separate from timing.

Reuse a compare_tqbmm_service study as a configuration template, NOT as mutable
state. Make a NEW diagnostic study after synchronizing code. Restart once for
each (implementation,C) case. Collect fixed token-ID prompts directly, without
bench serve's hidden test requests. Analyze ONE matching DP/TP rank at a time.
"""

import argparse
import copy
import csv
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import analyze_tqbmm_profile as focus
import compare_tqbmm_service as bench
from tools.dsv4_a3_wo.service_trace_analysis import Durations, analyze_trace, focus_family

MAX_HTTP_BYTES = 4 * 1024 * 1024
MAX_REPORT_BYTES = 48 * 1024
HTTP_TIMEOUT = 300
STOP_TIMEOUT = 1800
CASE_CONCURRENCIES = (1, 64)


def session_name(args):
    return f"{args.variant}-c{args.concurrency}"


def profile_config(trace_dir):
    return {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir),
        "ignore_frontend": True,
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": False,
    }


def make_diagnostic_plan(base, output_len):
    plan = copy.deepcopy(base)
    plan.pop("plan_sha256")
    plan["source_plan_sha256"] = base["plan_sha256"]
    plan["measurement_mode"] = "bottleneck_capture_not_performance"
    plan["identity"] = bench.runtime_identity(plan["model"])
    plan["kernel_config"] = bench.kernel_config()
    plan["environment_hashes"] = bench.selected_env(bench.child_env("composition"))
    if "--profiler-config" in plan["server_args"]:
        raise ValueError("Base plan already contains profiler options")
    config = bench.read_json(Path(plan["model"]) / "config.json")
    vocab_size = config.get("vocab_size", config.get("text_config", {}).get("vocab_size"))
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 2:
        raise ValueError("Cannot find valid vocab_size in exported config.json")
    input_len = plan["workload"]["input_len"]
    if input_len + output_len > int(plan["server_args"][plan["server_args"].index("--max-model-len") + 1]):
        raise ValueError("Diagnostic input+output exceeds server max-model-len")
    generator = random.Random(123)
    prompts = [[generator.randrange(vocab_size) for _ in range(input_len)] for _ in range(max(CASE_CONCURRENCIES))]
    plan["diagnostic_protocol"] = {
        "concurrencies": list(CASE_CONCURRENCIES),
        "input_len": input_len,
        "output_len": output_len,
        "requests_by_c": {"1": 2, "64": 64},
        "warmup_min_requests": 8,
        "seed": 123,
        "prompts_sha256": bench.digest(prompts),
        "request_body": {
            "temperature": 0,
            "top_p": 1,
            "ignore_eos": True,
            "stream": False,
            "add_special_tokens": False,
        },
        "note": "Synthetic token IDs; not an accuracy dataset. Internal DP batches/Graph padding may differ.",
    }
    plan["limitations"] = [
        "Profiling perturbs execution; never rank TTFT/TPOT/throughput from this capture.",
        "Fixed client token IDs/counts do not force identical internal batch/DP allocation or kernel counts.",
        "record_shapes exposes available framework shapes, not necessarily Triton kernel arguments/strides.",
        "Default decode-only Graph mode is retained; this is a mixed request capture, not a pure-decode window.",
    ]
    plan["plan_sha256"] = bench.digest(plan)
    return plan, prompts


def load_diagnostic(study):
    plan = bench.load_plan(study)
    if plan.get("measurement_mode") != "bottleneck_capture_not_performance":
        raise ValueError("Use a NEW diagnostic study created by this script's init command")
    return plan


def rendered_plan(plan, folder):
    result = copy.deepcopy(plan)
    result["server_args"] += ["--profiler-config", json.dumps(profile_config(folder / "trace"), separators=(",", ":"))]
    return result


def post(url, body=None, request_id=None, timeout=HTTP_TIMEOUT, expect_json=True):
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-Id"] = request_id
    request = urllib.request.Request(url, data=json.dumps(body or {}).encode(), headers=headers, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(MAX_HTTP_BYTES + 1)
        if len(raw) > MAX_HTTP_BYTES:
            raise ValueError("HTTP response exceeds diagnostic size limit")
    return json.loads(raw) if expect_json and raw.strip() else {}


def payloads(plan, prompts, count):
    protocol = plan["diagnostic_protocol"]
    return [
        {
            "model": plan["model_name"],
            "prompt": prompts[index % len(prompts)],
            "max_tokens": protocol["output_len"],
            "seed": protocol["seed"],
            **protocol["request_body"],
        }
        for index in range(count)
    ]


def send_one(url, payload, request_id, protocol):
    response = post(url + "/v1/completions", payload, request_id)
    usage = response.get("usage", {})
    if usage.get("prompt_tokens") != protocol["input_len"] or usage.get("completion_tokens") != protocol["output_len"]:
        raise ValueError(f"Request {request_id}: token usage mismatch, got {usage}")
    if not response.get("choices") or response.get("error"):
        raise ValueError(f"Request {request_id}: missing choices or API error")
    return {
        "request_id": request_id,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
    }


def send_batch(pool, url, bodies, prefix, protocol):
    # All payloads, imports, and the warmup are prepared before /start_profile.
    futures = [pool.submit(send_one, url, body, f"{prefix}-{index:04d}", protocol) for index, body in enumerate(bodies)]
    records, failures = [], []
    for future in futures:
        try:
            records.append(future.result())
        except Exception as error:
            failures.append(str(error))
    # Wait for ALL submitted requests before stopping the profiler, even if one failed.
    if failures:
        raise ValueError(f"{len(failures)} requests failed; first error: {failures[0]}")
    return records


def collect(args, plan):
    folder = args.study / "sessions" / session_name(args)
    manifest = bench.read_json(folder / "server.json")
    expected_command = bench.server_command(rendered_plan(plan, folder))
    if manifest["plan_sha256"] != plan["plan_sha256"] or manifest["command"] != expected_command:
        raise ValueError("Server was not launched with this exact diagnostic configuration")
    if manifest["variant"] != args.variant:
        raise ValueError("Wrong implementation session")
    bench.check_identity(plan, bench.child_env(args.variant))
    bench.require_ready(plan, manifest)
    process_env = bench.process_environment(manifest["pid"])
    if (
        bench.selected_env(process_env) != plan["environment_hashes"]
        or process_env.get(bench.VARIANT_ENV) != manifest["wo_a_switch"]
    ):
        raise ValueError("Running server environment differs from the frozen plan")
    prompts = bench.read_json(args.study / "prompts.json")
    protocol = plan["diagnostic_protocol"]
    if bench.digest(prompts) != protocol["prompts_sha256"]:
        raise ValueError("Frozen token-ID prompts changed")
    with (folder / "capture-once.lock").open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    # Keep the lock permanently: a session is captured exactly once, not resumed.
    count = protocol["requests_by_c"][str(args.concurrency)]
    warmup_count = max(protocol["warmup_min_requests"], args.concurrency)
    warmup_bodies = payloads(plan, prompts, warmup_count)
    measured_bodies = payloads(plan, prompts, count)
    state = {
        "session": session_name(args),
        "plan_sha256": plan["plan_sha256"],
        "protocol_sha256": bench.digest(protocol),
        "concurrency": args.concurrency,
        "expected_requests": count,
        "expected_input_tokens": count * protocol["input_len"],
        "expected_output_tokens": count * protocol["output_len"],
        "requests": [],
        "status": "failed",
        "start_profile_attempted": False,
        "stop_profile_attempted": False,
        "note": "No hidden benchmark test requests or retries. Capture includes prefill and decode.",
    }
    failure = None
    bench.npu_snapshot(folder / "npu-before.txt")
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            print(f"Warmup: {warmup_count} requests, C={args.concurrency}", flush=True)
            send_batch(pool, plan["base_url"], warmup_bodies, "warmup", protocol)
            print(f"Collecting: {count} requests, output={protocol['output_len']} tokens each", flush=True)
            state["start_profile_attempted"] = True
            state["start_control_begin_unix_ns"] = time.time_ns()
            post(plan["base_url"] + "/start_profile", expect_json=False)
            state["start_control_end_unix_ns"] = time.time_ns()
            state["requests"] = send_batch(pool, plan["base_url"], measured_bodies, "capture", protocol)
            state["requests_finished_unix_ns"] = time.time_ns()
            state["status"] = "completed"
    except BaseException as error:
        failure = error
        state["error"] = str(error) or type(error).__name__
    finally:
        if state["start_profile_attempted"]:
            state["stop_profile_attempted"] = True
            print(
                "Stopping/exporting once; do not send another stop request. Parsing may take several minutes.",
                flush=True,
            )
            try:
                post(plan["base_url"] + "/stop_profile", timeout=STOP_TIMEOUT, expect_json=False)
            except Exception as error:
                state["status"] = "failed"
                state["stop_error"] = str(error)
                if failure is None:
                    failure = error
        state["export_control_finished_unix_ns"] = time.time_ns()
        bench.write_json(folder / "capture.json", state)
        bench.npu_snapshot(folder / "npu-after.txt")
    if failure is not None:
        raise failure
    print(f"Capture completed: {folder}\nAnalyze only after export has finished.", flush=True)


def select_rank(folder, dp, tp):
    candidates = []
    for path in (folder / "trace").rglob("kernel_details.csv"):
        rank = path.parent.parent.name
        if re.search(rf"(?:^|_)dp{dp}_", rank) and re.search(rf"(?:^|_)tp{tp}_", rank):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one parsed DP{dp}/TP{tp} CSV; found {len(candidates)}. "
            "Wait/check export; no automatic all-rank reparse."
        )
    return candidates[0]


def focused_csv_durations(source):
    groups = {}
    before = source.stat()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        for row in reader:
            try:
                duration = focus.number(focus.pick(row, "Duration(us)", "Task Duration(us)"))
            except ValueError:
                continue
            name = focus.pick(row, "Name", "Op Name")
            category = focus_family(name)
            if category != "other":
                groups.setdefault(category, Durations()).add(duration)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("CSV changed while reading")
    return {name: stats.report() for name, stats in groups.items()}


def log_warnings(path):
    patterns = ("collected data has been lost", "tlv", "failed to", "error", "replaying aclgraph")
    found = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lower = line.lower()
            for pattern in patterns:
                if pattern in lower:
                    item = found.setdefault(pattern, {"lines": 0, "sample": focus.clipped(line.strip(), 300)})
                    item["lines"] += 1
    return found


def bounded_payload(report):
    lists = [report["device_csv"]["top_operators_by_sum"], *report["device_csv"]["focus_shapes_by_sum"].values()]
    trace = report.get("trace") or {}
    for key in ("host_top_inclusive", "unknown_top", "observed_input_shapes"):
        if key in trace:
            lists.append(trace[key])
    if trace.get("flow_evidence"):
        lists.append(trace["flow_evidence"]["focus_associations"])
    if trace.get("device_task_coverage", {}).get("largest_uncovered_intervals"):
        lists.append(trace["device_task_coverage"]["largest_uncovered_intervals"])
    while True:
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if len(payload.encode()) <= MAX_REPORT_BYTES:
            return payload
        longest = max(lists, key=len)
        if not longest:
            raise ValueError("Report metadata exceeds size guard")
        longest.pop()
        report["lists_trimmed_for_size"] = True


def analyze(args, plan):
    folder = args.study / "sessions" / session_name(args)
    capture = bench.read_json(folder / "capture.json")
    if capture.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("Capture belongs to another plan")
    source = select_rank(folder, args.dp, args.tp)
    before = source.stat()
    aggregate, upstream = focus.read_source(source)
    actual_totals = {
        "completed_requests": len(capture.get("requests", [])),
        "input_tokens": sum(request["prompt_tokens"] for request in capture.get("requests", [])),
        "output_tokens": sum(request["completion_tokens"] for request in capture.get("requests", [])),
    }
    report = {
        "schema_version": 1,
        "session": session_name(args),
        "rank_dp_tp": [args.dp, args.tp],
        "plan_sha256": plan["plan_sha256"],
        "protocol_sha256": capture.get("protocol_sha256"),
        "capture_status": capture["status"],
        "request_totals": {
            key: capture.get(key)
            for key in ("expected_requests", "expected_input_tokens", "expected_output_tokens", "error", "stop_error")
        },
        "device_csv": focus.make_report(aggregate, source, 1, upstream, 3, session_name(args)),
        "device_focus_duration_histograms": focused_csv_durations(source),
        "server_log_evidence": log_warnings(folder / "server.log"),
        "trace": None,
        "lists_trimmed_for_size": False,
        "actual_request_totals": actual_totals,
        "warnings": [
            "Profiling data only, not a performance score. Compare matching C, DP/TP and the same protocol hash.",
            "CPU/flow/gap evidence is incomplete when events are missing; no automatic root-cause diagnosis.",
        ],
    }
    traces = [
        path for path in (source.parent / "trace_view.json", source.parent / "trace_view.json.gz") if path.is_file()
    ]
    if len(traces) == 1:
        try:
            report["trace"] = analyze_trace(traces[0])
        except (OSError, ValueError, TypeError, AttributeError) as error:
            report["warnings"].append("Trace unavailable/incomplete: " + focus.clipped(error, 300))
    else:
        report["warnings"].append("No unique trace_view.json/json.gz for this rank; Host and gap evidence unavailable.")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("CSV changed across analysis passes; wait for export to finish")
    output = args.output.resolve() if args.output else folder / f"bottleneck-dp{args.dp}-tp{args.tp}.json"
    payload = bounded_payload(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(f"Share: {output} ({output.stat().st_size / 1024:.1f} KiB)", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    init = commands.add_parser("init")
    init.add_argument("--base-study", type=Path, required=True)
    init.add_argument("--study", type=Path, required=True)
    init.add_argument("--output-len", type=int, default=32, help="Short diagnostic output, max 128")
    for action in ("serve", "collect", "analyze"):
        item = commands.add_parser(action)
        item.add_argument("--study", type=Path, required=True)
        item.add_argument("--variant", choices=bench.VARIANTS, required=True)
        item.add_argument("--concurrency", type=int, choices=CASE_CONCURRENCIES, required=True)
        if action == "analyze":
            item.add_argument("--dp", type=int, default=0)
            item.add_argument("--tp", type=int, default=0)
            item.add_argument("--output", type=Path, help="Optional new filename when reanalyzing; never overwrites")
    args = parser.parse_args(argv)
    args.study = args.study.resolve()
    if args.action == "init" and not 2 <= args.output_len <= 128:
        parser.error("Diagnostic output-len must be in [2,128]")
    if args.action == "analyze" and min(args.dp, args.tp) < 0:
        parser.error("Rank selectors must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.action == "init":
            base = bench.load_plan(args.base_study.resolve())
            plan, prompts = make_diagnostic_plan(base, args.output_len)
            args.study.mkdir(parents=True, exist_ok=False)
            bench.write_json(args.study / "plan.json", plan)
            bench.write_json(args.study / "prompts.json", prompts)
            print(
                f"New diagnostic plan: {args.study}\n"
                "Restart per variant/C case; do not modify the old performance study."
            )
            return 0
        plan = load_diagnostic(args.study)
        folder = args.study / "sessions" / session_name(args)
        if args.action == "serve":
            args.session = session_name(args)
            bench.serve(args, rendered_plan(plan, folder))
        elif args.action == "collect":
            if sys.platform != "linux":
                raise ValueError("collect requires the same Linux container as the server")
            collect(args, plan)
        else:
            analyze(args, plan)
    except KeyboardInterrupt:
        print("Interrupted; retain raw files and capture.json, do not reuse this capture session", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

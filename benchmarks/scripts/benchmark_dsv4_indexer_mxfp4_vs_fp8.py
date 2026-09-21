#!/usr/bin/env python3
"""Benchmark DeepSeek V4 Indexer MXFP4 against the native A5 FP8 path.

The script starts two otherwise identical three-layer vLLM services in
sequence. Each service is warmed up before streaming requests are measured,
so model loading and first-use Triton compilation are excluded from results.

Run this script from an activated vLLM Ascend environment with the required
CANN and PYTHONPATH settings already exported.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

import requests


MODE_MXFP4 = "mxfp4"
MODE_FP8 = "fp8"
MIN_C4_TEST_LAYERS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start sequential DeepSeek V4 services and compare the Indexer "
            "MXFP4 path with the native A5 FP8 path."
        )
    )
    parser.add_argument("model", help="Local DeepSeek-V4-Flash-0731 model directory")
    parser.add_argument("--vllm-command", default="vllm", help="vLLM CLI executable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="deepseek-v4-indexer-ab")
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt-repeat", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory; defaults to benchmark_results/<timestamp>",
    )
    parser.add_argument(
        "--extra-serve-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments appended to both vllm serve commands",
    )
    return parser.parse_args()


def ensure_port_is_free(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(
                f"{host}:{port} is already in use. Stop the existing service "
                "or select another --port."
            )


def build_serve_command(args: argparse.Namespace, mode: str) -> list[str]:
    use_mxfp4 = mode == MODE_MXFP4
    hf_overrides = json.dumps(
        {
            "num_hidden_layers": args.num_hidden_layers,
            "use_mxfp4_indexer": use_mxfp4,
        },
        separators=(",", ":"),
    )
    additional_config = json.dumps(
        {"sparse_kv_offload_config": {"enabled": False}},
        separators=(",", ":"),
    )

    return [
        args.vllm_command,
        "serve",
        str(Path(args.model).expanduser().resolve()),
        "--served-model-name",
        args.served_model_name,
        "--trust-remote-code",
        "--load-format",
        "auto",
        "--dtype",
        "auto",
        "--hf-overrides",
        hf_overrides,
        "--additional-config",
        additional_config,
        "--tensor-parallel-size",
        "1",
        "--block-size",
        str(args.block_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "--host",
        args.host,
        "--port",
        str(args.port),
        *args.extra_serve_args,
    ]


def tail_file(path: Path, line_count: int = 80) -> str:
    if not path.exists():
        return "<log file was not created>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def wait_until_ready(
    process: subprocess.Popen[bytes],
    health_url: str,
    timeout: float,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"vLLM exited with code {return_code} before becoming ready.\n"
                f"Last log lines:\n{tail_file(log_path)}"
            )
        try:
            response = requests.get(health_url, timeout=2.0)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2.0)

    raise TimeoutError(
        f"vLLM did not become ready within {timeout:.0f}s.\n"
        f"Last log lines:\n{tail_file(log_path)}"
    )


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    for sig, timeout in (
        (signal.SIGINT, 30.0),
        (signal.SIGTERM, 15.0),
        (signal.SIGKILL, 5.0),
    ):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def run_streaming_request(
    url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    begin = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    nonempty_chunks = 0

    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break

            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]

            choices = event.get("choices") or []
            text = choices[0].get("text", "") if choices else ""
            if text:
                nonempty_chunks += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()

    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("The request completed without a non-empty token chunk")

    completion_tokens = int(usage.get("completion_tokens") or nonempty_chunks)
    if completion_tokens <= 0:
        raise RuntimeError("The request reported zero completion tokens")

    decode_tokens = max(completion_tokens - 1, 0)
    decode_seconds = max(end - first_token_at, 0.0)
    tpot_ms = None
    output_tps = None
    if decode_tokens > 0 and decode_seconds > 0:
        tpot_ms = decode_seconds * 1000.0 / decode_tokens
        output_tps = decode_tokens / decode_seconds

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_s": end - begin,
        "ttft_s": first_token_at - begin,
        "tpot_ms": tpot_ms,
        "output_tps": output_tps,
    }


def mean_present(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return statistics.mean(values) if values else None


def median_present(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return statistics.median(values) if values else None


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(samples),
        "prompt_tokens": samples[0].get("prompt_tokens") if samples else None,
        "completion_tokens": samples[0].get("completion_tokens") if samples else None,
        "mean_total_s": mean_present(samples, "total_s"),
        "median_total_s": median_present(samples, "total_s"),
        "mean_ttft_s": mean_present(samples, "ttft_s"),
        "median_ttft_s": median_present(samples, "ttft_s"),
        "mean_tpot_ms": mean_present(samples, "tpot_ms"),
        "median_tpot_ms": median_present(samples, "tpot_ms"),
        "mean_output_tps": mean_present(samples, "output_tps"),
        "median_output_tps": median_present(samples, "output_tps"),
    }


def inspect_mode_log(mode: str, log_path: Path) -> str:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if mode == MODE_MXFP4:
        marker = "runtime MXFP4 path enabled"
    else:
        marker = "using the legacy cache and QK path"
    if marker not in log_text:
        print(
            f"WARNING: expected mode marker {marker!r} was not found in {log_path}",
            file=sys.stderr,
        )
    return marker


def run_mode(
    args: argparse.Namespace,
    mode: str,
    output_dir: Path,
    prompt: str,
) -> dict[str, Any]:
    ensure_port_is_free(args.host, args.port)
    command = build_serve_command(args, mode)
    log_path = output_dir / f"server_{mode}.log"
    health_url = f"http://{args.host}:{args.port}/health"
    completion_url = f"http://{args.host}:{args.port}/v1/completions"

    print(f"\n===== starting {mode.upper()} service =====")
    print("command:", " ".join(command))
    print("log:", log_path)

    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
            wait_until_ready(
                process,
                health_url,
                args.startup_timeout,
                log_path,
            )

            print(f"{mode.upper()} service is ready; running warmups")
            for index in range(args.warmup_runs):
                metric = run_streaming_request(
                    completion_url,
                    args.served_model_name,
                    prompt,
                    args.max_tokens,
                    args.request_timeout,
                )
                print(f"warmup {index + 1}/{args.warmup_runs}: {metric}")

            samples = []
            for index in range(args.benchmark_runs):
                metric = run_streaming_request(
                    completion_url,
                    args.served_model_name,
                    prompt,
                    args.max_tokens,
                    args.request_timeout,
                )
                samples.append(metric)
                print(f"run {index + 1}/{args.benchmark_runs}: {metric}")

        expected_marker = inspect_mode_log(mode, log_path)
        return {
            "mode": mode,
            "command": command,
            "log_path": str(log_path),
            "expected_log_marker": expected_marker,
            "samples": samples,
            "summary": summarize(samples),
        }
    except Exception:
        print(f"\nLast lines from {log_path}:\n{tail_file(log_path)}", file=sys.stderr)
        raise
    finally:
        if process is not None:
            print(f"stopping {mode.upper()} service")
            stop_process_group(process)
        time.sleep(3.0)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def compare_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mxfp4 = results[MODE_MXFP4]["summary"]
    fp8 = results[MODE_FP8]["summary"]
    return {
        "total_latency_speedup_fp8_over_mxfp4": safe_ratio(
            fp8["mean_total_s"], mxfp4["mean_total_s"]
        ),
        "ttft_speedup_fp8_over_mxfp4": safe_ratio(
            fp8["mean_ttft_s"], mxfp4["mean_ttft_s"]
        ),
        "tpot_speedup_fp8_over_mxfp4": safe_ratio(
            fp8["mean_tpot_ms"], mxfp4["mean_tpot_ms"]
        ),
        "decode_throughput_speedup_mxfp4_over_fp8": safe_ratio(
            mxfp4["mean_output_tps"], fp8["mean_output_tps"]
        ),
    }


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid model directory: {model_path}")
    if args.num_hidden_layers < MIN_C4_TEST_LAYERS:
        raise ValueError(
            f"DeepSeek V4's first C4 layer is layer 2; use at least "
            f"--num-hidden-layers {MIN_C4_TEST_LAYERS}."
        )
    if shutil.which(args.vllm_command) is None:
        raise FileNotFoundError(f"vLLM executable not found: {args.vllm_command}")
    if args.warmup_runs < 1 or args.benchmark_runs < 1:
        raise ValueError("Both --warmup-runs and --benchmark-runs must be positive")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or Path("benchmark_results") / f"dsv4-indexer-ab-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    prompt = ("DeepSeek V4 C4 indexer performance benchmark. " * args.prompt_repeat).strip()
    results: dict[str, dict[str, Any]] = {}

    for mode in (MODE_MXFP4, MODE_FP8):
        results[mode] = run_mode(args, mode, output_dir, prompt)

    comparison = compare_results(results)
    report = {
        "model": str(model_path),
        "num_hidden_layers": args.num_hidden_layers,
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
        "max_tokens": args.max_tokens,
        "prompt_repeat": args.prompt_repeat,
        "results": results,
        "comparison": comparison,
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n===== MXFP4 vs native FP8 summary =====")
    print(json.dumps({mode: results[mode]["summary"] for mode in results}, indent=2))
    print("\n===== speedup (greater than 1 means MXFP4 is faster) =====")
    print(json.dumps(comparison, indent=2))
    print(f"\nFull report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

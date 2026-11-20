# SPDX-License-Identifier: Apache-2.0
"""Compile-only A3 V2 synchronization IR collection; never launches TQBMM.

Each variant runs in an isolated process/cache. No model weights, input
allocations, runtime correctness checks or timings are involved. Triton's
warmup=True means compile only, NOT inference warmup. A device context is
still required by the installed Triton driver.
"""

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
import tarfile
from collections import Counter, deque
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from tools.dsv4_a3_wo.tune_tqbmm import run_worker, write_json

ROOT = Path(__file__).resolve().parent
CASES = ("no_prefetch_s2", "prefetch_s1", "prefetch_s2")
TEXT_SUFFIXES = (".json", ".log", ".txt", ".mlir", ".ttir", ".ttadapter", ".bcmlir", ".llir")
SYNC_IR_FLAG = "--bishengir-print-ir-after=hivm-graph-sync-solver"
DEFAULT_PASTE_CHARS = 18000
PASTE_CONTEXT_SAMPLES = 3
MAX_IR_SNIPPET_CHARS = 650


def case_options(name):
    if name not in CASES:
        raise ValueError(f"Unknown case: {name}")
    return {"prefetch": name != "no_prefetch_s2", "num_stages": 1 if name == "prefetch_s1" else 2}


def check_python_environment():
    """Fail before creating artifacts; workers use this exact interpreter.

    This checks module visibility only, without importing torch, initializing
    an NPU context or changing installed packages.
    """
    if find_spec("torch_npu") is None:
        raise SystemExit(
            f"torch_npu is unavailable in this interpreter: {sys.executable}\n"
            "Run from the same vLLM container and Python environment as the NPU pytest run. "
            "Use that interpreter explicitly; do not upgrade/install torch or torch_npu just for this collector."
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New capture directory, or existing directory with --paste"
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=67)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--inner", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=65)
    parser.add_argument("--block-shape", nargs=3, type=int, default=[32, 32, 64])
    parser.add_argument("--panel-k", type=int, default=128)
    parser.add_argument("--programs", type=int, default=3)
    parser.add_argument("--weight-layout", choices=["kn", "nk"], default="kn")
    parser.add_argument("--timeout", type=int, default=300, help="Compilation timeout per variant, seconds")
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--paste", action="store_true", help="Read existing capture and print bounded text; no NPU needed"
    )
    parser.add_argument("--paste-case", choices=("all", *CASES), default="all")
    parser.add_argument("--paste-max-chars", type=int, default=DEFAULT_PASTE_CHARS)
    args = parser.parse_args(argv)
    if min(args.tokens, args.groups, args.inner, args.columns, args.programs, args.timeout) < 1:
        parser.error("Dimensions/programs/timeout must be positive")
    bm, bn, dot_k = args.block_shape
    if any(value < 16 or value > 128 or value & (value - 1) for value in (bm, bn, dot_k)):
        parser.error("Dot tile dimensions must be powers of two in [16,128]")
    if dot_k < 32 or args.panel_k not in (dot_k, 2 * dot_k, 4 * dot_k) or args.panel_k > 512:
        parser.error("Require DOT_K>=32 and panel K=1/2/4 times DOT_K, <=512")
    if args.worker and not args.case:
        parser.error("--worker requires --case")
    if args.paste and (args.worker or args.dry_run):
        parser.error("--paste cannot be combined with --worker or --dry-run")
    if args.paste_max_chars < 2000:
        parser.error("--paste-max-chars must be >=2000")
    return args


def compile_spec(args, case, dtypes):
    """Match the failing [T,G,K] / [G,K,N] layout and pointer dtypes."""
    t, g, k, n = args.tokens, args.groups, args.inner, args.columns
    bm, bn, dot_k = args.block_shape
    programs = min(args.programs, ((t + bm - 1) // bm) * ((n + bn - 1) // bn))
    weight_strides = (k * n, n, 1) if args.weight_layout == "kn" else (k * n, 1, k)
    positional = (*dtypes, t, g, k, n, g * k, k, 1, *weight_strides, g, 1, n, 1)
    options = case_options(case)
    keywords = {
        "BM": bm,
        "BN": bn,
        "DOT_K": dot_k,
        "PANEL_K": args.panel_k,
        "PROGRAMS": programs,
        "PREFETCH": options["prefetch"],
        "num_stages": options["num_stages"],
        "debug": False,
    }
    return positional, (programs, g), keywords


def compiler_capture(original_run, output, records):
    """Capture the compiler's stdout/stderr before the backend discards it.

    Installed backends differ in how debug IR is saved. This tool-only hook
    records BiShengIR subprocess output, adding only an IR-printing flag.
    Optimization flags and the Triton debug=False setting are unchanged.
    It is removed before leaving the isolated compile worker.
    """

    def run(*positional, **keywords):
        command = positional[0] if positional else keywords.get("args")
        capture = isinstance(command, (list, tuple)) and command and "bishengir" in Path(str(command[0])).name.lower()
        if capture:
            command = list(command)
            if SYNC_IR_FLAG not in command:
                command.insert(1, SYNC_IR_FLAG)
            if positional:
                positional = (command, *positional[1:])
            else:
                keywords = {**keywords, "args": command}
        try:
            result = original_run(*positional, **keywords)
        except subprocess.CalledProcessError as error:
            if capture:
                save(command, error.stdout, error.stderr, error.returncode)
            raise
        if capture:
            save(command, result.stdout, result.stderr, result.returncode)
        return result

    def save(command, stdout, stderr, returncode):
        index = len(records)
        prefix = f"compiler-{index:02}"
        for name, value in (("stdout", stdout), ("stderr", stderr)):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            (output / f"{prefix}.{name}.txt").write_text(value or "", encoding="utf-8")
        records.append({"command": list(map(str, command)), "returncode": returncode, "prefix": prefix})
        write_json(output / "compiler_commands.json", records)

    return run


def worker(args):
    # Compile-only dtype placeholders; no torch.empty/.to(), quantization,
    # .synchronize(), plan() or kernel[grid]() calls are allowed in this tool.
    print(f"Worker Python: {sys.executable} (prefix={sys.prefix})", flush=True)
    import torch
    import torch_npu

    implementation = importlib.import_module("vllm_ascend.ops.triton.transpose_quant_batch_matmul_v2")
    kernel = implementation._transpose_quant_batch_matmul_v2_kernel

    args.output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(args.device)
    dtypes = (torch.int8, torch.int8, torch.float32, torch.float32, torch.float32)
    positional, grid, keywords = compile_spec(args, args.case, dtypes)
    records = []
    print(f"Compile only: {args.case}, grid={grid}, options={keywords}", flush=True)
    with patch("subprocess.run", compiler_capture(subprocess.run, args.output, records)):
        compiled = kernel.warmup(*positional, grid=grid, **keywords)
    if compiled is None:
        raise RuntimeError("Triton warmup returned no compiled kernel")
    saved = []
    for name, value in compiled.asm.items():
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name):
            path = args.output / f"asm-{name}.mlir"
            path.write_text(value, encoding="utf-8")
            saved.append(path.name)
    report = {
        "compile_only": "completed",
        "case": args.case,
        "tqbmm_launches": 0,
        "shape_T_G_K_N": [args.tokens, args.groups, args.inner, args.columns],
        "weight_layout": args.weight_layout,
        "pointer_alignment": "dtype-only warmup assumes 16-byte alignment",
        "grid": grid,
        "options": keywords,
        "asm_text_files": saved,
        "compiler_invocations": records,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "backend_sync_ir_found": any(
            "hivm" in path.read_text(encoding="utf-8", errors="replace") for path in args.output.glob("compiler-*.txt")
        ),
        "kernel_source": str(Path(implementation.__file__).resolve()),
        "kernel_sha256": hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
        "correctness_and_performance": "not tested; no input buffers or model weights loaded",
    }
    write_json(args.output / "compile.json", report)
    return report


def worker_command(args, case, output):
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--case",
        case,
        "--output",
        str(output),
        "--device",
        args.device,
        "--tokens",
        str(args.tokens),
        "--groups",
        str(args.groups),
        "--inner",
        str(args.inner),
        "--columns",
        str(args.columns),
        "--block-shape",
        *map(str, args.block_shape),
        "--panel-k",
        str(args.panel_k),
        "--programs",
        str(args.programs),
        "--weight-layout",
        args.weight_layout,
    ]


def export_text(root):
    archive = root / "sync_ir_export.tar.gz"
    with tarfile.open(archive, "x:gz") as bundle:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and not path.is_symlink()
                and "cache" not in relative.parts
                and path.suffix in TEXT_SUFFIXES
                and path.resolve().is_relative_to(root.resolve())
            ):
                bundle.add(path, arcname=str(relative), recursive=False)
    return archive


def read_optional_json(path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        return {"read_error": str(error)}


def limited_text(value, limit):
    marker = "\n[TRUNCATED: request a single --paste-case or the referenced source lines]\n"
    return value if len(value) <= limit else value[: limit - len(marker)] + marker


def summarize_ir(path):
    """Stream counts plus first/last snippets; never infer runtime deadlock.

    Counts are STATIC IR occurrences, not executed operations or durations.
    Line references retain a path back to omitted loop bodies/event context.
    """
    counts = Counter()
    first = {name: [] for name in ("sync", "loops", "compute", "functions")}
    last = {name: deque(maxlen=PASTE_CONTEXT_SAMPLES) for name in first}
    total_lines = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for total_lines, line in enumerate(handle, 1):
            # Match instructions, not #hivm.address_space/type attributes.
            op = re.match(
                r'^\s*(?:%[^=]*=\s*)?"?(hivm(?:\.[A-Za-z0-9_]+)+|scf\.(?:for|if|while|yield)|func\.(?:func|return))'
                r'(?:"\s*\(|\s|\()',
                line,
            )
            if not op:
                continue
            name = op.group(1)
            counts[name] += 1
            if any(token in name.lower() for token in ("sync", "set_flag", "wait_flag", "barrier")):
                kind = "sync"
            elif name.startswith("scf."):
                kind = "loops"
            elif any(token in name.lower() for token in ("mmad", "fixpipe", "nd2nz")):
                kind = "compute"
            elif name.startswith("func."):
                kind = "functions"
            else:
                continue
            text = line.rstrip()
            symbol = re.search(r'sym_name\s*=\s*"([^"]+)"', text)
            if kind == "functions" and symbol:
                text = f"func.func sym_name={symbol.group(1)}"
            width = MAX_IR_SNIPPET_CHARS if kind == "sync" else MAX_IR_SNIPPET_CHARS // 2
            snippet = f"L{total_lines}: {text[:width]}" + (" [line shortened]" if len(text) > width else "")
            if len(first[kind]) < PASTE_CONTEXT_SAMPLES:
                first[kind].append((total_lines, snippet))
            last[kind].append((total_lines, snippet))
    samples = {}
    for kind in first:
        samples[kind] = [snippet for _, snippet in sorted(dict(first[kind] + list(last[kind])).items())]
    return {"lines": total_lines, "static_op_counts": dict(sorted(counts.items())), "samples": samples}


def choose_backend_ir(directory, commands):
    # Prefer the last successful compiler invocation over a failed UB-tuning
    # attempt. Only fall back to a debug dump if no captured HIVM output exists.
    prefixes = [record.get("prefix") for record in reversed(commands) if record.get("returncode") == 0]
    prefixes += [record.get("prefix") for record in reversed(commands) if record.get("returncode") != 0]
    candidates = []
    for prefix in prefixes:
        if isinstance(prefix, str) and re.fullmatch(r"compiler-\d+", prefix):
            candidates.extend(directory / f"{prefix}.{stream}.txt" for stream in ("stderr", "stdout"))
    candidates += sorted(directory.glob("compiler-*.stderr.txt")) + sorted(directory.glob("compiler-*.stdout.txt"))
    candidates += sorted((directory / "ir").rglob("*npuir*.mlir"))
    for path in dict.fromkeys(candidates):
        if path.is_file():
            with path.open(encoding="utf-8", errors="replace") as handle:
                if any("hivm." in line for line in handle):
                    return path
    return None


def paste_report(root, case="all", max_chars=DEFAULT_PASTE_CHARS):
    root = root.resolve(strict=True)
    manifest = read_optional_json(root / "manifest.json", {})
    header = [
        "TQBMM V2 COMPILE-ONLY SUMMARY",
        f"directory: {root}",
        "IR counts are static occurrences. Snippets are incomplete; they do not prove a runtime deadlock.",
        json.dumps(
            {key: manifest.get(key) for key in ("run_status", "python_executable", "python_prefix", "kernel_sha256")},
            ensure_ascii=False,
        ),
    ]
    names = CASES if case == "all" else (case,)
    case_entries = {entry.get("case"): entry for entry in manifest.get("cases", []) if isinstance(entry, dict)}
    case_budget = max(1000, (max_chars - sum(map(len, header)) - 32) // len(names))
    sections = []
    for name in names:
        directory = root / name
        report = read_optional_json(directory / "compile.json", case_entries.get(name, {}))
        commands = read_optional_json(directory / "compiler_commands.json", [])
        commands = commands if isinstance(commands, list) else []
        lines = [
            f"\n=== {name} ===",
            json.dumps(
                {
                    key: report.get(key)
                    for key in (
                        "compile_only",
                        "error",
                        "shape_T_G_K_N",
                        "grid",
                        "options",
                        "backend_sync_ir_found",
                        "torch",
                        "torch_npu",
                        "kernel_sha256",
                    )
                },
                ensure_ascii=False,
            ),
        ]
        flags = []
        for record in commands:
            selected = [
                arg
                for arg in record.get("command", [])
                if arg.startswith(
                    (
                        "--target=",
                        "--enable-auto-multi-buffer=",
                        "--enable-code-motion=",
                        "--enable-preload=",
                        "--enable-hivm-graph-sync-solver=",
                        "--limit-auto-multi-buffer",
                        "--bishengir-print-ir-after=",
                    )
                )
            ]
            flags.append({"prefix": record.get("prefix"), "returncode": record.get("returncode"), "flags": selected})
        lines.append("compiler flags: " + json.dumps(flags))
        path = choose_backend_ir(directory, commands)
        if path is None:
            lines.append("NO BACKEND HIVM IR: compilation/import may have failed; no synchronization conclusion.")
            log = directory / "compile.log"
            if log.is_file():
                with log.open(encoding="utf-8", errors="replace") as handle:
                    tail = deque(handle, maxlen=12)
                lines.append("compile.log tail:\n" + "".join(tail))
        else:
            summary = summarize_ir(path)
            lines += [
                f"source: {path.relative_to(root)}",
                f"source lines: {summary['lines']}",
                "static op counts: " + json.dumps(summary["static_op_counts"]),
            ]
            for kind in ("functions", "sync", "loops", "compute"):
                lines.append(f"{kind}: first/last samples (not the complete IR)")
                lines.extend(summary["samples"][kind])
        sections.append(limited_text("\n".join(lines), case_budget))
    return limited_text("\n".join(header + sections), max_chars)


def main(argv=None):
    args = parse_args(argv)
    if args.paste:
        print(paste_report(args.output, args.paste_case, args.paste_max_chars))
        return
    if args.worker:
        print(json.dumps(worker(args), indent=2))
        return
    commands = [(case, worker_command(args, case, args.output.resolve() / case)) for case in CASES]
    if args.dry_run:
        print(json.dumps({"compile_only": True, "commands": commands}, indent=2))
        return
    check_python_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    kernel_path = ROOT / "vllm_ascend/ops/triton/transpose_quant_batch_matmul_v2.py"
    manifest = {
        "run_status": "running",
        "compile_only": True,
        "tqbmm_launches": 0,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "cases": [],
    }
    write_json(args.output / "manifest.json", manifest)
    # Launch via a small wrapper so every worker gets a dedicated compiler
    # cache/dump location before importing Triton; existing caches are intact.
    wrapper = (
        "import os,runpy,sys; from pathlib import Path; "
        "out=Path(sys.argv[2]); "
        "os.environ.update(TRITON_CACHE_DIR=str(out/'cache'),TRITON_DUMP_DIR=str(out/'ir'),"
        "TRITON_KERNEL_DUMP='1',TRITON_ALWAYS_COMPILE='1'); "
        "sys.argv=[sys.argv[1]]+sys.argv[3:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    for case, command in commands:
        directory = args.output.resolve() / case
        directory.mkdir()
        wrapped = [command[0], "-u", "-c", wrapper, command[2], str(directory), *command[3:]]
        print(f"Compile only: {case}", flush=True)
        try:
            report = run_worker(wrapped, directory / "compile.log", args.timeout, "compile_only")
            if report.get("kernel_sha256") != manifest["kernel_sha256"]:
                raise RuntimeError("Worker imported a different kernel source than this repository")
            manifest["cases"].append(report)
        except Exception as error:
            manifest["cases"].append({"case": case, "compile_only": "failed", "error": str(error)})
        write_json(args.output / "manifest.json", manifest)
    manifest["run_status"] = (
        "completed" if all(case["compile_only"] == "completed" for case in manifest["cases"]) else "failed"
    )
    write_json(args.output / "manifest.json", manifest)
    print(export_text(args.output), flush=True)
    if manifest["run_status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

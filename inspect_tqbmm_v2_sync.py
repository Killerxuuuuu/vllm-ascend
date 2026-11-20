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
from pathlib import Path
from unittest.mock import patch

from tools.dsv4_a3_wo.tune_tqbmm import run_worker, write_json

ROOT = Path(__file__).resolve().parent
CASES = ("no_prefetch_s2", "prefetch_s1", "prefetch_s2")
TEXT_SUFFIXES = (".json", ".log", ".txt", ".mlir", ".ttir", ".ttadapter", ".bcmlir", ".llir")
SYNC_IR_FLAG = "--bishengir-print-ir-after=hivm-graph-sync-solver"


def case_options(name):
    if name not in CASES:
        raise ValueError(f"Unknown case: {name}")
    return {"prefetch": name != "no_prefetch_s2", "num_stages": 1 if name == "prefetch_s1" else 2}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; refuse to overwrite")
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


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        print(json.dumps(worker(args), indent=2))
        return
    commands = [(case, worker_command(args, case, args.output.resolve() / case)) for case in CASES]
    if args.dry_run:
        print(json.dumps({"compile_only": True, "commands": commands}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    kernel_path = ROOT / "vllm_ascend/ops/triton/transpose_quant_batch_matmul_v2.py"
    manifest = {
        "run_status": "running",
        "compile_only": True,
        "tqbmm_launches": 0,
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

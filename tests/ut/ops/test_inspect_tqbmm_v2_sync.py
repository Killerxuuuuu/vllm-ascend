# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for compile-only synchronization IR collection."""

import ast
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "inspect_tqbmm_v2_sync.py"


@pytest.fixture
def inspector():
    spec = importlib.util.spec_from_file_location("inspect_sync_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout,strides", [("kn", (266240, 65, 1)), ("nk", (266240, 1, 4096))])
@pytest.mark.parametrize(
    "case,prefetch,stages", [("no_prefetch_s2", False, 2), ("prefetch_s1", True, 1), ("prefetch_s2", True, 2)]
)
def test_exact_failing_shape_and_variant_axes(inspector, layout, strides, case, prefetch, stages):
    args = inspector.parse_args(["--output", "unused", "--weight-layout", layout])
    positional, grid, options = inspector.compile_spec(args, case, ("i8", "i8", "f32", "f32", "f32"))
    assert positional[5:9] == (67, 2, 4096, 65)
    assert positional[9:12] == (8192, 4096, 1)
    assert positional[12:15] == strides
    assert positional[15:] == (2, 1, 65, 1)
    assert grid == (3, 2)
    assert options == {
        "BM": 32,
        "BN": 32,
        "DOT_K": 64,
        "PANEL_K": 128,
        "PROGRAMS": 3,
        "PREFETCH": prefetch,
        "num_stages": stages,
        "debug": False,
    }


def test_compiler_output_hook_keeps_optimizations_and_captures_ir(inspector, tmp_path):
    records, called = [], []

    def original(command, **kwargs):
        called.append(command)
        return SimpleNamespace(stdout=b"UB size=42 bits", stderr=b"hivm.hir.sync_block_wait", returncode=0)

    command = ["/cann/bishengir-compile", "kernel.mlir", "--enable-auto-multi-buffer=True", "-o", "kernel.o"]
    wrapper = inspector.compiler_capture(original, tmp_path, records)
    wrapper(command, check=True)
    assert command[1] == "kernel.mlir"  # No mutation of the caller's command.
    assert called[0] == [command[0], inspector.SYNC_IR_FLAG, *command[1:]]
    assert records[0]["returncode"] == 0
    assert (tmp_path / "compiler-00.stderr.txt").read_text() == "hivm.hir.sync_block_wait"
    wrapper(["gcc", "callback.c"])
    assert len(records) == 1 and called[-1] == ["gcc", "callback.c"]


def test_failed_compile_keeps_original_exception_and_output(inspector, tmp_path):
    error = subprocess.CalledProcessError(1, ["bishengir-compile"], output=b"out", stderr=b"bad IR")

    def original(*args, **kwargs):
        raise error

    records = []
    with pytest.raises(subprocess.CalledProcessError) as caught:
        inspector.compiler_capture(original, tmp_path, records)(args=["bishengir-compile", "kernel.mlir"])
    assert caught.value is error
    assert records[0]["returncode"] == 1
    assert (tmp_path / "compiler-00.stderr.txt").read_text() == "bad IR"


def test_worker_has_no_runtime_kernel_launch_or_tensor_allocation():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "worker")
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Subscript) for node in calls)
    forbidden = {"empty", "ones", "randn", "to", "npu_dynamic_quant", "synchronize", "run"}
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in forbidden for node in calls)
    assert sum(isinstance(node.func, ast.Attribute) and node.func.attr == "warmup" for node in calls) == 1


def test_worker_uses_dtype_only_warmup_and_saves_ir(inspector, tmp_path, monkeypatch):
    calls = []
    torch = ModuleType("torch")
    torch.int8, torch.float32, torch.__version__ = "i8", "f32", "stub"
    torch.npu = SimpleNamespace(set_device=lambda device: calls.append(("set_device", device)))
    torch_npu = ModuleType("torch_npu")
    torch_npu.__version__ = "stub"

    def warmup(*args, **kwargs):
        calls.append(("warmup", args, kwargs))
        return SimpleNamespace(asm={"ttir": "module {}", "npubin": b"binary"})

    implementation = ModuleType("vllm_ascend.ops.triton.transpose_quant_batch_matmul_v2")
    implementation.__file__ = str(ROOT / "vllm_ascend/ops/triton/transpose_quant_batch_matmul_v2.py")
    implementation._transpose_quant_batch_matmul_v2_kernel = SimpleNamespace(warmup=warmup)
    for name, module in (("torch", torch), ("torch_npu", torch_npu), (implementation.__name__, implementation)):
        monkeypatch.setitem(sys.modules, name, module)
    args = inspector.parse_args(["--output", str(tmp_path), "--worker", "--case", "prefetch_s2"])
    result = inspector.worker(args)
    assert result["tqbmm_launches"] == 0
    assert [call[0] for call in calls] == ["set_device", "warmup"]
    assert calls[1][1][:5] == ("i8", "i8", "f32", "f32", "f32")
    assert calls[1][2]["debug"] is False
    assert (tmp_path / "asm-ttir.mlir").is_file()
    assert not (tmp_path / "asm-npubin.mlir").exists()
    assert result["backend_sync_ir_found"] is False


def test_export_excludes_compiler_cache_and_binary(inspector, tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "private.json").write_text("skip")
    (tmp_path / "kernel.npubin").write_bytes(b"binary")
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "compiler.stderr.txt").write_text("hivm")
    archive = inspector.export_text(tmp_path)
    with tarfile.open(archive) as handle:
        assert set(handle.getnames()) == {"manifest.json", "compiler.stderr.txt"}


def test_dry_run_does_not_import_torch_or_create_output(inspector, tmp_path, capsys):
    output = tmp_path / "new-run"
    inspector.main(["--output", str(output), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["compile_only"] is True and len(plan["commands"]) == 3
    assert not output.exists()
    for name, command in plan["commands"]:
        assert command[command.index("--case") + 1] == name
        assert command[command.index("--programs") + 1] == "3"


def test_parent_wraps_fresh_cache_and_exports_even_compile_failure(inspector, tmp_path, monkeypatch):
    calls = []

    def fake_run(command, log, timeout, expected_key):
        calls.append(command)
        assert "TRITON_CACHE_DIR" in command[3] and "TRITON_ALWAYS_COMPILE" in command[3]
        log.write_text("compile output")
        if len(calls) == 2:
            raise RuntimeError("compiler failed")
        return {
            "compile_only": "completed",
            "case": command[command.index("--case") + 1],
            "tqbmm_launches": 0,
            "kernel_sha256": inspector.hashlib.sha256(
                (ROOT / "vllm_ascend/ops/triton/transpose_quant_batch_matmul_v2.py").read_bytes()
            ).hexdigest(),
        }

    monkeypatch.setattr(inspector, "run_worker", fake_run)
    output = tmp_path / "result"
    with pytest.raises(SystemExit):
        inspector.main(["--output", str(output)])
    result = json.loads((output / "manifest.json").read_text())
    assert result["run_status"] == "failed" and len(calls) == 3
    assert result["cases"][1]["compile_only"] == "failed"
    assert (output / "sync_ir_export.tar.gz").is_file()

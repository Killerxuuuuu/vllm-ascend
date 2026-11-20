# SPDX-License-Identifier: Apache-2.0
"""CPU source interpreter/host contracts only; not an A3 compiler test."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tests.ut.ops import test_tqbmm_prepared_host as host_stubs
from tests.ut.ops.test_tqbmm_indexing import Array, Language, Pointer

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/ops/triton/transpose_quant_batch_matmul_v2.py"


@pytest.fixture
def host(monkeypatch):
    return host_stubs.host.__wrapped__(monkeypatch)


class V2Language(Language):
    static_range = staticmethod(range)

    @staticmethod
    def where(condition, lhs, rhs):
        return np.asarray(np.where(condition, lhs, rhs)).view(Array)

    @staticmethod
    def broadcast_to(array, shape):
        return np.broadcast_to(array, shape).view(Array)

    @staticmethod
    def gather(array, indices, axis):
        return np.take_along_axis(array, indices, axis).view(Array)


def load_kernels(language):
    nodes = []
    for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(
            ("_v2_coordinates", "_v2_load_panel", "_v2_dot_panel", "_transpose_")
        ):
            node.decorator_list = []
            for argument in node.args.args:
                argument.annotation = None
            nodes.append(node)
    namespace = {"tl": language, "GROUP_SIZE_M": 4}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace["_transpose_quant_batch_matmul_v2_kernel"]


@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 1), (67, 2, 32, 65), (67, 2, 131, 65), (129, 1, 257, 35), (17, 2, 4096, 33)]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("panel_ratio", [1, 2, 4])
@pytest.mark.parametrize("programs", [1, 3])
@pytest.mark.parametrize("block_shape", [(16, 32, 32), (32, 32, 64)])
def test_actual_v2_source_against_int64(shape, strided, prefetch, panel_ratio, programs, block_shape):
    t, g, k, n = shape
    rng = np.random.default_rng(5)
    x_storage = rng.integers(-128, 128, (t, g, k + 2), dtype=np.int8)
    x = x_storage[:, :, 1:-1]
    if strided:
        w_storage = rng.integers(-128, 128, (g, n, k + 2), dtype=np.int8)
        w = w_storage[:, :, 1:-1].transpose(0, 2, 1)
    else:
        w_storage = rng.integers(-128, 128, (g, k, n), dtype=np.int8)
        w = w_storage
    sx_storage = rng.random((g, t + 1), dtype=np.float32)
    sx = sx_storage[:, 1:].T
    sw_storage = rng.random((n + 1, g), dtype=np.float32)
    sw = sw_storage[1:].T
    actual = np.full((t, g, n), np.nan, dtype=np.float32)
    pointers = [
        Pointer(storage, (view.ctypes.data - storage.ctypes.data) // view.itemsize)
        for view, storage in ((x, x_storage), (w, w_storage), (sx, sx_storage), (sw, sw_storage))
    ]
    strides = tuple(stride // view.itemsize for view in (x, w, sx, sw) for stride in view.strides)
    bm, bn, dot_k = block_shape
    panel_k = dot_k * panel_ratio
    programs = min(programs, Language.cdiv(t, bm) * Language.cdiv(n, bn))
    for group in range(g):
        for pid in range(programs):
            kernel = load_kernels(V2Language(pid, group))
            kernel(*pointers, Pointer(actual), t, g, k, n, *strides, bm, bn, dot_k, panel_k, programs, prefetch)
    golden = np.einsum("tgk,gkn->tgn", x.astype(np.int64), w.astype(np.int64))
    expected = (golden.astype(np.float32) * sw[None]) * sx[:, :, None]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("inner", [32, 64, 65, 128, 129, 192, 193, 257])
@pytest.mark.parametrize("prefetch", [False, True])
def test_panel_pairs_load_ahead_once_and_handle_single_odd_tails(inner, prefetch):
    """Audit source ordering/load count, not physical DMA or pipeline overlap."""
    x = np.ones((1, 1, inner), dtype=np.int8)
    w = np.ones((1, inner, 1), dtype=np.int8)
    sx = np.ones((1, 1), dtype=np.float32)
    sw = np.ones((1, 1), dtype=np.float32)
    actual = np.full((1, 1, 1), np.nan, dtype=np.float32)
    events = []

    class TraceLanguage(V2Language):
        def load(self, pointer, mask, other):
            if np.shares_memory(pointer.storage, x):
                events.append("load_A")
            elif np.shares_memory(pointer.storage, w):
                events.append("load_B")
            return super().load(pointer, mask, other)

        def dot(self, a, b, acc, out_dtype):
            events.append("dot")
            return super().dot(a, b, acc, out_dtype)

    strides = tuple(stride // value.itemsize for value in (x, w, sx, sw) for stride in value.strides)
    load_kernels(TraceLanguage(0, 0))(
        *map(Pointer, (x, w, sx, sw, actual)), 1, 1, inner, 1, *strides, 16, 32, 32, 64, 1, prefetch
    )
    panels = Language.cdiv(inner, 64)
    if prefetch:
        expected_events = ["load_A", "load_B", "load_A", "load_B", "dot", "dot", "dot", "dot"] * (panels // 2)
        if panels % 2:
            expected_events += ["load_A", "load_B", "dot", "dot"]
    else:
        expected_events = ["load_A", "load_B", "dot", "dot"] * panels
    assert events == expected_events
    assert actual[0, 0, 0] == inner


def v2_host(host):
    module, backend = host
    namespace = module.__dict__
    namespace.update(_transpose_quant_batch_matmul_v2_kernel=backend)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.ClassDef))
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in ("validate_v2_options", "_v2_programs", "_launch_v2", "transpose_quant_batch_matmul_v2")
        )
    ]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return module, backend


def test_fixed_and_dynamic_v2_contracts(host):
    module, backend = v2_host(host)
    x, w, sx, sw = host_stubs.inputs(backend, tokens=129)
    config = module.V2Config(programs=3)
    plan = module.PreparedTQBMMV2(x, w, sx, sw, config=config)
    assert plan() is plan.output
    assert backend.launches[-1][0] == (3, 2)
    assert backend.launches[-1][2]["PANEL_K"] == 128
    assert backend.launches[-1][2]["DOT_K"] == 64
    queries = len(backend.queries)
    assert plan() is plan.output
    assert len(backend.queries) == queries
    dynamic = module.PreparedDynamicWoAV2(w, sw, config=config)
    a = backend.tensor(x.shape, "bf16")
    first, second = dynamic(a), dynamic(a)
    assert first._storage is not second._storage
    assert backend.launches[-1][1][0] is backend.quantizations[-1][1]
    assert backend.launches[-1][1][2] is backend.quantizations[-1][2]
    assert len(backend.quantizations) == 2


def test_v2_zero_tokens_and_invalid_activation(host):
    module, backend = v2_host(host)
    x, w, sx, sw = host_stubs.inputs(backend, tokens=0)
    config = module.V2Config(programs=2)
    assert module.PreparedTQBMMV2(x, w, sx, sw, config=config)().shape == (0, 2, 33)
    plan = module.PreparedDynamicWoAV2(w, sw, config=config)
    assert plan(backend.tensor((0, 2, 65), "bf16")).shape == (0, 2, 33)
    assert not backend.launches and not backend.quantizations
    with pytest.raises(ValueError, match="match"):
        plan(backend.tensor((1, 2, 64), "bf16"))
    with pytest.raises(ValueError, match="BF16"):
        plan(backend.tensor((1, 2, 65), "f32"))


def test_v2_checks_new_quantized_storage_each_call(host, monkeypatch):
    module, backend = v2_host(host)
    x, w, _, sw = host_stubs.inputs(backend)
    plan = module.PreparedDynamicWoAV2(w, sw, config=module.V2Config(programs=2))

    def nz_quant(tensor, **kwargs):
        quantized, scale = backend.dynamic_quant(tensor, **kwargs)
        quantized.fmt = 29
        return quantized, scale

    monkeypatch.setattr(module.torch_npu, "npu_dynamic_quant", nz_quant)
    with pytest.raises(ValueError, match="ND INT8/FP32"):
        plan(backend.tensor(x.shape, "bf16"))
    assert not backend.launches


@pytest.mark.parametrize("cores", [20, None, 0, True])
def test_v2_queries_aic_core_count_only_during_preparation(host, monkeypatch, cores):
    module, _ = v2_host(host)
    runtime = ModuleType("triton.runtime")
    runtime.driver = SimpleNamespace(
        active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=lambda device: {"num_aicore": cores}))
    )
    monkeypatch.setitem(sys.modules, "triton", ModuleType("triton"))
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime)
    if cores == 20:
        assert module._v2_programs(host_stubs.Device(), 0) == 20
    else:
        with pytest.raises(RuntimeError, match="num_aicore"):
            module._v2_programs(host_stubs.Device(), 0)
    assert module._v2_programs(host_stubs.Device(), 7) == 7


@pytest.mark.parametrize(
    "options",
    [
        {"panel_k": 48},
        {"block_shape": (16, 32, 16)},
        {"block_shape": (64, 33, 64)},
        {"prefetch": 1},
        {"programs": -1},
        {"num_stages": True},
        {"panel_k": 1024},
    ],
)
def test_bad_v2_configuration(host, options):
    module, _ = v2_host(host)
    with pytest.raises((ValueError, TypeError)):
        module.V2Config(**options)


def test_source_has_one_accumulator_and_immutable_gm_bases():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    kernel = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("_transpose_")
    )
    zeros = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "zeros"
    ]
    assert len(zeros) == 1
    assert not any(isinstance(node, ast.AugAssign) for node in ast.walk(kernel))
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "atomic_add"
        for node in ast.walk(tree)
    )


def test_prefetch_never_carries_panel_tensors_between_loop_iterations():
    """Regression guard for NZ initial memref vs ND yielded memref on A3.

    Each panel variable is born inside its consuming iteration/tail. There is
    no prologue panel or `current = next` recurrence for layout inference to
    turn into a panel-valued scf.for iter_arg. Device tests validate lowering.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    kernel = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("_transpose_")
    )
    tile_loop = next(node for node in kernel.body if isinstance(node, ast.For))
    before_tiles = kernel.body[: kernel.body.index(tile_loop)]
    assert not any(
        isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_v2_load_panel"
        for statement in before_tiles
        for node in ast.walk(statement)
    )
    panel_names = {"a_current", "b_current", "a_next", "b_next", "a_tail", "b_tail"}
    assignments = [
        node
        for node in ast.walk(tile_loop)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in panel_names
            for group in node.targets
            for target in ast.walk(group)
        )
    ]
    assert len(assignments) == 3
    assert all(
        isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "_v2_load_panel"
        for node in assignments
    )


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("benchmark_v2_cpu", ROOT / "benchmark_tqbmm_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_dry_run_and_command_forwarding(runner, tmp_path, capsys):
    options = [
        "--quant-model",
        "unused",
        "--output",
        str(tmp_path / "run"),
        "--tokens",
        "32",
        "8192",
        "--seeds",
        "0",
        "1",
        "--panel-k",
        "256",
        "--programs",
        "20",
        "--check-only",
        "--dry-run",
    ]
    runner.main(options)
    commands = json.loads(capsys.readouterr().out)["commands"]
    assert len(commands) == 4 and not (tmp_path / "run").exists()
    assert all("--check-only" in command for command in commands)
    assert all(command[command.index("--panel-k") + 1] == "256" for command in commands)
    assert all(command[command.index("--programs") + 1] == "20" for command in commands)


def test_runner_rotates_timing_and_computes_ratios(runner):
    args = runner.parse_args(["--quant-model", "unused", "--output", "unused", "--repeats", "3"])
    args.current_mode = "graph"
    calls = []

    def timer(function, *positional, **kwargs):
        calls.append(function)
        return {"composition": 2, "v1": 4, "v2": 1}[function]

    measurements = runner.measure({name: name for name in ("composition", "v1", "v2")}, args, timer)
    assert calls == ["composition", "v1", "v2", "v1", "v2", "composition", "v2", "composition", "v1"]
    report = {"T": 8192, "seed": 0, "timings": {"graph": {"prequantized": measurements}}}
    rows = list(runner.timing_rows(report))
    assert rows[-1]["speedup_vs_composition"] == 2
    assert rows[-1]["speedup_vs_v1"] == 4


def test_runner_preserves_failed_run_and_stops(runner, tmp_path, monkeypatch):
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("correctness gate failed")

    monkeypatch.setattr(runner, "run_worker", fail)
    output = tmp_path / "run"
    options = ["--quant-model", str(tmp_path), "--output", str(output), "--tokens", "32", "8192"]
    with pytest.raises(RuntimeError, match="gate failed"):
        runner.main(options)
    result = json.loads((output / "results.json").read_text())
    assert result["run_status"] == "failed" and not result["reports"]
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        runner.main(options)

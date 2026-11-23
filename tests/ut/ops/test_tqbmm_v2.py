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
    @staticmethod
    def cdiv(x, y):
        # Match tl.cdiv's tensor return, not Python integer division. A plain
        # int result here previously hid the device static_range compile bug.
        return np.asarray((x + y - 1) // y, dtype=np.int32).view(Array)

    @staticmethod
    def static_range(*bounds):
        if any(type(value) is not int for value in bounds):
            raise TypeError("static_range requires compile-time integers, not Tensor scalars")
        return range(*bounds)

    @staticmethod
    def extract_slice(array, offsets, sizes, strides):
        assert strides == (1, 1)
        return array[tuple(slice(start, start + size) for start, size in zip(offsets, sizes))]


def test_static_range_rejects_tensor_cdiv_result():
    language = V2Language(0, 0)
    with pytest.raises(TypeError, match="compile-time integers"):
        language.static_range(language.cdiv(32, 64))
    assert list(language.static_range((32 + 64 - 1) // 64)) == [0]


def test_v2_jit_has_no_chained_boolean_operators():
    """Python accepts three-way BoolOps; the A3 Triton AST-to-TTIR frontend rejects them."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    violations = [
        (function.name, node.lineno, ast.unparse(node))
        for function in tree.body
        if isinstance(function, ast.FunctionDef)
        and any(ast.unparse(d) == "triton.jit" for d in function.decorator_list)
        for node in ast.walk(function)
        if isinstance(node, ast.BoolOp) and len(node.values) > 2
    ]
    assert not violations, violations


def test_shape_control_and_micro_loop_bound_are_explicit_constexpr():
    """NumPy cannot model Triton's unannotated-Assign conversion to tensors.

    Protect that compiler contract directly as well as executing arithmetic.
    AnnAssign constexpr values must be pure arithmetic, without JIT tl.cdiv.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for function_name, required in (
        ("_v2_accumulate_panel", {"micro_tiles", "issue_width"}),
        ("_transpose_quant_batch_matmul_v2_kernel", {"wide", "num_tiles", "step_k", "full_steps", "tail_k"}),
    ):
        function = functions[function_name]
        constants = {
            node.target.id: node
            for node in ast.walk(function)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert required <= constants.keys()
        for name in required:
            node = constants[name]
            assert isinstance(node.annotation, ast.Attribute) and node.annotation.attr == "constexpr"
            assert isinstance(node.annotation.value, ast.Name) and node.annotation.value.id == "tl"
            assert not any(isinstance(value, ast.Call) for value in ast.walk(node.value))
    micro_loop = next(node for node in ast.walk(functions["_v2_accumulate_panel"]) if isinstance(node, ast.For))
    assert ast.unparse(micro_loop.iter) == "tl.static_range(micro_tiles // issue_width)"


def load_kernel_namespace(language):
    nodes = []
    for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(("_v2_", "_transpose_")):
            node.decorator_list = []
            for argument in node.args.args:
                argument.annotation = None
            nodes.append(node)
    namespace = {"tl": language, "GROUP_SIZE_M": 4, "extract_slice": language.extract_slice}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace


def load_kernels(language):
    return load_kernel_namespace(language)["_transpose_quant_batch_matmul_v2_kernel"]


@pytest.mark.parametrize("block_shape", [(16, 32, 32), (32, 32, 64), (64, 16, 128), (128, 64, 128)])
@pytest.mark.parametrize("panel_ratio", [1, 2, 4])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_direct_loads_preserve_exact_int8_k_order(block_shape, panel_ratio, transposed, signed):
    """Execute the kernel's offset/shape selection, including quarter order.

    No compiler or device claim: these tests establish exact byte selection
    for all supported panel ratios, full-range INT8, and both W layouts.
    """
    bm, bn, dot_k = block_shape
    panel_k = dot_k * panel_ratio
    rng = np.random.default_rng(7)
    low = -128 if signed else 0
    a = rng.integers(low, 128, (bm, panel_k), dtype=np.int8).view(Array)
    if transposed:
        b = rng.integers(low, 128, (bn, panel_k), dtype=np.int8).T.view(Array)
    else:
        b = rng.integers(low, 128, (panel_k, bn), dtype=np.int8).view(Array)
    namespace = load_kernel_namespace(V2Language(0, 0))
    load = namespace["_v2_load_operands"]
    rows = np.arange(bm).view(Array)
    cols = np.arange(bn).view(Array)
    # Pointer takes physical storage, not a flattened copy of a transposed view.
    w_storage = b.T if transposed else b
    a_parts, b_parts = [], []
    for micro in range(panel_ratio):
        k_start = micro * dot_k
        actual_a, actual_b = load(
            Pointer(a),
            Pointer(w_storage),
            rows,
            cols,
            0,
            k_start,
            bm,
            panel_k,
            bn,
            panel_k,
            0,
            1,
            0,
            *b.strides,
            dot_k,
        )
        np.testing.assert_array_equal(actual_a, a[:, k_start : k_start + dot_k])
        np.testing.assert_array_equal(actual_b, b[k_start : k_start + dot_k, :])
        assert actual_a.dtype == actual_b.dtype == np.int8
        a_parts.append(actual_a)
        b_parts.append(actual_b)
    np.testing.assert_array_equal(np.concatenate(a_parts, axis=1), a)
    np.testing.assert_array_equal(np.concatenate(b_parts, axis=0), b)
    wide_result = namespace["_v2_dot_full_panel"](a, b, Language.zeros((bm, bn), np.int32), dot_k, panel_k)
    np.testing.assert_array_equal(wide_result, a.astype(np.int64) @ b.astype(np.int64))


def test_dot_operands_are_direct_loads():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    load = functions["_v2_load_operands"]
    ret = next(node for node in load.body if isinstance(node, ast.Return))
    assignments = {ast.unparse(node.targets[0]): node.value for node in load.body if isinstance(node, ast.Assign)}
    assert all(ast.unparse(assignments[name.id].func) == "tl.load" for name in ret.value.elts)
    # No slice/gather in the direct-load code; the aligned helper can slice.
    for fn in (load, functions["_v2_accumulate_panel"]):
        assert not any(
            isinstance(node, ast.Call)
            and ast.unparse(node.func) in {"extract_slice", "tl.gather", "tl.split", "tl.permute"}
            for node in ast.walk(fn)
        )


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


@pytest.mark.parametrize("t,k,n", [(67, 4096, 65), (67, 32, 65), (64, 384, 96), (64, 257, 96), (64, 4096, 96)])
@pytest.mark.parametrize("prefetch", [False, True])
def test_exact_tile_ownership_and_load_width(t, k, n, prefetch):
    """Check both shape specializations, not just equality of final values."""
    bm, bn, dot_k, panel_k, groups, programs = 32, 32, 64, 128, 2, 3
    rng = np.random.default_rng(0)
    x = rng.integers(-8, 9, (t, groups, k), dtype=np.int8)
    w = rng.integers(-8, 9, (groups, k, n), dtype=np.int8)
    sx = np.ones((t, groups), dtype=np.float32)
    sw = np.ones((groups, n), dtype=np.float32)
    actual = np.full((t, groups, n), np.nan, dtype=np.float32)
    writes = np.zeros(actual.size, dtype=np.int32)
    widths = []

    class AuditLanguage(V2Language):
        def store(self, pointer, value, mask):
            offsets, valid = np.broadcast_arrays(pointer.offset, mask)
            np.add.at(writes, offsets[valid], 1)
            super().store(pointer, value, mask)

        def load(self, pointer, mask, other):
            if np.shares_memory(pointer.storage, w):
                widths.append(pointer.offset.shape[0])
            return super().load(pointer, mask, other)

    strides = tuple(s // v.itemsize for v in (x, w, sx, sw) for s in v.strides)
    for group in range(groups):
        for pid in reversed(range(programs)):
            load_kernels(AuditLanguage(pid, group))(
                *map(Pointer, (x, w, sx, sw, actual)),
                t,
                groups,
                k,
                n,
                *strides,
                bm,
                bn,
                dot_k,
                panel_k,
                programs,
                prefetch,
            )
    width = panel_k if t % bm == 0 and n % bn == 0 and k % panel_k == 0 else dot_k
    tiles = groups * Language.cdiv(t, bm) * Language.cdiv(n, bn)
    assert widths == [width] * (tiles * Language.cdiv(k, width))
    np.testing.assert_array_equal(writes, 1)
    golden = np.einsum("tgk,gkn->tgn", x.astype(np.int64), w.astype(np.int64))
    np.testing.assert_array_equal(actual, golden)


@pytest.mark.parametrize("inner", [1, 32, 64, 65, 128, 129, 192, 193, 256, 257, 513])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("panel_k,dot_k", [(64, 32), (128, 32), (64, 64), (128, 64), (256, 64)])
def test_panel_pairs_load_ahead_once_and_handle_single_odd_tails(inner, prefetch, panel_k, dot_k):
    """Audit source ordering/load count, not physical DMA or pipeline overlap."""
    x = np.ones((1, 1, inner), dtype=np.int8)
    w = np.ones((1, inner, 1), dtype=np.int8)
    sx = np.ones((1, 1), dtype=np.float32)
    sw = np.ones((1, 1), dtype=np.float32)
    actual = np.full((1, 1, 1), np.nan, dtype=np.float32)
    events = []
    load_widths = {"A": [], "B": []}

    class TraceLanguage(V2Language):
        def load(self, pointer, mask, other):
            if np.shares_memory(pointer.storage, x):
                events.append("load_A")
                load_widths["A"].append(pointer.offset.shape[1])
            elif np.shares_memory(pointer.storage, w):
                events.append("load_B")
                load_widths["B"].append(pointer.offset.shape[0])
            return super().load(pointer, mask, other)

        def dot(self, a, b, acc, out_dtype):
            events.append("dot")
            return super().dot(a, b, acc, out_dtype)

    strides = tuple(stride // value.itemsize for value in (x, w, sx, sw) for stride in value.strides)
    load_kernels(TraceLanguage(0, 0))(
        *map(Pointer, (x, w, sx, sw, actual)), 1, 1, inner, 1, *strides, 32, 32, dot_k, panel_k, 1, prefetch
    )
    dot_count = Language.cdiv(inner, dot_k)
    load_pair = ["load_A", "load_B"]
    if prefetch:
        expected_events = (load_pair * 2 + ["dot", "dot"]) * (dot_count // 2)
        if dot_count % 2:
            expected_events += load_pair + ["dot"]
    else:
        expected_events = (load_pair + ["dot"]) * dot_count
    assert events == expected_events
    expected_widths = [dot_k] * dot_count
    assert load_widths["A"] == load_widths["B"] == expected_widths
    assert events.count("dot") == Language.cdiv(inner, dot_k)
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
    # Nested helper calls isolate operand lifetimes. The dynamic K loop
    # returns only the accumulator, never a panel as an scf.for iter_arg.
    tile_loop = next(node for node in kernel.body if isinstance(node, ast.For))
    k_loop = next(node for node in ast.walk(tile_loop) if isinstance(node, ast.For) and node is not tile_loop)
    assert len(k_loop.body) == 1
    update = k_loop.body[0]
    assert isinstance(update, ast.Assign) and ast.unparse(update.targets[0]) == "acc"
    assert ast.unparse(update.value.func) == "_v2_accumulate_panel"


@pytest.mark.parametrize("failure", [None, "input-sync", "launch", "device-sync", "cpu-copy"])
def test_device_stage_diagnostic_reports_the_blocking_boundary(failure, capsys):
    path = ROOT / "tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul_v2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_plan_to_cpu_with_stages"
    )
    calls = []

    def record(stage):
        calls.append(stage)
        if stage == failure:
            raise RuntimeError(f"blocked at {stage}")

    def sync():
        record("device-sync" if "launch" in calls else "input-sync")

    def copy():
        record("cpu-copy")
        return "cpu-result"

    def plan():
        record("launch")
        return SimpleNamespace(cpu=copy)

    namespace = {"torch": SimpleNamespace(npu=SimpleNamespace(synchronize=sync))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    diagnostic = namespace[function.name]
    if failure is not None:
        with pytest.raises(RuntimeError, match=failure):
            diagnostic(plan)
        assert capsys.readouterr().out.rstrip().endswith(f"{failure}.begin")
        assert calls[-1] == failure
    else:
        assert diagnostic(plan) == "cpu-result"
        assert calls == ["input-sync", "launch", "device-sync", "cpu-copy"]
        assert capsys.readouterr().out.rstrip().endswith("cpu-copy.done")


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

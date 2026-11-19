# SPDX-License-Identifier: Apache-2.0
"""CPU audit of the actual kernel body, NOT a Triton compiler/NPU test.

Run with pytest --noconftest on this file; no torch/vLLM import is needed.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

KERNEL_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/transpose_quant_batch_matmul.py"


class Array(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Array)


class Pointer:
    def __init__(self, storage, offset=0):
        self.storage = storage.reshape(-1)
        self.offset = offset
        self.dtype = SimpleNamespace(element_ty=storage.dtype)

    def __add__(self, offset):
        return Pointer(self.storage, self.offset + offset)


class Language:
    int64 = np.int64
    int32 = np.int32
    float32 = np.float32

    def __init__(self, pid, group):
        self.pid = pid
        self.group = group

    def program_id(self, axis):
        return np.asarray(self.pid if axis == 0 else self.group).view(Array)

    @staticmethod
    def cdiv(x, y):
        return (x + y - 1) // y

    @staticmethod
    def minimum(x, y):
        return min(x, y)

    @staticmethod
    def arange(start, end):
        return np.arange(start, end).view(Array)

    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype).view(Array)

    @staticmethod
    def trans(value):
        return value.T.view(Array)

    @staticmethod
    def load(pointer, mask, other):
        offsets, valid = np.broadcast_arrays(pointer.offset, mask)
        result = np.full(offsets.shape, other, dtype=pointer.storage.dtype)
        assert np.all((offsets[valid] >= 0) & (offsets[valid] < pointer.storage.size))
        result[valid] = pointer.storage[offsets[valid]]
        return result.view(Array)

    @staticmethod
    def store(pointer, value, mask):
        offsets, values, valid = np.broadcast_arrays(pointer.offset, value, mask)
        assert np.all((offsets[valid] >= 0) & (offsets[valid] < pointer.storage.size))
        pointer.storage[offsets[valid]] = values[valid]

    @staticmethod
    def dot(a, b, accumulator, out_dtype):
        assert a.dtype == b.dtype == np.int8
        assert accumulator.dtype == out_dtype == np.int32
        return (accumulator + a.astype(np.int32) @ b.astype(np.int32)).view(Array)


def load_function(name, namespace):
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    function.returns = None
    for argument in (*function.args.args, *function.args.kwonlyargs):
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(KERNEL_PATH), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 1), (17, 2, 65, 33), (65, 3, 129, 131), (513, 2, 33, 129), (16, 1, 4096, 32)]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "block_shape", [(16, 32, 32), (32, 32, 32), (64, 32, 32), (16, 32, 64), (32, 32, 64), (16, 64, 32), (32, 64, 64)]
)
@pytest.mark.parametrize("transpose_weight_tile", [False, True])
@pytest.mark.parametrize("reuse_m", [1, 2])
@pytest.mark.parametrize("reuse_n", [1, 2])
def test_actual_kernel_addressing_and_int8_arithmetic(
    shape, strided, block_shape, transpose_weight_tile, reuse_m, reuse_n
):
    tokens, groups, inner, columns = shape
    rng = np.random.default_rng(0)
    if strided:
        x_storage = rng.integers(-128, 128, (groups, tokens + 1, inner * 2 + 2), dtype=np.int8)
        x = x_storage[:, 1:, 1 : 1 + inner * 2 : 2].transpose(1, 0, 2)
        w_storage = rng.integers(-128, 128, (groups, columns, inner + 2), dtype=np.int8)
        weight = w_storage[:, :, 1:-1].transpose(0, 2, 1)
        sx_storage = rng.random((groups, tokens * 2 + 2), dtype=np.float32)
        sx = sx_storage[:, 1 : 1 + tokens * 2 : 2].T
        sw_storage = rng.random((columns + 1, groups + 1), dtype=np.float32)
        sw = sw_storage[1:, 1:].T
    else:
        x_storage = rng.integers(-128, 128, (tokens, groups, inner), dtype=np.int8)
        w_storage = rng.integers(-128, 128, (groups, inner, columns), dtype=np.int8)
        sx_storage = rng.random((tokens, groups), dtype=np.float32)
        sw_storage = rng.random((groups, columns), dtype=np.float32)
        x, weight, sx, sw = x_storage, w_storage, sx_storage, sw_storage
    actual = np.full((tokens, groups, columns), np.nan, dtype=np.float32)
    strides = tuple(step // value.itemsize for value in (x, weight, sx, sw) for step in value.strides)
    block_m, block_n, block_k = block_shape
    num_tiles = Language.cdiv(tokens, block_m * reuse_m) * Language.cdiv(columns, block_n * reuse_n)
    for group in range(groups):
        for pid in range(num_tiles):
            kernel = load_function("_transpose_quant_batch_matmul_kernel", {"tl": Language(pid, group)})
            kernel(
                Pointer(x_storage, (x.ctypes.data - x_storage.ctypes.data) // x.itemsize),
                Pointer(w_storage, (weight.ctypes.data - w_storage.ctypes.data) // weight.itemsize),
                Pointer(sx_storage, (sx.ctypes.data - sx_storage.ctypes.data) // sx.itemsize),
                Pointer(sw_storage, (sw.ctypes.data - sw_storage.ctypes.data) // sw.itemsize),
                Pointer(actual),
                tokens,
                groups,
                inner,
                columns,
                *strides,
                block_m,
                block_n,
                block_k,
                4,
                transpose_weight_tile,
                reuse_m,
                reuse_n,
            )
    integer_golden = np.einsum("tgk,gkn->tgn", x.astype(np.int64), weight.astype(np.int64))
    expected = (integer_golden.astype(np.float32) * sw[None]) * sx[:, :, None]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("start,end", [(0, 32), (32, 64), (2048, 2080), (4064, 4096), (0, 64), (0, 65), (0, 4096)])
@pytest.mark.parametrize("transpose_weight_tile", [False, True])
@pytest.mark.parametrize("reuse_m", [1, 2])
@pytest.mark.parametrize("reuse_n", [1, 2])
def test_fixed_physical_weight_pitch_with_logical_k_slices(start, end, transpose_weight_tile, reuse_m, reuse_n):
    """Physical column stride stays 4096 even when logical K is 32 or 65."""
    tokens, groups, physical_k, columns = 17, 2, 4096, 33
    rng = np.random.default_rng(1)
    x_storage = rng.integers(-128, 128, (tokens, groups, physical_k), dtype=np.int8)
    w_storage = rng.integers(-128, 128, (groups, columns, physical_k), dtype=np.int8)
    x = x_storage[:, :, start:end]
    weight = w_storage[:, :, start:end].transpose(0, 2, 1)
    sx = rng.random((tokens, groups), dtype=np.float32)
    sw = rng.random((groups, columns), dtype=np.float32)
    actual = np.full((tokens, groups, columns), np.nan, dtype=np.float32)
    strides = tuple(step // value.itemsize for value in (x, weight, sx, sw) for step in value.strides)
    block_m, block_n, block_k = 16, 32, 32
    num_tiles = Language.cdiv(tokens, block_m * reuse_m) * Language.cdiv(columns, block_n * reuse_n)
    for group in range(groups):
        for pid in range(num_tiles):
            kernel = load_function("_transpose_quant_batch_matmul_kernel", {"tl": Language(pid, group)})
            kernel(
                Pointer(x_storage, start),
                Pointer(w_storage, start),
                Pointer(sx),
                Pointer(sw),
                Pointer(actual),
                tokens,
                groups,
                end - start,
                columns,
                *strides,
                block_m,
                block_n,
                block_k,
                4,
                transpose_weight_tile,
                reuse_m,
                reuse_n,
            )
    golden = np.einsum("tgk,gkn->tgn", x.astype(np.int64), weight.astype(np.int64))
    expected = (golden.astype(np.float32) * sw[None]) * sx[:, :, None]
    np.testing.assert_array_equal(actual, expected)


def test_paired_m_shares_b_loads_without_reducing_dot_count():
    """SSA load reuse, not a claim about compiler L1 placement or DRAM bytes."""
    rng = np.random.default_rng(42)
    x = rng.integers(-128, 128, (64, 1, 65), dtype=np.int8)
    w_storage = rng.integers(-128, 128, (1, 32, 65), dtype=np.int8)
    weight = w_storage.transpose(0, 2, 1)
    sx = np.ones((64, 1), dtype=np.float32)
    sw = np.ones((1, 32), dtype=np.float32)
    strides = tuple(s // a.itemsize for a in (x, weight, sx, sw) for s in a.strides)
    counts, outputs = {}, {}

    class CountLanguage(Language):
        def load(self, pointer, mask, other):
            if np.shares_memory(pointer.storage, w_storage):
                counts["b_loads"] += 1
            return super().load(pointer, mask, other)

        def dot(self, a, b, accumulator, out_dtype):
            counts["dots"] += 1
            return super().dot(a, b, accumulator, out_dtype)

    for reuse_m in (1, 2):
        counts.update(b_loads=0, dots=0)
        output = np.full((64, 1, 32), np.nan, dtype=np.float32)
        for pid in range(Language.cdiv(64, 32 * reuse_m)):
            kernel = load_function("_transpose_quant_batch_matmul_kernel", {"tl": CountLanguage(pid, 0)})
            kernel(
                Pointer(x),
                Pointer(w_storage),
                Pointer(sx),
                Pointer(sw),
                Pointer(output),
                64,
                1,
                65,
                32,
                *strides,
                32,
                32,
                32,
                4,
                False,
                reuse_m,
            )
        outputs[reuse_m] = output, dict(counts)
    np.testing.assert_array_equal(outputs[1][0], outputs[2][0])
    assert outputs[2][1]["b_loads"] * 2 == outputs[1][1]["b_loads"]
    assert outputs[2][1]["dots"] == outputs[1][1]["dots"] == 6


@pytest.mark.parametrize("block_shape", [(128, 128, 64), (64, 128, 64)])
@pytest.mark.parametrize("reuse_m", [1, 2])
@pytest.mark.parametrize("transposed", [False, True])
def test_wider_n_and_reuse_cover_full_regions_and_masked_tails(block_shape, reuse_m, transposed):
    test_actual_kernel_addressing_and_int8_arithmetic((257, 2, 129, 257), True, block_shape, transposed, reuse_m, 2)


def test_two_axis_reuse_reduces_a_and_b_loads_but_preserves_dots():
    """Count source SSA loads, not actual HBM traffic or compiler buffers."""
    rng = np.random.default_rng(3)
    x = rng.integers(-128, 128, (64, 1, 65), dtype=np.int8)
    w_storage = rng.integers(-128, 128, (1, 64, 65), dtype=np.int8)
    weight = w_storage.transpose(0, 2, 1)
    sx, sw = np.ones((64, 1), dtype=np.float32), np.ones((1, 64), dtype=np.float32)
    strides = tuple(s // a.itemsize for a in (x, weight, sx, sw) for s in a.strides)
    counter = {}

    class CountLanguage(Language):
        def load(self, pointer, mask, other):
            if np.shares_memory(pointer.storage, x):
                counter["a"] += 1
            if np.shares_memory(pointer.storage, w_storage):
                counter["b"] += 1
            return super().load(pointer, mask, other)

        def dot(self, a, b, accumulator, out_dtype):
            counter["dots"] += 1
            return super().dot(a, b, accumulator, out_dtype)

    all_counts = {}
    for reuse_m in (1, 2):
        for reuse_n in (1, 2):
            counter.update(a=0, b=0, dots=0)
            output = np.full((64, 1, 64), np.nan, dtype=np.float32)
            for pid in range(Language.cdiv(64, 32 * reuse_m) * Language.cdiv(64, 32 * reuse_n)):
                kernel = load_function("_transpose_quant_batch_matmul_kernel", {"tl": CountLanguage(pid, 0)})
                kernel(
                    Pointer(x),
                    Pointer(w_storage),
                    Pointer(sx),
                    Pointer(sw),
                    Pointer(output),
                    64,
                    1,
                    65,
                    64,
                    *strides,
                    32,
                    32,
                    32,
                    4,
                    False,
                    reuse_m,
                    reuse_n,
                )
            golden = np.einsum("tgk,gkn->tgn", x.astype(np.int64), weight.astype(np.int64)).astype(np.float32)
            np.testing.assert_array_equal(output, golden)
            all_counts[reuse_m, reuse_n] = dict(counter)
    assert all_counts[1, 1] == {"a": 12, "b": 12, "dots": 12}
    assert all_counts[1, 2] == {"a": 6, "b": 12, "dots": 12}
    assert all_counts[2, 1] == {"a": 12, "b": 6, "dots": 12}
    assert all_counts[2, 2] == {"a": 6, "b": 6, "dots": 12}


@pytest.mark.parametrize("num_stages", [1, 2, 0, -1, 3, "1", None, True, 1.0])
def test_num_stages_validation_without_runtime_imports(num_stages):
    validate = load_function("_validate_num_stages", {})
    if type(num_stages) is int and num_stages in (1, 2):
        validate(num_stages)
    else:
        with pytest.raises(ValueError, match="num_stages"):
            validate(num_stages)


@pytest.mark.parametrize("num_stages", [None, 1, 2])
@pytest.mark.parametrize("transpose_weight_tile", [None, False, True])
def test_launch_uses_small_tiles_and_forwards_num_stages(num_stages, transpose_weight_tile):
    """Check host launch wiring only, not device/compiler correctness."""
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs))

            return launch

    device = SimpleNamespace(type="npu")
    x = SimpleNamespace(shape=(16, 1, 4096), device=device, stride=lambda: (4096, 4096, 1))
    weight = SimpleNamespace(stride=lambda: (131072, 1, 4096), shape=(1, 4096, 32))
    sx = SimpleNamespace(stride=lambda: (1, 1))
    sw = SimpleNamespace(stride=lambda: (32, 1))
    output = object()
    namespace = {
        "torch": SimpleNamespace(bfloat16="bf16", empty=lambda *args, **kwargs: output),
        "torch_npu": SimpleNamespace(get_npu_format=lambda tensor: 2),
        "triton": SimpleNamespace(cdiv=Language.cdiv),
        "_validate_inputs": lambda *args: None,
        "_validate_num_stages": load_function("_validate_num_stages", {}),
        "_transpose_quant_batch_matmul_kernel": Kernel(),
        "DEFAULT_BLOCK_N": 32,
        "DEFAULT_BLOCK_K": 32,
        "DEFAULT_GROUP_SIZE_M": 4,
    }
    load_function("_resolve_launch_options", namespace)
    operator = load_function("transpose_quant_batch_matmul", namespace)
    kwargs = {} if num_stages is None else {"num_stages": num_stages}
    if transpose_weight_tile is not None:
        kwargs["transpose_weight_tile"] = transpose_weight_tile
    assert operator(x, weight, sx, sw, **kwargs) is output
    assert len(launches) == 1
    grid, launch = launches[0]
    assert grid == (1, 1)
    assert (launch["BLOCK_M"], launch["BLOCK_N"], launch["BLOCK_K"]) == (16, 32, 32)
    assert launch["num_stages"] == (1 if num_stages is None else num_stages)
    assert launch["TRANSPOSE_WEIGHT_TILE"] == (False if transpose_weight_tile is None else transpose_weight_tile)


@pytest.mark.parametrize(
    "case,error",
    [
        ("valid", None),
        ("empty_tokens", None),
        ("floating_x", TypeError),
        ("floating_w", TypeError),
        ("rank", ValueError),
        ("zero_groups", ValueError),
        ("zero_k", ValueError),
        ("zero_n", ValueError),
        ("mismatched_gk", ValueError),
        ("overflow", ValueError),
        ("scale_shape", ValueError),
        ("scale_dtype", TypeError),
        ("output_dtype", TypeError),
        ("device", ValueError),
    ],
)
def test_host_validation_without_runtime_imports(case, error):
    torch_metadata = SimpleNamespace(int8="int8", float32="fp32", float16="fp16", bfloat16="bf16")
    validate = load_function("_validate_inputs", {"torch": torch_metadata, "MAX_INT8_ACCUMULATION_K": 131071})

    def tensor(shape, dtype):
        return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device="npu:0")

    x = tensor((17, 2, 65), "int8")
    w = tensor((2, 65, 33), "int8")
    sx = tensor((17, 2), "fp32")
    sw = tensor((2, 33), "fp32")
    output_dtype = "bf16"
    if case == "empty_tokens":
        x.shape, sx.shape = (0, 2, 65), (0, 2)
    elif case == "floating_x":
        x.dtype = "bf16"
    elif case == "floating_w":
        w.dtype = "bf16"
    elif case == "rank":
        x.ndim = 2
    elif case == "zero_groups":
        x.shape = (17, 0, 65)
    elif case == "zero_k":
        x.shape = (17, 2, 0)
    elif case == "zero_n":
        w.shape = (2, 65, 0)
    elif case == "mismatched_gk":
        w.shape = (3, 65, 33)
    elif case == "overflow":
        x.shape, w.shape = (17, 2, 131072), (2, 131072, 33)
    elif case == "scale_shape":
        sx.shape = (17,)
    elif case == "scale_dtype":
        sw.dtype = "bf16"
    elif case == "output_dtype":
        output_dtype = "int8"
    elif case == "device":
        w.device = "npu:1"
    if error is None:
        validate(x, w, sx, sw, output_dtype)
    else:
        with pytest.raises(error):
            validate(x, w, sx, sw, output_dtype)

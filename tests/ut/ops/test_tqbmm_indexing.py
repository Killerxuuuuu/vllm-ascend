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
    for argument in function.args.args:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(KERNEL_PATH), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (17, 2, 65, 33), (65, 3, 129, 131), (513, 2, 33, 129)])
@pytest.mark.parametrize("strided", [False, True])
def test_actual_kernel_addressing_and_int8_arithmetic(shape, strided):
    tokens, groups, inner, columns = shape
    rng = np.random.default_rng(0)
    if strided:
        x_storage = rng.integers(-128, 128, (groups, tokens, inner * 2), dtype=np.int8)
        x = x_storage.transpose(1, 0, 2)[:, :, ::2]
        w_storage = rng.integers(-128, 128, (groups, columns, inner), dtype=np.int8)
        weight = w_storage.transpose(0, 2, 1)
        sx_storage = rng.random((groups, tokens * 2), dtype=np.float32)
        sx = sx_storage[:, ::2].T
        sw_storage = rng.random((columns, groups), dtype=np.float32)
        sw = sw_storage.T
    else:
        x_storage = rng.integers(-128, 128, (tokens, groups, inner), dtype=np.int8)
        w_storage = rng.integers(-128, 128, (groups, inner, columns), dtype=np.int8)
        sx_storage = rng.random((tokens, groups), dtype=np.float32)
        sw_storage = rng.random((groups, columns), dtype=np.float32)
        x, weight, sx, sw = x_storage, w_storage, sx_storage, sw_storage
    actual = np.full((tokens, groups, columns), np.nan, dtype=np.float32)
    strides = tuple(step // value.itemsize for value in (x, weight, sx, sw) for step in value.strides)
    block_m, block_n, block_k = 32, 64, 64
    num_tiles = Language.cdiv(tokens, block_m) * Language.cdiv(columns, block_n)
    for group in range(groups):
        for pid in range(num_tiles):
            kernel = load_function("_transpose_quant_batch_matmul_kernel", {"tl": Language(pid, group)})
            kernel(
                Pointer(x_storage),
                Pointer(w_storage),
                Pointer(sx_storage),
                Pointer(sw_storage),
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
            )
    integer_golden = np.einsum("tgk,gkn->tgn", x.astype(np.int64), weight.astype(np.int64))
    expected = (integer_golden.astype(np.float32) * sw[None]) * sx[:, :, None]
    np.testing.assert_array_equal(actual, expected)


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

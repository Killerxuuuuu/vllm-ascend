# SPDX-License-Identifier: Apache-2.0
"""Execute actual host code against metadata stubs, without torch/NPU imports.

These cover API contracts and wiring only, not NPU numerical correctness.
"""

import ast
import math
import sys
from dataclasses import FrozenInstanceError, dataclass
from itertools import count
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/transpose_quant_batch_matmul.py"


@dataclass(frozen=True)
class Device:
    type: str = "npu"
    index: int = 0


class Storage:
    def __init__(self, address, size):
        self.address, self.size, self.value = address, size, 0

    def data_ptr(self):
        return self.address

    def nbytes(self):
        return self.size


class Tensor:
    def __init__(self, shape, dtype, storage, *, device=None, strides=None, contiguous=True, fmt=2, grad=False):
        self.shape, self.dtype, self._storage = tuple(shape), dtype, storage
        self.device, self.requires_grad, self.fmt = device if device is not None else Device(), grad, fmt
        self._contiguous = contiguous
        self._strides = (
            tuple(strides) if strides is not None else tuple(math.prod(shape[axis + 1 :]) for axis in range(len(shape)))
        )

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self):
        return self._strides

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return math.prod(self.shape)

    def untyped_storage(self):
        return self._storage

    def detach(self):
        return Tensor(
            self.shape,
            self.dtype,
            self._storage,
            device=self.device,
            strides=self._strides,
            contiguous=self._contiguous,
            fmt=self.fmt,
        )


class Backend:
    def __init__(self):
        self.addresses = count(1 << 40, 1 << 40)
        self.allocations, self.queries, self.launches = [], [], []

    def tensor(self, shape, dtype, **kwargs):
        tensor = Tensor(shape, dtype, Storage(next(self.addresses), math.prod(shape) * 4), **kwargs)
        self.allocations.append(tensor)
        return tensor

    def empty(self, shape, *, dtype, device):
        return self.tensor(shape, dtype, device=device)

    def get_format(self, tensor):
        self.queries.append(tensor)
        return tensor.fmt

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


@pytest.fixture
def host(monkeypatch):
    backend = Backend()
    module = ModuleType("tqbmm_host_under_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = module.__dict__
    namespace.update(
        dataclass=dataclass,
        torch=SimpleNamespace(
            Tensor=Tensor,
            dtype=str,
            int8="i8",
            float32="f32",
            bfloat16="bf16",
            float16="f16",
            empty=backend.empty,
        ),
        torch_npu=SimpleNamespace(get_npu_format=backend.get_format),
        triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        _transpose_quant_batch_matmul_kernel=backend,
    )
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.ClassDef))
        or (isinstance(node, ast.FunctionDef) and node.name != "_transpose_quant_batch_matmul_kernel")
    ]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return module, backend


def inputs(backend, tokens=17):
    return (
        backend.tensor((tokens, 2, 65), "i8"),
        backend.tensor((2, 65, 33), "i8", strides=(2145, 1, 65), contiguous=False),
        backend.tensor((tokens, 2), "f32"),
        backend.tensor((2, 33), "f32"),
    )


def test_safe_out_returns_supplied_tensor_and_keeps_validation(host):
    module, backend = host
    tensors = inputs(backend)
    out = backend.tensor((17, 2, 33), "bf16")
    allocations = len(backend.allocations)
    assert module.transpose_quant_batch_matmul(*tensors, out=out) is out
    assert len(backend.allocations) == allocations
    assert len(backend.queries) == 5
    assert backend.launches[0][1][4] is out
    tensors[0].dtype = "bf16"
    with pytest.raises(TypeError, match="INT8"):
        module.transpose_quant_batch_matmul(*tensors, out=out)
    assert len(backend.launches) == 1


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("bad", ["shape", "dtype", "device", "strided", "nz", "grad", "alias", "partial_alias"])
def test_rejects_invalid_or_aliasing_output_before_launch(host, prepared, bad):
    module, backend = host
    tensors = inputs(backend)
    out = backend.tensor((17, 2, 33), "bf16")
    if bad == "shape":
        out.shape = (17, 33, 2)
    elif bad == "dtype":
        out.dtype = "f32"
    elif bad == "device":
        out.device = Device(index=1)
    elif bad == "strided":
        out._contiguous = False
    elif bad == "nz":
        out.fmt = 29
    elif bad == "grad":
        out.requires_grad = True
    elif bad == "alias":
        out._storage = tensors[1]._storage
    elif bad == "partial_alias":
        source = tensors[1].untyped_storage()
        out._storage = Storage(source.data_ptr() + 8, source.nbytes())
    function = module.PreparedTQBMM if prepared else module.transpose_quant_batch_matmul
    with pytest.raises(ValueError):
        function(*tensors, out=out)
    assert not backend.launches


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_prepared_rejects_nz_input(host, position):
    module, backend = host
    tensors = inputs(backend)
    tensors[position].fmt = 29
    with pytest.raises(ValueError, match="ND storage"):
        module.PreparedTQBMM(*tensors)
    assert not backend.launches


def test_prepared_caches_geometry_but_keeps_shared_values_alive(host):
    module, backend = host
    tensors = inputs(backend)
    block = [64, 64, 128]
    prepared = module.PreparedTQBMM(*tensors, block_shape=block, num_stages=2)
    assert not backend.launches
    allocations, queries = len(backend.allocations), len(backend.queries)
    block[0] = 16
    tensors[0]._storage.value = 42
    for _ in range(3):
        assert prepared() is prepared.output
    assert len(backend.allocations) == allocations
    assert len(backend.queries) == queries
    grid, arguments, kwargs = backend.launches[-1]
    assert grid == (1, 2)
    assert arguments[5:] == (17, 2, 65, 33, 130, 65, 1, 2145, 1, 65, 2, 1, 33, 1)
    assert arguments[0] is not tensors[0]
    assert arguments[0]._storage is tensors[0]._storage
    assert arguments[0]._storage.value == 42
    assert kwargs == {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 128,
        "GROUP_SIZE_M": 4,
        "TRANSPOSE_WEIGHT_TILE": False,
        "num_stages": 2,
    }
    with pytest.raises(FrozenInstanceError):
        prepared._num_stages = 1
    # Replacing caller tensor metadata must not change the detached binding.
    tensors[0].shape = (1, 1, 1)
    prepared()
    assert backend.launches[-1][1][0].shape == (17, 2, 65)


def test_prepared_output_aliases_explicit_buffer_without_copy(host):
    module, backend = host
    tensors = inputs(backend)
    out = backend.tensor((17, 2, 33), "bf16")
    count_before = len(backend.allocations)
    prepared = module.PreparedTQBMM(*tensors, out=out)
    assert prepared.output is not out
    assert prepared.output._storage is out._storage
    assert len(backend.allocations) == count_before


@pytest.mark.parametrize("prepared", [False, True])
def test_zero_tokens_do_not_launch(host, prepared):
    module, backend = host
    tensors = inputs(backend, tokens=0)
    out = backend.tensor((0, 2, 33), "bf16")
    actual = (
        module.PreparedTQBMM(*tensors, out=out)()
        if prepared
        else module.transpose_quant_batch_matmul(*tensors, out=out)
    )
    assert actual.shape == (0, 2, 33)
    assert not backend.launches


@pytest.mark.parametrize(
    "options",
    [
        {"num_stages": True},
        {"num_stages": 3},
        {"transpose_weight_tile": 1},
        {"block_shape": (24, 32, 64)},
        {"block_shape": (16, 32, 16)},
        {"output_dtype": "i8"},
    ],
)
def test_prepared_checks_options_at_construction(host, options):
    module, backend = host
    with pytest.raises((ValueError, TypeError)):
        module.PreparedTQBMM(*inputs(backend), **options)
    assert not backend.launches

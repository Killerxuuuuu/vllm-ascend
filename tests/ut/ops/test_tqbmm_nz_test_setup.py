# SPDX-License-Identifier: Apache-2.0
"""Check the NPU regression test's setup/cleanup without importing torch."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "e2e/pull_request/one_card/test_transpose_quant_batch_matmul.py"


class WriteOnlyConfig:
    def __init__(self, option):
        self.option = option
        self.updates = []

    @property
    def allow_internal_format(self):
        raise AssertionError("The torch_npu 2.10 config attribute is write-only")

    @allow_internal_format.setter
    def allow_internal_format(self, enabled):
        self.updates.append(enabled)
        self.option = b"enable" if enabled else b"disable"


def load_test(initial_option, outcome):
    config = WriteOnlyConfig(initial_option)
    launches = []

    def get_option(name):
        assert name == "ALLOW_INTERNAL_FORMAT"
        return config.option

    def format_cast(tensor, fmt):
        assert config.option == b"enable"
        assert fmt == 29
        if outcome == "cast_error":
            raise RuntimeError("format cast failed")
        return SimpleNamespace(fmt=2 if outcome == "nd_fallback" else fmt)

    def matmul(*args):
        launches.append(args)
        if outcome == "wrong_error":
            raise ValueError("unexpected error")
        if outcome != "missing_rejection":
            raise ValueError("TQBMM requires ND storage; NZ is not supported")

    namespace = {
        "pytest": pytest,
        "torch": SimpleNamespace(ones=lambda *args, **kwargs: object(), int8="int8"),
        "torch_npu": SimpleNamespace(
            _C=SimpleNamespace(_npu_getOption=get_option),
            npu=SimpleNamespace(config=config),
            npu_format_cast=format_cast,
            get_npu_format=lambda tensor: tensor.fmt,
        ),
        "transpose_quant_batch_matmul": matmul,
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "test_nz_storage_is_rejected"
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name], config, launches


@pytest.mark.parametrize("initial_option", [b"enable", b"disable"])
@pytest.mark.parametrize("outcome", ["rejected", "cast_error", "nd_fallback", "missing_rejection", "wrong_error"])
def test_nz_setup_restores_actual_option(initial_option, outcome):
    run, config, launches = load_test(initial_option, outcome)
    if outcome == "rejected":
        run()
    elif outcome == "cast_error":
        with pytest.raises(RuntimeError, match="format cast failed"):
            run()
    elif outcome == "nd_fallback":
        with pytest.raises(AssertionError, match="Expected FRACTAL_NZ storage"):
            run()
    elif outcome == "wrong_error":
        with pytest.raises(AssertionError):
            run()
    else:
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            run()
    assert config.option == initial_option
    assert config.updates == [True, initial_option == b"enable"]
    assert len(launches) == (0 if outcome in ("cast_error", "nd_fallback") else 1)


def test_unknown_initial_option_fails_before_changing_config():
    run, config, launches = load_test(None, "rejected")
    with pytest.raises(AssertionError, match="Cannot snapshot internal format option"):
        run()
    assert config.option is None
    assert config.updates == []
    assert launches == []

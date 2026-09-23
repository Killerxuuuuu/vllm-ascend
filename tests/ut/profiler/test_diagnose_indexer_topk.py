# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks/scripts/diagnose_indexer_topk.py"
SPEC = importlib.util.spec_from_file_location("topk_diagnosis_under_test", SCRIPT)
diagnosis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnosis
SPEC.loader.exec_module(diagnosis)


class TopkDiagnosisTest(unittest.TestCase):
    def test_cases_are_valid_and_include_production_shape(self):
        cases = diagnosis.make_cases("all")
        self.assertEqual(len(cases), 13)
        self.assertEqual(len({c.name for c in cases}), len(cases))
        self.assertEqual(cases[0], diagnosis.Case("baseline", 192, 512, 1024, 192))
        for case in cases:
            diagnosis.validate_case(case)

    def test_width_sweep_preserves_input_and_k(self):
        cases = [c for c in diagnosis.make_cases("all") if c.name.startswith("width_")]
        self.assertEqual({(c.n, c.k, c.valid) for c in cases}, {(192, 512, 192)})
        self.assertEqual({c.width for c in cases}, {256, 512})

    def test_baseline_only(self):
        self.assertEqual(len(diagnosis.make_cases("baseline")), 1)

    def test_invalid_shapes(self):
        for kwargs in (
            {"n": 2048},
            {"k": 0},
            {"k": 513},
            {"width": 300},
            {"valid": -1},
            {"valid": 193},
            {"kind": "wrong"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                case = {"name": "bad", "n": 192, "k": 512, "width": 1024, "valid": 192}
                diagnosis.validate_case(diagnosis.Case(**(case | kwargs)))

    def test_percentile(self):
        self.assertEqual(diagnosis.percentile([30, 10, 20], 0.5), 20)
        self.assertEqual(diagnosis.percentile([10, 20], 0.1), 11)

    def write_csv(self, root, name, headers, rows):
        path = root / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        return path

    def test_task_csv_does_not_double_count_op_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self.write_csv(
                root, "kernel_details.csv", ["Name", "Duration(us)"], [["target", 4], ["other", 99], ["target", 6]]
            )
            self.write_csv(root, "op_summary_1.csv", ["Op Name", "Task Duration(us)"], [["target", 4], ["target", 6]])
            source, durations = diagnosis.extract_durations(root, "target")
            self.assertEqual(source, expected)
            self.assertEqual(durations, [4.0, 6.0])

    def test_op_summary_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_csv(root, "op_summary_1.csv", ["Op Name", "Task Duration(us)"], [["target", 4]])
            self.assertEqual(diagnosis.extract_durations(root, "target")[1], [4.0])

    def test_no_kernel_or_invalid_duration_fails(self):
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.write_csv(root, "kernel_details.csv", ["Name", "Duration(us)"], [["target", value]])
                with self.assertRaises(RuntimeError):
                    diagnosis.extract_durations(root, "target")
                with self.assertRaises(RuntimeError):
                    diagnosis.extract_durations(root, "missing")

    def test_missing_files_fail(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(RuntimeError):
            diagnosis.extract_durations(Path(tmp), "target")

    def test_metric_selection_and_no_silent_fallback(self):
        profiler = SimpleNamespace(
            ProfilerLevel=SimpleNamespace(Level0=0, Level1=1),
            AiCMetrics=SimpleNamespace(AiCoreNone=0, PipeUtilization=1),
            ExportType=SimpleNamespace(Text="text"),
            _ExperimentalConfig=Mock(),
        )
        diagnosis.profiler_config(profiler, "PipeUtilization")
        profiler._ExperimentalConfig.assert_called_with(profiler_level=1, aic_metrics=1, export_type="text")
        diagnosis.profiler_config(profiler, "none")
        profiler._ExperimentalConfig.assert_called_with(profiler_level=0, aic_metrics=0, export_type="text")
        with self.assertRaises(RuntimeError):
            diagnosis.profiler_config(profiler, "MemoryUB")


if __name__ == "__main__":
    unittest.main()

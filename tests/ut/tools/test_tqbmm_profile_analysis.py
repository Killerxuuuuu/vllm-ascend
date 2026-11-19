# SPDX-License-Identifier: Apache-2.0
"""CPU-only profile reduction tests; run with --noconftest."""

import csv
import json

import pytest

import analyze_tqbmm_profile as analysis


def write_csv(path, rows, headers=None):
    headers = headers or ["Name", "Duration(us)", "Input Shapes", "aic_mac_ratio"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def test_csv_weighted_aggregation_and_counter_coverage(tmp_path):
    source = write_csv(
        tmp_path / "kernel_details.csv",
        [["QuantBatchMatMul", " 10\t", "1,64,32", ".2"], ["QuantBatchMatMul", "30", "1,64,32", "N/A"]],
    )
    aggregate, upstream = analysis.read_source(source)
    report = analysis.make_report(aggregate, source, 1, upstream, 6, "composition-c1")
    entry = report["top_operators_by_sum"][0]
    assert entry["calls"] == 2
    assert entry["sum_us"] == 40
    assert entry["mean_us"] == 20
    assert entry["counter_call_means"]["aic_mac_ratio"] == {"mean": 0.2, "covered_calls": 1}
    assert entry["family"] == "native_quant_matmul_candidate"
    assert "Wo_a" not in entry["family"]


def test_header_aliases_bom_and_zero_duration(tmp_path):
    source = write_csv(
        tmp_path / "summary.csv",
        [["Transpose", "0", "AIV", "", "0"]],
        [" Op Name ", "Task Duration(us)", "Task Type", "Input Shapes", "Device_id"],
    )
    aggregate, _ = analysis.read_source(source)
    assert aggregate.families["layout"].count == 1
    assert aggregate.families["layout"].total_us == 0
    assert aggregate.missing_shapes == 1
    assert aggregate.devices == {"0": 1}


def test_pipeline_times_above_one_and_missing_coverage(tmp_path):
    headers = ["Name", "Duration(us)", "aic_mte2_time(us)", "aic_mac_time(us)", "aic_scalar_time(us)"]
    source = write_csv(
        tmp_path / "kernel_details.csv",
        [["QuantBatchMatMul", 100, 90, 20, "N/A"], ["QuantBatchMatMul", 200, 180, "N/A", "N/A"]],
        headers,
    )
    aggregate, _ = analysis.read_source(source)
    counters = next(iter(aggregate.names.values())).report(300)["counter_call_means"]
    assert counters["aic_mte2_time(us)"] == {"mean": 135, "covered_calls": 2}
    assert counters["aic_mac_time(us)"] == {"mean": 20, "covered_calls": 1}
    assert "aic_scalar_time(us)" not in counters


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "N/A", "", "junk"])
def test_invalid_rows_are_counted(tmp_path, bad):
    source = write_csv(tmp_path / "kernel_details.csv", [["a", bad, "", ""], ["b", 5, "", ""]])
    aggregate, _ = analysis.read_source(source)
    assert aggregate.invalid == 1
    assert aggregate.records == 2
    assert sum(stats.count for stats in aggregate.names.values()) == 1


def test_earlier_json_weighted_mean_not_mean_of_means(tmp_path):
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "source": "/rank/kernel_details.csv",
                "total_rows": 11,
                "invalid_duration_rows": 2,
                "kernels": [
                    {"name": "a", "count": 10, "total_us": 100, "min_us": 10, "max_us": 10},
                    {"name": "a", "count": 1, "total_us": 40, "min_us": 40, "max_us": 40},
                ],
            }
        ),
        encoding="utf-8",
    )
    aggregate, upstream = analysis.read_source(source)
    stats = next(iter(aggregate.names.values()))
    assert stats.report(140)["mean_us"] == round(140 / 11, 3)
    assert stats.min_us == 10
    assert stats.max_us == 40
    assert upstream["invalid_duration_rows"] == "2"


def test_bad_summary_records_skipped(tmp_path):
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "kernels": [
                    None,
                    {"name": "a", "count": 0, "total_us": 0, "min_us": 0, "max_us": 0},
                    {"name": "a", "count": 1.5, "total_us": 10, "min_us": 0, "max_us": 10},
                    {"name": "a", "count": 1, "total_us": 100, "min_us": 1, "max_us": 2},
                    {"name": "a", "count": 1, "total_us": 1, "min_us": 1, "max_us": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    aggregate, _ = analysis.read_source(source)
    assert aggregate.invalid == 4
    assert aggregate.records == 5


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("_transpose_quant_batch_matmul_kernel", "triton_tqbmm"),
        ("TransposeQuantBatchMatMul", "native_quant_matmul_candidate"),
        ("QuantBatchMatMulV3", "native_quant_matmul_candidate"),
        ("hccl_allGather", "communication"),
        ("DynamicQuant", "quantization"),
        ("Stack", "layout"),
        ("FlashAttentionScore", "attention"),
        ("Gemm", "other_matmul"),
    ],
)
def test_family_heuristics(name, expected):
    assert analysis.family(name, "") == expected


def test_directory_selects_one_csv_prefers_tp0_no_double_count(tmp_path):
    write_csv(tmp_path / "dp0_tp3_rank3" / "kernel_details.csv", [["a", 3, "", ""]])
    preferred = write_csv(tmp_path / "dp1_tp0_rank0" / "kernel_details.csv", [["a", 1, "", ""]])
    write_csv(tmp_path / "dp1_tp0_rank0" / "op_summary_0.csv", [["a", 100, "", ""]])
    selected, discovered = analysis.select_source(tmp_path)
    assert selected == preferred
    assert discovered == 2


def test_bounded_output_and_top_limits(tmp_path):
    aggregate = analysis.Aggregator()
    for index in range(300):
        for name in ("_transpose_quant_batch_matmul_kernel", "QuantBatchMatMul", "DynamicQuant", "Transpose", "MatMul"):
            aggregate.add({"Name": name + "x" * 500, "Duration(us)": index + 1, "Input Shapes": str(index) + "y" * 500})
    source = tmp_path / "kernel_details.csv"
    source.touch()
    report = analysis.make_report(aggregate, source, 1, {}, 20, "label" * 500)
    payload = analysis.serialize_bounded(report, 8)
    assert len(payload.encode("utf-8")) <= 8 * 1024
    assert report["output_limits"]["trimmed_for_size"]
    assert report["overview"]["valid_calls"] == 1500
    assert report["families"]["triton_tqbmm"]["calls"] == 300
    assert "truncated" in payload


def test_cli_creates_report_and_does_not_overwrite(tmp_path):
    source = write_csv(tmp_path / "kernel_details.csv", [["a", 5, "", ""]])
    output = tmp_path / "focus" / "focus.json"
    arguments = ["--input", str(source), "--output", str(output)]
    assert analysis.main(arguments) == 0
    before = output.read_bytes()
    assert json.loads(before)["overview"]["valid_calls"] == 1
    assert analysis.main(arguments) == 1
    assert output.read_bytes() == before
    original = source.read_bytes()
    assert analysis.main(["--input", str(source), "--output", str(source)]) == 1
    assert source.read_bytes() == original


def test_source_change_rejected(tmp_path, monkeypatch):
    source = write_csv(tmp_path / "kernel_details.csv", [["a", 5, "", ""]])
    original = analysis.Aggregator.add

    def change_source(self, row, summary=False):
        original(self, row, summary)
        with source.open("a", encoding="utf-8") as handle:
            handle.write("\n")

    monkeypatch.setattr(analysis.Aggregator, "add", change_source)
    with pytest.raises(ValueError, match="Source changed"):
        analysis.read_source(source)


@pytest.mark.parametrize("contents", ["", "Name,Wrong\na,5\n", "Name,Duration(us)\na,nan\n"])
def test_unusable_csv_fails(tmp_path, contents):
    source = tmp_path / "kernel_details.csv"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.read_source(source)


@pytest.mark.parametrize("contents", ["{}", "[]", '{"kernels": {}}', "{broken"])
def test_unusable_json_fails(tmp_path, contents):
    source = tmp_path / "summary.json"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.read_source(source)


@pytest.mark.parametrize("options", [["--top", "0"], ["--top", "100"], ["--max-report-kib", "1"]])
def test_invalid_options(options):
    with pytest.raises(SystemExit):
        analysis.parse_args(["--input", "a", "--output", "b", *options])

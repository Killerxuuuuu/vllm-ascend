# SPDX-License-Identifier: Apache-2.0
"""Produce a bounded, shareable TQBMM profile report (stdlib only).

Input: kernel_details.csv / op_summary*.csv, or the earlier JSON containing
{"kernels": [{"name": ..., "count": ..., "total_us": ..., ...}]}.
CSV rows are streamed; memory grows with distinct operator/shape combinations.
JSON is loaded into memory; prefer the original CSV for very large inputs.
Directories select ONE kernel_details.csv, never sum all TP/DP ranks or combine
kernel_details with op_summary (which normally describe the same tasks).
"""

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TOP = 6
MAX_TOP = 20
DEFAULT_REPORT_KIB = 32
MIN_REPORT_KIB = 8
MAX_REPORT_KIB = 128
TEXT_LIMIT = 180
PATH_LIMIT = 600
CSV_FIELD_LIMIT = 16 * 1024 * 1024
FOCUS_FAMILIES = ("triton_tqbmm", "native_quant_matmul_candidate", "other_matmul", "layout", "quantization")
COUNTER_FIELDS = ("aic_mac_ratio", "aic_mte2_ratio", "aiv_vec_ratio")
PIPELINE_TIME_FIELDS = (
    "aicore_time(us)",
    "aic_mte2_time(us)",
    "aic_mac_time(us)",
    "aic_mte1_time(us)",
    "aic_scalar_time(us)",
    "aic_fixpipe_time(us)",
    "aiv_vec_time(us)",
    "aiv_scalar_time(us)",
)
MISSING_VALUES = ("", "n/a", "na", "none", "null", "--")


def clipped(value, limit=TEXT_LIMIT):
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def number(value):
    result = float(str(value).strip().replace(",", ""))
    if not math.isfinite(result) or result < 0:
        raise ValueError("Expected a finite, nonnegative number")
    return result


def pick(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip().lower() not in MISSING_VALUES:
            return str(value).strip()
    return ""


def family(name, kind):
    """Heuristics, NOT layer attribution or a proof of the bottleneck."""
    text = re.sub(r"[^a-z0-9]", "", (name + " " + kind).lower())
    if "transposequantbatchmatmul" in text:
        # Only the known Triton entry point is unambiguous.
        return "triton_tqbmm" if "_transpose_quant_batch_matmul_kernel" in name else "native_quant_matmul_candidate"
    if any(word in text for word in ("hccl", "allreduce", "allgather", "reducescatter", "alltoall", "sendrecv")):
        return "communication"
    if any(word in text for word in ("quantbatchmatmul", "quantmatmul", "qbmm")):
        return "native_quant_matmul_candidate"
    if any(word in text for word in ("matmul", "gemm", "batchmatmul")):
        return "other_matmul"
    if any(word in text for word in ("quant", "dequant")):
        return "quantization"
    if any(word in text for word in ("transpose", "transdata", "stack", "contiguous", "memcpy")):
        return "layout"
    if any(word in text for word in ("attention", "flashatt", "pagedatt")):
        return "attention"
    return "other"


@dataclass
class Stats:
    count: int = 0
    total_us: float = 0.0
    min_us: float = math.inf
    max_us: float = 0.0
    counters: dict = field(default_factory=dict)

    def add(self, count, total, minimum, maximum, counters):
        self.count += count
        self.total_us += total
        self.min_us = min(self.min_us, minimum)
        self.max_us = max(self.max_us, maximum)
        for key, value in counters.items():
            weighted, samples = self.counters.get(key, (0.0, 0))
            self.counters[key] = (weighted + value * count, samples + count)

    def report(self, denominator):
        result = {
            "calls": self.count,
            "sum_us": round(self.total_us, 3),
            "mean_us": round(self.total_us / self.count, 3),
            "min_us": round(self.min_us, 3),
            "max_us": round(self.max_us, 3),
            "task_time_share_pct": round(100 * self.total_us / denominator, 3) if denominator else 0,
        }
        if self.counters:
            result["counter_call_means"] = {
                key: {"mean": round(total / samples, 4), "covered_calls": samples}
                for key, (total, samples) in self.counters.items()
            }
        return result


class Aggregator:
    def __init__(self):
        self.names = {}
        self.shapes = {}
        self.families = {}
        self.devices = Counter()
        self.invalid = 0
        self.records = 0
        self.missing_shapes = 0

    def add(self, row, summary=False):
        self.records += 1
        try:
            count_value = number(row.get("count", "")) if summary else 1
            if not count_value or int(count_value) != count_value:
                raise ValueError("Invalid count")
            count = int(count_value)
            total = (
                number(row.get("total_us", "")) if summary else number(pick(row, "Duration(us)", "Task Duration(us)"))
            )
            minimum = number(row.get("min_us", "")) if summary else total
            maximum = number(row.get("max_us", "")) if summary else total
            if not minimum <= total / count <= maximum:
                if not (math.isclose(total / count, minimum) or math.isclose(total / count, maximum)):
                    raise ValueError("Inconsistent summary bounds")
            if minimum > maximum:
                raise ValueError("Invalid bounds")
            name = pick(row, "name") if summary else pick(row, "Name", "Op Name")
            if not name:
                raise ValueError("Missing operator name")
        except (ValueError, TypeError, OverflowError):
            self.invalid += 1
            return
        kind = pick(row, "type") if summary else pick(row, "Type", "OP Type")
        core = pick(row, "core") if summary else pick(row, "Accelerator Core", "Task Type")
        shapes = pick(row, "input_shapes") if summary else pick(row, "Input Shapes")
        outputs = pick(row, "output_shapes") if summary else pick(row, "Output Shapes")
        device = pick(row, "Device_id", "Device ID")
        self.devices[device or "unknown"] += count
        if not shapes:
            self.missing_shapes += count
        counters = {}
        if not summary:
            for key in (*COUNTER_FIELDS, *PIPELINE_TIME_FIELDS):
                try:
                    value = number(row.get(key, ""))
                    if key in PIPELINE_TIME_FIELDS or value <= 1:
                        counters[key] = value
                except (ValueError, TypeError):
                    pass
        category = family(name, kind)
        for mapping, key in (
            (self.names, (name, kind, core)),
            (self.shapes, (name, kind, core, shapes, outputs)),
            (self.families, category),
        ):
            mapping.setdefault(key, Stats()).add(count, total, minimum, maximum, counters)


def select_source(path):
    path = path.resolve(strict=True)
    if not path.is_dir():
        return path, 1
    candidates = sorted(path.rglob("kernel_details.csv"))
    if not candidates:
        raise ValueError("No kernel_details.csv found; pass a CSV or the earlier summary JSON explicitly")
    # Prefer a TP0 rank, then stable path ordering. Do not mix ranks/duplicate exports.
    candidates.sort(key=lambda item: (not bool(re.search(r"(?:^|_)tp0(?:_|$)", str(item))), str(item)))
    return candidates[0], len(candidates)


def read_source(source):
    before = source.stat()
    aggregate = Aggregator()
    upstream = {}
    if source.suffix.lower() == ".json":
        with source.open(encoding="utf-8-sig") as handle:
            document = json.load(handle)
        if not isinstance(document, dict) or not isinstance(document.get("kernels"), list):
            raise ValueError('JSON must be the earlier summary with a "kernels" array; use the original CSV otherwise')
        for row in document["kernels"]:
            if not isinstance(row, dict):
                aggregate.invalid += 1
                aggregate.records += 1
                continue
            aggregate.add(row, summary=True)
        upstream = {
            key: clipped(document.get(key), PATH_LIMIT)
            for key in ("source", "total_rows", "invalid_duration_rows")
            if key in document
        }
    elif source.suffix.lower() == ".csv":
        csv.field_size_limit(CSV_FIELD_LIMIT)
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
            if not {"Duration(us)", "Task Duration(us)"}.intersection(reader.fieldnames):
                raise ValueError("CSV lacks Duration(us) / Task Duration(us)")
            for row in reader:
                aggregate.add(row)
    else:
        raise ValueError("Input must be a CSV, summary JSON, or directory containing kernel_details.csv")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while reading; wait until profiler export/analysis finishes")
    if not aggregate.names:
        raise ValueError("No valid timing records found")
    return aggregate, upstream


def entry(key, stats, total):
    name, kind, core, *shapes = key
    result = {
        "key_hash": hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16],
        "name": clipped(name),
        "type": clipped(kind),
        "core": clipped(core),
        "family": family(name, kind),
        **stats.report(total),
    }
    if shapes:
        result.update(input_shapes=clipped(shapes[0]), output_shapes=clipped(shapes[1]))
    return result


def make_report(aggregate, source, discovered, upstream, top, label):
    total = sum(item.total_us for item in aggregate.names.values())
    calls = sum(item.count for item in aggregate.names.values())
    ranked_names = heapq.nlargest(top, aggregate.names.items(), key=lambda pair: pair[1].total_us)
    sections = {}
    for category in FOCUS_FAMILIES:
        matching = ((key, stats) for key, stats in aggregate.shapes.items() if family(key[0], key[1]) == category)
        sections[category] = [
            entry(key, stats, total) for key, stats in heapq.nlargest(top, matching, key=lambda p: p[1].total_us)
        ]
    warnings = [
        "sum_us/share 是所读任务耗时的累加/占比，不是请求延迟、关键路径或设备利用率；并行任务可能重叠。",
        "类别由名称启发式匹配。原生 QuantMatMul/Transpose/Stack 不能仅凭名字认定为 Wo_a；也不能直接相加当 Wo_a 耗时。",
        "CSV/汇总 JSON 不能证明 Graph 回放、Prepared 命中或采集完整，也无法独立定位 Python Host 开销。",
        "不能从均值/最值重建 P50/P99；counter_call_means 是有数据调用的计数加权平均，不是带宽/峰值算力利用率。",
        "流水 time(us) 可相互重叠，不能相加或从 Duration 相减推算空闲；缺失 PMU 字段不按零处理。",
        "若采集日志出现 collected data has been lost/TLV 警告，本报告仍可能基于不完整数据；请一并提供该警告。",
        "跨实验仅比较相同 rank、负载、采集区间和配置；不要将不同 TP/DP rank 累加成端到端耗时。",
    ]
    if aggregate.missing_shapes:
        warnings.append("存在缺失输入形状，不能据此区分 prefill/decode 或确认 Wo_a；需额外 shape/调用关联证据。")
    if len(aggregate.devices) > 1:
        warnings.append("输入含多个设备，时长是跨设备累加，不宜直接与单 rank 报告比较。")
    if discovered > 1:
        warnings.append(f"发现 {discovered} 个 CSV，只读取 source 所示文件；如需指定 rank，请直接传该 CSV。")
    return {
        "schema_version": 1,
        "label": clipped(label),
        "source": clipped(source, PATH_LIMIT),
        "source_bytes": source.stat().st_size,
        "input_kind": "summary_json" if source.suffix.lower() == ".json" else "csv",
        "upstream": upstream,
        "overview": {
            "input_records": aggregate.records,
            "invalid_records_skipped": aggregate.invalid,
            "valid_calls": calls,
            "calls_missing_input_shapes": aggregate.missing_shapes,
            "operator_groups": len(aggregate.names),
            "shape_groups": len(aggregate.shapes),
            "sum_task_us": round(total, 3),
            "device_count": len(aggregate.devices),
        },
        "warnings": warnings,
        "families": {key: stats.report(total) for key, stats in sorted(aggregate.families.items())},
        "top_operators_by_sum": [entry(key, stats, total) for key, stats in ranked_names],
        "focus_shapes_by_sum": sections,
        "output_limits": {"requested_top_per_list": top, "text_limit": TEXT_LIMIT, "trimmed_for_size": False},
        "next_evidence": [
            "先对照两个报告的 triton_tqbmm 与原生候选形状、次数、mean_us；次数/负载不同时不能比较累计时长。",
            "关注 layout、quantization、communication 是否增加；类别包含整网其他层，需关联调用才能归因。",
            "若设备任务差异解释不了 TTFT/TPOT，再看短区间 CPU+NPU 时间线、调度/排队以及实际 Graph/Prepared 路径。",
        ],
    }


def serialize_bounded(report, max_kib):
    limit = max_kib * 1024
    while True:
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if len(payload.encode("utf-8")) <= limit:
            return payload
        lists = [report["top_operators_by_sum"], *report["focus_shapes_by_sum"].values()]
        longest = max(lists, key=len)
        if not longest:
            raise ValueError("Report metadata exceeds requested size limit")
        longest.pop()
        report["output_limits"]["trimmed_for_size"] = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV, earlier summary JSON, or profile directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are not overwritten")
    parser.add_argument("--label", default="unspecified", help="e.g. composition-c1 or triton-prepared-c64")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--max-report-kib", type=int, default=DEFAULT_REPORT_KIB)
    args = parser.parse_args(argv)
    if not 1 <= args.top <= MAX_TOP:
        parser.error(f"--top must be between 1 and {MAX_TOP}")
    if not MIN_REPORT_KIB <= args.max_report_kib <= MAX_REPORT_KIB:
        parser.error(f"--max-report-kib must be between {MIN_REPORT_KIB} and {MAX_REPORT_KIB}")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        source, discovered = select_source(args.input)
        output = args.output.resolve()
        if output == source or output.exists():
            raise ValueError("Output already exists or aliases input; choose a new output filename")
        aggregate, upstream = read_source(source)
        report = make_report(aggregate, source, discovered, upstream, args.top, args.label)
        payload = serialize_bounded(report, args.max_report_kib)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        print(f"Source: {source}")
        print(f"Report: {output} ({len(payload.encode('utf-8')) / 1024:.1f} KiB)")
        print(f"Valid calls: {report['overview']['valid_calls']}; skipped records: {aggregate.invalid}")
        print("Share this focus JSON only; keep original files and collection logs.")
    except (OSError, ValueError, csv.Error, OverflowError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

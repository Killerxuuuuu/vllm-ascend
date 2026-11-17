# SPDX-License-Identifier: Apache-2.0
"""Bounded offline CPU/NPU trace evidence, not a critical-path/root-cause oracle."""

import gzip
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from analyze_tqbmm_profile import Stats, clipped, family

CHUNK_SIZE = 64 * 1024
MAX_EVENT_CHARS = 4 * 1024 * 1024
MAX_TIMELINE_EVENTS = 200_000
MAX_FLOW_EVENTS = 200_000
BUCKET_US = (1, 5, 10, 20, 50, 100, 500, 1000, 10000)
TOP = 5


class JSONStream:
    """Incremental decoder: bounded per-value size, no whole-file json.load."""

    def __init__(self, handle):
        self.handle = handle
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def more(self):
        self.buffer = self.buffer[self.position :]
        self.position = 0
        chunk = self.handle.read(CHUNK_SIZE)
        self.eof = not chunk
        self.buffer += chunk
        if len(self.buffer) > MAX_EVENT_CHARS:
            raise ValueError("One trace JSON value exceeds size guard")

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.more()

    def consume(self, character):
        if self.peek() != character:
            raise ValueError(f"Incomplete/malformed trace: expected {character!r}")
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # A scalar at a chunk boundary may have more digits in the next chunk.
                if end == len(self.buffer) and not self.eof:
                    self.more()
                    continue
                self.position = end
                return value
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError("Incomplete/malformed trace JSON") from error
                self.more()

    def array(self):
        self.consume("[")
        if self.peek() != "]":
            while True:
                yield self.value()
                if self.peek() != ",":
                    break
                self.consume(",")
        self.consume("]")


def iter_events(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        stream = JSONStream(handle)
        found = False
        if stream.peek() == "[":
            found = True
            yield from stream.array()
        else:
            stream.consume("{")
            if stream.peek() != "}":
                while True:
                    key = stream.value()
                    if not isinstance(key, str):
                        raise ValueError("Trace object key must be a string")
                    stream.consume(":")
                    if key == "traceEvents":
                        if found:
                            raise ValueError("Duplicate traceEvents array")
                        found = True
                        yield from stream.array()
                    else:
                        stream.value()
                    if stream.peek() != ",":
                        break
                    stream.consume(",")
            stream.consume("}")
        if not found or stream.peek():
            raise ValueError("Missing traceEvents array or trailing data")


def focus_family(name):
    lower = name.lower()
    if "fill" in lower or "zero_" in lower:
        return "fill_or_zero_candidate"
    if any(word in lower for word in ("synchroniz", "streamwait", "eventwait")):
        return "synchronization_candidate"
    return family(name, "")


class Durations:
    def __init__(self):
        self.stats = Stats()
        self.buckets = Counter()

    def add(self, duration):
        self.stats.add(1, duration, duration, duration, {})
        label = next((f"<={bound}" for bound in BUCKET_US if duration <= bound), ">10000")
        self.buckets[label] += 1

    def report(self):
        result = self.stats.report(self.stats.total_us)
        result.pop("task_time_share_pct")
        return {**result, "exclusive_duration_buckets_us": dict(self.buckets)}


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite trace time")
    return value


def event_key(event):
    return str(event.get("pid", "")), str(event.get("tid", "")), finite(event["ts"])


def event_kind(event, processes):
    category = str(event.get("cat", "")).lower()
    process = processes.get(str(event.get("pid", "")), "").lower()
    if "kernel" in category or category in ("aicore", "ai_core", "ai_vector_core") or "ascend hardware" in process:
        return "device"
    if any(word in category for word in ("cpu_op", "python", "runtime", "user_annotation")) or process == "python":
        return "host"
    return "unknown"


def merged_interval_report(intervals):
    """Only coverage between first/last observed task; never hardware utilization."""
    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    if not ordered:
        return {"available": False}
    first, end, previous = ordered[0]
    covered = end - first
    gaps = []
    for start, finish, name in ordered[1:]:
        if start > end:
            gap = start - end
            if len(gaps) < TOP:
                heapq.heappush(gaps, (gap, end, start, previous, name))
            elif gap > gaps[0][0]:
                heapq.heapreplace(gaps, (gap, end, start, previous, name))
            covered += finish - start
        elif finish > end:
            covered += finish - end
        if finish > end:
            end, previous = finish, name
    span = end - first
    return {
        "available": True,
        "observed_span_us": round(span, 3),
        "union_covered_us": round(covered, 3),
        "uncovered_between_tasks_us": round(max(0.0, span - covered), 3),
        "largest_uncovered_intervals": [
            {
                "gap_us": round(gap, 3),
                "previous_task": clipped(before),
                "next_task": clipped(after),
                "start_us": start,
                "end_us": finish,
            }
            for gap, start, finish, before, after in sorted(gaps, reverse=True)
        ],
        "meaning": "Observed task coverage only; gaps are NOT proof of CPU starvation or hardware idle.",
    }


def analyze_trace(path):
    path = Path(path)
    before = path.stat()
    processes = {}
    categories = Counter()
    total_events = 0
    # Metadata can appear after events: identify lanes in a separate streaming pass.
    for event in iter_events(path):
        total_events += 1
        if not isinstance(event, dict):
            raise ValueError("Trace array contains a non-object")
        if event.get("ph") == "M" and event.get("name") == "process_name":
            processes[str(event.get("pid", ""))] = str(event.get("args", {}).get("name", ""))
        categories[clipped(event.get("cat", ""))] += 1
    groups = {"host": {}, "device": {}, "unknown": {}}
    focus = {"host": {}, "device": {}}
    starts = {"host": defaultdict(list), "device": defaultdict(list)}
    start_counts = Counter()
    intervals = []
    flows = defaultdict(lambda: {"s": [], "f": []})
    flow_count = invalid = omitted_intervals = omitted_flows = non_x = omitted_starts = 0
    shapes = Counter()
    for event in iter_events(path):
        phase = event.get("ph")
        if phase in ("s", "f") and event.get("name") == "torch_to_npu":
            if flow_count >= MAX_FLOW_EVENTS:
                omitted_flows += 1
                continue
            try:
                key = event_key(event)
                # Include category and scope; never match unrelated flow categories by id.
                identifier = json.dumps(
                    [event.get("cat"), event.get("scope"), event.get("id"), event.get("id2")], sort_keys=True
                )
                flows[identifier][phase].append(key)
                flow_count += 1
            except (ValueError, TypeError, KeyError):
                invalid += 1
            continue
        if phase != "X":
            non_x += phase in ("B", "E")
            continue
        try:
            timestamp, duration = finite(event["ts"]), finite(event["dur"])
            if duration < 0:
                raise ValueError("Negative duration")
        except (ValueError, TypeError, KeyError):
            invalid += 1
            continue
        kind = event_kind(event, processes)
        name = str(event.get("name", "unknown"))
        groups[kind].setdefault(name, Durations()).add(duration)
        if kind == "unknown":
            continue
        category = focus_family(name)
        if category != "other":
            focus[kind].setdefault(category, Durations()).add(duration)
        args = event.get("args") or {}
        dims = args.get("Input Dims", args.get("Input Shapes", args.get("input_shapes", "")))
        dims_text = clipped(json.dumps(dims, ensure_ascii=True)) if dims else ""
        if dims_text and category != "other":
            shapes[(kind, clipped(name), dims_text)] += 1
        start_key = (str(event.get("pid", "")), str(event.get("tid", "")), timestamp)
        if start_counts[kind] < MAX_TIMELINE_EVENTS:
            starts[kind][start_key].append((name, duration, dims_text))
            start_counts[kind] += 1
        else:
            omitted_starts += 1
        if kind == "device":
            if len(intervals) < MAX_TIMELINE_EVENTS:
                intervals.append((timestamp, timestamp + duration, name))
            else:
                omitted_intervals += 1
    associations = {}
    matched = ambiguous = 0
    for endpoints in flows.values():
        if len(endpoints["s"]) != 1 or len(endpoints["f"]) != 1:
            ambiguous += 1
            continue
        host_key, device_key = endpoints["s"][0], endpoints["f"][0]
        hosts, devices = starts["host"].get(host_key, []), starts["device"].get(device_key, [])
        if len(hosts) != 1 or len(devices) != 1:
            ambiguous += 1
            continue
        host_name, host_duration, dims = hosts[0]
        device_name, device_duration, _ = devices[0]
        matched += 1
        if focus_family(device_name) == "other":
            continue
        key = (host_name, device_name, dims)
        item = associations.setdefault(key, {"host": Durations(), "device": Durations(), "start_delta": Durations()})
        item["host"].add(host_duration)
        item["device"].add(device_duration)
        delta = device_key[2] - host_key[2]
        if delta >= 0:
            item["start_delta"].add(delta)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Trace changed during analysis; wait for export to finish")
    coverage = (
        merged_interval_report(intervals)
        if not omitted_intervals
        else {
            "available": False,
            "reason": "Event cap exceeded; partial gaps would be misleading",
        }
    )
    return {
        "source": str(path),
        "events": total_events,
        "invalid_events": invalid,
        "category_samples": dict(categories.most_common(TOP)),
        "non_X_B_E_events_not_analyzed": non_x,
        "omitted_interval_events": omitted_intervals,
        "omitted_flow_events": omitted_flows,
        "omitted_start_entries": omitted_starts,
        "host_top_inclusive": [
            {"name": clipped(name), **stats.report()}
            for name, stats in heapq.nlargest(TOP, groups["host"].items(), key=lambda pair: pair[1].stats.total_us)
        ],
        "host_focus_inclusive": {name: stats.report() for name, stats in focus["host"].items()},
        "device_focus": {name: stats.report() for name, stats in focus["device"].items()},
        "unknown_top": [
            {"name": clipped(name), **stats.report()}
            for name, stats in heapq.nlargest(TOP, groups["unknown"].items(), key=lambda pair: pair[1].stats.total_us)
        ],
        "observed_input_shapes": [
            {"kind": kind, "name": name, "input_dims": dims, "calls": count}
            for (kind, name, dims), count in shapes.most_common(TOP)
        ],
        "device_task_coverage": coverage,
        "flow_evidence": {
            "matched_unique_pairs": matched,
            "unmatched_or_ambiguous_pairs": ambiguous,
            "focus_associations": [
                {
                    "host_name": clipped(key[0]),
                    "device_name": clipped(key[1]),
                    "host_input_dims": key[2],
                    "host_inclusive": value["host"].report(),
                    "device": value["device"].report(),
                    "host_start_to_device_start": value["start_delta"].report()
                    if value["start_delta"].stats.count
                    else None,
                }
                for key, value in heapq.nlargest(
                    TOP, associations.items(), key=lambda pair: pair[1]["device"].stats.total_us
                )
            ],
            "meaning": "Only unique exported torch_to_npu flow endpoints; start delta is NOT launch/queue latency.",
        },
        "warnings": [
            "CPU durations are inclusive and may nest/overlap; their sum is not Python self time.",
            "Event-lane classification is heuristic. Unknown lanes and missing shapes remain unknown.",
            "Graph replay may expose device kernels without individual CPU operator/shape associations.",
            "Observed gaps cannot distinguish scheduling, synchronization, missing events or CPU delays.",
            "Trace parsing success does not establish completeness; retain TLV/data-loss warnings.",
        ],
    }

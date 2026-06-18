#!/usr/bin/env python3
"""Summarize LLMServingSim per-request CSV output.

The simulator writes one row per completed request. This helper rebuilds the
final aggregate metrics that can be derived from those rows using the same
units and distribution semantics as Scheduler.print_result().
usage: python summarize_sim_csv.py outputs/<run>.csv [--summary-csv outputs/<run>_summary.csv]
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

FREQ = 1_000_000_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build LLMServingSim summary metrics from a per-request CSV."
    )
    parser.add_argument("csv_path", help="Path to outputs/<run>.csv")
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional path for a machine-readable summary CSV.",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.csv_path))
    if not rows:
        raise SystemExit(f"No request rows found in {args.csv_path}")

    summary_rows: list[dict[str, object]] = []
    lines = render_summary(rows, summary_rows)
    print("\n".join(lines))

    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), summary_rows)

    return 0


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {
                "instance_id": int(number(raw, "instance id", "instance_id")),
                "request_id": int(number(raw, "request id", "request_id")),
                "model": raw.get("model", ""),
                "input": int(number(raw, "input")),
                "output": int(number(raw, "output")),
                "arrival": int(number(raw, "arrival")),
                "end_time": int(number(raw, "end_time")),
                "latency": int(number(raw, "latency")),
                "queuing_delay": int(number(raw, "queuing_delay")),
                "TTFT": int(number(raw, "TTFT")),
                "TPOT": int(number(raw, "TPOT")),
                "ITL": parse_itl(raw.get("ITL", "[]")),
            }
            attach_optional_prefix_fields(raw, row)
            rows.append(row)
    return rows


def render_summary(rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    lines.append("Simulation results from CSV")
    lines.append("")
    lines.extend(render_throughput(rows, summary_rows))
    lines.append("")
    lines.extend(render_prefix(rows, summary_rows))

    lines.append("")
    lines.append("Overall")
    lines.extend(render_latency_metrics("Overall", "all", rows, summary_rows))

    by_instance: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_instance[int(row["instance_id"])].append(row)

    for instance_id in sorted(by_instance):
        lines.append("")
        lines.append(f"Instance [{instance_id}]")
        lines.extend(render_latency_metrics("Instance", instance_id, by_instance[instance_id], summary_rows))

    return lines


def render_throughput(rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> list[str]:
    req_cnt = len(rows)
    total_clocks = max(int(r["end_time"]) for r in rows)
    total_latency_s = total_clocks / FREQ
    total_prompt = sum(int(r["input"]) for r in rows)
    total_gen = sum(int(r["output"]) for r in rows)

    metrics = [
        ("Total requests", req_cnt, "count"),
        ("Total clocks", total_clocks, "ns"),
        ("Total latency", total_latency_s, "s"),
        ("Total input tokens", total_prompt, "tokens"),
        ("Total generated tokens", total_gen, "tokens"),
        ("Request throughput", safe_div(req_cnt, total_latency_s), "req/s"),
        ("Average prompt throughput", safe_div(total_prompt, total_latency_s), "tok/s"),
        ("Average generation throughput", safe_div(total_gen, total_latency_s), "tok/s"),
        ("Total token throughput", safe_div(total_prompt + total_gen, total_latency_s), "tok/s"),
    ]

    lines = ["Throughput Results"]
    for name, value, unit in metrics:
        lines.append(f"{name}: {format_value(value)} {unit}")
        summary_rows.append(row_for_summary("Throughput Results", "all", name, "", value, unit))
    return lines


def render_prefix(rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> list[str]:
    if not all("prefix_requested_tokens" in row for row in rows):
        return [
            "Prefix Caching Results",
            "N/A: this CSV does not contain prefix-cache hit fields.",
        ]

    requested = sum(int(r["prefix_requested_tokens"]) for r in rows)
    npu_hits = sum(int(r.get("npu_prefix_hit_tokens", 0)) for r in rows)
    storage_hits = sum(int(r.get("storage_prefix_hit_tokens", 0)) for r in rows)
    total_hits = npu_hits + storage_hits

    metrics = [
        ("Total requested prompt tokens", requested, "tokens"),
        ("NPU prefix hit prompt tokens", npu_hits, "tokens"),
        ("NPU prefix hit ratio", safe_div(npu_hits * 100, requested), "%"),
        ("Storage prefix hit prompt tokens", storage_hits, "tokens"),
        ("Storage prefix hit ratio", safe_div(storage_hits * 100, requested), "%"),
        ("Total prefix hit ratio", safe_div(total_hits * 100, requested), "%"),
    ]

    lines = ["Prefix Caching Results"]
    for name, value, unit in metrics:
        lines.append(f"{name}: {format_value(value)} {unit}")
        summary_rows.append(row_for_summary("Prefix Caching Results", "all", name, "", value, unit))
    return lines


def render_latency_metrics(
    section: str,
    instance_id: object,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> list[str]:
    metrics = [
        ("Time to First Token", "TTFT", [int(r["TTFT"]) for r in rows]),
        ("Time per Output Token (excl. 1st token)", "TPOT", [int(r["TPOT"]) for r in rows]),
        ("Inter-token Latency", "ITL", [int(v) for r in rows for v in r["ITL"]]),
    ]

    lines: list[str] = []
    for title, short, values_ns in metrics:
        lines.append(title)
        if not values_ns:
            lines.append(f"No {short} data available")
            continue
        values_ms = [v / 1_000_000 for v in values_ns]
        stats = [
            ("Mean", mean(values_ms)),
            ("Median", percentile(values_ms, 50)),
            ("P99", percentile(values_ms, 99)),
        ]
        for stat_name, value in stats:
            lines.append(f"{stat_name} {short} (ms): {value:.2f}")
            summary_rows.append(
                row_for_summary(section, instance_id, short, stat_name, value, "ms")
            )
    return lines


def attach_optional_prefix_fields(raw: dict[str, str], row: dict[str, object]) -> None:
    requested = optional_number(raw, "prefix_requested_tokens", "prefix requested tokens")
    npu = optional_number(raw, "npu_prefix_hit_tokens", "npu prefix hit tokens")
    storage = optional_number(raw, "storage_prefix_hit_tokens", "storage prefix hit tokens")
    total = optional_number(raw, "total_prefix_hit_tokens", "total prefix hit tokens")

    if requested is None:
        return
    if npu is None:
        npu = 0
    if storage is None:
        storage = max(0, (total or 0) - npu)

    row["prefix_requested_tokens"] = int(requested)
    row["npu_prefix_hit_tokens"] = int(npu)
    row["storage_prefix_hit_tokens"] = int(storage)


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "instance_id", "metric", "stat", "value", "unit"],
        )
        writer.writeheader()
        writer.writerows(rows)


def row_for_summary(section: str, instance_id: object, metric: str, stat: str, value: object, unit: str) -> dict[str, object]:
    return {
        "section": section,
        "instance_id": instance_id,
        "metric": metric,
        "stat": stat,
        "value": value,
        "unit": unit,
    }


def parse_itl(value: str) -> list[int]:
    if not value:
        return []
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        return []
    return [int(v) for v in parsed]


def number(row: dict[str, str], *names: str) -> float:
    value = optional_number(row, *names)
    if value is None:
        joined = ", ".join(names)
        raise KeyError(f"Missing numeric column: {joined}")
    return value


def optional_number(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    return None


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    idx = (q / 100.0) * (len(values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def format_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.2f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())

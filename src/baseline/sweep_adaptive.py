"""Run a lightweight adaptive-count parameter sweep on Kaggle train data."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import pandas as pd


SUMMARY_PATTERN = re.compile(
    r"OFFICIAL sweep_metrics: dataset_count=(?P<dataset_count>\d+) "
    r"score=(?P<score>\S+) node_recall=(?P<node_recall>\S+) "
    r"adj_edge_jaccard=(?P<adj_edge_jaccard>\S+)"
)
CSV_COLUMNS = (
    "detection_policy",
    "threshold_percentile",
    "sigma",
    "min_distance_xy",
    "min_distance_z",
    "adaptive_mad_k",
    "adaptive_min_detections",
    "adaptive_max_detections",
    "adaptive_count_smoothing_window",
    "link_max_um",
    "seed",
    "requested_dataset_limit",
    "dataset_count",
    "official_summary_score",
    "node_recall",
    "adjusted_edge_jaccard",
    "runtime_seconds",
    "status",
    "return_code",
    "error",
)


def _comma_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _config(value: str) -> tuple[float, int, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("config must be MAD_K,MAX_DETECTIONS,LINK_MAX_UM")
    return float(parts[0]), int(parts[1]), float(parts[2])


def parse_official_summary(output: str) -> dict[str, float | int] | None:
    match = SUMMARY_PATTERN.search(output)
    if match is None:
        return None

    def metric(name: str) -> float:
        value = match.group(name)
        try:
            return float(value)
        except ValueError:
            return math.nan

    return {
        "dataset_count": int(match.group("dataset_count")),
        "official_summary_score": metric("score"),
        "node_recall": metric("node_recall"),
        "adjusted_edge_jaccard": metric("adj_edge_jaccard"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep adaptive-count detection parameters on train datasets.")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("adaptive_sweep.csv"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--adaptive-mad-k-values", default="2.5,3.0,3.5")
    parser.add_argument("--adaptive-max-detections-values", default="300,400")
    parser.add_argument("--link-max-um-values", default="5.0,7.0")
    parser.add_argument(
        "--config",
        action="append",
        type=_config,
        help="Explicit MAD_K,MAX_DETECTIONS,LINK_MAX_UM tuple; repeat for a staged final sweep.",
    )
    parser.add_argument("--threshold-percentile", type=float, default=99.7)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--min-distance-xy", type=int, default=5)
    parser.add_argument("--min-distance-z", type=int, default=2)
    parser.add_argument("--adaptive-min-detections", type=int, default=80)
    parser.add_argument("--adaptive-count-smoothing-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = args.config or list(
        itertools.product(
            _comma_floats(args.adaptive_mad_k_values),
            _comma_ints(args.adaptive_max_detections_values),
            _comma_floats(args.link_max_um_values),
        )
    )
    if not configs:
        raise ValueError("adaptive sweep has no parameter configurations to evaluate")
    rows: list[dict[str, object]] = []
    for mad_k, adaptive_max, link_max_um in configs:
        command = [
            sys.executable,
            "-m",
            "src.baseline.make_baseline_submission",
            "--mode",
            "eval-on-train",
            "--train-root",
            str(args.train_root),
            "--limit",
            str(args.limit),
            "--detection-policy",
            "adaptive-count",
            "--threshold-percentile",
            str(args.threshold_percentile),
            "--sigma",
            str(args.sigma),
            "--min-distance-xy",
            str(args.min_distance_xy),
            "--min-distance-z",
            str(args.min_distance_z),
            "--adaptive-mad-k",
            str(mad_k),
            "--adaptive-min-detections",
            str(args.adaptive_min_detections),
            "--adaptive-max-detections",
            str(adaptive_max),
            "--adaptive-count-smoothing-window",
            str(args.adaptive_count_smoothing_window),
            "--link-max-um",
            str(link_max_um),
            "--seed",
            str(args.seed),
        ]
        started = time.perf_counter()
        launch_error = ""
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            return_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except OSError as exc:
            return_code = -1
            stdout = ""
            stderr = ""
            launch_error = f"could not launch eval-on-train subprocess: {exc}"
        runtime = time.perf_counter() - started
        metrics = parse_official_summary(stdout)
        status = "success" if return_code == 0 and metrics is not None else "failure"
        output_tail = (stderr or stdout)[-2000:]
        if launch_error:
            error = launch_error
        elif return_code != 0:
            error = f"eval-on-train exited with code {return_code}: {output_tail}"
        elif metrics is None:
            error = f"official machine-readable summary metrics were not found: {output_tail}"
        else:
            error = ""
        row: dict[str, object] = {
            "detection_policy": "adaptive-count",
            "threshold_percentile": args.threshold_percentile,
            "sigma": args.sigma,
            "min_distance_xy": args.min_distance_xy,
            "min_distance_z": args.min_distance_z,
            "adaptive_mad_k": mad_k,
            "adaptive_min_detections": args.adaptive_min_detections,
            "adaptive_max_detections": adaptive_max,
            "adaptive_count_smoothing_window": args.adaptive_count_smoothing_window,
            "link_max_um": link_max_um,
            "seed": args.seed,
            "requested_dataset_limit": args.limit,
            "dataset_count": 0,
            "official_summary_score": math.nan,
            "node_recall": math.nan,
            "adjusted_edge_jaccard": math.nan,
            "runtime_seconds": runtime,
            "status": status,
            "return_code": return_code,
            "error": error,
        }
        if metrics is not None:
            row.update(metrics)
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(args.output, index=False)
        print(
            f"{status}: mad_k={mad_k} adaptive_max={adaptive_max} link_max_um={link_max_um} "
            f"score={row['official_summary_score']} runtime_seconds={runtime:.1f}"
        )
    return 0 if rows and all(row["status"] == "success" for row in rows) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

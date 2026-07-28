"""Generate FlowSense's final motion-prediction accuracy report."""

from __future__ import annotations

import argparse
from pathlib import Path

from flowsense.prediction_evaluation import (
    evaluate_datasets,
    write_accuracy_csv,
    write_markdown_report,
)


DEFAULT_INPUTS = sorted(
    Path("tracking_data").glob("member2_canonical_tracks*.csv")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FlowSense position predictions against future observed "
            "track centers and a stationary-position baseline."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Canonical tracking CSVs (default: all four repository datasets).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("easy_results/prediction_accuracy.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("PREDICTION_EVALUATION.md"),
    )
    parser.add_argument("--history-points", type=int, default=30)
    parser.add_argument("--velocity-window", type=int, default=5)
    parser.add_argument("--prediction-horizon", type=int, default=15)
    parser.add_argument("--inactive-timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.inputs:
        raise SystemExit("No canonical tracking CSVs were found")
    summaries, errors = evaluate_datasets(
        args.inputs,
        history_points=args.history_points,
        velocity_window=args.velocity_window,
        prediction_horizon_frames=args.prediction_horizon,
        inactive_timeout_frames=args.inactive_timeout,
    )
    write_accuracy_csv(args.output, summaries)
    write_markdown_report(
        args.report,
        summaries,
        history_points=args.history_points,
        velocity_window=args.velocity_window,
        prediction_horizon_frames=args.prediction_horizon,
    )
    print(f"Prediction samples evaluated: {len(errors)}")
    print(f"Accuracy table: {args.output.resolve()}")
    print(f"Markdown report: {args.report.resolve()}")


if __name__ == "__main__":
    main()

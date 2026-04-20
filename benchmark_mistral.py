#!/usr/bin/env python3
"""Benchmark Mistral inference across rocprofiler-sdk output formats.

This script mirrors the direct LD_PRELOAD profiling strategy from
run_benchmark.py, but profiles `mistral.py` instead of the synthetic HIP
benchmark binary.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import run_benchmark as rb

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_COUNTER_FILE = SCRIPT_DIR / "benchmark" / "pmc_perf_0.txt"
DEFAULT_SDK_BUILD_DIR_NAME = "build-hypothesis-rocprofiler-sdk"
DEFAULT_MISTRAL_SCRIPT = SCRIPT_DIR / "mistral.py"
DEFAULT_MODEL_TAG = "Mistral-7B-v0.1"
DEFAULT_PROFILE_MARKER = 50

PER_HYPOTHESIS_HEADER = (
    "hypothesis",
    "format",
    "model",
    "counter_file",
    "unique_counter_count",
    "output_bytes",
    "output_human",
    "exec_sec",
    "write_sec",
    "read_sec",
    "rows",
    "unique_counters",
    "output_dir",
    "log_file",
)

CONSOLIDATED_HEADER = (
    "hypothesis",
    "format",
    "output_bytes",
    "output_human",
    "exec_sec",
    "write_sec",
    "read_sec",
    "rows",
    "unique_counters",
)


def log(message: str) -> None:
    """Print an informational message."""
    print(message)


def warn(message: str) -> None:
    """Print a warning message."""
    print(f"WARNING: {message}", file=sys.stderr)


def format_seconds(seconds: float) -> str:
    """Format a floating-point second value."""
    return f"{seconds:.6f}"


def format_command(command: Iterable[str]) -> str:
    """Format a subprocess command for display."""
    return shlex.join([str(part) for part in command])


def resolve_path(raw_path: str, cwd: Path) -> Path:
    """Resolve user-provided paths relative to the current working directory."""
    resolved = Path(raw_path).expanduser()
    if resolved.is_absolute():
        return resolved
    return (cwd / resolved).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Create and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile mistral.py with rocprofiler-sdk-tool via LD_PRELOAD and compare "
            "output format size/read/write performance."
        )
    )
    parser.add_argument(
        "--hypotheses",
        default=None,
        help=(
            "Comma-separated hypothesis names to run "
            "(default: auto-discover from run_benchmark.py known hypotheses)"
        ),
    )
    parser.add_argument(
        "--formats",
        default=None,
        help=(
            "Optional comma-separated format filter "
            "(subset of csv,rocpd,feather,parquet,rocksdb)"
        ),
    )
    parser.add_argument(
        "--counter-file",
        default=str(DEFAULT_COUNTER_FILE),
        help=f"Counter file path (default: {DEFAULT_COUNTER_FILE})",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--sdk-build-dir",
        default=DEFAULT_SDK_BUILD_DIR_NAME,
        help=f"SDK build dir name inside worktree (default: {DEFAULT_SDK_BUILD_DIR_NAME})",
    )
    parser.add_argument(
        "--mistral-script",
        default=str(DEFAULT_MISTRAL_SCRIPT),
        help=f"Path to mistral inference script (default: {DEFAULT_MISTRAL_SCRIPT})",
    )
    parser.add_argument(
        "--python-exe",
        default="python3",
        help="Python executable used to run mistral script (default: python3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print run plan without profiling, measuring, or writing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra command logging",
    )
    return parser.parse_args(argv)


def run_mistral_profile(
    *,
    hypothesis: str,
    output_format: str,
    output_dir: Path,
    command: list[str],
    profile_env: dict[str, str],
    dry_run: bool,
    verbose: bool,
) -> tuple[float, Path]:
    """Run one profiled mistral invocation and return elapsed seconds + log path."""
    log_file = output_dir / f"profile_{output_format}.log"
    if dry_run:
        log(f"[DRY RUN] hypothesis={hypothesis} format={output_format}")
        log(f"[DRY RUN]   command: {format_command(command)}")
        log(f"[DRY RUN]   output_dir: {output_dir}")
        return 0.0, log_file

    rb.ensure_dir(output_dir, dry_run=False)
    if verbose:
        log(f"[profile] hypothesis={hypothesis} format={output_format}")
        log(f"[profile] command: {format_command(command)}")
    start = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            env=profile_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        warn(
            f"Mistral profile run exited with code {completed.returncode} "
            f"for hypothesis={hypothesis} format={output_format}. Log: {log_file}"
        )
    return elapsed, log_file


def write_csv(
    output_file: Path,
    header: tuple[str, ...],
    rows: list[list[str]],
    *,
    dry_run: bool,
) -> None:
    """Write CSV rows to disk or emit dry-run log."""
    if dry_run:
        log(f"[DRY RUN] would write: {output_file}")
        return
    rb.ensure_dir(output_file.parent, dry_run=False)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    cwd = Path.cwd()
    try:
        active_hypotheses = rb.hypothesis_pairs(args.hypotheses)
        format_filter = rb.parse_format_filter(args.formats)
        counter_file = resolve_path(args.counter_file, cwd)
        results_dir = resolve_path(args.results_dir, cwd)
        mistral_script = resolve_path(args.mistral_script, cwd)

        if not args.dry_run and not mistral_script.is_file():
            raise rb.BenchmarkError(f"Mistral script not found: {mistral_script}")

        counter_names = rb.parse_counter_file(counter_file)
        unique_counter_count = len(set(counter_names))
        counter_line = "pmc: " + " ".join(counter_names)

        if not args.dry_run:
            rb.ensure_dir(results_dir, dry_run=False)

        log(f"==> Counter file: {counter_file}")
        log(f"==> Counters ({unique_counter_count}): {' '.join(counter_names)}")
        log(f"==> Mistral script: {mistral_script}")
        log(f"==> Model: {DEFAULT_MODEL_TAG}")
        log(
            "==> Hypotheses: "
            + " ".join([name for name, _worktree in active_hypotheses])
        )
        if args.dry_run:
            log("==> Dry run: enabled")
        log("")

        consolidated_rows: list[list[str]] = []
        for hypothesis, worktree in active_hypotheses:
            formats = rb.formats_for_hypothesis(hypothesis, format_filter)
            sdk_build_dir = worktree / args.sdk_build_dir
            tool_lib = rb.find_tool_library(sdk_build_dir, args.dry_run)
            sdk_lib = rb.find_sdk_library(sdk_build_dir, args.dry_run)

            log(f"==> Hypothesis: {hypothesis}")
            log(f"    worktree: {worktree}")
            log(f"    formats: {','.join(formats)}")

            per_hypothesis_rows: list[list[str]] = []
            for output_format in formats:
                output_dir = results_dir / "raw" / "mistral" / hypothesis / output_format
                rb.clean_dir(output_dir, args.dry_run)
                profile_env = rb.build_profile_env(
                    build_dir=sdk_build_dir,
                    tool_lib=tool_lib,
                    sdk_lib=sdk_lib,
                    output_dir=output_dir,
                    output_name=f"mistral_{output_format}",
                    output_format=output_format,
                    counter_line=counter_line,
                )
                command = [args.python_exe, str(mistral_script)]
                exec_sec, log_file = run_mistral_profile(
                    hypothesis=hypothesis,
                    output_format=output_format,
                    output_dir=output_dir,
                    command=command,
                    profile_env=profile_env,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )
                rb.write_profile_metadata(
                    output_dir / rb.PROFILE_METADATA_FILE,
                    hypothesis=hypothesis,
                    dispatches=DEFAULT_PROFILE_MARKER,
                    output_format=output_format,
                    exec_sec=exec_sec,
                    dry_run=args.dry_run,
                )

                metrics = rb.measure_format(output_format, output_dir, exec_sec, args.dry_run)
                log(
                    f"    {output_format}: bytes={metrics.output_bytes} "
                    f"exec={format_seconds(metrics.exec_sec)}s "
                    f"write={format_seconds(metrics.write_sec)}s "
                    f"read={format_seconds(metrics.read_sec)}s "
                    f"rows={metrics.row_count} "
                    f"unique_counters={metrics.unique_counters}"
                )

                per_hypothesis_rows.append(
                    [
                        hypothesis,
                        output_format,
                        DEFAULT_MODEL_TAG,
                        str(counter_file),
                        str(unique_counter_count),
                        str(metrics.output_bytes),
                        rb.human_size(metrics.output_bytes),
                        format_seconds(metrics.exec_sec),
                        format_seconds(metrics.write_sec),
                        format_seconds(metrics.read_sec),
                        str(metrics.row_count),
                        str(metrics.unique_counters),
                        str(output_dir),
                        str(log_file),
                    ]
                )
                consolidated_rows.append(
                    [
                        hypothesis,
                        output_format,
                        str(metrics.output_bytes),
                        rb.human_size(metrics.output_bytes),
                        format_seconds(metrics.exec_sec),
                        format_seconds(metrics.write_sec),
                        format_seconds(metrics.read_sec),
                        str(metrics.row_count),
                        str(metrics.unique_counters),
                    ]
                )

            hypothesis_file = results_dir / f"mistral_{hypothesis}.csv"
            write_csv(
                hypothesis_file,
                PER_HYPOTHESIS_HEADER,
                per_hypothesis_rows,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                log(f"    wrote: {hypothesis_file}")

        consolidated_file = results_dir / "mistral_results.csv"
        write_csv(
            consolidated_file,
            CONSOLIDATED_HEADER,
            consolidated_rows,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            log(f"==> Wrote consolidated results: {consolidated_file}")

        log("")
        log(f"Done. Results are in: {results_dir}")
        return 0
    except rb.BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Benchmark Mistral with the rocprofiler-sdk libraries from an install tree.

This is the single-SDK variant of benchmark_mistral.py. It keeps the same
LD_PRELOAD profiling path and measurement helpers, but points them at a local
rocprofiler-sdk install prefix instead of hypothesis worktrees.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Iterable

import run_benchmark as rb

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_COUNTER_FILE = SCRIPT_DIR / "benchmark" / "pmc_perf_0.txt"
DEFAULT_MISTRAL_SCRIPT = SCRIPT_DIR / "mistral.py"
DEFAULT_SDK_PREFIX = Path(
    "/home/abchoudh/dev/sqlite_zstd/projects/rocprofiler-sdk/install"
)
DEFAULT_FORMATS = "csv,rocpd"
DEFAULT_MODEL_TAG = "Mistral-7B-v0.1"
INSTALL_LABEL = "install"
PROFILE_MARKER = 0

PER_INSTALL_HEADER = (
    "hypothesis",
    "format",
    "model",
    "counter_file",
    "unique_counter_count",
    "output_bytes",
    "output_human",
    "exec_sec",
    "exec_min_sec",
    "exec_max_sec",
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


def parse_format_list(raw_value: str) -> list[str]:
    """Parse and validate selected output formats."""
    formats = rb.parse_csv_list(raw_value)
    if not formats:
        raise rb.BenchmarkError("No formats were provided")
    unsupported = [fmt for fmt in formats if fmt not in rb.SUPPORTED_FORMATS]
    if unsupported:
        raise rb.BenchmarkError(
            "Unsupported format(s): "
            + ",".join(unsupported)
            + f". Supported: {','.join(sorted(rb.SUPPORTED_FORMATS))}"
        )
    return formats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Create and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile mistral.py with rocprofiler-sdk-tool from an install prefix "
            "and compare output format size/read/write performance."
        )
    )
    parser.add_argument(
        "--sdk-prefix",
        default=str(DEFAULT_SDK_PREFIX),
        help=(
            "rocprofiler-sdk install prefix containing lib/librocprofiler-sdk.so "
            "and lib/rocprofiler-sdk/librocprofiler-sdk-tool.so "
            f"(default: {DEFAULT_SDK_PREFIX})"
        ),
    )
    parser.add_argument(
        "--formats",
        default=DEFAULT_FORMATS,
        help=(
            "Comma-separated format list "
            f"(subset of {','.join(sorted(rb.SUPPORTED_FORMATS))}; "
            f"default: {DEFAULT_FORMATS})"
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
        "--iterations",
        type=int,
        default=3,
        help="Timed iterations per format (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed warmup iterations per format (default: 1)",
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
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        raise rb.BenchmarkError("--iterations must be > 0")
    if args.warmup < 0:
        raise rb.BenchmarkError("--warmup must be >= 0")
    return args


def read_model_path(mistral_script: Path) -> Path | None:
    """Extract the literal MODEL_PATH value from mistral.py, if present."""
    source = mistral_script.read_text(encoding="utf-8")
    match = re.search(r"^MODEL_PATH\s*=\s*['\"]([^'\"]+)['\"]", source, re.MULTILINE)
    if match is None:
        return None
    return Path(match.group(1)).expanduser()


def require_optional_measurement_deps(formats: list[str], *, dry_run: bool) -> None:
    """Fail early when a requested format cannot be measured locally."""
    if dry_run:
        return
    if any(fmt in formats for fmt in ("feather", "parquet")):
        if importlib.util.find_spec("pyarrow") is None:
            raise rb.BenchmarkError(
                "pyarrow is required to measure feather/parquet outputs"
            )
    if "rocksdb" in formats and importlib.util.find_spec("rocksdb") is None:
        raise rb.BenchmarkError("python-rocksdb is required to measure rocksdb outputs")


def validate_inputs(
    *,
    sdk_prefix: Path,
    counter_file: Path,
    mistral_script: Path,
    formats: list[str],
    dry_run: bool,
) -> tuple[list[str], Path, Path]:
    """Validate static inputs and resolve the install libraries."""
    if not mistral_script.is_file():
        raise rb.BenchmarkError(f"Mistral script not found: {mistral_script}")
    model_path = read_model_path(mistral_script)
    if model_path is not None and not dry_run and not model_path.exists():
        raise rb.BenchmarkError(f"Mistral model path not found: {model_path}")

    counter_names = rb.parse_counter_file(counter_file)
    tool_lib = rb.find_tool_library(sdk_prefix, dry_run=False)
    sdk_lib = rb.find_sdk_library(sdk_prefix, dry_run=False)
    require_optional_measurement_deps(formats, dry_run=dry_run)
    return counter_names, tool_lib, sdk_lib


def profile_env_for_run(
    *,
    sdk_prefix: Path,
    tool_lib: Path,
    sdk_lib: Path,
    output_dir: Path,
    output_format: str,
    output_name: str,
    counter_line: str,
) -> dict[str, str]:
    """Build the LD_PRELOAD profiling environment for one run."""
    return rb.build_profile_env(
        build_dir=sdk_prefix,
        tool_lib=tool_lib,
        sdk_lib=sdk_lib,
        output_dir=output_dir,
        output_name=output_name,
        output_format=output_format,
        counter_line=counter_line,
    )


def run_mistral_profile(
    *,
    output_format: str,
    output_dir: Path,
    log_file: Path,
    command: list[str],
    profile_env: dict[str, str],
    dry_run: bool,
    verbose: bool,
) -> float:
    """Run one profiled mistral invocation and return elapsed seconds."""
    if dry_run:
        log(f"[DRY RUN] format={output_format}")
        log(f"[DRY RUN]   command: {format_command(command)}")
        log(f"[DRY RUN]   output_dir: {output_dir}")
        log(f"[DRY RUN]   log_file: {log_file}")
        return 0.0

    rb.ensure_dir(output_dir, dry_run=False)
    if verbose:
        log(f"[profile] format={output_format}")
        log(f"[profile] command: {format_command(command)}")
        log(f"[profile] log: {log_file}")

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
        raise rb.BenchmarkError(
            "Mistral profile run failed with exit code "
            f"{completed.returncode} for format={output_format}. Log: {log_file}"
        )
    return elapsed


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
    try:
        args = parse_args(argv)
        cwd = Path.cwd()
        sdk_prefix = resolve_path(args.sdk_prefix, cwd)
        counter_file = resolve_path(args.counter_file, cwd)
        results_dir = resolve_path(args.results_dir, cwd)
        mistral_script = resolve_path(args.mistral_script, cwd)
        formats = parse_format_list(args.formats)

        counter_names, tool_lib, sdk_lib = validate_inputs(
            sdk_prefix=sdk_prefix,
            counter_file=counter_file,
            mistral_script=mistral_script,
            formats=formats,
            dry_run=args.dry_run,
        )
        unique_counter_count = len(set(counter_names))
        counter_line = "pmc: " + " ".join(counter_names)

        if not args.dry_run:
            rb.ensure_dir(results_dir, dry_run=False)

        log(f"==> SDK prefix: {sdk_prefix}")
        log(f"==> Tool library: {tool_lib}")
        log(f"==> SDK library: {sdk_lib}")
        log(f"==> Counter file: {counter_file}")
        log(f"==> Counters ({unique_counter_count}): {' '.join(counter_names)}")
        log(f"==> Mistral script: {mistral_script}")
        log(f"==> Model: {DEFAULT_MODEL_TAG}")
        log(f"==> Formats: {','.join(formats)}")
        log(f"==> Warmup iterations: {args.warmup}")
        log(f"==> Timed iterations: {args.iterations}")
        if args.dry_run:
            log("==> Dry run: enabled")
        log("")

        command = [args.python_exe, str(mistral_script)]
        per_install_rows: list[list[str]] = []
        consolidated_rows: list[list[str]] = []

        for output_format in formats:
            output_dir = results_dir / "raw" / "mistral" / INSTALL_LABEL / output_format
            warmup_dir = (
                results_dir
                / "raw"
                / "mistral"
                / INSTALL_LABEL
                / f"{output_format}_warmup"
            )

            rb.clean_dir(output_dir, args.dry_run)
            if args.warmup:
                rb.clean_dir(warmup_dir, args.dry_run)
                for warmup_idx in range(args.warmup):
                    warmup_env = profile_env_for_run(
                        sdk_prefix=sdk_prefix,
                        tool_lib=tool_lib,
                        sdk_lib=sdk_lib,
                        output_dir=warmup_dir,
                        output_format=output_format,
                        output_name=f"mistral_{output_format}_warmup_{warmup_idx}",
                        counter_line=counter_line,
                    )
                    run_mistral_profile(
                        output_format=output_format,
                        output_dir=warmup_dir,
                        log_file=warmup_dir / f"warmup_{warmup_idx}.log",
                        command=command,
                        profile_env=warmup_env,
                        dry_run=args.dry_run,
                        verbose=args.verbose,
                    )

            exec_times: list[float] = []
            log_files: list[Path] = []
            for iteration_idx in range(args.iterations):
                timed_env = profile_env_for_run(
                    sdk_prefix=sdk_prefix,
                    tool_lib=tool_lib,
                    sdk_lib=sdk_lib,
                    output_dir=output_dir,
                    output_format=output_format,
                    output_name=f"mistral_{output_format}_{iteration_idx}",
                    counter_line=counter_line,
                )
                log_file = output_dir / f"profile_{iteration_idx}.log"
                exec_times.append(
                    run_mistral_profile(
                        output_format=output_format,
                        output_dir=output_dir,
                        log_file=log_file,
                        command=command,
                        profile_env=timed_env,
                        dry_run=args.dry_run,
                        verbose=args.verbose,
                    )
                )
                log_files.append(log_file)

            exec_sec = mean(exec_times)
            rb.write_profile_metadata(
                output_dir / rb.PROFILE_METADATA_FILE,
                hypothesis=INSTALL_LABEL,
                dispatches=PROFILE_MARKER,
                output_format=output_format,
                exec_sec=exec_sec,
                dry_run=args.dry_run,
            )

            metrics = rb.measure_format(output_format, output_dir, exec_sec, args.dry_run)
            exec_min = min(exec_times)
            exec_max = max(exec_times)
            log(
                f"    {output_format}: bytes={metrics.output_bytes} "
                f"exec={format_seconds(metrics.exec_sec)}s "
                f"min={format_seconds(exec_min)}s "
                f"max={format_seconds(exec_max)}s "
                f"write={format_seconds(metrics.write_sec)}s "
                f"read={format_seconds(metrics.read_sec)}s "
                f"rows={metrics.row_count} "
                f"unique_counters={metrics.unique_counters}"
            )

            per_install_rows.append(
                [
                    INSTALL_LABEL,
                    output_format,
                    DEFAULT_MODEL_TAG,
                    str(counter_file),
                    str(unique_counter_count),
                    str(metrics.output_bytes),
                    rb.human_size(metrics.output_bytes),
                    format_seconds(metrics.exec_sec),
                    format_seconds(exec_min),
                    format_seconds(exec_max),
                    format_seconds(metrics.write_sec),
                    format_seconds(metrics.read_sec),
                    str(metrics.row_count),
                    str(metrics.unique_counters),
                    str(output_dir),
                    ";".join(str(path) for path in log_files),
                ]
            )
            consolidated_rows.append(
                [
                    INSTALL_LABEL,
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

        install_file = results_dir / "mistral_install.csv"
        write_csv(
            install_file,
            PER_INSTALL_HEADER,
            per_install_rows,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            log(f"    wrote: {install_file}")

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

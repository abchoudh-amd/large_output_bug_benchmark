#!/usr/bin/env python3
"""Benchmark harness for large output hypothesis experiments.

This script replicates the behavior of run_benchmark.sh while adding granular
subcommands and using rocprofiler-sdk-tool directly (via LD_PRELOAD) instead of
invoking rocprofv3.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SRC = SCRIPT_DIR / "benchmark" / "synthetic_dispatch_benchmark.cpp"
BENCHMARK_BUILD_DIR = SCRIPT_DIR / "benchmark" / "build"
BENCHMARK_BIN = BENCHMARK_BUILD_DIR / "synthetic_dispatch_benchmark"
DEFAULT_COUNTER_FILE = SCRIPT_DIR / "benchmark" / "pmc_perf_0.txt"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_SDK_BUILD_DIR_NAME = "build-hypothesis-rocprofiler-sdk"
DEFAULT_DISPATCHES = "100,10000,100000"
CONSOLIDATED_HEADER = ("name", "num_dispatch", "size", "read_time", "write_time", "exec_time")
PROFILE_METADATA_FILE = "profile_metadata.json"
WRITE_TIMING_FILE = "write_timing.json"

KNOWN_HYPOTHESES = (
    "baseline",
    "counter_rows_per_dispatch",
    "csv_per_dispatch_flushed",
    "rocpd_wide_pmc",
    "rocpd_aggregated_pmc",
    "feather",
    "parquet",
    "parquet_per_instance",
    "rocksdb",
    "rocksdb_aggregated",
    "lz4",
    "zstd",
)
FORMAT_MAP = {
    "baseline": ("csv", "rocpd"),
    "counter_rows_per_dispatch": ("csv",),
    "csv_per_dispatch_flushed": ("csv",),
    "rocpd_wide_pmc": ("rocpd",),
    "rocpd_aggregated_pmc": ("rocpd",),
    "feather": ("feather",),
    "parquet": ("parquet",),
    "parquet_per_instance": ("parquet",),
    "rocksdb": ("rocksdb",),
    "rocksdb_aggregated": ("rocksdb",),
    "lz4": ("rocpd",),
    "zstd": ("rocpd",),
}
FORMAT_GLOB_MAP = {
    "csv": ("*.csv",),
    "rocpd": ("*.db",),
    "feather": ("*.feather",),
    "parquet": ("*.parquet",),
    "rocksdb": ("*.rocksdb",),
}
CONSOLIDATED_NAME_MAP = {
    ("baseline", "csv"): "csv_baseline",
    ("baseline", "rocpd"): "rocpd_baseline",
    ("counter_rows_per_dispatch", "csv"): "csv_per_dispatch",
    ("csv_per_dispatch_flushed", "csv"): "csv_per_dispatch_flushed",
    ("rocpd_wide_pmc", "rocpd"): "rocpd_wide_pmc",
    ("rocpd_aggregated_pmc", "rocpd"): "rocpd_aggregated_pmc",
    ("feather", "feather"): "feather",
    ("parquet", "parquet"): "parquet",
    ("parquet_per_instance", "parquet"): "parquet_per_instance",
    ("rocksdb", "rocksdb"): "rocksdb",
    ("rocksdb_aggregated", "rocksdb"): "rocksdb_aggregated",
    ("lz4", "rocpd"): "rocpd_lz4",
    ("zstd", "rocpd"): "rocpd_zstd",
}
SUPPORTED_FORMATS = set(FORMAT_GLOB_MAP)


class BenchmarkError(RuntimeError):
    """Error raised for expected command/runtime failures."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime options shared by command handlers."""

    hypotheses: list[str]
    dispatches: list[int]
    counter_file: Path | None
    counter_names: list[str]
    unique_counter_count: int
    results_dir: Path
    sdk_build_dir_name: str
    dry_run: bool
    verbose: bool


@dataclass(frozen=True)
class FormatMetrics:
    """Measured metrics for one output format."""

    output_bytes: int
    exec_sec: float
    read_sec: float
    write_sec: float
    row_count: int
    unique_counters: int
    output_dir: Path
    decompress_sec: float = 0.0


def log(message: str) -> None:
    """Print an informational message."""
    print(message)


def warn(message: str) -> None:
    """Print a warning message."""
    print(f"WARNING: {message}", file=sys.stderr)


def die(message: str) -> None:
    """Raise a benchmark error."""
    raise BenchmarkError(message)


def parse_csv_list(raw_value: str) -> list[str]:
    """Parse a comma separated list while preserving order."""
    result: list[str] = []
    for value in raw_value.split(","):
        token = value.strip()
        if token and token not in result:
            result.append(token)
    return result


def read_cmake_cache_entry(cache_file: Path, key: str) -> str | None:
    """Read a single key from CMakeCache.txt."""
    prefix = f"{key}:"
    for raw_line in cache_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.startswith(prefix):
            continue
        _entry, _separator, value = raw_line.partition("=")
        if not value:
            return None
        return value.strip()
    return None


def validate_sdk_build_dir_source(
    build_dir: Path, expected_source_dir: Path, *, hypothesis: str | None = None
) -> None:
    """Fail fast when build_dir cache points at a different source tree."""
    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.is_file():
        return

    configured_source = read_cmake_cache_entry(cache_file, "CMAKE_HOME_DIRECTORY")
    if not configured_source:
        return

    configured_path = Path(configured_source).expanduser().resolve()
    expected_path = expected_source_dir.expanduser().resolve()
    if configured_path == expected_path:
        return

    if hypothesis is None:
        rebuild_hint = (
            f"rm -rf {build_dir} && cmake -S {expected_source_dir} -B {build_dir} "
            "-DCMAKE_BUILD_TYPE=Release"
        )
        scope = "SDK build directory source mismatch detected."
    else:
        rebuild_hint = (
            f"rm -rf {build_dir} && python3 run_benchmark.py build-sdk "
            f"--hypotheses {hypothesis}"
        )
        scope = f"SDK build directory source mismatch for hypothesis '{hypothesis}'."

    die(
        f"{scope} Build dir '{build_dir}' is configured for '{configured_path}', "
        f"expected '{expected_path}'. Recreate the build directory with: {rebuild_hint}"
    )


def parse_dispatches(raw_value: str) -> list[int]:
    """Parse and validate dispatch counts from a comma separated list."""
    tokens = parse_csv_list(raw_value)
    if not tokens:
        die("No dispatch values provided")
    dispatches: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise BenchmarkError(f"Dispatch must be an integer, got: {token}") from exc
        if value <= 0:
            die(f"Dispatch must be > 0, got: {value}")
        dispatches.append(value)
    return dispatches


def parse_counter_file(counter_file: Path) -> list[str]:
    """Parse the first `pmc:` line in the counter file."""
    if not counter_file.is_file():
        die(f"Counter file not found: {counter_file}")

    for raw_line in counter_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
        if line.lower().startswith("pmc:"):
            counters = line.split(":", 1)[1].strip().split()
            if counters:
                return counters
            break

    die(f"No counters found in counter file: {counter_file}")
    return []


def hypothesis_pairs(hypothesis_csv: str | None) -> list[tuple[str, Path]]:
    """Resolve hypothesis names to worktree directories."""
    if hypothesis_csv:
        requested = parse_csv_list(hypothesis_csv)
        for hypothesis in requested:
            if hypothesis not in KNOWN_HYPOTHESES:
                die(f"Unknown hypothesis in --hypotheses: {hypothesis}")
    else:
        requested = list(KNOWN_HYPOTHESES)

    active: list[tuple[str, Path]] = []
    for hypothesis in requested:
        hypothesis_dir = SCRIPT_DIR / hypothesis
        sdk_dir = hypothesis_dir / "projects" / "rocprofiler-sdk"
        if not hypothesis_dir.is_dir():
            warn(f"Hypothesis directory not found, skipping: {hypothesis_dir}")
            continue
        if not sdk_dir.is_dir():
            warn(f"Missing projects/rocprofiler-sdk, skipping: {hypothesis_dir}")
            continue
        active.append((hypothesis, hypothesis_dir))

    if not active:
        die("No runnable hypothesis worktrees were found")
    return active


def parse_format_filter(raw_value: str | None) -> list[str] | None:
    """Parse and validate optional format filter."""
    if raw_value is None:
        return None
    requested = parse_csv_list(raw_value)
    if not requested:
        die("No values provided to --formats")
    for fmt in requested:
        if fmt not in SUPPORTED_FORMATS:
            die(f"Unsupported format in --formats: {fmt}")
    return requested


def formats_for_hypothesis(hypothesis: str, requested: list[str] | None) -> list[str]:
    """Get valid formats for one hypothesis with optional filter applied."""
    defaults = list(FORMAT_MAP[hypothesis])
    if requested is None:
        return defaults

    selected: list[str] = []
    for fmt in requested:
        if fmt in defaults and fmt not in selected:
            selected.append(fmt)
    if not selected:
        die(
            f"--formats filter has no valid entries for hypothesis '{hypothesis}'. "
            f"Allowed: {','.join(defaults)}"
        )
    return selected


def human_size(num_bytes: int | float) -> str:
    """Format bytes with binary units matching bash script output."""
    value = float(num_bytes)
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{int(num_bytes)} B"


def append_env_list(env: dict[str, str], key: str, value: str, sep: str = ":") -> None:
    """Append path-like values to an env variable without duplicates."""
    current = env.get(key, "")
    existing = [item for item in current.split(sep) if item]
    for item in value.split(sep):
        if item and item not in existing:
            existing.append(item)
    if existing:
        env[key] = sep.join(existing)


def format_seconds(seconds: float) -> str:
    """Format time in seconds."""
    return f"{seconds:.6f}"


def format_command(command: Iterable[os.PathLike[str] | str]) -> str:
    """Return a shell-escaped command string for logging."""
    return shlex.join([str(part) for part in command])


def run_checked(
    command: list[str],
    *,
    dry_run: bool,
    verbose: bool,
    cwd: Path | None = None,
) -> None:
    """Run a command and raise on failure."""
    display = format_command(command)
    if dry_run:
        log(f"[DRY RUN] {display}")
        return
    if verbose:
        log(f"[exec] {display}")
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None, check=False)
    if completed.returncode != 0:
        die(f"Command failed with exit code {completed.returncode}: {display}")


def ensure_dir(path: Path, dry_run: bool) -> None:
    """Create a directory if needed."""
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def clean_dir(path: Path, dry_run: bool) -> None:
    """Remove and recreate a directory."""
    if dry_run:
        log(f"[DRY RUN] reset output directory: {path}")
        return
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def sum_bytes_for_glob(root_dir: Path, patterns: Iterable[str], dry_run: bool) -> int:
    """Sum file sizes for one or more glob patterns under root_dir."""
    if dry_run or not root_dir.exists():
        return 0
    total = 0
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root_dir.rglob(pattern):
            if path.is_file() and path not in seen:
                total += path.stat().st_size
                seen.add(path)
    return total


def discover_rocksdb_dirs(root_dir: Path) -> list[Path]:
    """Find RocksDB directory outputs under root_dir."""
    if not root_dir.exists():
        return []
    return sorted(path for path in root_dir.rglob("*.rocksdb") if path.is_dir())


def sum_bytes_for_rocksdb(root_dir: Path, dry_run: bool) -> int:
    """Sum file sizes for RocksDB directory outputs."""
    if dry_run:
        return 0
    total = 0
    for db_dir in discover_rocksdb_dirs(root_dir):
        for path in db_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    return total


def read_write_timing(output_dir: Path, format_name: str, dry_run: bool) -> float:
    """Read format write time from write_timing.json in output_dir."""
    if dry_run:
        return 0.0
    timing_file = output_dir / WRITE_TIMING_FILE
    if not timing_file.is_file():
        return 0.0
    try:
        payload = json.loads(timing_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        warn(f"Malformed write timing file: {timing_file}")
        return 0.0
    key = f"{format_name}_write_sec"
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        warn(f"Invalid write timing value for key '{key}' in {timing_file}")
        return 0.0


def measure_csv_read(root_dir: Path, dry_run: bool) -> tuple[float, int, int]:
    """Measure CSV read time, row count, and unique counter names."""
    if dry_run:
        return 0.0, 0, 0
    csv_files = sorted(path for path in root_dir.rglob("*.csv") if path.is_file())
    rows = 0
    unique_counters: set[str] = set()
    start = time.perf_counter()
    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows += 1
                if "Counter_Name" in row and row["Counter_Name"]:
                    unique_counters.add(str(row["Counter_Name"]))
                elif "counter_name" in row and row["counter_name"]:
                    unique_counters.add(str(row["counter_name"]))
    elapsed = time.perf_counter() - start
    return elapsed, rows, len(unique_counters)


def measure_rocpd_read(root_dir: Path, dry_run: bool) -> tuple[float, int, int]:
    """Measure RocPD DB read time, row count, and distinct counters."""
    if dry_run:
        return 0.0, 0, 0
    db_files = sorted(path for path in root_dir.rglob("*.db") if path.is_file())
    pmc_rows = 0
    unique_counters = 0
    start = time.perf_counter()
    for db_path in db_files:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_pmc_event%'"
            )
            pmc_tables = [row[0] for row in cur.fetchall()]

            for table_name in pmc_tables:
                try:
                    cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        pmc_rows += int(row[0])
                except sqlite3.Error:
                    continue
                try:
                    cur.execute(f'SELECT COUNT(DISTINCT pmc_id) FROM "{table_name}"')
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        unique_counters = max(unique_counters, int(row[0]))
                except sqlite3.Error:
                    continue
        finally:
            conn.close()
    elapsed = time.perf_counter() - start
    return elapsed, pmc_rows, unique_counters


_LZ4_BLOB_MAGIC = b"LZ4F"

LZ4_DECOMPRESS_VIEWS = (
    "rocpd_info_process",
    "rocpd_info_thread",
    "rocpd_info_agent",
    "rocpd_info_queue",
    "rocpd_info_stream",
    "rocpd_info_pmc",
    "rocpd_info_code_object",
    "rocpd_info_kernel_symbol",
    "rocpd_track",
    "rocpd_event",
    "rocpd_arg",
    "rocpd_pmc_event",
    "rocpd_region",
    "rocpd_sample",
    "rocpd_kernel_dispatch",
    "rocpd_memory_copy",
    "rocpd_memory_allocate",
)

ZSTD_COMPRESSED_TABLES = (
    "rocpd_string",
    "rocpd_info_process",
    "rocpd_info_thread",
    "rocpd_info_agent",
    "rocpd_info_queue",
    "rocpd_info_stream",
    "rocpd_region",
    "rocpd_sample",
    "rocpd_kernel_dispatch",
    "rocpd_memory_copy",
    "rocpd_memory_allocate",
)


def _lz4_blob_decompress(value: bytes | None) -> bytes:
    """Decompress an LZ4-frame BLOB written with the rocpd magic+length prefix."""
    if value is None:
        return b""
    payload = bytes(value)
    if not payload:
        return b""
    if len(payload) < 8 or payload[:4] != _LZ4_BLOB_MAGIC:
        raise ValueError("invalid lz4 blob magic")
    expected = struct.unpack("<I", payload[4:8])[0]
    import lz4.frame  # type: ignore[import-not-found]

    output = lz4.frame.decompress(payload[8:])
    if len(output) != expected:
        raise ValueError("lz4 length mismatch")
    return output


def detect_rocpd_compression(db_path: Path) -> str | None:
    """Return 'lz4', 'zstd', or None based on rocpd_metadata tags in db_path."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'rocpd_metadata%'"
            )
            metadata_tables = [row[0] for row in cur.fetchall()]
        except sqlite3.Error:
            return None
        for table_name in metadata_tables:
            try:
                cur.execute(f'SELECT tag, value FROM "{table_name}"')
                tags = {str(row[0]): str(row[1]) for row in cur.fetchall()}
            except sqlite3.Error:
                continue
            compression = tags.get("compression", "").lower()
            if compression.startswith("lz4"):
                return "lz4"
            sqlite_zstd = tags.get("sqlite_zstd", "").lower()
            if sqlite_zstd in ("1", "true", "enabled", "on"):
                return "zstd"
    finally:
        conn.close()
    return None


def find_sqlite_zstd_lib(sdk_prefix: Path | None) -> Path | None:
    """Locate libsqlite_zstd.so via env var or known SDK install layout."""
    env_path = os.environ.get("ROCPROFILER_SQLITE_ZSTD_LIBPATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate
    if sdk_prefix is None:
        return None
    for libdir in ("lib", "lib64"):
        candidate = sdk_prefix / libdir / "rocprofiler-sdk-rocpd" / "libsqlite_zstd.so"
        if candidate.is_file():
            return candidate
    return None


def register_rocpd_compression(
    conn: sqlite3.Connection,
    mode: str,
    *,
    sdk_prefix: Path | None = None,
) -> bool:
    """Register lz4_decompress or load libsqlite_zstd.so on conn for this mode."""
    if mode == "lz4":
        try:
            import lz4.frame  # type: ignore[import-not-found, unused-ignore]  # noqa: F401
        except ModuleNotFoundError:
            warn(
                "lz4 python package missing; skipping decompression timing for LZ4 db. "
                "Install with: pip install 'lz4>=4.0'"
            )
            return False
        try:
            conn.create_function(
                "lz4_decompress", 1, _lz4_blob_decompress, deterministic=True
            )
        except TypeError:
            conn.create_function("lz4_decompress", 1, _lz4_blob_decompress)
        return True
    if mode == "zstd":
        lib_path = find_sqlite_zstd_lib(sdk_prefix)
        if lib_path is None:
            warn(
                "libsqlite_zstd.so not found "
                f"(sdk_prefix={sdk_prefix}, ROCPROFILER_SQLITE_ZSTD_LIBPATH unset); "
                "skipping decompression timing for ZSTD db"
            )
            return False
        try:
            conn.enable_load_extension(True)
            conn.load_extension(str(lib_path))
        except (
            sqlite3.OperationalError,
            sqlite3.NotSupportedError,
            AttributeError,
        ) as exc:
            warn(f"failed to load libsqlite_zstd.so from {lib_path}: {exc}")
            return False
        return True
    return False


def measure_rocpd_decompression(
    root_dir: Path,
    *,
    sdk_prefix: Path | None = None,
    dry_run: bool = False,
) -> float:
    """Measure end-to-end decompression-while-reading time across rocpd .db files."""
    if dry_run or not root_dir.exists():
        return 0.0
    db_files = sorted(path for path in root_dir.rglob("*.db") if path.is_file())
    total_elapsed = 0.0
    for db_path in db_files:
        mode = detect_rocpd_compression(db_path)
        if mode is None:
            continue
        conn = sqlite3.connect(str(db_path))
        try:
            if not register_rocpd_compression(conn, mode, sdk_prefix=sdk_prefix):
                continue
            views = LZ4_DECOMPRESS_VIEWS if mode == "lz4" else ZSTD_COMPRESSED_TABLES
            cur = conn.cursor()
            start = time.perf_counter()
            for view_name in views:
                try:
                    cur.execute(f'SELECT * FROM "{view_name}"')
                except sqlite3.Error:
                    continue
                while True:
                    rows = cur.fetchmany(1024)
                    if not rows:
                        break
            total_elapsed += time.perf_counter() - start
        finally:
            conn.close()
    return total_elapsed


def measure_feather_read(root_dir: Path, dry_run: bool) -> tuple[float, int, int]:
    """Measure Feather read time, row count, and unique counters."""
    if dry_run:
        return 0.0, 0, 0
    try:
        import pyarrow.feather as feather  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise BenchmarkError(
            f"pyarrow is required for feather read measurement: {exc}"
        ) from exc

    files = sorted(path for path in root_dir.rglob("*.feather") if path.is_file())
    rows = 0
    unique_counters: set[str] = set()
    start = time.perf_counter()
    for path in files:
        table = feather.read_table(path)
        rows += table.num_rows
        names = table.column_names
        if "Counter_Name" in names:
            for value in table.column("Counter_Name").to_pylist():
                if value is not None:
                    unique_counters.add(str(value))
        elif "counter_name" in names:
            for value in table.column("counter_name").to_pylist():
                if value is not None:
                    unique_counters.add(str(value))
    elapsed = time.perf_counter() - start
    return elapsed, rows, len(unique_counters)


def measure_parquet_read(root_dir: Path, dry_run: bool) -> tuple[float, int, int]:
    """Measure Parquet read time, row count, and unique counters."""
    if dry_run:
        return 0.0, 0, 0
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise BenchmarkError(
            f"pyarrow is required for parquet read measurement: {exc}"
        ) from exc

    files = sorted(path for path in root_dir.rglob("*.parquet") if path.is_file())
    rows = 0
    unique_counters: set[str] = set()
    start = time.perf_counter()
    for path in files:
        table = parquet.read_table(path)
        rows += table.num_rows
        names = table.column_names
        if "Counter_Name" in names:
            for value in table.column("Counter_Name").to_pylist():
                if value is not None:
                    unique_counters.add(str(value))
        elif "counter_name" in names:
            for value in table.column("counter_name").to_pylist():
                if value is not None:
                    unique_counters.add(str(value))
    elapsed = time.perf_counter() - start
    return elapsed, rows, len(unique_counters)


def parse_rocksdb_counter_blob(payload: bytes) -> tuple[int, set[int]]:
    """Decode RocksDB value payload emitted by rocprofiler-sdk-tool."""
    header_fmt = "<IQQQQQQQQQI"
    entry_fmt = "<Qd"
    header_size = struct.calcsize(header_fmt)
    entry_size = struct.calcsize(entry_fmt)
    if len(payload) < header_size:
        return 0, set()

    (
        _version,
        _thread_id,
        _stream_id,
        _correlation_id,
        _start_timestamp,
        _end_timestamp,
        _dispatch_id,
        _agent_id,
        _queue_id,
        _kernel_id,
        counter_count,
    ) = struct.unpack_from(header_fmt, payload, 0)

    available = max(0, len(payload) - header_size) // entry_size
    valid_count = min(counter_count, available)
    unique_counters: set[int] = set()
    offset = header_size
    for _ in range(valid_count):
        counter_id, _value = struct.unpack_from(entry_fmt, payload, offset)
        unique_counters.add(counter_id)
        offset += entry_size

    return int(valid_count), unique_counters


def measure_rocksdb_read(root_dir: Path, dry_run: bool) -> tuple[float, int, int]:
    """Measure RocksDB read time, decoded counter rows, and unique counters."""
    if dry_run:
        return 0.0, 0, 0
    try:
        import rocksdb  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        warn(
            "python-rocksdb is not installed; skipping RocksDB read measurement "
            f"(detail: {exc})"
        )
        return 0.0, 0, 0

    if not hasattr(rocksdb, "DB") or not hasattr(rocksdb, "Options"):
        warn(
            "Imported 'rocksdb' module does not expose DB/Options API "
            "(likely shadowed by local rocksdb/ directory); "
            "skipping RocksDB read measurement"
        )
        return 0.0, 0, 0

    db_dirs = discover_rocksdb_dirs(root_dir)
    rows = 0
    unique_counters: set[int] = set()
    start = time.perf_counter()
    for db_dir in db_dirs:
        options = rocksdb.Options(create_if_missing=False)
        try:
            db = rocksdb.DB(str(db_dir), options, read_only=True)
        except TypeError:
            # Some python-rocksdb builds do not expose read_only.
            db = rocksdb.DB(str(db_dir), options)
        try:
            iterator = db.iteritems()
            iterator.seek_to_first()
            for _key, value in iterator:
                decoded_rows, decoded_unique = parse_rocksdb_counter_blob(bytes(value))
                rows += decoded_rows
                unique_counters.update(decoded_unique)
        finally:
            del db
    elapsed = time.perf_counter() - start
    return elapsed, rows, len(unique_counters)


def measure_format(
    format_name: str,
    output_dir: Path,
    exec_sec: float,
    dry_run: bool,
) -> FormatMetrics:
    """Measure read performance and output size for a format directory."""
    if format_name == "rocksdb":
        output_bytes = sum_bytes_for_rocksdb(output_dir, dry_run)
    else:
        output_bytes = sum_bytes_for_glob(output_dir, FORMAT_GLOB_MAP[format_name], dry_run)
    write_sec = read_write_timing(output_dir, format_name, dry_run)
    if format_name == "csv":
        read_sec, rows, unique = measure_csv_read(output_dir, dry_run)
    elif format_name == "rocpd":
        read_sec, rows, unique = measure_rocpd_read(output_dir, dry_run)
    elif format_name == "feather":
        read_sec, rows, unique = measure_feather_read(output_dir, dry_run)
    elif format_name == "parquet":
        read_sec, rows, unique = measure_parquet_read(output_dir, dry_run)
    elif format_name == "rocksdb":
        read_sec, rows, unique = measure_rocksdb_read(output_dir, dry_run)
    else:
        die(f"Unsupported format: {format_name}")
    return FormatMetrics(
        output_bytes=output_bytes,
        exec_sec=exec_sec,
        read_sec=read_sec,
        write_sec=write_sec,
        row_count=rows,
        unique_counters=unique,
        output_dir=output_dir,
    )


def build_synthetic_benchmark(*, dry_run: bool, verbose: bool) -> None:
    """Compile benchmark/synthetic_dispatch_benchmark.cpp with hipcc."""
    if not BENCHMARK_SRC.is_file():
        die(f"Synthetic benchmark source not found: {BENCHMARK_SRC}")
    hipcc_bin = os.environ.get("HIPCC", "hipcc")
    if dry_run:
        log(
            "[DRY RUN] build synthetic benchmark: "
            f"{hipcc_bin} -O2 -std=c++17 {BENCHMARK_SRC} -o {BENCHMARK_BIN}"
        )
        return
    ensure_dir(BENCHMARK_BUILD_DIR, dry_run=False)
    run_checked(
        [hipcc_bin, "-O2", "-std=c++17", str(BENCHMARK_SRC), "-o", str(BENCHMARK_BIN)],
        dry_run=False,
        verbose=verbose,
    )


def build_rocprofiler_sdk(
    sdk_src_dir: Path,
    sdk_build_dir: Path,
    *,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Configure and build rocprofiler-sdk via CMake."""
    validate_sdk_build_dir_source(sdk_build_dir, sdk_src_dir)
    if dry_run:
        log(
            "[DRY RUN] configure rocprofiler-sdk: "
            f"cmake -S {sdk_src_dir} -B {sdk_build_dir} -DCMAKE_BUILD_TYPE=Release"
        )
        log(
            "[DRY RUN] build rocprofiler-sdk: "
            f"cmake --build {sdk_build_dir} --parallel {os.cpu_count() or 1}"
        )
        return
    run_checked(
        [
            "cmake",
            "-S",
            str(sdk_src_dir),
            "-B",
            str(sdk_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        dry_run=False,
        verbose=verbose,
    )
    run_checked(
        [
            "cmake",
            "--build",
            str(sdk_build_dir),
            "--parallel",
            str(os.cpu_count() or 1),
        ],
        dry_run=False,
        verbose=verbose,
    )


def _find_library_candidate(
    build_dir: Path, search_roots: list[Path], filename_prefix: str
) -> Path | None:
    """Find first matching library candidate by exact or versioned file."""
    for root in search_roots:
        if not root.exists():
            continue
        exact = root / f"{filename_prefix}.so"
        if exact.is_file():
            return exact
        matches = sorted(
            path
            for path in root.glob(f"{filename_prefix}.so*")
            if path.is_file() and path.name.startswith(f"{filename_prefix}.so")
        )
        if matches:
            return matches[0]
    if build_dir.exists():
        matches = sorted(
            path
            for path in build_dir.rglob(f"{filename_prefix}.so*")
            if path.is_file() and path.name.startswith(f"{filename_prefix}.so")
        )
        if matches:
            return matches[0]
    return None


def find_tool_library(build_dir: Path, dry_run: bool) -> Path:
    """Find librocprofiler-sdk-tool.so in the SDK build tree."""
    default_path = build_dir / "lib" / "rocprofiler-sdk" / "librocprofiler-sdk-tool.so"
    if dry_run:
        return default_path
    candidate = _find_library_candidate(
        build_dir,
        [
            build_dir / "lib" / "rocprofiler-sdk",
            build_dir / "lib64" / "rocprofiler-sdk",
        ],
        "librocprofiler-sdk-tool",
    )
    if candidate is None:
        die(f"Could not find librocprofiler-sdk-tool.so in build dir: {build_dir}")
    return candidate


def find_sdk_library(build_dir: Path, dry_run: bool) -> Path:
    """Find librocprofiler-sdk.so in the SDK build tree."""
    default_path = build_dir / "lib" / "librocprofiler-sdk.so"
    if dry_run:
        return default_path
    candidate = _find_library_candidate(
        build_dir,
        [build_dir / "lib", build_dir / "lib64"],
        "librocprofiler-sdk",
    )
    if candidate is None:
        die(f"Could not find librocprofiler-sdk.so in build dir: {build_dir}")
    return candidate


def build_profile_env(
    *,
    build_dir: Path,
    tool_lib: Path,
    sdk_lib: Path,
    output_dir: Path,
    output_name: str,
    output_format: str,
    counter_line: str,
) -> dict[str, str]:
    """Create profiling environment equivalent to rocprofv3 env setup."""
    env = dict(os.environ)
    env["ROCPROFILER_LIBRARY_CTOR"] = "1"
    append_env_list(env, "ROCP_TOOL_LIBRARIES", str(tool_lib))
    append_env_list(env, "LD_PRELOAD", f"{tool_lib}:{sdk_lib}")
    append_env_list(env, "LD_LIBRARY_PATH", str(build_dir / "lib"))

    env["ROCPROF_OUTPUT_PATH"] = str(output_dir)
    env["ROCPROF_OUTPUT_FILE_NAME"] = output_name
    env["ROCPROF_OUTPUT_FORMAT"] = output_format.upper()
    env["ROCPROF_KERNEL_TRACE"] = "0"
    env["ROCPROF_COUNTER_COLLECTION"] = "1"
    env["ROCPROF_COUNTERS"] = counter_line
    return env


def write_profile_metadata(
    metadata_path: Path,
    *,
    hypothesis: str,
    dispatches: int,
    output_format: str,
    exec_sec: float,
    dry_run: bool,
) -> None:
    """Write profile metadata used by measure/report commands."""
    payload = {
        "hypothesis": hypothesis,
        "dispatches": dispatches,
        "format": output_format,
        "exec_sec": exec_sec,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    if dry_run:
        log(f"[DRY RUN] write profile metadata: {metadata_path}")
        return
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_profile_metadata(metadata_path: Path) -> float:
    """Read execution time from profile metadata, falling back to 0.0."""
    if not metadata_path.is_file():
        return 0.0
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        warn(f"Malformed profile metadata: {metadata_path}")
        return 0.0
    try:
        return float(payload.get("exec_sec", 0.0))
    except (TypeError, ValueError):
        return 0.0


def run_profile(
    *,
    hypothesis: str,
    dispatches: int,
    output_format: str,
    output_dir: Path,
    benchmark_bin: Path,
    grid_size: int,
    block_size: int,
    spin_iterations: int,
    profile_env: dict[str, str],
    dry_run: bool,
    verbose: bool,
) -> float:
    """Run benchmark with direct rocprofiler-sdk-tool instrumentation."""
    command = [
        str(benchmark_bin),
        str(dispatches),
        str(grid_size),
        str(block_size),
        str(spin_iterations),
    ]
    if dry_run:
        log(
            f"[DRY RUN] hypothesis={hypothesis} dispatches={dispatches} format={output_format}"
        )
        log(f"[DRY RUN]   command: {format_command(command)}")
        log(f"[DRY RUN]   output_dir: {output_dir}")
        return 0.0

    ensure_dir(output_dir, dry_run=False)
    log_file = output_dir / f"profile_{output_format}.log"
    if verbose:
        log(f"[profile] hypothesis={hypothesis} dispatches={dispatches} format={output_format}")
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
            f"Benchmark/profile run exited with code {completed.returncode} "
            f"for hypothesis={hypothesis} dispatches={dispatches} format={output_format}. "
            f"Log: {log_file}"
        )
    return elapsed


def build_per_dispatch_header_and_row(
    *,
    hypothesis: str,
    dispatches: int,
    counter_file: Path | None,
    unique_counter_count: int,
    metrics: dict[str, FormatMetrics],
) -> tuple[list[str], list[str]]:
    """Create per-dispatch CSV header and row."""
    header = ["worktree", "dispatches", "counter_file", "unique_counter_count"]
    row = [
        hypothesis,
        str(dispatches),
        str(counter_file) if counter_file else "",
        str(unique_counter_count),
    ]

    if "csv" in metrics:
        metric = metrics["csv"]
        header.extend(
            [
                "csv_output_bytes",
                "csv_exec_sec",
                "csv_write_sec",
                "csv_read_sec",
                "csv_rows",
                "csv_unique_counters",
                "csv_dir",
            ]
        )
        row.extend(
            [
                str(metric.output_bytes),
                format_seconds(metric.exec_sec),
                format_seconds(metric.write_sec),
                format_seconds(metric.read_sec),
                str(metric.row_count),
                str(metric.unique_counters),
                str(metric.output_dir),
            ]
        )

    if "rocpd" in metrics:
        metric = metrics["rocpd"]
        header.extend(
            [
                "rocpd_output_bytes",
                "rocpd_exec_sec",
                "rocpd_write_sec",
                "rocpd_read_sec",
                "rocpd_pmc_rows",
                "rocpd_unique_counters",
                "rocpd_dir",
            ]
        )
        row.extend(
            [
                str(metric.output_bytes),
                format_seconds(metric.exec_sec),
                format_seconds(metric.write_sec),
                format_seconds(metric.read_sec),
                str(metric.row_count),
                str(metric.unique_counters),
                str(metric.output_dir),
            ]
        )

    if "feather" in metrics:
        metric = metrics["feather"]
        header.extend(
            [
                "feather_output_bytes",
                "feather_exec_sec",
                "feather_write_sec",
                "feather_read_sec",
                "feather_rows",
                "feather_unique_counters",
                "feather_dir",
            ]
        )
        row.extend(
            [
                str(metric.output_bytes),
                format_seconds(metric.exec_sec),
                format_seconds(metric.write_sec),
                format_seconds(metric.read_sec),
                str(metric.row_count),
                str(metric.unique_counters),
                str(metric.output_dir),
            ]
        )

    if "parquet" in metrics:
        metric = metrics["parquet"]
        header.extend(
            [
                "parquet_output_bytes",
                "parquet_exec_sec",
                "parquet_write_sec",
                "parquet_read_sec",
                "parquet_rows",
                "parquet_unique_counters",
                "parquet_dir",
            ]
        )
        row.extend(
            [
                str(metric.output_bytes),
                format_seconds(metric.exec_sec),
                format_seconds(metric.write_sec),
                format_seconds(metric.read_sec),
                str(metric.row_count),
                str(metric.unique_counters),
                str(metric.output_dir),
            ]
        )

    if "rocksdb" in metrics:
        metric = metrics["rocksdb"]
        header.extend(
            [
                "rocksdb_output_bytes",
                "rocksdb_exec_sec",
                "rocksdb_write_sec",
                "rocksdb_read_sec",
                "rocksdb_rows",
                "rocksdb_unique_counters",
                "rocksdb_dir",
            ]
        )
        row.extend(
            [
                str(metric.output_bytes),
                format_seconds(metric.exec_sec),
                format_seconds(metric.write_sec),
                format_seconds(metric.read_sec),
                str(metric.row_count),
                str(metric.unique_counters),
                str(metric.output_dir),
            ]
        )
    return header, row


def write_single_row_csv(
    output_file: Path,
    header: list[str],
    row: list[str],
    dry_run: bool,
) -> None:
    """Write a two-row CSV file with header and one data row."""
    if dry_run:
        log(f"[DRY RUN] would write: {output_file}")
        return
    ensure_dir(output_file.parent, dry_run=False)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(row)


def build_consolidated_rows(
    hypothesis: str,
    dispatches: int,
    metrics: dict[str, FormatMetrics],
) -> list[list[str]]:
    """Convert per-format metrics into all_results.csv rows."""
    rows: list[list[str]] = []
    for output_format, metric in metrics.items():
        name = CONSOLIDATED_NAME_MAP.get((hypothesis, output_format))
        if name is None:
            die(
                f"No consolidated name mapping for hypothesis={hypothesis}, format={output_format}"
            )
        rows.append(
            [
                name,
                str(dispatches),
                human_size(metric.output_bytes),
                format_seconds(metric.read_sec),
                format_seconds(metric.write_sec),
                format_seconds(metric.exec_sec),
            ]
        )
    return rows


def write_consolidated_csv(
    results_dir: Path,
    consolidated_rows: list[list[str]],
    dry_run: bool,
) -> None:
    """Write all_results.csv."""
    output_file = results_dir / "all_results.csv"
    if dry_run:
        log(f"[DRY RUN] would write consolidated results: {output_file}")
        log(f"[DRY RUN] consolidated row count: {len(consolidated_rows)}")
        return
    ensure_dir(results_dir, dry_run=False)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(CONSOLIDATED_HEADER))
        writer.writerows(consolidated_rows)
    log(f"==> Wrote consolidated results: {output_file}")


def resolve_runtime_config(
    args: argparse.Namespace,
    *,
    require_counter: bool,
    require_dispatches: bool,
) -> RuntimeConfig:
    """Resolve and validate shared runtime configuration."""
    active = hypothesis_pairs(args.hypotheses)
    hypotheses = [name for name, _ in active]

    dispatches = parse_dispatches(args.dispatches) if require_dispatches else []

    counter_file: Path | None
    counter_names: list[str]
    unique_counter_count: int
    if require_counter:
        counter_file = Path(args.counter_file).expanduser()
        if not counter_file.is_absolute():
            counter_file = (Path.cwd() / counter_file).resolve()
        counter_names = parse_counter_file(counter_file)
        unique_counter_count = len(set(counter_names))
        log(f"==> Counter file: {counter_file}")
        log(f"==> Counters ({unique_counter_count}): {' '.join(counter_names)}")
    else:
        counter_file = None
        counter_names = []
        unique_counter_count = 0

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = (Path.cwd() / results_dir).resolve()

    return RuntimeConfig(
        hypotheses=hypotheses,
        dispatches=dispatches,
        counter_file=counter_file,
        counter_names=counter_names,
        unique_counter_count=unique_counter_count,
        results_dir=results_dir,
        sdk_build_dir_name=args.sdk_build_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def hypothesis_dir_map(hypothesis_csv: str | None) -> dict[str, Path]:
    """Resolve active hypothesis name to directory map."""
    return dict(hypothesis_pairs(hypothesis_csv))


def discover_dispatches(results_dir: Path, hypotheses: list[str]) -> list[int]:
    """Discover dispatch counts from existing raw output directories."""
    raw_root = results_dir / "raw"
    found: set[int] = set()
    for hypothesis in hypotheses:
        hypothesis_root = raw_root / hypothesis
        if not hypothesis_root.is_dir():
            continue
        for child in hypothesis_root.iterdir():
            if not child.is_dir() or not child.name.startswith("dispatches_"):
                continue
            suffix = child.name.removeprefix("dispatches_")
            if not suffix:
                continue
            try:
                value = int(suffix)
            except ValueError:
                continue
            if value > 0:
                found.add(value)
    return sorted(found)


def discover_dispatches_from_csvs(results_dir: Path, hypotheses: list[str]) -> list[int]:
    """Discover dispatch counts from existing per-dispatch CSV files."""
    if not results_dir.is_dir():
        return []
    found: set[int] = set()
    for hypothesis in hypotheses:
        prefix = f"{hypothesis}_"
        for path in results_dir.iterdir():
            if not path.is_file():
                continue
            if not path.name.startswith(prefix) or path.suffix != ".csv":
                continue
            suffix = path.name.removeprefix(prefix).removesuffix(".csv")
            if not suffix:
                continue
            try:
                value = int(suffix)
            except ValueError:
                continue
            if value > 0:
                found.add(value)
    return sorted(found)


def ensure_benchmark_binary(dry_run: bool) -> None:
    """Ensure benchmark binary exists and is executable."""
    if dry_run:
        return
    if not BENCHMARK_BIN.is_file() or not os.access(BENCHMARK_BIN, os.X_OK):
        die(
            "Benchmark binary missing or not executable. "
            "Run `./run_benchmark.py build-benchmark` first."
        )


def profile_and_measure_dispatch(
    *,
    hypothesis: str,
    worktree_dir: Path,
    dispatches: int,
    formats: list[str],
    config: RuntimeConfig,
    counter_line: str,
    grid_size: int,
    block_size: int,
    spin_iterations: int,
    do_profile: bool,
    perform_measurement: bool,
) -> dict[str, FormatMetrics]:
    """Run profile/measure pipeline for one hypothesis + dispatch count."""
    sdk_build_dir = worktree_dir / config.sdk_build_dir_name
    sdk_src_dir = worktree_dir / "projects" / "rocprofiler-sdk"
    tool_lib: Path | None = None
    sdk_lib: Path | None = None
    if do_profile:
        validate_sdk_build_dir_source(
            sdk_build_dir, sdk_src_dir, hypothesis=hypothesis
        )
        tool_lib = find_tool_library(sdk_build_dir, config.dry_run)
        sdk_lib = find_sdk_library(sdk_build_dir, config.dry_run)
    dispatch_root = (
        config.results_dir / "raw" / hypothesis / f"dispatches_{dispatches}"
    )

    metrics: dict[str, FormatMetrics] = {}
    for output_format in formats:
        format_dir = dispatch_root / output_format
        if do_profile:
            clean_dir(format_dir, config.dry_run)
            if tool_lib is None or sdk_lib is None:
                die("Internal error: profiling libraries were not resolved")
            profile_env = build_profile_env(
                build_dir=sdk_build_dir,
                tool_lib=tool_lib,
                sdk_lib=sdk_lib,
                output_dir=format_dir,
                output_name=f"dispatches_{dispatches}_{output_format}",
                output_format=output_format,
                counter_line=counter_line,
            )
            exec_sec = run_profile(
                hypothesis=hypothesis,
                dispatches=dispatches,
                output_format=output_format,
                output_dir=format_dir,
                benchmark_bin=BENCHMARK_BIN,
                grid_size=grid_size,
                block_size=block_size,
                spin_iterations=spin_iterations,
                profile_env=profile_env,
                dry_run=config.dry_run,
                verbose=config.verbose,
            )
            write_profile_metadata(
                format_dir / PROFILE_METADATA_FILE,
                hypothesis=hypothesis,
                dispatches=dispatches,
                output_format=output_format,
                exec_sec=exec_sec,
                dry_run=config.dry_run,
            )
        else:
            if not format_dir.exists() and not config.dry_run:
                warn(f"Missing format output directory: {format_dir}")
            exec_sec = read_profile_metadata(format_dir / PROFILE_METADATA_FILE)

        if not perform_measurement:
            continue

        measured = measure_format(output_format, format_dir, exec_sec, config.dry_run)
        metrics[output_format] = measured
        if output_format == "rocpd":
            log(
                "    rocpd: bytes="
                f"{measured.output_bytes} exec={format_seconds(measured.exec_sec)}s "
                f"write={format_seconds(measured.write_sec)}s "
                f"read={format_seconds(measured.read_sec)}s pmc_rows={measured.row_count} "
                f"unique_counters={measured.unique_counters}"
            )
        else:
            log(
                f"    {output_format}: bytes={measured.output_bytes} "
                f"exec={format_seconds(measured.exec_sec)}s "
                f"write={format_seconds(measured.write_sec)}s "
                f"read={format_seconds(measured.read_sec)}s rows={measured.row_count} "
                f"unique_counters={measured.unique_counters}"
            )
    return metrics


def cmd_build_sdk(args: argparse.Namespace) -> int:
    """Build rocprofiler-sdk for selected hypotheses."""
    config = resolve_runtime_config(args, require_counter=False, require_dispatches=False)
    worktrees = hypothesis_dir_map(args.hypotheses)
    for hypothesis in config.hypotheses:
        worktree = worktrees[hypothesis]
        sdk_src_dir = worktree / "projects" / "rocprofiler-sdk"
        sdk_build_dir = worktree / config.sdk_build_dir_name
        log(f"==> Building rocprofiler-sdk for {hypothesis}")
        build_rocprofiler_sdk(
            sdk_src_dir,
            sdk_build_dir,
            dry_run=config.dry_run,
            verbose=config.verbose,
        )
    return 0


def cmd_build_benchmark(args: argparse.Namespace) -> int:
    """Build synthetic dispatch benchmark binary."""
    build_synthetic_benchmark(dry_run=args.dry_run, verbose=args.verbose)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Profile selected hypothesis/dispatch combinations only."""
    config = resolve_runtime_config(args, require_counter=True, require_dispatches=True)
    format_filter = parse_format_filter(args.formats)
    ensure_benchmark_binary(config.dry_run)
    worktrees = hypothesis_dir_map(args.hypotheses)
    counter_line = "pmc: " + " ".join(config.counter_names)

    log(f"==> Hypotheses: {' '.join(config.hypotheses)}")
    log(f"==> Dispatches: {','.join(str(item) for item in config.dispatches)}")
    if config.dry_run:
        log("==> Dry run: enabled")
    for hypothesis in config.hypotheses:
        worktree = worktrees[hypothesis]
        formats = formats_for_hypothesis(hypothesis, format_filter)
        log(f"==> Hypothesis: {hypothesis}")
        log(f"    worktree: {worktree}")
        log(f"    formats: {','.join(formats)}")
        for dispatches in config.dispatches:
            log(f"  -> dispatches {dispatches}")
            _ = profile_and_measure_dispatch(
                hypothesis=hypothesis,
                worktree_dir=worktree,
                dispatches=dispatches,
                formats=formats,
                config=config,
                counter_line=counter_line,
                grid_size=args.grid_size,
                block_size=args.block_size,
                spin_iterations=args.spin_iterations,
                do_profile=True,
                perform_measurement=False,
            )
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    """Measure read/size and write per-dispatch CSV from existing raw output."""
    config = resolve_runtime_config(args, require_counter=True, require_dispatches=False)
    format_filter = parse_format_filter(args.formats)
    worktrees = hypothesis_dir_map(args.hypotheses)
    counter_line = "pmc: " + " ".join(config.counter_names)
    if args.dispatches is None:
        dispatches = discover_dispatches(config.results_dir, config.hypotheses)
        if not dispatches:
            die(
                "No dispatch directories found under results/raw for selected hypotheses. "
                "Provide --dispatches explicitly or run profile first."
            )
    else:
        dispatches = parse_dispatches(args.dispatches)

    if not config.dry_run:
        ensure_dir(config.results_dir, dry_run=False)

    log(f"==> Hypotheses: {' '.join(config.hypotheses)}")
    log(f"==> Dispatches: {','.join(str(item) for item in dispatches)}")
    for hypothesis in config.hypotheses:
        worktree = worktrees[hypothesis]
        formats = formats_for_hypothesis(hypothesis, format_filter)
        log(f"==> Hypothesis: {hypothesis}")
        for dispatch_count in dispatches:
            log(f"  -> dispatches {dispatch_count}")
            metrics = profile_and_measure_dispatch(
                hypothesis=hypothesis,
                worktree_dir=worktree,
                dispatches=dispatch_count,
                formats=formats,
                config=config,
                counter_line=counter_line,
                grid_size=args.grid_size,
                block_size=args.block_size,
                spin_iterations=args.spin_iterations,
                do_profile=False,
                perform_measurement=True,
            )
            header, row = build_per_dispatch_header_and_row(
                hypothesis=hypothesis,
                dispatches=dispatch_count,
                counter_file=config.counter_file,
                unique_counter_count=config.unique_counter_count,
                metrics=metrics,
            )
            output_file = config.results_dir / f"{hypothesis}_{dispatch_count}.csv"
            write_single_row_csv(output_file, header, row, config.dry_run)
            if not config.dry_run:
                log(f"    wrote: {output_file}")
    return 0


def parse_float_field(row: dict[str, str], key: str) -> float:
    """Parse a float field from a CSV row."""
    value = row.get(key, "")
    if value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        warn(f"Failed to parse float field '{key}' with value '{value}'")
        return 0.0


def parse_int_field(row: dict[str, str], key: str) -> int | None:
    """Parse an int field from a CSV row."""
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        warn(f"Failed to parse integer field '{key}' with value '{value}'")
        return None


def cmd_report(args: argparse.Namespace) -> int:
    """Generate consolidated all_results.csv from per-dispatch CSV files."""
    config = resolve_runtime_config(args, require_counter=False, require_dispatches=False)
    format_filter = parse_format_filter(args.formats)
    if args.dispatches is None:
        dispatches = discover_dispatches_from_csvs(config.results_dir, config.hypotheses)
        if not dispatches:
            die(
                "No per-dispatch CSV files found for selected hypotheses in results dir. "
                "Provide --dispatches explicitly or run measure first."
            )
    else:
        dispatches = parse_dispatches(args.dispatches)

    consolidated_rows: list[list[str]] = []
    for hypothesis in config.hypotheses:
        formats = formats_for_hypothesis(hypothesis, format_filter)
        for dispatch_count in dispatches:
            result_file = config.results_dir / f"{hypothesis}_{dispatch_count}.csv"
            if not result_file.is_file():
                warn(f"Per-dispatch result missing, skipping: {result_file}")
                continue
            with result_file.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                row = next(reader, None)
            if row is None:
                warn(f"No data row found in result file: {result_file}")
                continue

            for output_format in formats:
                bytes_key = f"{output_format}_output_bytes"
                read_key = f"{output_format}_read_sec"
                write_key = f"{output_format}_write_sec"
                exec_key = f"{output_format}_exec_sec"
                output_bytes = parse_int_field(row, bytes_key)
                if output_bytes is None:
                    continue
                consolidated_name = CONSOLIDATED_NAME_MAP.get((hypothesis, output_format))
                if consolidated_name is None:
                    die(
                        f"No consolidated name mapping for hypothesis={hypothesis}, "
                        f"format={output_format}"
                    )
                consolidated_rows.append(
                    [
                        consolidated_name,
                        str(dispatch_count),
                        human_size(output_bytes),
                        format_seconds(parse_float_field(row, read_key)),
                        format_seconds(parse_float_field(row, write_key)),
                        format_seconds(parse_float_field(row, exec_key)),
                    ]
                )

    write_consolidated_csv(config.results_dir, consolidated_rows, config.dry_run)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute full build/profile/measure/report pipeline."""
    config = resolve_runtime_config(args, require_counter=True, require_dispatches=True)
    format_filter = parse_format_filter(args.formats)
    worktrees = hypothesis_dir_map(args.hypotheses)
    counter_line = "pmc: " + " ".join(config.counter_names)

    if args.skip_benchmark_build:
        ensure_benchmark_binary(config.dry_run)
    else:
        build_synthetic_benchmark(dry_run=config.dry_run, verbose=config.verbose)
        ensure_benchmark_binary(config.dry_run)

    if not config.dry_run:
        ensure_dir(config.results_dir, dry_run=False)

    log(f"==> Hypotheses: {' '.join(config.hypotheses)}")
    log(f"==> Dispatches: {','.join(str(item) for item in config.dispatches)}")
    log(f"==> Counter file: {config.counter_file}")
    log(f"==> Unique counters in group: {config.unique_counter_count}")
    if config.dry_run:
        log("==> Dry run: enabled")
    log("")

    consolidated_rows: list[list[str]] = []
    for hypothesis in config.hypotheses:
        worktree = worktrees[hypothesis]
        sdk_src_dir = worktree / "projects" / "rocprofiler-sdk"
        sdk_build_dir = worktree / config.sdk_build_dir_name
        formats = formats_for_hypothesis(hypothesis, format_filter)

        if not args.skip_sdk_build:
            build_rocprofiler_sdk(
                sdk_src_dir,
                sdk_build_dir,
                dry_run=config.dry_run,
                verbose=config.verbose,
            )

        log(f"==> Hypothesis: {hypothesis}")
        log(f"    worktree: {worktree}")
        log(f"    formats: {','.join(formats)}")

        for dispatches in config.dispatches:
            log(f"  -> dispatches {dispatches}")
            metrics = profile_and_measure_dispatch(
                hypothesis=hypothesis,
                worktree_dir=worktree,
                dispatches=dispatches,
                formats=formats,
                config=config,
                counter_line=counter_line,
                grid_size=args.grid_size,
                block_size=args.block_size,
                spin_iterations=args.spin_iterations,
                do_profile=True,
                perform_measurement=True,
            )
            header, row = build_per_dispatch_header_and_row(
                hypothesis=hypothesis,
                dispatches=dispatches,
                counter_file=config.counter_file,
                unique_counter_count=config.unique_counter_count,
                metrics=metrics,
            )
            per_dispatch_file = config.results_dir / f"{hypothesis}_{dispatches}.csv"
            write_single_row_csv(per_dispatch_file, header, row, config.dry_run)
            if not config.dry_run:
                log(f"    wrote: {per_dispatch_file}")
            consolidated_rows.extend(
                build_consolidated_rows(hypothesis, dispatches, metrics)
            )

    write_consolidated_csv(config.results_dir, consolidated_rows, config.dry_run)
    log("")
    log(f"Done. Results are in: {config.results_dir}")
    return 0


def add_shared_options(parser: argparse.ArgumentParser) -> None:
    """Add options shared by most subcommands."""
    parser.add_argument(
        "--hypotheses",
        default=None,
        help=(
            "Comma-separated hypothesis names to run "
            "(default: auto-discover baseline,counter_rows_per_dispatch,"
            "csv_per_dispatch_flushed,rocpd_wide_pmc,feather,parquet,rocksdb,"
            "lz4,zstd)"
        ),
    )
    parser.add_argument(
        "--dispatches",
        default=DEFAULT_DISPATCHES,
        help="Comma-separated dispatch counts (default: 100,10000,100000)",
    )
    parser.add_argument(
        "--counter-file",
        default=str(DEFAULT_COUNTER_FILE),
        help=f"Counter file path (default: {DEFAULT_COUNTER_FILE})",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="HIP grid size for synthetic benchmark (default: 128)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="HIP block size for synthetic benchmark (default: 64)",
    )
    parser.add_argument(
        "--spin-iterations",
        type=int,
        default=5,
        help="Spin iterations inside kernel (default: 5)",
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
        "--dry-run",
        action="store_true",
        help="Print run plan without building/running/writing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra command logging",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create CLI parser and subcommands."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark harness for large output hypothesis worktrees. "
            "Uses rocprofiler-sdk-tool directly via LD_PRELOAD."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared_parent = argparse.ArgumentParser(add_help=False)
    add_shared_options(shared_parent)

    build_sdk = subparsers.add_parser(
        "build-sdk",
        parents=[shared_parent],
        help="Configure and build rocprofiler-sdk in selected worktrees",
    )
    build_sdk.set_defaults(func=cmd_build_sdk)

    build_benchmark = subparsers.add_parser(
        "build-benchmark",
        parents=[shared_parent],
        help="Build synthetic dispatch benchmark binary",
    )
    build_benchmark.set_defaults(func=cmd_build_benchmark)

    profile = subparsers.add_parser(
        "profile",
        parents=[shared_parent],
        help="Run profiling only (no consolidated report generation)",
    )
    profile.add_argument(
        "--formats",
        default=None,
        help="Optional comma-separated format filter (subset of csv,rocpd,feather,parquet,rocksdb)",
    )
    profile.set_defaults(func=cmd_profile)

    measure = subparsers.add_parser(
        "measure",
        parents=[shared_parent],
        help="Measure existing raw outputs and write per-dispatch CSV files",
    )
    measure.add_argument(
        "--formats",
        default=None,
        help="Optional comma-separated format filter (subset of csv,rocpd,feather,parquet,rocksdb)",
    )
    measure.set_defaults(func=cmd_measure, dispatches=None)

    report = subparsers.add_parser(
        "report",
        parents=[shared_parent],
        help="Generate consolidated all_results.csv from per-dispatch CSV files",
    )
    report.add_argument(
        "--formats",
        default=None,
        help="Optional comma-separated format filter (subset of csv,rocpd,feather,parquet,rocksdb)",
    )
    report.set_defaults(func=cmd_report, dispatches=None)

    run = subparsers.add_parser(
        "run",
        parents=[shared_parent],
        help="Run full pipeline (build-sdk, build-benchmark, profile, measure, report)",
    )
    run.add_argument(
        "--formats",
        default=None,
        help="Optional comma-separated format filter (subset of csv,rocpd,feather,parquet,rocksdb)",
    )
    run.add_argument(
        "--skip-sdk-build",
        action="store_true",
        help="Skip cmake configure/build of rocprofiler-sdk",
    )
    run.add_argument(
        "--skip-benchmark-build",
        action="store_true",
        help="Skip compiling synthetic benchmark binary",
    )
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

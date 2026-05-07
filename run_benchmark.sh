#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_benchmark.sh [options]

Builds rocprofiler-sdk across hypothesis worktrees, runs synthetic profiling
workloads at multiple dispatch tiers using a fixed multi-counter group, and
writes per-tier CSV results plus one consolidated CSV.

Options:
  --hypotheses LIST         Comma-separated hypothesis names to run
                            (default: auto-discover baseline,counter_rows_per_dispatch,feather,parquet,lz4,zstd)
  --tiers LIST              Comma-separated dispatch tiers (default: 100,10000,100000)
  --counter-file PATH       TXT file with pmc counter group (required for multi-counter;
                            default: <script>/benchmark/pmc_perf_0.txt)
  --grid-size N             HIP grid size for synthetic benchmark (default: 128)
  --block-size N            HIP block size for synthetic benchmark (default: 64)
  --spin-iterations N       Spin iterations inside kernel (single-kernel mode, default: 256)
  --multi-kernel            Run benchmark in multi-kernel mode (default)
  --single-kernel           Run benchmark in legacy single-kernel mode
  --results-dir PATH        Results directory (default: <script>/results)
  --sdk-build-dir NAME      SDK build dir name inside worktree (default: build-hypothesis-rocprofiler-sdk)
  --skip-sdk-build          Skip cmake configure/build of rocprofiler-sdk
  --skip-benchmark-build    Skip compiling synthetic benchmark binary
  --dry-run                 Print full run plan without building/running/writing
  -h, --help                Show this help

Output files:
  <results-dir>/<hypothesis>_<tier>.csv
  <results-dir>/all_results.csv
  <results-dir>/raw/<hypothesis>/tier_<tier>/<format>/
  all_results.csv schema: name,num_dispatch,size,read_time,exec_time
  all_results.csv names: csv_baseline,rocpd_baseline,csv_per_dispatch,feather,parquet,rocpd_lz4,rocpd_zstd

Default output formats per hypothesis:
  baseline                  csv,rocpd
  counter_rows_per_dispatch csv
  feather                   feather
  parquet                   parquet
  lz4                       rocpd
  zstd                      rocpd
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SRC="${SCRIPT_DIR}/benchmark/synthetic_dispatch_benchmark.cpp"
BENCHMARK_BUILD_DIR="${SCRIPT_DIR}/benchmark/build"
BENCHMARK_BIN="${BENCHMARK_BUILD_DIR}/synthetic_dispatch_benchmark"
KNOWN_HYPOTHESES=(baseline counter_rows_per_dispatch feather parquet lz4 zstd)
CONSOLIDATED_HEADER="name,num_dispatch,size,read_time,exec_time"

HYPOTHESES_CSV=""
TIERS_CSV="100,10000,100000"
COUNTER_FILE=""
GRID_SIZE=128
BLOCK_SIZE=64
SPIN_ITERATIONS=256
RESULTS_DIR="${SCRIPT_DIR}/results"
SDK_BUILD_DIR_NAME="build-hypothesis-rocprofiler-sdk"
MULTI_KERNEL=1
SKIP_SDK_BUILD=0
SKIP_BENCHMARK_BUILD=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hypotheses)
            HYPOTHESES_CSV="$2"
            shift 2
            ;;
        --tiers)
            TIERS_CSV="$2"
            shift 2
            ;;
        --counter-file)
            COUNTER_FILE="$2"
            shift 2
            ;;
        --grid-size)
            GRID_SIZE="$2"
            shift 2
            ;;
        --block-size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --spin-iterations)
            SPIN_ITERATIONS="$2"
            shift 2
            ;;
        --multi-kernel)
            MULTI_KERNEL=1
            shift
            ;;
        --single-kernel)
            MULTI_KERNEL=0
            shift
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --sdk-build-dir)
            SDK_BUILD_DIR_NAME="$2"
            shift 2
            ;;
        --skip-sdk-build)
            SKIP_SDK_BUILD=1
            shift
            ;;
        --skip-benchmark-build)
            SKIP_BENCHMARK_BUILD=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ -f "${BENCHMARK_SRC}" ]] || die "Synthetic benchmark source not found: ${BENCHMARK_SRC}"

if [[ -z "${COUNTER_FILE}" ]]; then
    COUNTER_FILE="${SCRIPT_DIR}/benchmark/pmc_perf_0.txt"
fi
[[ -f "${COUNTER_FILE}" ]] || die "Counter file not found: ${COUNTER_FILE}"

COUNTER_NAMES="$(grep -oP '(?<=^pmc:\s).*' "${COUNTER_FILE}" | tr -s ' ')"
[[ -n "${COUNTER_NAMES}" ]] || die "No counters found in counter file: ${COUNTER_FILE}"
UNIQUE_COUNTER_COUNT="$(echo "${COUNTER_NAMES}" | tr ' ' '\n' | sort -u | wc -l)"

echo "==> Counter file: ${COUNTER_FILE}"
echo "==> Counters (${UNIQUE_COUNTER_COUNT}): ${COUNTER_NAMES}"

IFS=',' read -r -a TIERS <<< "${TIERS_CSV}"
[[ "${#TIERS[@]}" -gt 0 ]] || die "No dispatch tiers provided"

for tier in "${TIERS[@]}"; do
    [[ "${tier}" =~ ^[0-9]+$ ]] || die "Tier must be an integer, got: ${tier}"
    [[ "${tier}" -gt 0 ]] || die "Tier must be > 0, got: ${tier}"
done

is_known_hypothesis() {
    local hypothesis="$1"
    local known
    for known in "${KNOWN_HYPOTHESES[@]}"; do
        if [[ "${known}" == "${hypothesis}" ]]; then
            return 0
        fi
    done
    return 1
}

get_formats_for_hypothesis() {
    local hypothesis="$1"
    case "${hypothesis}" in
        baseline) echo "csv,rocpd" ;;
        counter_rows_per_dispatch) echo "csv" ;;
        feather) echo "feather" ;;
        parquet) echo "parquet" ;;
        lz4) echo "rocpd" ;;
        zstd) echo "rocpd" ;;
        *) die "Unknown hypothesis for format mapping: ${hypothesis}" ;;
    esac
}

get_glob_for_format() {
    local format="$1"
    case "${format}" in
        csv) echo "*.csv" ;;
        rocpd) echo "*.db" ;;
        feather) echo "*.feather" ;;
        parquet) echo "*.parquet" ;;
        *) die "Unsupported format: ${format}" ;;
    esac
}

get_consolidated_name() {
    local hypothesis="$1"
    local format="$2"
    case "${hypothesis}:${format}" in
        baseline:csv) echo "csv_baseline" ;;
        baseline:rocpd) echo "rocpd_baseline" ;;
        counter_rows_per_dispatch:csv) echo "csv_per_dispatch" ;;
        feather:feather) echo "feather" ;;
        parquet:parquet) echo "parquet" ;;
        lz4:rocpd) echo "rocpd_lz4" ;;
        zstd:rocpd) echo "rocpd_zstd" ;;
        *) die "No consolidated name mapping for hypothesis=${hypothesis}, format=${format}" ;;
    esac
}

human_size() {
    local bytes="$1"
    python3 - "${bytes}" <<'PY'
import sys

try:
    value = float(sys.argv[1])
except (ValueError, TypeError):
    value = 0.0

units = ["B", "KB", "MB", "GB", "TB", "PB"]
for unit in units:
    if abs(value) < 1024.0 or unit == units[-1]:
        if unit == "B":
            print(f"{int(value)} {unit}")
        else:
            print(f"{value:.1f} {unit}")
        break
    value /= 1024.0
PY
}

canonicalize_hypotheses() {
    local -a requested=()
    local -a selected=()
    local raw

    if [[ -n "${HYPOTHESES_CSV}" ]]; then
        IFS=',' read -r -a requested <<< "${HYPOTHESES_CSV}"
        for raw in "${requested[@]}"; do
            raw="${raw//[[:space:]]/}"
            [[ -n "${raw}" ]] || continue
            is_known_hypothesis "${raw}" || die "Unknown hypothesis in --hypotheses: ${raw}"
            selected+=("${raw}")
        done
        [[ "${#selected[@]}" -gt 0 ]] || die "No valid hypotheses in --hypotheses: ${HYPOTHESES_CSV}"
    else
        selected=("${KNOWN_HYPOTHESES[@]}")
    fi

    local hypothesis
    for hypothesis in "${selected[@]}"; do
        local hypothesis_dir="${SCRIPT_DIR}/${hypothesis}"
        if [[ ! -d "${hypothesis_dir}" ]]; then
            echo "WARNING: hypothesis directory not found, skipping: ${hypothesis_dir}" >&2
            continue
        fi
        if [[ ! -d "${hypothesis_dir}/projects/rocprofiler-sdk" ]]; then
            echo "WARNING: missing projects/rocprofiler-sdk, skipping: ${hypothesis_dir}" >&2
            continue
        fi
        ACTIVE_HYPOTHESES+=("${hypothesis}")
        ACTIVE_WORKTREES+=("${hypothesis_dir}")
    done

    [[ "${#ACTIVE_HYPOTHESES[@]}" -gt 0 ]] || die "No runnable hypothesis worktrees were found"
}

build_synthetic_benchmark() {
    local hipcc_bin
    hipcc_bin="${HIPCC:-hipcc}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY RUN] build synthetic benchmark: ${hipcc_bin} -O2 -std=c++17 ${BENCHMARK_SRC} -o ${BENCHMARK_BIN}"
        return 0
    fi

    command -v "${hipcc_bin}" >/dev/null 2>&1 || die "hipcc not found (tried: ${hipcc_bin})"

    mkdir -p "${BENCHMARK_BUILD_DIR}"
    echo "==> Building synthetic benchmark with ${hipcc_bin}"
    "${hipcc_bin}" -O2 -std=c++17 "${BENCHMARK_SRC}" -o "${BENCHMARK_BIN}"
}

build_rocprofiler_sdk() {
    local sdk_src_dir="$1"
    local sdk_build_dir="$2"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY RUN] configure rocprofiler-sdk: cmake -S ${sdk_src_dir} -B ${sdk_build_dir} -DCMAKE_BUILD_TYPE=Release"
        echo "[DRY RUN] build rocprofiler-sdk: cmake --build ${sdk_build_dir} --parallel $(nproc)"
        return 0
    fi

    echo "==> Configuring rocprofiler-sdk at ${sdk_build_dir}"
    cmake -S "${sdk_src_dir}" -B "${sdk_build_dir}" -DCMAKE_BUILD_TYPE=Release
    echo "==> Building rocprofiler-sdk"
    cmake --build "${sdk_build_dir}" --parallel "$(nproc)"
}

find_rocprofv3() {
    local sdk_build_dir="$1"
    local candidates=(
        "${sdk_build_dir}/bin/rocprofv3"
        "${sdk_build_dir}/source/bin/rocprofv3"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "${candidates[0]}"
        return 0
    fi

    die "Could not find rocprofv3 in build directory: ${sdk_build_dir}"
}

sum_bytes_for_glob() {
    local root_dir="$1"
    local pattern="$2"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "0"
        return 0
    fi

    python3 - "$root_dir" "$pattern" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
pattern = sys.argv[2]
total = 0
if root.exists():
    for path in root.rglob(pattern):
        if path.is_file():
            total += path.stat().st_size
print(total)
PY
}

measure_csv_read() {
    local root_dir="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "0.000000,0,0"
        return 0
    fi

    python3 - "$root_dir" <<'PY'
from pathlib import Path
import csv
import sys
import time

root = Path(sys.argv[1])
csv_files = sorted(path for path in root.rglob("*.csv") if path.is_file())

rows = 0
unique_counters = set()
start = time.perf_counter()
for csv_file in csv_files:
    with csv_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            if "Counter_Name" in row:
                unique_counters.add(row["Counter_Name"])
            elif "counter_name" in row:
                unique_counters.add(row["counter_name"])
elapsed = time.perf_counter() - start

print(f"{elapsed:.6f},{rows},{len(unique_counters)}")
PY
}

measure_rocpd_read() {
    local root_dir="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "0.000000,0,0"
        return 0
    fi

    python3 - "$root_dir" <<'PY'
from pathlib import Path
import sqlite3
import sys
import time

root = Path(sys.argv[1])
db_files = sorted(path for path in root.rglob("*.db") if path.is_file())

pmc_rows = 0
unique_counters = 0
start = time.perf_counter()
for db_path in db_files:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_pmc_event%'"
    )
    pmc_tables = [r[0] for r in cur.fetchall()]

    for tbl in pmc_tables:
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
            row = cur.fetchone()
            if row and row[0] is not None:
                pmc_rows += int(row[0])
        except sqlite3.Error:
            pass

        try:
            cur.execute(f'SELECT COUNT(DISTINCT pmc_id) FROM "{tbl}"')
            row = cur.fetchone()
            if row and row[0] is not None:
                unique_counters = max(unique_counters, int(row[0]))
        except sqlite3.Error:
            pass

    conn.close()

elapsed = time.perf_counter() - start
print(f"{elapsed:.6f},{pmc_rows},{unique_counters}")
PY
}

measure_feather_read() {
    local root_dir="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "0.000000,0,0"
        return 0
    fi

    python3 - "$root_dir" <<'PY'
from pathlib import Path
import sys
import time

try:
    import pyarrow.feather as feather
except ModuleNotFoundError as exc:
    raise SystemExit(f"pyarrow is required for feather read measurement: {exc}")

root = Path(sys.argv[1])
files = sorted(path for path in root.rglob("*.feather") if path.is_file())

rows = 0
unique_counters = set()
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
print(f"{elapsed:.6f},{rows},{len(unique_counters)}")
PY
}

measure_parquet_read() {
    local root_dir="$1"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "0.000000,0,0"
        return 0
    fi

    python3 - "$root_dir" <<'PY'
from pathlib import Path
import sys
import time

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(f"pyarrow is required for parquet read measurement: {exc}")

root = Path(sys.argv[1])
files = sorted(path for path in root.rglob("*.parquet") if path.is_file())

rows = 0
unique_counters = set()
start = time.perf_counter()
for path in files:
    table = pq.read_table(path)
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
print(f"{elapsed:.6f},{rows},{len(unique_counters)}")
PY
}

run_profile() {
    local hypothesis="$1"
    local rocprofv3_bin="$2"
    local format="$3"
    local tier="$4"
    local out_dir="$5"
    local counter_file="$6"

    local benchmark_args=()
    if [[ "${MULTI_KERNEL}" -eq 1 ]]; then
        benchmark_args=(--multi "${tier}")
    else
        benchmark_args=("${tier}" "${GRID_SIZE}" "${BLOCK_SIZE}" "${SPIN_ITERATIONS}")
    fi

    local run_cmd=(
        "${rocprofv3_bin}"
        -d "${out_dir}"
        -o "tier_${tier}_${format}"
        --output-format "${format}"
        --kernel-trace
        -i "${counter_file}"
        -- "${BENCHMARK_BIN}" "${benchmark_args[@]}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY RUN] hypothesis=${hypothesis} tier=${tier} format=${format}" >&2
        echo "[DRY RUN]   command: $(printf '%q ' "${run_cmd[@]}")" >&2
        echo "[DRY RUN]   result_dir: ${out_dir}" >&2
        echo "0.000000"
        return 0
    fi

    local log_file="${out_dir}/rocprof_${format}.log"
    local time_file
    time_file="$(mktemp)"

    mkdir -p "${out_dir}"

    /usr/bin/time -f "%e" -o "${time_file}" \
        "${run_cmd[@]}" \
        >"${log_file}" 2>&1

    local exit_code=$?
    local elapsed
    elapsed="$(awk '/^[0-9]+([.][0-9]+)?$/ { val = $0 } END { if (val == "") val = "0.000000"; print val }' "${time_file}")"
    rm -f "${time_file}"

    if [[ ${exit_code} -ne 0 ]]; then
        echo "WARNING: rocprofv3 exited with code ${exit_code} for format=${format} tier=${tier}" >&2
        echo "         Log: ${log_file}" >&2
    fi

    echo "${elapsed}"
}

format_summary_line() {
    local format="$1"
    local output_bytes="$2"
    local exec_sec="$3"
    local read_sec="$4"
    local row_count="$5"
    local unique_counter_count="$6"

    case "${format}" in
        rocpd)
            echo "    ${format}: bytes=${output_bytes} exec=${exec_sec}s read=${read_sec}s pmc_rows=${row_count} unique_counters=${unique_counter_count}"
            ;;
        *)
            echo "    ${format}: bytes=${output_bytes} exec=${exec_sec}s read=${read_sec}s rows=${row_count} unique_counters=${unique_counter_count}"
            ;;
    esac
}

ACTIVE_HYPOTHESES=()
ACTIVE_WORKTREES=()
canonicalize_hypotheses

if [[ "${SKIP_BENCHMARK_BUILD}" -eq 0 ]]; then
    build_synthetic_benchmark
fi
if [[ "${DRY_RUN}" -eq 0 ]]; then
    [[ -x "${BENCHMARK_BIN}" ]] || die "Benchmark binary missing or not executable: ${BENCHMARK_BIN}"
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${RESULTS_DIR}"
fi

echo "==> Hypotheses: ${ACTIVE_HYPOTHESES[*]}"
echo "==> Tiers: ${TIERS_CSV}"
echo "==> Counter file: ${COUNTER_FILE}"
echo "==> Unique counters in group: ${UNIQUE_COUNTER_COUNT}"
if [[ "${MULTI_KERNEL}" -eq 1 ]]; then
    echo "==> Benchmark mode: multi-kernel (--multi <dispatches_per_config>)"
else
    echo "==> Benchmark mode: single-kernel (dispatches grid block spin)"
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "==> Dry run: enabled"
fi
echo

consolidated_result_file="${RESULTS_DIR}/all_results.csv"
declare -a CONSOLIDATED_ROWS=()

for i in "${!ACTIVE_HYPOTHESES[@]}"; do
    hypothesis="${ACTIVE_HYPOTHESES[$i]}"
    worktree="${ACTIVE_WORKTREES[$i]}"
    sdk_src_dir="${worktree}/projects/rocprofiler-sdk"
    sdk_build_dir="${worktree}/${SDK_BUILD_DIR_NAME}"
    formats_csv="$(get_formats_for_hypothesis "${hypothesis}")"
    IFS=',' read -r -a formats <<< "${formats_csv}"

    if [[ "${SKIP_SDK_BUILD}" -eq 0 ]]; then
        build_rocprofiler_sdk "${sdk_src_dir}" "${sdk_build_dir}"
    fi

    rocprofv3_bin="$(find_rocprofv3 "${sdk_build_dir}")"
    echo "==> Hypothesis: ${hypothesis}"
    echo "    worktree: ${worktree}"
    echo "    rocprofv3: ${rocprofv3_bin}"
    echo "    formats: ${formats_csv}"

    for tier in "${TIERS[@]}"; do
        echo "  -> tier ${tier}"

        tier_root="${RESULTS_DIR}/raw/${hypothesis}/tier_${tier}"
        tier_result_file="${RESULTS_DIR}/${hypothesis}_${tier}.csv"

        per_tier_header=(
            "worktree" "tier" "counter_file" "unique_counter_count"
        )
        per_tier_row=(
            "${hypothesis}" "${tier}" "${COUNTER_FILE}" "${UNIQUE_COUNTER_COUNT}"
        )

        csv_output_bytes="" csv_exec_sec="" csv_read_sec="" csv_rows="" csv_unique_counters="" csv_dir=""
        rocpd_output_bytes="" rocpd_exec_sec="" rocpd_read_sec="" rocpd_pmc_rows="" rocpd_unique_counters="" rocpd_dir=""
        feather_output_bytes="" feather_exec_sec="" feather_read_sec="" feather_rows="" feather_unique_counters="" feather_dir=""
        parquet_output_bytes="" parquet_exec_sec="" parquet_read_sec="" parquet_rows="" parquet_unique_counters="" parquet_dir=""

        for format in "${formats[@]}"; do
            format_dir="${tier_root}/${format}"
            if [[ "${DRY_RUN}" -eq 0 ]]; then
                rm -rf "${format_dir}"
                mkdir -p "${format_dir}"
            fi

            exec_sec="$(run_profile "${hypothesis}" "${rocprofv3_bin}" "${format}" "${tier}" "${format_dir}" "${COUNTER_FILE}")"
            output_bytes="$(sum_bytes_for_glob "${format_dir}" "$(get_glob_for_format "${format}")")"

            case "${format}" in
                csv)
                    IFS=',' read -r read_sec rows unique_counters <<< "$(measure_csv_read "${format_dir}")"
                    csv_output_bytes="${output_bytes}"
                    csv_exec_sec="${exec_sec}"
                    csv_read_sec="${read_sec}"
                    csv_rows="${rows}"
                    csv_unique_counters="${unique_counters}"
                    csv_dir="${format_dir}"
                    per_tier_header+=("csv_output_bytes" "csv_exec_sec" "csv_read_sec" "csv_rows" "csv_unique_counters" "csv_dir")
                    per_tier_row+=("${csv_output_bytes}" "${csv_exec_sec}" "${csv_read_sec}" "${csv_rows}" "${csv_unique_counters}" "${csv_dir}")
                    echo "$(format_summary_line "csv" "${csv_output_bytes}" "${csv_exec_sec}" "${csv_read_sec}" "${csv_rows}" "${csv_unique_counters}")"
                    ;;
                rocpd)
                    IFS=',' read -r read_sec pmc_rows unique_counters <<< "$(measure_rocpd_read "${format_dir}")"
                    rocpd_output_bytes="${output_bytes}"
                    rocpd_exec_sec="${exec_sec}"
                    rocpd_read_sec="${read_sec}"
                    rocpd_pmc_rows="${pmc_rows}"
                    rocpd_unique_counters="${unique_counters}"
                    rocpd_dir="${format_dir}"
                    per_tier_header+=("rocpd_output_bytes" "rocpd_exec_sec" "rocpd_read_sec" "rocpd_pmc_rows" "rocpd_unique_counters" "rocpd_dir")
                    per_tier_row+=("${rocpd_output_bytes}" "${rocpd_exec_sec}" "${rocpd_read_sec}" "${rocpd_pmc_rows}" "${rocpd_unique_counters}" "${rocpd_dir}")
                    echo "$(format_summary_line "rocpd" "${rocpd_output_bytes}" "${rocpd_exec_sec}" "${rocpd_read_sec}" "${rocpd_pmc_rows}" "${rocpd_unique_counters}")"
                    ;;
                feather)
                    IFS=',' read -r read_sec rows unique_counters <<< "$(measure_feather_read "${format_dir}")"
                    feather_output_bytes="${output_bytes}"
                    feather_exec_sec="${exec_sec}"
                    feather_read_sec="${read_sec}"
                    feather_rows="${rows}"
                    feather_unique_counters="${unique_counters}"
                    feather_dir="${format_dir}"
                    per_tier_header+=("feather_output_bytes" "feather_exec_sec" "feather_read_sec" "feather_rows" "feather_unique_counters" "feather_dir")
                    per_tier_row+=("${feather_output_bytes}" "${feather_exec_sec}" "${feather_read_sec}" "${feather_rows}" "${feather_unique_counters}" "${feather_dir}")
                    echo "$(format_summary_line "feather" "${feather_output_bytes}" "${feather_exec_sec}" "${feather_read_sec}" "${feather_rows}" "${feather_unique_counters}")"
                    ;;
                parquet)
                    IFS=',' read -r read_sec rows unique_counters <<< "$(measure_parquet_read "${format_dir}")"
                    parquet_output_bytes="${output_bytes}"
                    parquet_exec_sec="${exec_sec}"
                    parquet_read_sec="${read_sec}"
                    parquet_rows="${rows}"
                    parquet_unique_counters="${unique_counters}"
                    parquet_dir="${format_dir}"
                    per_tier_header+=("parquet_output_bytes" "parquet_exec_sec" "parquet_read_sec" "parquet_rows" "parquet_unique_counters" "parquet_dir")
                    per_tier_row+=("${parquet_output_bytes}" "${parquet_exec_sec}" "${parquet_read_sec}" "${parquet_rows}" "${parquet_unique_counters}" "${parquet_dir}")
                    echo "$(format_summary_line "parquet" "${parquet_output_bytes}" "${parquet_exec_sec}" "${parquet_read_sec}" "${parquet_rows}" "${parquet_unique_counters}")"
                    ;;
                *)
                    die "Unsupported format in loop: ${format}"
                    ;;
            esac

            consolidated_name="$(get_consolidated_name "${hypothesis}" "${format}")"
            consolidated_size="$(human_size "${output_bytes}")"
            consolidated_entry="${consolidated_name},${tier},${consolidated_size},${read_sec},${exec_sec}"
            CONSOLIDATED_ROWS+=("${consolidated_entry}")
            if [[ "${DRY_RUN}" -eq 1 ]]; then
                echo "[DRY RUN]   consolidated row: ${consolidated_entry}"
            fi
        done

        if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "[DRY RUN] would write per-tier result: ${tier_result_file}"
        else
            (
                IFS=','
                printf '%s\n' "${per_tier_header[*]}"
                printf '%s\n' "${per_tier_row[*]}"
            ) > "${tier_result_file}"
            echo "    wrote: ${tier_result_file}"
        fi
    done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY RUN] would write consolidated results: ${consolidated_result_file}"
    echo "[DRY RUN] consolidated row count: ${#CONSOLIDATED_ROWS[@]}"
else
    {
        echo "${CONSOLIDATED_HEADER}"
        for row in "${CONSOLIDATED_ROWS[@]}"; do
            echo "${row}"
        done
    } > "${consolidated_result_file}"
    echo "==> Wrote consolidated results: ${consolidated_result_file}"
fi

echo
echo "Done. Results are in: ${RESULTS_DIR}"

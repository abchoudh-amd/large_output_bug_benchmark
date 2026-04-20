# large_output_bug_benchmark

Benchmark and testing harness for the rocprofiler-sdk "large output bug"
investigation. This repo holds the synthetic workload used to stress-test
output paths (CSV, RocksDB, Feather, Parquet, ...) along with the
orchestration scripts that drive a comparative run across hypothesis
worktrees.

## Contents

```
benchmark/
  synthetic_dispatch_benchmark.cpp   HIP-based synthetic dispatch workload
                                     used to generate counter-collection
                                     output at controllable scale.
  measure_rocksdb_baseline.cpp       Standalone tool that measures decoding
                                     of the RocksDB output produced by the
                                     `rocksdb` / `rocksdb_aggregated`
                                     hypotheses (newer SST footer that
                                     python-rocksdb cannot parse).
  pmc_perf_0.txt                     Counter set used during profiling runs.

run_benchmark.sh                     Bash driver: builds the synthetic
                                     benchmark, runs each hypothesis under
                                     rocprofv3 with the chosen counter set,
                                     and aggregates results into CSVs.
run_benchmark.py                     Python equivalent / extended version of
                                     the same orchestration.
benchmark_mistral.py                 Mistral-flavoured variant of the
                                     benchmark harness.
mistral.py                           Mistral helper used by the above.
```

## Important: hypothesis worktree path assumption

`run_benchmark.sh` (around line 244) and the Python drivers locate the
hypothesis branches as siblings of themselves, via
`"${SCRIPT_DIR}/${hypothesis}"`.

In the working environment those worktrees live under
`~/dev/hypothesis_testing/large_output_bug/<hypothesis>/`. If you check
this repo out somewhere else and run `run_benchmark.sh` from that
checkout, it will not find the hypothesis worktrees and the run will
fail. Treat this commit as a snapshot of the harness for version control
and sharing; making the hypothesis root configurable (e.g. a
`HYPOTHESIS_ROOT` env var or `--hypothesis-root` CLI flag) is a
follow-up.

## Building the synthetic benchmark manually

```
hipcc -O2 -std=c++17 benchmark/synthetic_dispatch_benchmark.cpp \
    -o benchmark/build/synthetic_dispatch_benchmark
```

`run_benchmark.sh --skip-benchmark-build` will reuse a pre-built binary
if present.

## Building measure_rocksdb_baseline manually

Requires the same RocksDB static library that the `rocksdb` /
`rocksdb_aggregated` hypotheses link against. Refer to those hypothesis
branches for the exact link flags.

## Excluded from the repo

- `build/` (compiled binaries)
- `__pycache__/`, `*.pyc`, `*.pyo`
- `.venv/`
- `.rocprofv3/` (rocprofv3 runtime data)
- `results/` (per-run output)

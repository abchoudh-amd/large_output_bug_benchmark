# Large Output Bug: rocprofiler-sdk Output Format Investigation

## Background

When `rocprofv3` collects PMC counters across a real workload, the output
artifact size grows out of proportion with the work being measured. A
single Mistral-7B inference run that generates 50 tokens with 28 PMC
counters produces multi-gigabyte output files: 1.4 GB of CSV under
`ROCPROF_OUTPUT_FORMAT=CSV`, 2.1 GB of SQLite under
`ROCPROF_OUTPUT_FORMAT=ROCPD`. That is impractical for users to ship,
load, query, or keep around for repeated runs.

The investigation tracked here tests output format and schema changes
inside `projects/rocprofiler-sdk/`. The question being answered is:
**which output format and schema can shrink the artifact substantially
while keeping the data downstream tooling actually needs?**

Each hypothesis is implemented as a feature branch on
`ROCm/rocm-systems` with its own modifications to the rocprofiler-sdk
output path. The harness in this repo runs the same workload against
each hypothesis and records output size and timing.

## Methodology

- Workload: Mistral-7B-v0.1 inference, prompt `"The capital of France is"`, `max_new_tokens=50`.
- Counters: 28 PMC counters captured in a single pass, listed in [benchmark/pmc_perf_0.txt](../benchmark/pmc_perf_0.txt).
- Driver: [benchmark_mistral.py](../benchmark_mistral.py) wrapping [mistral.py](../mistral.py); rocprofv3 invoked once per hypothesis with the matching `ROCPROF_OUTPUT_FORMAT`.
- Per-run measurements: output size in bytes, kernel/inference exec time, write time, read time, row count, unique counters.
- One inference run per layout. No averaging over multiple runs.

## Hypotheses

All hypothesis branches live under
`users/abchoudh/hypothesis/large_output_bug/<suffix>` on
`ROCm/rocm-systems`. The two `(baseline)` rows correspond to the
unmodified collector running with the indicated `ROCPROF_OUTPUT_FORMAT`.

| Layout | Branch | Format | What changed |
| --- | --- | --- | --- |
| CSV (baseline) | `users/abchoudh/hypothesis/large_output_bug/baseline` | csv | Default collector with `ROCPROF_OUTPUT_FORMAT=CSV`. Long-form text: one row per (dispatch, counter, instance). |
| ROCPD/SQLite (baseline) | `users/abchoudh/hypothesis/large_output_bug/baseline` | rocpd | Default collector with `ROCPROF_OUTPUT_FORMAT=ROCPD`. Long-form `rocpd_pmc_event` table with indexes, one row per (dispatch, counter, instance). |
| CSV wide per-dispatch | `users/abchoudh/hypothesis/large_output_bug/counter_rows_per_dispatch` | csv | One CSV row per dispatch with counters as columns; per-instance values dropped. |
| Parquet (aggregated) | `users/abchoudh/hypothesis/large_output_bug/parquet` | parquet | Single Parquet file per run, values summed across instances, dictionary + snappy compression. |
| Parquet (per-instance) | `users/abchoudh/hypothesis/large_output_bug/parquet_per_instance` | parquet | Parquet, one row per (dispatch, counter, instance); preserves per-instance values. |
| Feather (uncompressed) | `users/abchoudh/hypothesis/large_output_bug/feather` | feather | Apache Arrow Feather, values summed across instances, no compression. |
| RocksDB (aggregated) | `users/abchoudh/hypothesis/large_output_bug/rocksdb_aggregated` | rocksdb | One RocksDB key per dispatch, values summed across instances. |
| RocksDB (per-instance) | `users/abchoudh/hypothesis/large_output_bug/rocksdb` | rocksdb | One RocksDB key per dispatch; per-instance counter entries packed into a binary blob value. |
| ROCPD aggregated PMC | `users/abchoudh/hypothesis/large_output_bug/rocpd_aggregated_pmc` | rocpd | rocpd/sqlite default schema, values summed across instances before insert. |
| ROCPD wide JSONB PMC | `users/abchoudh/hypothesis/large_output_bug/rocpd_wide_pmc` | rocpd | rocpd/sqlite with `value REAL` replaced by `values JSONB` array holding per-instance values. |

## Results

Sorted smallest output first. `vs CSV` and `vs ROCPD` are
size-reduction multiples against the two baselines. Sizes are reported
in human units (1 MB = 1,000,000 B, 1 GB = 1,000,000,000 B).

| Layout | Output size | vs CSV | vs ROCPD | write (s) | read (s) | rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parquet (aggregated) | 12.2 MB | 124x | 180x | 3.48 | 2.78 | 2,607,108 |
| RocksDB (aggregated) | 49.7 MB | 30x | 44x | 1.04 | 0.83 | 2,607,108 |
| Parquet (per-instance) | 51.5 MB | 29x | 42x | 35.18 | 30.10 | 35,472,240 |
| CSV wide per-dispatch | 81.6 MB | 18x | 27x | 0.00 | 0.76 | 93,114 |
| ROCPD aggregated PMC | 277.9 MB | 5.4x | 7.9x | 4.09 | 0.34 | 2,607,108 |
| ROCPD wide JSONB PMC | 409.3 MB | 3.7x | 5.4x | 6.40 | 0.49 | 2,607,108 |
| RocksDB (per-instance) | 551.6 MB | 2.7x | 4.0x | 2.85 | 0.59 | 35,382,180 |
| Feather (uncompressed) | 1.4 GB | 1.0x | 1.5x | 3.40 | 2.16 | 2,607,108 |
| CSV (baseline) | 1.4 GB | 1x | 1.5x | 0.00 | 10.08 | 2,607,111 |
| ROCPD/SQLite (baseline) | 2.1 GB | 0.7x | 1x | 30.10 | 2.47 | 35,382,180 |

## Findings

Three groupings show up clearly in the data.

**Row-count groupings.** The 10 layouts cluster into three row-count
buckets, and the bucket alone explains a large part of the size spread:

| Row count | Layouts |
| --- | --- |
| ~35.4M rows (per-instance preserved) | ROCPD/SQLite (baseline), RocksDB (per-instance), Parquet (per-instance) |
| ~2.6M rows (per-counter aggregated across instances) | CSV (baseline), Parquet (aggregated), Feather (uncompressed), RocksDB (aggregated), ROCPD aggregated PMC, ROCPD wide JSONB PMC |
| ~93k rows (per-dispatch wide) | CSV wide per-dispatch |

**Schema shape alone is a major lever.** Going from "one row per (dispatch, counter, instance)" to "one row per dispatch with counters as columns" cuts CSV from 1.4 GB to 81.6 MB without changing storage format or sacrificing text compatibility. The row count drops from 2.6M to 93k.

**Format choice still matters within the same schema shape.** Inside the ~2.6M-row aggregated group, picking the right binary format dominates: Parquet (aggregated) is 12.2 MB, RocksDB (aggregated) is 49.7 MB, ROCPD aggregated PMC is 277.9 MB, ROCPD wide JSONB PMC is 409.3 MB, and Feather (uncompressed) stays at 1.4 GB. That is a ~115x spread within identical row counts, driven entirely by encoding and compression.

**Per-instance fidelity is affordable in columnar binary.** Parquet (per-instance) keeps the full 35.5M-row representation but lands at 51.5 MB, while the other per-instance-preserving layouts (RocksDB at 551.6 MB, baseline ROCPD at 2.1 GB) are roughly an order of magnitude larger.

## Reproducing

The harness lives at <https://github.com/abchoudh-amd/large_output_bug_benchmark>.
Each hypothesis is a separate branch on `ROCm/rocm-systems` (see the
Hypotheses table for the full branch name of each layout).

Steps:

1. Clone the harness:

   ```
   git clone https://github.com/abchoudh-amd/large_output_bug_benchmark.git harness
   cd harness
   ```

2. For each hypothesis you want to measure, clone `ROCm/rocm-systems`
   on the matching branch and build the rocprofiler-sdk from
   `projects/rocprofiler-sdk/` using its standard CMake build. Example
   for `parquet`:

   ```
   git clone --branch users/abchoudh/hypothesis/large_output_bug/parquet \
             --depth 1 \
             https://github.com/ROCm/rocm-systems.git rocm-systems-parquet
   cmake -S rocm-systems-parquet/projects/rocprofiler-sdk -B build-parquet
   cmake --build build-parquet --target install
   ```

3. Drive the Mistral run with `benchmark_mistral.py`, pointing at the
   freshly built rocprofiler-sdk install. The script invokes
   `rocprofv3` once with the 28 PMC counters from
   [`benchmark/pmc_perf_0.txt`](../benchmark/pmc_perf_0.txt) and
   captures `output_bytes`, `exec_sec`, `write_sec`, `read_sec`,
   `rows`, and `unique_counters` into a CSV that lines up directly with
   the Results table above.

Note: `run_benchmark.sh` in this harness assumes
`${SCRIPT_DIR}/<branch_suffix>/` sibling directories holding each
hypothesis worktree. The harness [README](../README.md) calls this
out. The minimal Mistral reproduction described above can be driven
through `benchmark_mistral.py` directly without that layout, by
pointing it at one rocprofiler-sdk install at a time.

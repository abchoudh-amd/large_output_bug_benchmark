// Tool to measure baseline rocksdb output that python-rocksdb cannot decode
// (newer RocksDB v9.10.0 SST footer). Links against the same RocksDB static lib
// used by the rocksdb/rocksdb_aggregated hypotheses.
//
// Binary layout of the value blob (from serialize_rocksdb_value in tool.cpp):
//   uint32_t format_version
//   uint64_t thread_id
//   uint64_t stream_id
//   uint64_t correlation_id
//   uint64_t start_timestamp
//   uint64_t end_timestamp
//   uint64_t dispatch_id
//   uint64_t agent_id
//   uint64_t queue_id
//   uint64_t kernel_id
//   uint32_t counter_count
//   [counter_count] * { uint64_t counter_id, double value }

#include <rocksdb/db.h>
#include <rocksdb/options.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr std::size_t header_size = sizeof(uint32_t) + 9 * sizeof(uint64_t) + sizeof(uint32_t);
constexpr std::size_t entry_size  = sizeof(uint64_t) + sizeof(double);
}

struct DecodeResult
{
    std::size_t rows                       = 0;
    std::set<uint64_t> unique_counter_ids  = {};
};

DecodeResult
decode_value(const rocksdb::Slice& slice)
{
    DecodeResult out;
    if(slice.size() < header_size) return out;

    auto const* data = reinterpret_cast<const uint8_t*>(slice.data());
    uint32_t counter_count = 0;
    std::memcpy(&counter_count, data + header_size - sizeof(uint32_t), sizeof(uint32_t));

    auto available = slice.size() >= header_size
                         ? (slice.size() - header_size) / entry_size
                         : 0;
    auto valid = static_cast<std::size_t>(counter_count);
    if(valid > available) valid = available;

    std::size_t offset = header_size;
    for(std::size_t i = 0; i < valid; ++i)
    {
        uint64_t cid = 0;
        std::memcpy(&cid, data + offset, sizeof(uint64_t));
        out.unique_counter_ids.insert(cid);
        offset += entry_size;
    }
    out.rows = valid;
    return out;
}

std::uintmax_t
directory_size_bytes(const fs::path& dir)
{
    std::uintmax_t total = 0;
    if(!fs::exists(dir)) return 0;
    for(const auto& entry : fs::recursive_directory_iterator(dir))
    {
        if(entry.is_regular_file())
        {
            std::error_code ec;
            auto size = entry.file_size(ec);
            if(!ec) total += size;
        }
    }
    return total;
}

int
main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::fprintf(stderr, "usage: %s <rocksdb_path>\n", argv[0]);
        return 2;
    }

    fs::path db_path = argv[1];
    if(!fs::is_directory(db_path))
    {
        std::fprintf(stderr, "not a directory: %s\n", db_path.c_str());
        return 2;
    }

    auto options           = rocksdb::Options{};
    options.create_if_missing = false;

    rocksdb::DB* db_raw = nullptr;
    auto status = rocksdb::DB::OpenForReadOnly(options, db_path.string(), &db_raw);
    if(!status.ok())
    {
        std::fprintf(stderr, "DB::OpenForReadOnly failed: %s\n", status.ToString().c_str());
        return 1;
    }
    std::unique_ptr<rocksdb::DB> db(db_raw);

    auto bytes = directory_size_bytes(db_path);

    std::size_t total_rows = 0;
    std::set<uint64_t> unique_counter_ids;
    std::size_t dispatch_entries = 0;

    auto t0 = std::chrono::steady_clock::now();

    auto read_options = rocksdb::ReadOptions{};
    read_options.verify_checksums = false;
    read_options.fill_cache       = false;
    {
        std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(read_options));
        for(it->SeekToFirst(); it->Valid(); it->Next())
        {
            ++dispatch_entries;
            auto decoded = decode_value(it->value());
            total_rows += decoded.rows;
            unique_counter_ids.insert(decoded.unique_counter_ids.begin(),
                                      decoded.unique_counter_ids.end());
        }

        if(!it->status().ok())
        {
            std::fprintf(stderr, "iterator error: %s\n", it->status().ToString().c_str());
            return 1;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    db.reset();

    std::printf("bytes=%llu\n", static_cast<unsigned long long>(bytes));
    std::printf("dispatches=%zu\n", dispatch_entries);
    std::printf("rows=%zu\n", total_rows);
    std::printf("unique_counters=%zu\n", unique_counter_ids.size());
    std::printf("read_sec=%.6f\n", elapsed_sec);

    return 0;
}

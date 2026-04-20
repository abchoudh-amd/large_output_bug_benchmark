#include <hip/hip_runtime.h>

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
void
check_hip(hipError_t err, const char* expr)
{
    if(err != hipSuccess)
    {
        std::cerr << "[synthetic-dispatch-benchmark] HIP error: " << hipGetErrorString(err)
                  << " while executing: " << expr << '\n';
        throw std::runtime_error("hip_error");
    }
}

#define HIP_CHECK(EXPR) check_hip((EXPR), #EXPR)

enum class KernelKind : int
{
    Spin      = 0,
    MemRead   = 1,
    Reduction = 2,
    Branchy   = 3,
};

struct DispatchConfig
{
    KernelKind   kernel;
    unsigned int grid_size;
    unsigned int block_size;
    int          spin_iterations;
};

constexpr std::array<DispatchConfig, 18> kDispatchConfigs = {{
    {KernelKind::Spin, 32, 64, 128},
    {KernelKind::Spin, 64, 128, 192},
    {KernelKind::Spin, 128, 64, 256},
    {KernelKind::Spin, 256, 128, 320},
    {KernelKind::Spin, 512, 64, 512},
    {KernelKind::MemRead, 64, 64, 128},
    {KernelKind::MemRead, 128, 128, 160},
    {KernelKind::MemRead, 192, 256, 192},
    {KernelKind::MemRead, 256, 128, 224},
    {KernelKind::Reduction, 64, 64, 64},
    {KernelKind::Reduction, 128, 128, 96},
    {KernelKind::Reduction, 256, 128, 128},
    {KernelKind::Reduction, 128, 256, 160},
    {KernelKind::Branchy, 48, 64, 128},
    {KernelKind::Branchy, 96, 128, 192},
    {KernelKind::Branchy, 160, 64, 224},
    {KernelKind::Branchy, 224, 128, 256},
    {KernelKind::Branchy, 320, 64, 320},
}};

static_assert(kDispatchConfigs.size() >= 15 && kDispatchConfigs.size() <= 20,
              "Multi-kernel mode should include 15-20 dispatch configurations");

constexpr bool
is_power_of_two(unsigned int value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

constexpr bool
reduction_configs_are_valid()
{
    for(const auto& cfg : kDispatchConfigs)
    {
        if(cfg.kernel == KernelKind::Reduction && !is_power_of_two(cfg.block_size)) return false;
    }
    return true;
}

static_assert(reduction_configs_are_valid(),
              "Reduction kernel configurations require power-of-two block sizes");

__global__ void
spin_kernel(float* data, int n, int spin_iterations)
{
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if(idx >= n) return;

    float value = data[idx];
    for(int i = 0; i < spin_iterations; ++i)
    {
        value = (value * 1.000001f) + 0.000001f;
    }
    data[idx] = value;
}

__global__ void
mem_read_kernel(const float* input, float* output, int n, int spin_iterations)
{
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if(idx >= n) return;

    float value = 0.0f;
    for(int i = 0; i < spin_iterations; ++i)
    {
        const int read_idx = (idx + (i * 17)) % n;
        value += input[read_idx] * 0.999999f + 0.000001f;
    }
    output[idx] = value;
}

__global__ void
reduction_kernel(const float* input, float* output, int n, int spin_iterations)
{
    extern __shared__ float scratch[];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;

    float value = 0.0f;
    if(gid < static_cast<unsigned int>(n))
    {
        for(int i = 0; i < spin_iterations; ++i)
        {
            const int read_idx = static_cast<int>((gid + (i * blockDim.x)) % static_cast<unsigned int>(n));
            value += input[read_idx] * 0.5f + 0.000001f;
        }
    }

    scratch[tid] = value;
    __syncthreads();

    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }

    if(tid == 0) output[blockIdx.x] = scratch[0];
}

__global__ void
branchy_kernel(float* data, int n, int spin_iterations)
{
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if(idx >= n) return;

    float value = data[idx];
    for(int i = 0; i < spin_iterations; ++i)
    {
        const int selector = (idx + i) & 0x3;
        if(selector == 0)
        {
            value = value * 1.000013f + 0.0000007f;
        }
        else if(selector == 1)
        {
            value = value * 0.999991f - 0.0000003f;
        }
        else if(selector == 2)
        {
            value = value + 0.0000020f;
        }
        else
        {
            value = value * 1.000001f + 0.0000001f;
        }
    }
    data[idx] = value;
}

int
to_int(const char* arg, const char* name)
{
    const auto v = std::strtol(arg, nullptr, 10);
    if(v <= 0)
    {
        std::cerr << "Invalid " << name << ": " << arg << '\n';
        throw std::invalid_argument("invalid argument");
    }
    return static_cast<int>(v);
}

const char*
kernel_name(KernelKind kernel)
{
    switch(kernel)
    {
        case KernelKind::Spin: return "spin";
        case KernelKind::MemRead: return "mem_read";
        case KernelKind::Reduction: return "reduction";
        case KernelKind::Branchy: return "branchy";
    }
    return "unknown";
}

size_t
max_element_count_for_multi()
{
    size_t max_element_count = 0;
    for(const auto& cfg : kDispatchConfigs)
    {
        const size_t element_count = static_cast<size_t>(cfg.grid_size) * static_cast<size_t>(cfg.block_size);
        if(element_count > max_element_count) max_element_count = element_count;
    }
    return max_element_count;
}

bool
looks_like_option(const char* arg)
{
    return arg != nullptr && arg[0] == '-';
}

void
launch_dispatch_config(const DispatchConfig& cfg, float*& primary, float*& secondary, hipStream_t stream)
{
    const int  element_count = static_cast<int>(cfg.grid_size * cfg.block_size);
    const dim3 grid(cfg.grid_size);
    const dim3 block(cfg.block_size);

    switch(cfg.kernel)
    {
        case KernelKind::Spin:
            hipLaunchKernelGGL(spin_kernel, grid, block, 0, stream, primary, element_count, cfg.spin_iterations);
            break;
        case KernelKind::MemRead:
            hipLaunchKernelGGL(mem_read_kernel,
                               grid,
                               block,
                               0,
                               stream,
                               primary,
                               secondary,
                               element_count,
                               cfg.spin_iterations);
            std::swap(primary, secondary);
            break;
        case KernelKind::Reduction:
            hipLaunchKernelGGL(reduction_kernel,
                               grid,
                               block,
                               cfg.block_size * sizeof(float),
                               stream,
                               primary,
                               secondary,
                               element_count,
                               cfg.spin_iterations);
            std::swap(primary, secondary);
            break;
        case KernelKind::Branchy:
            hipLaunchKernelGGL(branchy_kernel, grid, block, 0, stream, primary, element_count, cfg.spin_iterations);
            break;
    }

    HIP_CHECK(hipPeekAtLastError());
}

void
print_help()
{
    std::cout << "Usage:\n"
                 "  synthetic_dispatch_benchmark [dispatches] [grid_size] [block_size] [spin_iterations]\n"
                 "  synthetic_dispatch_benchmark --multi [dispatches_per_config]\n"
                 "\n"
                 "Single-kernel defaults:\n"
                 "  dispatches=10000 grid_size=128 block_size=64 spin_iterations=256\n"
                 "\n"
                 "Multi-kernel defaults:\n"
                 "  dispatches_per_config=100 configs="
              << kDispatchConfigs.size()
              << " total_dispatches=(dispatches_per_config * configs)\n";
}
}  // namespace

int
main(int argc, char** argv)
{
    try
    {
        bool multi_mode = false;

        int dispatches               = 10000;
        int grid_size                = 128;
        int block_size               = 64;
        int spin_iterations          = 256;
        int dispatches_per_config    = 100;
        std::vector<std::string> raw = {};

        for(int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];

            if(arg == "--help" || arg == "-h")
            {
                print_help();
                return 0;
            }

            if(arg == "--multi")
            {
                multi_mode = true;
                if(i + 1 < argc && !looks_like_option(argv[i + 1]))
                {
                    dispatches_per_config = to_int(argv[++i], "dispatches_per_config");
                }
                continue;
            }

            if(arg.rfind("--", 0) == 0)
            {
                std::cerr << "Unknown option: " << arg << '\n';
                return 1;
            }

            raw.push_back(arg);
        }

        if(multi_mode && !raw.empty())
        {
            std::cerr << "Positional arguments cannot be combined with --multi\n";
            return 1;
        }

        if(!multi_mode)
        {
            if(raw.size() > 4)
            {
                std::cerr << "Too many positional arguments. Run with --help for usage.\n";
                return 1;
            }
            if(raw.size() > 0) dispatches = to_int(raw[0].c_str(), "dispatches");
            if(raw.size() > 1) grid_size = to_int(raw[1].c_str(), "grid_size");
            if(raw.size() > 2) block_size = to_int(raw[2].c_str(), "block_size");
            if(raw.size() > 3) spin_iterations = to_int(raw[3].c_str(), "spin_iterations");
        }

        const size_t element_count = multi_mode ? max_element_count_for_multi()
                                                : static_cast<size_t>(grid_size) * static_cast<size_t>(block_size);
        const size_t num_bytes = element_count * sizeof(float);

        float* d_primary   = nullptr;
        float* d_secondary = nullptr;
        HIP_CHECK(hipMalloc(&d_primary, num_bytes));
        HIP_CHECK(hipMalloc(&d_secondary, num_bytes));
        HIP_CHECK(hipMemset(d_primary, 0, num_bytes));
        HIP_CHECK(hipMemset(d_secondary, 0, num_bytes));

        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreate(&stream));

        if(multi_mode)
        {
            const int total_dispatches =
                dispatches_per_config * static_cast<int>(kDispatchConfigs.size());
            std::cout << "[synthetic-dispatch-benchmark] mode=multi dispatches_per_config="
                      << dispatches_per_config << " config_count=" << kDispatchConfigs.size()
                      << " total_dispatches=" << total_dispatches << '\n';

            for(const auto& cfg : kDispatchConfigs)
            {
                std::cout << "[synthetic-dispatch-benchmark] config kernel=" << kernel_name(cfg.kernel)
                          << " grid_size=" << cfg.grid_size << " block_size=" << cfg.block_size
                          << " spin_iterations=" << cfg.spin_iterations << '\n';
            }

            for(int i = 0; i < dispatches_per_config; ++i)
            {
                for(const auto& cfg : kDispatchConfigs)
                {
                    launch_dispatch_config(cfg, d_primary, d_secondary, stream);
                }
            }
        }
        else
        {
            std::cout << "[synthetic-dispatch-benchmark] mode=single dispatches=" << dispatches
                      << " grid_size=" << grid_size << " block_size=" << block_size
                      << " spin_iterations=" << spin_iterations << '\n';

            const int  single_element_count = grid_size * block_size;
            const dim3 grid(static_cast<unsigned int>(grid_size));
            const dim3 block(static_cast<unsigned int>(block_size));

            for(int i = 0; i < dispatches; ++i)
            {
                hipLaunchKernelGGL(
                    spin_kernel, grid, block, 0, stream, d_primary, single_element_count, spin_iterations);
                HIP_CHECK(hipPeekAtLastError());
            }
        }

        HIP_CHECK(hipStreamSynchronize(stream));

        float result = 0.0f;
        HIP_CHECK(hipMemcpy(&result, d_primary, sizeof(float), hipMemcpyDeviceToHost));
        std::cout << std::fixed << std::setprecision(6)
                  << "[synthetic-dispatch-benchmark] sample_result=" << result << '\n';

        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipFree(d_secondary));
        HIP_CHECK(hipFree(d_primary));
        return 0;
    } catch(const std::exception& e)
    {
        std::cerr << "[synthetic-dispatch-benchmark] fatal error: " << e.what() << '\n';
        return 1;
    }
}

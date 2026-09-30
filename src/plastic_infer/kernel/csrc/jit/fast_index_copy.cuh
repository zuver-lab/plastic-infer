// De-tvm_ffi copy of FreeToken's kernel/jit/fast_index_copy.cuh.
//
// Every __global__ kernel and device helper below is VERBATIM from the
// original. What changed is the host glue: FreeToken's tvm_ffi TensorView
// entry points (TensorMatcher validation, LaunchKernel) are replaced by
// extern "C" C-ABI entry points taking raw pointers + a stream, because this
// port JIT-compiles with standalone nvcc + ctypes instead of tvm_ffi's
// load_inline (see kernel/jit.py). The validation the original did via
// TensorMatcher now lives on the Python side (kernel/fast_index_copy.py).
//
// Linux/UVA (including WSL2) maps registered host memory at its host VA, so
// device_alias is identity in practice; it is kept faithful (probe + fallback)
// rather than deleted.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace device {

template <typename T, std::size_t N>
struct device_vec {
    T data[N];
};

}  // namespace device

namespace details {

template <std::size_t kUnit>
inline constexpr auto get_mem_package() {
    if constexpr (kUnit == 16) {
    return uint4{};
    } else if constexpr (kUnit == 8) {
    return uint2{};
    } else if constexpr (kUnit == 4) {
    return uint1{};
    } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
    }
}

__always_inline __device__ auto load_nc(const uint1* __restrict__ src) -> uint1 {
    uint32_t tmp;
    asm volatile("ld.global.L1::no_allocate.b32 %0,[%1];" : "=r"(tmp) : "l"(src));
    return uint1{tmp};
}

__always_inline __device__ auto load_nc(const uint2* __restrict__ src) -> uint2 {
    uint32_t tmp0, tmp1;
    asm volatile("ld.global.L1::no_allocate.v2.b32 {%0,%1},[%2];" : "=r"(tmp0), "=r"(tmp1) : "l"(src));
    return uint2{tmp0, tmp1};
}

__always_inline __device__ auto load_nc(const uint4* __restrict__ src) -> uint4 {
    uint32_t tmp0, tmp1, tmp2, tmp3;
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3) : "l"(src));
    return uint4{tmp0, tmp1, tmp2, tmp3};
}

__always_inline __device__ void store_nc(uint1* __restrict__ dst, const uint1& value) {
    uint32_t tmp = value.x;
    asm volatile("st.global.wt.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
}

__always_inline __device__ void store_nc(uint2* __restrict__ dst, const uint2& value) {
    uint32_t tmp0 = value.x;
    uint32_t tmp1 = value.y;
    asm volatile("st.global.wt.v2.b32 [%0],{%1,%2};" ::"l"(dst), "r"(tmp0), "r"(tmp1));
}

__always_inline __device__ void store_nc(uint4* __restrict__ dst, const uint4& value) {
    uint32_t tmp0 = value.x;
    uint32_t tmp1 = value.y;
    uint32_t tmp2 = value.z;
    uint32_t tmp3 = value.w;
    asm volatile("st.global.wt.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
}

__always_inline __device__ void wait_flag_clear(const int32_t* __restrict__ flag_ptr) {
    // Exponential backoff to avoid hammering a global atomic in a tight loop.
    auto* flag = reinterpret_cast<int*>(const_cast<int32_t*>(flag_ptr));
    uint32_t sleep_ns = 128;
    while (atomicAdd(flag, 0) > 0) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(sleep_ns);
#endif
        sleep_ns = sleep_ns < 2048 ? (sleep_ns << 1) : 2048;
    }
}

template <std::size_t kUnit>
using mem_package_t = decltype(get_mem_package<kUnit>());

template <std::size_t kBytes, std::size_t kUnit, std::size_t kThreads>
__always_inline __device__ auto load_vec(const void* __restrict__ src) {
    using Package = mem_package_t<kUnit>;
    constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
    constexpr auto kLoopCount = kBytes / kBytesPerLoop;
    static_assert(kBytes % kBytesPerLoop == 0, "kBytes must be multiple of 128 bytes");

    const auto src_packed = static_cast<const Package*>(src);
    const auto lane_id = threadIdx.x % kThreads;
    device::device_vec<Package, kLoopCount> vec;

#pragma unroll kLoopCount
    for (std::size_t i = 0; i < kLoopCount; ++i) {
        const auto j = i * kThreads + lane_id;
        vec.data[i] = load_nc(src_packed + j);
    }

    return vec;
}

template <std::size_t kBytes, std::size_t kUnit, std::size_t kThreads, typename Tp>
__always_inline __device__ void store_vec(void* __restrict__ dst, const Tp& vec) {
    using Package = mem_package_t<kUnit>;
    constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
    constexpr auto kLoopCount = kBytes / kBytesPerLoop;
    static_assert(kBytes % kBytesPerLoop == 0, "kBytes must be multiple of 128 bytes");
    static_assert(std::is_same_v<Tp, device::device_vec<Package, kLoopCount>>);

    const auto dst_packed = static_cast<Package*>(dst);
    const auto lane_id = threadIdx.x % kThreads;

#pragma unroll kLoopCount
    for (std::size_t i = 0; i < kLoopCount; ++i) {
        const auto j = i * kThreads + lane_id;
        details::store_nc(dst_packed + j, vec.data[i]);
    }
}

}  // namespace details

// freetoken/utils.cuh pointer::offset, inlined here (its only other tvm_ffi-free use).
namespace pointer {

__device__ inline void* offset(void* base, std::size_t bytes) {
    return static_cast<char*>(base) + bytes;
}

__device__ inline const void* offset(const void* base, std::size_t bytes) {
    return static_cast<const char*>(base) + bytes;
}

}  // namespace pointer


// Pinned host memory is GPU-dereferenceable at its host VA only where UVA identity
// holds (Linux). On Windows/WDDM, cudaHostRegister'd memory maps to a different device
// address, so host-resident tensors are translated here -- the one point their pointer
// enters kernel params. Cached once per process: PlasticInfer pins one CUDA device per
// process (set at engine launch).
inline bool host_ptr_identity() {
    static const bool identity = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            return false;  // fail closed: translate (and surface errors), don't assume identity
        }
        int uva = 0, reg = 0;
        cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device);
        cudaDeviceGetAttribute(&reg, cudaDevAttrCanUseHostPointerForRegisteredMem, device);
        return uva == 1 && reg == 1;
    }();
    return identity;
}

inline void* device_alias(void* ptr, bool is_cuda) {
    // CUDA tensors always dereference at their own address; only host tensors can
    // need translation (WSL2/WDDM report CanUseHostPointerForRegisteredMem = 0, so
    // registered host memory is not guaranteed dereferenceable at its host VA).
    if (is_cuda || host_ptr_identity()) {
        return ptr;
    }
    void* mapped = nullptr;
    const auto err = cudaHostGetDevicePointer(&mapped, ptr, 0);
    if (err != cudaSuccess) {
        // Python-side validation guarantees registered, mapped host tensors on the
        // paths that translate; surface loudly rather than deref garbage.
        fprintf(stderr, "fast_index_copy: host tensor must be pinned+mapped "
                        "(cudaHostGetDevicePointer: %s)\n", cudaGetErrorString(err));
        return nullptr;
    }
    return mapped;
}

struct IndexKernelParams {
    void* __restrict__ dst;
    const void* __restrict__ indices_dst;
    void* __restrict__ src;
    const void* __restrict__ indices_src;
    std::size_t length;
    const int64_t* __restrict__ valid_length;
};

/*
Each worker has `kWorkerThreads` threads and handles `kWorkersFeatures` features for one index.
For one index that num_feat is large, we split it and use multiple workers to handle it in parallel.

cost of pre index read = [read index] + [copy kWorkersFeatures data]

For num_feat be small (<=1024) while length is large, set kWorkerThreads = 8:
    - so a warp (32 threads) can handle 4 indices in parallel.
    - cost one step to read 4 indices (reduce index read cost)

For num_feat be large (>2048), while length is small, set kWorkerThreads = 32, kWorkersFeatures=1024:
    - so enough threads parrallel to copy data for one index.

In one worker:
unroll `kUnrollCount` times to copy data.
each thread will use sizeof(Dtype) * kUnrollCount Bytes in one iteration.

launch (kNumBlocks, kNumThreads)

*/

template <
    typename IdType,
    std::size_t kFeatureBytes,
    std::size_t kWorkerThreads,
    std::size_t kWorkersFeatures,
    std::size_t kNumThreads, // should equal to blockDim.x
    std::size_t kNumBlocks,  // should equal to gridDim.x
    std::size_t kMaxOccupancy,
    bool kWaitOnFlag
>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void fast_index_copy(
    IndexKernelParams params,
    int32_t* sync_flag_ptr
) {
    using namespace device;
    static_assert(kNumThreads % kWorkerThreads == 0);
    constexpr auto kWorkersPerBlock = kNumThreads / kWorkerThreads;
    constexpr auto kWorkers = kWorkersPerBlock * kNumBlocks;

    const auto& [
        dst_ptr, indices_dst, src_ptr, indices_src,
        length, valid_length_ptr
    ] = params;

    const auto length_limit = valid_length_ptr ? static_cast<std::size_t>(valid_length_ptr[0]) : length;

    static_assert(kFeatureBytes % kWorkersFeatures == 0, "kFeatureBytes must be multiple of kWorkersFeatures");
    const auto kWorkersPerIndex = ((kFeatureBytes + kWorkersFeatures - 1) / kWorkersFeatures); // TODO: support not divisible

    const auto worker_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kWorkerThreads;

    constexpr auto kGranularity = 128 / kWorkerThreads;


    const auto total_work_items = length_limit * kWorkersPerIndex;
    const auto max_loops = (total_work_items + kWorkers - 1) / kWorkers;

    // Keep loop trip count identical across workers so future synchronization points
    // (e.g. block/warp polling logic) can be inserted safely.
    for (std::size_t loop = 0; loop < max_loops; ++loop) {
        if constexpr (kWaitOnFlag) {
            // One thread per block polls the shared flag and gates this loop.
            // All threads must execute this barrier the same number of times.
            if (threadIdx.x == 0) {
                details::wait_flag_clear(sync_flag_ptr);
            }
            __syncthreads();
        }

        const auto i = worker_id + loop * kWorkers;
        if (i >= total_work_items) {
            continue;
        }

        const auto index_id = i / kWorkersPerIndex;
        const auto index_subid = i % kWorkersPerIndex;
        const auto pos_src = static_cast<const IdType*>(indices_src)[index_id];
        const auto pos_dst = static_cast<const IdType*>(indices_dst)[index_id];

        const auto col = index_subid * kWorkersFeatures;
        const auto src_base = pointer::offset(src_ptr, pos_src * kFeatureBytes + col);
        const auto dst_base = pointer::offset(dst_ptr, pos_dst * kFeatureBytes + col);

        const auto vec = details::load_vec<kWorkersFeatures, kGranularity, kWorkerThreads>(src_base);
        details::store_vec<kWorkersFeatures, kGranularity, kWorkerThreads>(dst_base, vec);
    }
}

__global__ void update_copy_flag_kernel(int32_t* flag_ptr, int delta) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<int*>(flag_ptr), delta);
    }
}

/// Manually update the sync flag by delta. Use for signaling normal-priority
/// workers to pause (delta > 0) or resume (delta < 0). Mirrors the original
/// update_copy_flag (TensorMatcher validation moved to Python).
inline void update_copy_flag(int32_t* flag_ptr, int32_t delta, cudaStream_t stream) {
    update_copy_flag_kernel<<<1, 1, 0, stream>>>(flag_ptr, delta);
}


template <
    std::size_t kFeatureBytes,
    std::size_t kWorkerThreads,
    std::size_t kWorkersFeatures,
    std::size_t kNumThreads, // should equal to blockDim.x
    std::size_t kNumBlocks,  // should equal to gridDim.x
    std::size_t kMaxOccupancy
>
struct FastIndexCopyKernel {

    template <typename IdType>
    static constexpr auto _kernel_nowait = fast_index_copy<
        IdType,
        kFeatureBytes,
        kWorkerThreads,
        kWorkersFeatures,
        kNumThreads,
        kNumBlocks,
        kMaxOccupancy,
        false
    >;

    template <typename IdType>
    static constexpr auto _kernel_wait = fast_index_copy<
        IdType,
        kFeatureBytes,
        kWorkerThreads,
        kWorkersFeatures,
        kNumThreads,
        kNumBlocks,
        kMaxOccupancy,
        true
    >;

    enum class PriorityMode {
        kDefault,
        kHigh,
        kNormal
    };

    // mode: 0 default, 1 high, 2 normal (the original PriorityMode order).
    static void run_impl(
        void* dst,
        const void* dst_indices,
        void* src,
        const void* src_indices,
        std::size_t length,
        const int64_t* num_indices,
        bool use_int32,
        int32_t* sync_flag,
        int mode,
        cudaStream_t stream
    ) {
        // dst is always the CUDA slot cache; src is always a host bank in the
        // offload cache's usage (FreeToken's kDLCUDA / kDLCUDAHost distinction).
        void* dst_ptr = device_alias(dst, true);
        void* src_ptr = device_alias(src, false);

        const auto params = IndexKernelParams{
            dst_ptr,
            dst_indices,
            src_ptr,
            src_indices,
            length,
            num_indices
        };

        if (mode == 1) {  // kHigh: pause normal-priority workers, copy, resume
            update_copy_flag(sync_flag, 1, stream);
            const auto kernel = use_int32 ? _kernel_nowait<int32_t> : _kernel_nowait<int64_t>;
            kernel<<<kNumBlocks, kNumThreads, 0, stream>>>(params, static_cast<int32_t*>(nullptr));
            update_copy_flag(sync_flag, -1, stream);
            return;
        }

        if (mode == 2) {  // kNormal: gate each loop on the flag being clear
            const auto kernel = use_int32 ? _kernel_wait<int32_t> : _kernel_wait<int64_t>;
            kernel<<<kNumBlocks, kNumThreads, 0, stream>>>(params, sync_flag);
            return;
        }

        const auto kernel = use_int32 ? _kernel_nowait<int32_t> : _kernel_nowait<int64_t>;
        kernel<<<kNumBlocks, kNumThreads, 0, stream>>>(params, static_cast<int32_t*>(nullptr));
    }

    static void run(
        void* dst,
        const void* dst_indices,
        void* src,
        const void* src_indices,
        std::size_t length,
        const int64_t* num_indices
    ) {
        run_impl(dst, dst_indices, src, src_indices, length, num_indices,
                 true, nullptr, 0, nullptr);
    }

    static void run_high(
        void* dst,
        const void* dst_indices,
        void* src,
        const void* src_indices,
        std::size_t length,
        const int64_t* num_indices,
        int32_t* sync_flag
    ) {
        run_impl(dst, dst_indices, src, src_indices, length, num_indices,
                 true, sync_flag, 1, nullptr);
    }

    static void run_normal(
        void* dst,
        const void* dst_indices,
        void* src,
        const void* src_indices,
        std::size_t length,
        const int64_t* num_indices,
        int32_t* sync_flag
    ) {
        run_impl(dst, dst_indices, src, src_indices, length, num_indices,
                 true, sync_flag, 2, nullptr);
    }
};


// ---------------------------------------------------------------------------
// Multi-bank fused index copy. The offload cache copies the SAME rows
// (dst_indices=evict_slots <- src_indices) for every registered bank, but the
// banks have distinct per-row feature byte sizes, so the single-bank kernel above
// needs one launch per bank (e.g. 6 banks * 36 layers = 216 launches/decode step,
// all near-empty at a warm/full cache). This fuses every bank into one launch:
// block b copies bank `b = blockIdx.x / kBlocksPerBank`, grid-striding over that
// bank's (num_indices * feat/16) 16-byte units. Pointers + feature sizes are passed
// as small device arrays (built once by the cache), so it stays CUDA-graph capturable.
struct MultiIndexCopyParams {
    const int64_t* __restrict__ dst_ptrs;     // [B] device, each base addr of a bank slot cache
    const int64_t* __restrict__ src_ptrs;     // [B] device, each GPU-visible base addr of a bank host source
    const int64_t* __restrict__ feat_bytes;   // [B] device, per-row bytes (multiple of 16)
    const void* __restrict__ dst_indices;     // [L]
    const void* __restrict__ src_indices;     // [L]
    const int64_t* __restrict__ valid_length; // [1] or null
    int64_t length;                           // max L
    int num_banks;
};

template <typename IdType, std::size_t kNumThreads, std::size_t kBlocksPerBank>
__global__ __launch_bounds__(kNumThreads) void fast_index_copy_multi(
    const __grid_constant__ MultiIndexCopyParams p
) {
    const int b = static_cast<int>(blockIdx.x / kBlocksPerBank);
    if (b >= p.num_banks) {
        return;
    }
    const int blk = static_cast<int>(blockIdx.x % kBlocksPerBank);
    const auto* src = reinterpret_cast<const uint8_t*>(p.src_ptrs[b]);
    auto* dst = reinterpret_cast<uint8_t*>(p.dst_ptrs[b]);
    const int64_t feat = p.feat_bytes[b];
    const int64_t n = p.valid_length ? p.valid_length[0] : p.length;
    const int64_t units = feat >> 4;  // 16-byte (uint4) units per row; feat % 16 == 0
    const int64_t total = n * units;
    const auto* di = static_cast<const IdType*>(p.dst_indices);
    const auto* si = static_cast<const IdType*>(p.src_indices);
    const int64_t stride = static_cast<int64_t>(kBlocksPerBank) * kNumThreads;
    for (int64_t u = static_cast<int64_t>(blk) * kNumThreads + threadIdx.x; u < total; u += stride) {
        const int64_t row = u / units;
        const int64_t col = (u - row * units) << 4;  // byte offset within the row
        const int64_t pd = static_cast<int64_t>(di[row]);
        const int64_t ps = static_cast<int64_t>(si[row]);
        const uint4 v = *reinterpret_cast<const uint4*>(src + ps * feat + col);
        *reinterpret_cast<uint4*>(dst + pd * feat + col) = v;
    }
}

template <std::size_t kNumThreads, std::size_t kBlocksPerBank>
struct MultiIndexCopyKernel {
    static void run(
        const int64_t* dst_ptrs,
        const int64_t* src_ptrs,
        const int64_t* feat_bytes,
        const void* dst_indices,
        const void* src_indices,
        std::size_t length,
        const int64_t* num_indices,
        int num_banks,
        bool use_int32,
        cudaStream_t stream
    ) {
        const auto params = MultiIndexCopyParams{
            dst_ptrs,
            src_ptrs,
            feat_bytes,
            dst_indices,
            src_indices,
            num_indices,
            static_cast<int64_t>(length),
            num_banks,
        };
        const auto kernel = use_int32
            ? fast_index_copy_multi<int32_t, kNumThreads, kBlocksPerBank>
            : fast_index_copy_multi<int64_t, kNumThreads, kBlocksPerBank>;
        kernel<<<static_cast<std::size_t>(kBlocksPerBank) * num_banks, kNumThreads,
                 0, stream>>>(params);
    }
};

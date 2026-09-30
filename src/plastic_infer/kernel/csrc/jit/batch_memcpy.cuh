#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

// Host wrapper over cudaMemcpyBatchAsync (CUDA >= 13.0, the 8-argument signature;
// 12.8/12.9 carried an extra failIdx parameter): enqueue N independent
// pointer-to-pointer copies with ONE runtime call, on an explicit (non-legacy)
// stream. Callers hand pre-resolved raw addresses; copies within a batch are
// unordered, so entries must be pairwise independent.
//
// De-tvm_ffi port: FreeToken's tvm::ffi::TensorView args become raw C pointers
// (int64 -> the C-ABI entry passes n separately), and CUDA_CHECK's abort is
// replaced by a loud stderr print -- a failed copy is caught by the Python probe
// (wrong bytes) and routes to the caller's legacy fallback path.
struct BatchMemcpy {
    static void run(
        const void* const* dst_ptrs,
        const void* const* src_ptrs,
        const std::size_t* sizes,
        std::size_t n,
        cudaStream_t stream
    ) {
#if CUDART_VERSION >= 13000
        if (n == 0) {
            return;
        }
        if (stream == nullptr) {
            fprintf(stderr, "batch_memcpy: cudaMemcpyBatchAsync rejects the legacy NULL stream\n");
            return;
        }
        auto attr = ::cudaMemcpyAttributes{};
        attr.srcAccessOrder = ::cudaMemcpySrcAccessOrderStream;
        std::size_t attr_idx = 0;
        const auto err = ::cudaMemcpyBatchAsync(
            const_cast<void* const*>(dst_ptrs),
            src_ptrs,
            sizes,
            n,
            &attr,
            &attr_idx,
            1,
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "batch_memcpy: cudaMemcpyBatchAsync failed: %s\n",
                    cudaGetErrorString(err));
        }
#else
        fprintf(stderr, "batch_memcpy: cudaMemcpyBatchAsync requires CUDA >= 13.0 at build time\n");
#endif
    }
};

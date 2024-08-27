/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/gpu_flash_attn_helper_kernel.h"

#include <cub/cub.cuh>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace xla {
namespace gpu {

#define unlikely(x) __builtin_expect(!!(x), 0)

#define RETURN_IF_CUDA_ERROR(expr)       \
  do {                                   \
    cudaError_t _res = (expr);           \
    if (unlikely(_res != cudaSuccess)) { \
      return _res;                       \
    }                                    \
  } while (0)

template <typename InputT, typename OutputT, typename IndexT>
cudaError_t RunFlashAttnRowSum(cudaStream_t stream, const InputT* mask,
                               OutputT* batch_seqlens, const IndexT batch_size,
                               const IndexT seqlen, void* temp_storage_ptr,
                               size_t temp_storage_bytes) {
  bool malloced_temp_storage = false;
  size_t actual_temp_storage_bytes;
  // Calculate the actual temp storage size
  auto offset_it = thrust::make_transform_iterator(
      thrust::counting_iterator<IndexT>(0), thrust::placeholders::_1 * seqlen);
  RETURN_IF_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(
      nullptr, actual_temp_storage_bytes, mask, batch_seqlens, batch_size + 1,
      offset_it, offset_it + 1, stream));
  if (actual_temp_storage_bytes > temp_storage_bytes) {
    RETURN_IF_CUDA_ERROR(
        cudaMallocAsync(&temp_storage_ptr, actual_temp_storage_bytes, stream));
    malloced_temp_storage = true;
  }
  // Calculate sum of each row(batch_seqlens)
  RETURN_IF_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(
      temp_storage_ptr, actual_temp_storage_bytes, mask, batch_seqlens,
      batch_size + 1, offset_it, offset_it + 1, stream));
  if (malloced_temp_storage) {
    RETURN_IF_CUDA_ERROR(cudaFreeAsync(temp_storage_ptr, stream));
  }
  return cudaSuccess;
}
template cudaError_t RunFlashAttnRowSum<bool, int, int>(
    cudaStream_t stream, const bool* mask, int* batch_seqlens,
    const int batch_size, const int seqlen, void* temp_storage_ptr,
    size_t temp_storage_bytes);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnCumSum(cudaStream_t stream, const ElementT* input,
                               ElementT* output, const IndexT input_size,
                               void* temp_storage_ptr,
                               size_t temp_storage_bytes) {
  bool malloced_temp_storage = false;
  size_t actual_temp_storage_bytes;
  // Calculate the actual temp storage size
  RETURN_IF_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(
      nullptr, actual_temp_storage_bytes, input, output, input_size, stream));
  if (actual_temp_storage_bytes > temp_storage_bytes) {
    RETURN_IF_CUDA_ERROR(
        cudaMallocAsync(&temp_storage_ptr, actual_temp_storage_bytes, stream));
    malloced_temp_storage = true;
  }
  // Calculate cumulative sum(cu_seqlen)
  RETURN_IF_CUDA_ERROR(
      cub::DeviceScan::ExclusiveSum(temp_storage_ptr, actual_temp_storage_bytes,
                                    input, output, input_size, stream));
  if (malloced_temp_storage) {
    RETURN_IF_CUDA_ERROR(cudaFreeAsync(temp_storage_ptr, stream));
  }
  return cudaSuccess;
}
template cudaError_t RunFlashAttnCumSum<int, int>(cudaStream_t stream,
                                                  const int* input, int* output,
                                                  const int input_size,
                                                  void* temp_storage_ptr,
                                                  size_t temp_storage_bytes);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnMax(cudaStream_t stream, const ElementT* input,
                            ElementT* output, const IndexT input_size,
                            void* temp_storage_ptr, size_t temp_storage_bytes) {
  bool malloced_temp_storage = false;
  size_t actual_temp_storage_bytes;
  // Calculate the actual temp storage size
  RETURN_IF_CUDA_ERROR(cub::DeviceReduce::Max(
      nullptr, actual_temp_storage_bytes, input, output, input_size, stream));
  if (actual_temp_storage_bytes > temp_storage_bytes) {
    RETURN_IF_CUDA_ERROR(
        cudaMallocAsync(&temp_storage_ptr, actual_temp_storage_bytes, stream));
    malloced_temp_storage = true;
  }
  // Calculate max value(max_seqlen)
  RETURN_IF_CUDA_ERROR(cub::DeviceReduce::Max(temp_storage_ptr,
                                              actual_temp_storage_bytes, input,
                                              output, input_size, stream));
  if (malloced_temp_storage) {
    RETURN_IF_CUDA_ERROR(cudaFreeAsync(temp_storage_ptr, stream));
  }
  return cudaSuccess;
}
template cudaError_t RunFlashAttnMax<int, int>(cudaStream_t stream,
                                               const int* input, int* output,
                                               const int input_size,
                                               void* temp_storage_ptr,
                                               size_t temp_storage_bytes);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnNonzero(cudaStream_t stream, const ElementT* mask,
                                IndexT* indices, IndexT* nnz,
                                const IndexT max_nnz, void* temp_storage_ptr,
                                size_t temp_storage_bytes) {
  bool malloced_temp_storage = false;
  size_t actual_temp_storage_bytes;
  auto counting_itr = thrust::make_counting_iterator<IndexT>(0);
  auto nonzero_itr = thrust::make_transform_iterator(
      mask, thrust::placeholders::_1 != ElementT(0));
  // Calculate the actual temp storage size
  RETURN_IF_CUDA_ERROR(cub::DeviceSelect::Flagged(
      nullptr, actual_temp_storage_bytes, counting_itr, nonzero_itr, indices,
      nnz, max_nnz, stream));
  if (actual_temp_storage_bytes > temp_storage_bytes) {
    RETURN_IF_CUDA_ERROR(
        cudaMallocAsync(&temp_storage_ptr, actual_temp_storage_bytes, stream));
    malloced_temp_storage = true;
  }
  auto reverse_counting_itr = thrust::make_transform_iterator(
      counting_itr, max_nnz - thrust::placeholders::_1 - 1);
  auto reverse_mask =
      thrust::make_reverse_iterator<const ElementT*>(mask + max_nnz);
  auto zero_itr = thrust::make_transform_iterator(
      reverse_mask, thrust::placeholders::_1 == ElementT(0));
  auto reverse_indices =
      thrust::make_reverse_iterator<IndexT*>(indices + max_nnz);

  // First, place the indices of all zero elements in the last `max_nzz - nnz`
  // positions of `indices`, then put the indices of all non-zero elements in
  // the first nnz positions of `indices`. The order of these operations cannot
  // be reversed because ultimately we need to obtain the number of all non-zero
  // elements, which will be stored in the memory pointed to by the `nnz`
  // pointer.
  RETURN_IF_CUDA_ERROR(cub::DeviceSelect::Flagged(
      temp_storage_ptr, actual_temp_storage_bytes, reverse_counting_itr,
      zero_itr, reverse_indices, nnz, max_nnz, stream));
  RETURN_IF_CUDA_ERROR(cub::DeviceSelect::Flagged(
      temp_storage_ptr, actual_temp_storage_bytes, counting_itr, nonzero_itr,
      indices, nnz, max_nnz, stream));

  if (malloced_temp_storage) {
    RETURN_IF_CUDA_ERROR(cudaFreeAsync(temp_storage_ptr, stream));
  }
  return cudaSuccess;
}
template cudaError_t RunFlashAttnNonzero<bool, int>(
    cudaStream_t stream, const bool* mask, int* indices, int* nnz,
    const int max_nnz, void* temp_storage_ptr, size_t temp_storage_bytes);

namespace {

template <typename ElementT, typename IndexT>
__global__ void flash_attn_gather_kernel(const ElementT* __restrict__ input,
                                         ElementT* __restrict__ output_unpad,
                                         const IndexT* __restrict__ indices,
                                         const IndexT* __restrict__ nnz,
                                         const IndexT hidden_size) {
  __shared__ IndexT src_row;
  const IndexT dst_row = blockIdx.x;
  if (dst_row < __ldg(nnz)) {
    if (threadIdx.x == 0) {
      src_row = __ldg(&indices[dst_row]);
    }
    __syncthreads();
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    const IndexT dst_idx = dst_row * hidden_size + col;
    const IndexT src_idx = src_row * hidden_size + col;
    output_unpad[dst_idx] = input[src_idx];
  }
}

template <typename ElementT, typename IndexT>
__global__ void flash_attn_gather3_kernel(
    const ElementT* __restrict__ query, ElementT* __restrict__ query_unpad,
    const ElementT* __restrict__ key, ElementT* __restrict__ key_unpad,
    const ElementT* __restrict__ value, ElementT* __restrict__ value_unpad,
    const IndexT* __restrict__ indices, const IndexT* __restrict__ nnz,
    const IndexT hidden_size) {
  __shared__ IndexT src_row;
  const IndexT dst_row = blockIdx.x;
  if (dst_row < __ldg(nnz)) {
    if (threadIdx.x == 0) {
      src_row = __ldg(&indices[dst_row]);
    }
    __syncthreads();
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    const IndexT dst_idx = dst_row * hidden_size + col;
    const IndexT src_idx = src_row * hidden_size + col;
    query_unpad[dst_idx] = query[src_idx];
    key_unpad[dst_idx] = key[src_idx];
    value_unpad[dst_idx] = value[src_idx];
  }
}

template <typename ElementT, typename IndexT>
__global__ void flash_attn_scatter_kernel(
    const ElementT* __restrict__ input_unpad, ElementT* __restrict__ output,
    const IndexT* __restrict__ indices, const IndexT* __restrict__ nnz,
    const IndexT hidden_size) {
  __shared__ IndexT dst_row;
  const IndexT src_row = blockIdx.x;
  if (threadIdx.x == 0) {
    dst_row = __ldg(&indices[src_row]);
  }
  __syncthreads();
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  const IndexT dst_idx = dst_row * hidden_size + col;
  if (src_row < __ldg(nnz)) {
    const IndexT src_idx = src_row * hidden_size + col;
    output[dst_idx] = input_unpad[src_idx];
  } else {
    output[dst_idx] = ElementT();
  }
}

template <typename ElementT, typename IndexT>
__global__ void flash_attn_scatter3_kernel(
    const ElementT* __restrict__ query_unpad, ElementT* __restrict__ query,
    const ElementT* __restrict__ key_unpad, ElementT* __restrict__ key,
    const ElementT* __restrict__ value_unpad, ElementT* __restrict__ value,
    const IndexT* __restrict__ indices, const IndexT* __restrict__ nnz,
    const IndexT hidden_size) {
  __shared__ IndexT dst_row;
  const IndexT src_row = blockIdx.x;
  if (threadIdx.x == 0) {
    dst_row = __ldg(&indices[src_row]);
  }
  __syncthreads();
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  const IndexT dst_idx = dst_row * hidden_size + col;
  if (src_row < __ldg(nnz)) {
    const IndexT src_idx = src_row * hidden_size + col;
    query[dst_idx] = query_unpad[src_idx];
    key[dst_idx] = key_unpad[src_idx];
    value[dst_idx] = value_unpad[src_idx];
  } else {
    query[dst_idx] = ElementT();
    key[dst_idx] = ElementT();
    value[dst_idx] = ElementT();
  }
}

template <typename ElementT, typename IndexT>
__global__ void flash_attn_mqa_gqa_sum_kernel(
    const ElementT* __restrict__ dk_expanded, ElementT* __restrict__ dk,
    const ElementT* __restrict__ dv_expanded, ElementT* __restrict__ dv,
    const IndexT* __restrict__ total_k, const IndexT ngroups,
    const IndexT stride0, const IndexT stride1, const IndexT stride2,
    const IndexT new_stride0, const IndexT new_stride1) {
  if (total_k && blockIdx.x >= __ldg(total_k)) {
    return;
  }
  using ComputeT = float;
  ComputeT dk_sum = ComputeT();
  ComputeT dv_sum = ComputeT();
  cutlass::NumericConverter<ComputeT, ElementT> to_computeT;
  cutlass::NumericConverter<ElementT, ComputeT> to_elementT;
  const IndexT src_idx_base =
      blockIdx.x * stride0 + blockIdx.y * stride1 + threadIdx.x;
  for (IndexT i = 0; i < ngroups; ++i) {
    const IndexT src_idx = src_idx_base + i * stride2;
    dk_sum += to_computeT(dk_expanded[src_idx]);
    dv_sum += to_computeT(dv_expanded[src_idx]);
  }
  const IndexT dst_idx =
      blockIdx.x * new_stride0 + blockIdx.y * new_stride1 + threadIdx.x;
  dk[dst_idx] = to_elementT(dk_sum);
  dv[dst_idx] = to_elementT(dv_sum);
}

}  // namespace

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnGather(cudaStream_t stream, const ElementT* input,
                               ElementT* output_unpad, const IndexT* indices,
                               const IndexT* nnz, const IndexT max_nnz,
                               const IndexT hidden_size) {
  // Upcast to uint128_t data type
  using UpcastT = cutlass::uint128_t;
  static_assert(!std::is_same_v<ElementT, bool>);
  const IndexT factor = cutlass::sizeof_bits<UpcastT>::value /
                        cutlass::sizeof_bits<ElementT>::value;
  assert(hidden_size % factor == 0);
  const IndexT hidden_size_upcast = hidden_size / factor;

  const void* kernel =
      reinterpret_cast<void*>(&flash_attn_gather_kernel<UpcastT, IndexT>);
  const int threads_per_block = std::min(hidden_size_upcast, 1024);
  dim3 grid(max_nnz,
            (hidden_size_upcast + threads_per_block - 1) / threads_per_block);
  dim3 block(threads_per_block);

  void* kernel_args[] = {
      reinterpret_cast<void*>(&input),
      reinterpret_cast<void*>(&output_unpad),
      reinterpret_cast<void*>(&indices),
      reinterpret_cast<void*>(&nnz),
      reinterpret_cast<void*>(const_cast<IndexT*>(&hidden_size_upcast)),
  };

  RETURN_IF_CUDA_ERROR(
      cudaLaunchKernel(kernel, grid, block, kernel_args, 0, stream));

  return cudaSuccess;
}
template cudaError_t RunFlashAttnGather<cutlass::half_t, int>(
    cudaStream_t stream, const cutlass::half_t* input,
    cutlass::half_t* output_unpad, const int* indices, const int* nnz,
    const int max_nnz, const int hidden_size);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnGather3(cudaStream_t stream, const ElementT* query,
                                ElementT* query_unpad, const ElementT* key,
                                ElementT* key_unpad, const ElementT* value,
                                ElementT* value_unpad, const IndexT* indices,
                                const IndexT* nnz, const IndexT max_nnz,
                                const IndexT hidden_size) {
  // Upcast to uint128_t data type
  using UpcastT = cutlass::uint128_t;
  static_assert(!std::is_same_v<ElementT, bool>);
  const IndexT factor = cutlass::sizeof_bits<UpcastT>::value /
                        cutlass::sizeof_bits<ElementT>::value;
  assert(hidden_size % factor == 0);
  const IndexT hidden_size_upcast = hidden_size / factor;

  const void* kernel =
      reinterpret_cast<void*>(&flash_attn_gather3_kernel<UpcastT, IndexT>);
  const int threads_per_block = std::min(hidden_size_upcast, 1024);
  dim3 grid(max_nnz,
            (hidden_size_upcast + threads_per_block - 1) / threads_per_block);
  dim3 block(threads_per_block);

  void* kernel_args[] = {
      reinterpret_cast<void*>(&query),
      reinterpret_cast<void*>(&query_unpad),
      reinterpret_cast<void*>(&key),
      reinterpret_cast<void*>(&key_unpad),
      reinterpret_cast<void*>(&value),
      reinterpret_cast<void*>(&value_unpad),
      reinterpret_cast<void*>(&indices),
      reinterpret_cast<void*>(&nnz),
      reinterpret_cast<void*>(const_cast<IndexT*>(&hidden_size_upcast)),
  };

  RETURN_IF_CUDA_ERROR(
      cudaLaunchKernel(kernel, grid, block, kernel_args, 0, stream));

  return cudaSuccess;
}
template cudaError_t RunFlashAttnGather3<cutlass::half_t, int>(
    cudaStream_t stream, const cutlass::half_t* query,
    cutlass::half_t* query_unpad, const cutlass::half_t* key,
    cutlass::half_t* key_unpad, const cutlass::half_t* value,
    cutlass::half_t* value_unpad, const int* indices, const int* nnz,
    const int max_nnz, const int hidden_size);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnScatter(cudaStream_t stream,
                                const ElementT* input_unpad, ElementT* output,
                                const IndexT* indices, const IndexT* nnz,
                                const IndexT max_nnz,
                                const IndexT hidden_size) {
  // Upcast to uint128_t data type
  using UpcastT = cutlass::uint128_t;
  static_assert(!std::is_same_v<ElementT, bool>);
  const IndexT factor = cutlass::sizeof_bits<UpcastT>::value /
                        cutlass::sizeof_bits<ElementT>::value;
  assert(hidden_size % factor == 0);
  const IndexT hidden_size_upcast = hidden_size / factor;

  const void* kernel =
      reinterpret_cast<void*>(&flash_attn_scatter_kernel<UpcastT, IndexT>);
  const int threads_per_block = std::min(hidden_size_upcast, 1024);
  dim3 grid(max_nnz,
            (hidden_size_upcast + threads_per_block - 1) / threads_per_block);
  dim3 block(threads_per_block);
  void* kernel_args[] = {
      reinterpret_cast<void*>(&input_unpad),
      reinterpret_cast<void*>(&output),
      reinterpret_cast<void*>(&indices),
      reinterpret_cast<void*>(&nnz),
      reinterpret_cast<void*>(const_cast<IndexT*>(&hidden_size_upcast)),
  };

  RETURN_IF_CUDA_ERROR(
      cudaLaunchKernel(kernel, grid, block, kernel_args, 0, stream));

  return cudaSuccess;
}
template cudaError_t RunFlashAttnScatter<cutlass::half_t, int>(
    cudaStream_t stream, const cutlass::half_t* input_unpad,
    cutlass::half_t* output, const int* indices, const int* nnz,
    const int max_nnz, const int hidden_size);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnScatter3(cudaStream_t stream,
                                 const ElementT* query_unpad, ElementT* query,
                                 const ElementT* key_unpad, ElementT* key,
                                 const ElementT* value_unpad, ElementT* value,
                                 const IndexT* indices, const IndexT* nnz,
                                 const IndexT max_nnz,
                                 const IndexT hidden_size) {
  // Upcast to uint128_t data type
  using UpcastT = cutlass::uint128_t;
  static_assert(!std::is_same_v<ElementT, bool>);
  const IndexT factor = cutlass::sizeof_bits<UpcastT>::value /
                        cutlass::sizeof_bits<ElementT>::value;
  assert(hidden_size % factor == 0);
  const IndexT hidden_size_upcast = hidden_size / factor;

  const void* kernel =
      reinterpret_cast<void*>(&flash_attn_scatter3_kernel<UpcastT, IndexT>);
  const int threads_per_block = std::min(hidden_size_upcast, 1024);
  dim3 grid(max_nnz,
            (hidden_size_upcast + threads_per_block - 1) / threads_per_block);
  dim3 block(threads_per_block);
  void* kernel_args[] = {
      reinterpret_cast<void*>(&query_unpad),
      reinterpret_cast<void*>(&query),
      reinterpret_cast<void*>(&key_unpad),
      reinterpret_cast<void*>(&key),
      reinterpret_cast<void*>(&value_unpad),
      reinterpret_cast<void*>(&value),
      reinterpret_cast<void*>(&indices),
      reinterpret_cast<void*>(&nnz),
      reinterpret_cast<void*>(const_cast<IndexT*>(&hidden_size_upcast)),
  };

  RETURN_IF_CUDA_ERROR(
      cudaLaunchKernel(kernel, grid, block, kernel_args, 0, stream));

  return cudaSuccess;
}
template cudaError_t RunFlashAttnScatter3<cutlass::half_t, int>(
    cudaStream_t stream, const cutlass::half_t* query_unpad,
    cutlass::half_t* query, const cutlass::half_t* key_unpad,
    cutlass::half_t* key, const cutlass::half_t* value_unpad,
    cutlass::half_t* value, const int* indices, const int* nnz,
    const int max_nnz, const int hidden_size);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnMqaGqaSum(
    cudaStream_t stream, const ElementT* dk_expanded, ElementT* dk,
    const ElementT* dv_expanded, ElementT* dv, const IndexT* total_k,
    const IndexT max_total_k, const IndexT num_heads_k, const IndexT ngroups,
    const IndexT head_size) {
  assert(head_size <= 1024);
  const IndexT stirde0 = num_heads_k * ngroups * head_size;
  const IndexT stride1 = ngroups * head_size;
  const IndexT stride2 = head_size;
  const IndexT new_stride0 = num_heads_k * head_size;
  const IndexT new_stride1 = head_size;

  const void* kernel =
      reinterpret_cast<void*>(&flash_attn_mqa_gqa_sum_kernel<ElementT, IndexT>);
  const int thread_per_block = head_size;
  dim3 block(thread_per_block);
  dim3 grid(max_total_k, num_heads_k);
  void* kernel_args[] = {
      reinterpret_cast<void*>(&dk_expanded),
      reinterpret_cast<void*>(&dk),
      reinterpret_cast<void*>(&dv_expanded),
      reinterpret_cast<void*>(&dv),
      reinterpret_cast<void*>(&total_k),
      reinterpret_cast<void*>(const_cast<IndexT*>(&ngroups)),
      reinterpret_cast<void*>(const_cast<IndexT*>(&stirde0)),
      reinterpret_cast<void*>(const_cast<IndexT*>(&stride1)),
      reinterpret_cast<void*>(const_cast<IndexT*>(&stride2)),
      reinterpret_cast<void*>(const_cast<IndexT*>(&new_stride0)),
      reinterpret_cast<void*>(const_cast<IndexT*>(&new_stride1)),
  };

  RETURN_IF_CUDA_ERROR(
      cudaLaunchKernel(kernel, grid, block, kernel_args, 0, stream));

  return cudaSuccess;
}
template cudaError_t RunFlashAttnMqaGqaSum<cutlass::half_t, int>(
    cudaStream_t stream, const cutlass::half_t* dk_expanded,
    cutlass::half_t* dk, const cutlass::half_t* dv_expanded,
    cutlass::half_t* dv, const int* total_k, const int max_total_k,
    const int num_heads_k, const int ngroups, const int head_size);
template cudaError_t RunFlashAttnMqaGqaSum<cutlass::bfloat16_t, int>(
    cudaStream_t stream, const cutlass::bfloat16_t* dk_expanded,
    cutlass::bfloat16_t* dk, const cutlass::bfloat16_t* dv_expanded,
    cutlass::bfloat16_t* dv, const int* total_k, const int max_total_k,
    const int num_heads_k, const int ngroups, const int head_size);

}  // namespace gpu
}  // namespace xla

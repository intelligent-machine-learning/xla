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

#ifndef XLA_SERVICE_GPU_GPU_FLASH_ATTN_HELPER_KERNEL_H_
#define XLA_SERVICE_GPU_GPU_FLASH_ATTN_HELPER_KERNEL_H_

#include <cuda_runtime.h>

namespace xla {
namespace gpu {

template <typename InputT, typename OutputT, typename IndexT>
cudaError_t RunFlashAttnRowSum(cudaStream_t stream, const InputT* mask,
                               OutputT* batch_seqlens, const IndexT batch_size,
                               const IndexT seqlen, void* temp_storage_ptr,
                               size_t temp_storage_bytes);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnCumSum(cudaStream_t stream, const ElementT* input,
                               ElementT* output, const IndexT input_size,
                               void* temp_storage_ptr,
                               size_t temp_storage_bytes);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnMax(cudaStream_t stream, const ElementT* input,
                            ElementT* output, const IndexT input_size,
                            void* temp_storage_ptr, size_t temp_storage_bytes);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnNonzero(cudaStream_t stream, const ElementT* mask,
                                IndexT* indices, IndexT* nnz,
                                const IndexT max_nnz, void* temp_storage_ptr,
                                size_t temp_storage_bytes);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnGather(cudaStream_t stream, const ElementT* input,
                               ElementT* output_unpad, const IndexT* indices,
                               const IndexT* nnz, const IndexT max_nnz,
                               const IndexT hidden_size);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnGather3(cudaStream_t stream, const ElementT* query,
                                ElementT* query_unpad, const ElementT* key,
                                ElementT* key_unpad, const ElementT* value,
                                ElementT* value_unpad, const IndexT* indices,
                                const IndexT* nnz, const IndexT max_nnz,
                                const IndexT hidden_size);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnScatter(cudaStream_t stream,
                                const ElementT* input_unpad, ElementT* output,
                                const IndexT* indices, const IndexT* nnz,
                                const IndexT max_nnz, const IndexT hidden_size);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnScatter3(cudaStream_t stream,
                                 const ElementT* query_unpad, ElementT* query,
                                 const ElementT* key_unpad, ElementT* key,
                                 const ElementT* value_unpad, ElementT* value,
                                 const IndexT* indices, const IndexT* nnz,
                                 const IndexT max_nnz,
                                 const IndexT hidden_size);

template <typename ElementT, typename IndexT>
cudaError_t RunFlashAttnMqaGqaSum(cudaStream_t stream,
                                  const ElementT* dk_expanded, ElementT* dk,
                                  const ElementT* dv_expanded, ElementT* dv,
                                  const IndexT* total_k,
                                  const IndexT max_total_k,
                                  const IndexT num_heads_k,
                                  const IndexT ngroups, const IndexT head_size);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_FLASH_ATTN_HELPER_KERNEL_H_

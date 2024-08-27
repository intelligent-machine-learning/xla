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

#ifndef XLA_SERVICE_GPU_RUNTIME_FLASH_ATTN_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_FLASH_ATTN_THUNK_H_

#include "absl/status/status.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_flash_attn.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

class FlashAttnForwardThunk : public Thunk {
 public:
  FlashAttnForwardThunk(
      ThunkInfo thunk_info, FlashAttnConfig config,
      BufferAllocation::Slice query_slice, BufferAllocation::Slice key_slice,
      BufferAllocation::Slice value_slice,
      BufferAllocation::Slice alibi_slopes_slice, /* may be null */
      BufferAllocation::Slice output_slice,
      BufferAllocation::Slice softmax_lse_slice,
      BufferAllocation::Slice rng_state_slice,
      BufferAllocation::Slice s_dmask_slice /* may be null */,
      BufferAllocation::Slice output_accum_slice, /* may be null */
      BufferAllocation::Slice softmax_lse_accum_slice /* may be null */);

  FlashAttnForwardThunk(const FlashAttnForwardThunk &) = delete;
  FlashAttnForwardThunk &operator=(const FlashAttnForwardThunk &) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams &params) override;

 protected:
  const FlashAttnConfig config_;

  BufferAllocation::Slice query_buffer_;         // input
  BufferAllocation::Slice key_buffer_;           // input
  BufferAllocation::Slice value_buffer_;         // input
  BufferAllocation::Slice alibi_slopes_buffer_;  // input(optional)

  BufferAllocation::Slice output_buffer_;             // output
  BufferAllocation::Slice softmax_lse_buffer_;        // output
  BufferAllocation::Slice rng_state_buffer_;          // output
  BufferAllocation::Slice s_dmask_buffer_;            // output(optional)
  BufferAllocation::Slice output_accum_buffer_;       // input(optional,temp)
  BufferAllocation::Slice softmax_lse_accum_buffer_;  // input(optional,temp)
};

class FlashAttnVarlenForwardThunk : public Thunk {
 public:
  FlashAttnVarlenForwardThunk(
      ThunkInfo thunk_info, FlashAttnConfig config,
      BufferAllocation::Slice query_slice, BufferAllocation::Slice key_slice,
      BufferAllocation::Slice value_slice,
      BufferAllocation::Slice key_padding_mask_slice,
      BufferAllocation::Slice query_padding_mask_slice, /* may be null */
      BufferAllocation::Slice alibi_slopes_slice,       /* may be null */
      BufferAllocation::Slice output_slice,
      BufferAllocation::Slice query_unpad_slice,
      BufferAllocation::Slice key_unpad_slice,
      BufferAllocation::Slice value_unpad_slice,
      BufferAllocation::Slice output_unpad_slice,
      BufferAllocation::Slice softmax_lse_slice,
      BufferAllocation::Slice indices_k_slice,
      BufferAllocation::Slice indices_len_k_slice,
      BufferAllocation::Slice batch_seqlens_k_slice,
      BufferAllocation::Slice indices_q_slice,       /* may be null */
      BufferAllocation::Slice indices_len_q_slice,   /* may be null */
      BufferAllocation::Slice batch_seqlens_q_slice, /* may be null */
      BufferAllocation::Slice rng_state_slice,
      BufferAllocation::Slice s_dmask_slice, /* may be null */
      BufferAllocation::Slice cu_seqlens_k_slice,
      BufferAllocation::Slice max_seqlen_k_slice,
      BufferAllocation::Slice cu_seqlens_q_slice, /* may be null */
      BufferAllocation::Slice max_seqlen_q_slice /* may be null */);

  FlashAttnVarlenForwardThunk(const FlashAttnVarlenForwardThunk &) = delete;
  FlashAttnVarlenForwardThunk &operator=(const FlashAttnVarlenForwardThunk &) =
      delete;

  absl::Status ExecuteOnStream(const ExecuteParams &params) override;

 protected:
  const FlashAttnConfig config_;

  BufferAllocation::Slice query_buffer_;               // input
  BufferAllocation::Slice key_buffer_;                 // input
  BufferAllocation::Slice value_buffer_;               // input
  BufferAllocation::Slice key_padding_mask_buffer_;    // input
  BufferAllocation::Slice query_padding_mask_buffer_;  // input(optional)
  BufferAllocation::Slice alibi_slopes_buffer_;        // input(optional)

  BufferAllocation::Slice output_buffer_;           // output
  BufferAllocation::Slice query_unpad_buffer_;      // output
  BufferAllocation::Slice key_unpad_buffer_;        // output
  BufferAllocation::Slice value_unpad_buffer_;      // output
  BufferAllocation::Slice output_unpad_buffer_;     // output
  BufferAllocation::Slice softmax_lse_buffer_;      // output
  BufferAllocation::Slice indices_k_buffer_;        // output
  BufferAllocation::Slice indices_len_k_buffer_;    // output
  BufferAllocation::Slice batch_seqlens_k_buffer_;  // output
  BufferAllocation::Slice indices_q_buffer_;        // output(optional)
  BufferAllocation::Slice indices_len_q_buffer_;    // output(optional)
  BufferAllocation::Slice batch_seqlens_q_buffer_;  // output(optional)
  BufferAllocation::Slice rng_state_buffer_;        // output
  BufferAllocation::Slice s_dmask_buffer_;          // output(optional)
  BufferAllocation::Slice cu_seqlens_k_buffer_;     // output(temp)
  BufferAllocation::Slice max_seqlen_k_buffer_;     // output(temp)
  BufferAllocation::Slice cu_seqlens_q_buffer_;     // output(temp)
  BufferAllocation::Slice max_seqlen_q_buffer_;     // output(temp)
};

class FlashAttnBackwardThunk : public Thunk {
 public:
  FlashAttnBackwardThunk(
      ThunkInfo thunk_info, FlashAttnConfig config,
      BufferAllocation::Slice grad_output_slice,
      BufferAllocation::Slice query_slice, BufferAllocation::Slice key_slice,
      BufferAllocation::Slice value_slice, BufferAllocation::Slice output_slice,
      BufferAllocation::Slice softmax_lse_slice,
      BufferAllocation::Slice rng_state_slice,
      BufferAllocation::Slice alibi_slopes_slice, /* may be null */
      BufferAllocation::Slice grad_query_slice,
      BufferAllocation::Slice grad_key_slice,
      BufferAllocation::Slice grad_value_slice,
      BufferAllocation::Slice grad_softmax_slice,
      BufferAllocation::Slice grad_query_accum_slice,
      BufferAllocation::Slice grad_key_expanded_slice, /* may be null */
      BufferAllocation::Slice grad_value_expanded_slice /* may be null */);

  FlashAttnBackwardThunk(const FlashAttnBackwardThunk &) = delete;
  FlashAttnBackwardThunk &operator=(const FlashAttnBackwardThunk &) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams &params) override;

 protected:
  const FlashAttnConfig config_;

  BufferAllocation::Slice grad_output_buffer_;   // input
  BufferAllocation::Slice query_buffer_;         // input
  BufferAllocation::Slice key_buffer_;           // input
  BufferAllocation::Slice value_buffer_;         // input
  BufferAllocation::Slice output_buffer_;        // input
  BufferAllocation::Slice softmax_lse_buffer_;   // input
  BufferAllocation::Slice rng_state_buffer_;     // input
  BufferAllocation::Slice alibi_slopes_buffer_;  // input(optional)

  BufferAllocation::Slice grad_query_buffer_;           // output
  BufferAllocation::Slice grad_key_buffer_;             // output
  BufferAllocation::Slice grad_value_buffer_;           // output
  BufferAllocation::Slice grad_softmax_buffer_;         // output
  BufferAllocation::Slice grad_query_accum_buffer_;     // output(temp)
  BufferAllocation::Slice grad_key_expanded_buffer_;    // output(temp)
  BufferAllocation::Slice grad_value_expanded_buffer_;  // output(temp)
};

class FlashAttnVarlenBackwardThunk : public Thunk {
 public:
  FlashAttnVarlenBackwardThunk(
      ThunkInfo thunk_info, FlashAttnConfig config,
      BufferAllocation::Slice grad_output_slice,
      BufferAllocation::Slice query_unpad_slice,
      BufferAllocation::Slice key_unpad_slice,
      BufferAllocation::Slice value_unpad_slice,
      BufferAllocation::Slice output_unpad_slice,
      BufferAllocation::Slice softmax_lse_slice,
      BufferAllocation::Slice indices_k_slice,
      BufferAllocation::Slice indices_len_k_slice,
      BufferAllocation::Slice batch_seqlens_k_slice,
      BufferAllocation::Slice indices_q_slice,
      BufferAllocation::Slice indices_len_q_slice,
      BufferAllocation::Slice batch_seqlens_q_slice,
      BufferAllocation::Slice rng_state_slice,
      BufferAllocation::Slice alibi_slopes_slice, /* may be null */
      BufferAllocation::Slice grad_query_slice,
      BufferAllocation::Slice grad_key_slice,
      BufferAllocation::Slice grad_value_slice,
      BufferAllocation::Slice grad_softmax_slice,
      BufferAllocation::Slice cu_seqlens_k_slice,
      BufferAllocation::Slice max_seqlen_k_slice,
      BufferAllocation::Slice cu_seqlens_q_slice, /* may be null */
      BufferAllocation::Slice max_seqlen_q_slice, /* may be null */
      BufferAllocation::Slice grad_query_unpad_slice,
      BufferAllocation::Slice grad_key_unpad_slice,
      BufferAllocation::Slice grad_value_unpad_slice,
      BufferAllocation::Slice grad_output_unpad_slice,
      BufferAllocation::Slice grad_query_accum_slice,
      BufferAllocation::Slice grad_key_expanded_slice, /* may be null */
      BufferAllocation::Slice grad_value_expanded_slice /* may be null */);

  FlashAttnVarlenBackwardThunk(const FlashAttnVarlenBackwardThunk &) = delete;
  FlashAttnVarlenBackwardThunk &operator=(
      const FlashAttnVarlenBackwardThunk &) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams &params) override;

 protected:
  const FlashAttnConfig config_;

  BufferAllocation::Slice grad_output_buffer_;      // input
  BufferAllocation::Slice query_unpad_buffer_;      // input
  BufferAllocation::Slice key_unpad_buffer_;        // input
  BufferAllocation::Slice value_unpad_buffer_;      // input
  BufferAllocation::Slice output_unpad_buffer_;     // input
  BufferAllocation::Slice softmax_lse_buffer_;      // input
  BufferAllocation::Slice indices_k_buffer_;        // input
  BufferAllocation::Slice indices_len_k_buffer_;    // input
  BufferAllocation::Slice batch_seqlens_k_buffer_;  // input
  BufferAllocation::Slice indices_q_buffer_;        // input(optional)
  BufferAllocation::Slice indices_len_q_buffer_;    // input(optional)
  BufferAllocation::Slice batch_seqlens_q_buffer_;  // input(optional)
  BufferAllocation::Slice rng_state_buffer_;        // input
  BufferAllocation::Slice alibi_slopes_buffer_;     // input(optional)

  BufferAllocation::Slice grad_query_buffer_;           // output
  BufferAllocation::Slice grad_key_buffer_;             // output
  BufferAllocation::Slice grad_value_buffer_;           // output
  BufferAllocation::Slice grad_softmax_buffer_;         // output
  BufferAllocation::Slice cu_seqlens_k_buffer_;         // output(temp)
  BufferAllocation::Slice max_seqlen_k_buffer_;         // output(temp)
  BufferAllocation::Slice cu_seqlens_q_buffer_;         // output(temp)
  BufferAllocation::Slice max_seqlen_q_buffer_;         // output(temp)
  BufferAllocation::Slice grad_query_unpad_buffer_;     // output(temp)
  BufferAllocation::Slice grad_key_unpad_buffer_;       // output(temp)
  BufferAllocation::Slice grad_value_unpad_buffer_;     // output(temp)
  BufferAllocation::Slice grad_output_unpad_buffer_;    // output(temp)
  BufferAllocation::Slice grad_query_accum_buffer_;     // output(temp)
  BufferAllocation::Slice grad_key_expanded_buffer_;    // output(temp)
  BufferAllocation::Slice grad_value_expanded_buffer_;  // output(temp)
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_FLASH_ATTN_THUNK_H_

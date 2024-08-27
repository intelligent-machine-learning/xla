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

#include "xla/service/gpu/runtime/flash_attn_thunk.h"

namespace xla {
namespace gpu {

FlashAttnForwardThunk::FlashAttnForwardThunk(
    ThunkInfo thunk_info, FlashAttnConfig config,
    BufferAllocation::Slice query_slice, BufferAllocation::Slice key_slice,
    BufferAllocation::Slice value_slice,
    BufferAllocation::Slice alibi_slopes_slice, /* may be null */
    BufferAllocation::Slice output_slice,
    BufferAllocation::Slice softmax_lse_slice,
    BufferAllocation::Slice rng_state_slice,
    BufferAllocation::Slice s_dmask_slice,          /* may be null */
    BufferAllocation::Slice output_accum_slice,     /* may be null */
    BufferAllocation::Slice softmax_lse_accum_slice /* may be null */
    )
    : Thunk(Kind::kFlashAttn, thunk_info),
      config_(std::move(config)),
      query_buffer_(query_slice),
      key_buffer_(key_slice),
      value_buffer_(value_slice),
      alibi_slopes_buffer_(alibi_slopes_slice),
      output_buffer_(output_slice),
      softmax_lse_buffer_(softmax_lse_slice),
      rng_state_buffer_(rng_state_slice),
      s_dmask_buffer_(s_dmask_slice),
      output_accum_buffer_(output_accum_slice),
      softmax_lse_accum_buffer_(softmax_lse_accum_slice) {}

static std::optional<se::DeviceMemoryBase> AssignBufferIfNotNull(
    const BufferAllocations& buffer_allocations,
    BufferAllocation::Slice& slice) {
  return slice.allocation() != nullptr
             ? std::optional<se::DeviceMemoryBase>{buffer_allocations
                                                       .GetDeviceAddress(slice)}
             : std::nullopt;
}

absl::Status FlashAttnForwardThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  // ======================== Inputs ========================

  se::DeviceMemoryBase query_buffer =
      buffer_allocations.GetDeviceAddress(query_buffer_);
  se::DeviceMemoryBase key_buffer =
      buffer_allocations.GetDeviceAddress(key_buffer_);
  se::DeviceMemoryBase value_buffer =
      buffer_allocations.GetDeviceAddress(value_buffer_);
  std::optional<se::DeviceMemoryBase> alibi_slopes_buffer =
      AssignBufferIfNotNull(buffer_allocations, alibi_slopes_buffer_);

  // ======================== Outputs ========================

  se::DeviceMemoryBase output_buffer =
      buffer_allocations.GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase softmax_lse_buffer =
      buffer_allocations.GetDeviceAddress(softmax_lse_buffer_);
  se::DeviceMemoryBase rng_state_buffer =
      buffer_allocations.GetDeviceAddress(rng_state_buffer_);
  std::optional<se::DeviceMemoryBase> s_dmask_buffer =
      AssignBufferIfNotNull(buffer_allocations, s_dmask_buffer_);
  std::optional<se::DeviceMemoryBase> output_accum_buffer =
      AssignBufferIfNotNull(buffer_allocations, output_accum_buffer_);
  std::optional<se::DeviceMemoryBase> softmax_lse_accum_buffer =
      AssignBufferIfNotNull(buffer_allocations, softmax_lse_accum_buffer_);

  TF_RETURN_IF_ERROR(RunFlashAttnForward(
      params.stream, config_, query_buffer, key_buffer, value_buffer,
      alibi_slopes_buffer, output_buffer, softmax_lse_buffer, rng_state_buffer,
      s_dmask_buffer, output_accum_buffer, softmax_lse_accum_buffer));

  if (!params.stream->ok()) {
    return Internal("FlashAttnForwardThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

FlashAttnVarlenForwardThunk::FlashAttnVarlenForwardThunk(
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
    BufferAllocation::Slice max_seqlen_q_slice /* may be null */)
    : Thunk(Kind::kFlashAttn, thunk_info),
      config_(std::move(config)),
      query_buffer_(query_slice),
      key_buffer_(key_slice),
      value_buffer_(value_slice),
      key_padding_mask_buffer_(key_padding_mask_slice),
      query_padding_mask_buffer_(query_padding_mask_slice),
      alibi_slopes_buffer_(alibi_slopes_slice),
      output_buffer_(output_slice),
      query_unpad_buffer_(query_unpad_slice),
      key_unpad_buffer_(key_unpad_slice),
      value_unpad_buffer_(value_unpad_slice),
      output_unpad_buffer_(output_unpad_slice),
      softmax_lse_buffer_(softmax_lse_slice),
      indices_k_buffer_(indices_k_slice),
      indices_len_k_buffer_(indices_len_k_slice),
      batch_seqlens_k_buffer_(batch_seqlens_k_slice),
      indices_q_buffer_(indices_q_slice),
      indices_len_q_buffer_(indices_len_q_slice),
      batch_seqlens_q_buffer_(batch_seqlens_q_slice),
      rng_state_buffer_(rng_state_slice),
      s_dmask_buffer_(s_dmask_slice),
      cu_seqlens_k_buffer_(cu_seqlens_k_slice),
      max_seqlen_k_buffer_(max_seqlen_k_slice),
      cu_seqlens_q_buffer_(cu_seqlens_q_slice),
      max_seqlen_q_buffer_(max_seqlen_q_slice) {}

absl::Status FlashAttnVarlenForwardThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  // ======================== Inputs ========================

  se::DeviceMemoryBase query_buffer =
      buffer_allocations.GetDeviceAddress(query_buffer_);
  se::DeviceMemoryBase key_buffer =
      buffer_allocations.GetDeviceAddress(key_buffer_);
  se::DeviceMemoryBase value_buffer =
      buffer_allocations.GetDeviceAddress(value_buffer_);
  se::DeviceMemoryBase key_padding_mask_buffer =
      buffer_allocations.GetDeviceAddress(key_padding_mask_buffer_);
  std::optional<se::DeviceMemoryBase> query_padding_mask_buffer =
      AssignBufferIfNotNull(buffer_allocations, query_padding_mask_buffer_);
  std::optional<se::DeviceMemoryBase> alibi_slopes_buffer =
      AssignBufferIfNotNull(buffer_allocations, alibi_slopes_buffer_);

  // ======================== Outputs ========================

  se::DeviceMemoryBase output_buffer =
      buffer_allocations.GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase query_unpad_buffer =
      buffer_allocations.GetDeviceAddress(query_unpad_buffer_);
  se::DeviceMemoryBase key_unpad_buffer =
      buffer_allocations.GetDeviceAddress(key_unpad_buffer_);
  se::DeviceMemoryBase value_unpad_buffer =
      buffer_allocations.GetDeviceAddress(value_unpad_buffer_);
  se::DeviceMemoryBase output_unpad_buffer =
      buffer_allocations.GetDeviceAddress(output_unpad_buffer_);
  se::DeviceMemoryBase softmax_lse_buffer =
      buffer_allocations.GetDeviceAddress(softmax_lse_buffer_);
  se::DeviceMemoryBase indices_k_buffer =
      buffer_allocations.GetDeviceAddress(indices_k_buffer_);
  se::DeviceMemoryBase indices_len_k_buffer =
      buffer_allocations.GetDeviceAddress(indices_len_k_buffer_);
  se::DeviceMemoryBase batch_seqlens_k_buffer =
      buffer_allocations.GetDeviceAddress(batch_seqlens_k_buffer_);
  std::optional<se::DeviceMemoryBase> indices_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, indices_q_buffer_);
  std::optional<se::DeviceMemoryBase> indices_len_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, indices_len_q_buffer_);
  std::optional<se::DeviceMemoryBase> batch_seqlens_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, batch_seqlens_q_buffer_);
  se::DeviceMemoryBase rng_state_buffer =
      buffer_allocations.GetDeviceAddress(rng_state_buffer_);
  std::optional<se::DeviceMemoryBase> s_dmask_buffer =
      AssignBufferIfNotNull(buffer_allocations, s_dmask_buffer_);
  se::DeviceMemoryBase cu_seqlens_k_buffer =
      buffer_allocations.GetDeviceAddress(cu_seqlens_k_buffer_);
  se::DeviceMemoryBase max_seqlen_k_buffer =
      buffer_allocations.GetDeviceAddress(max_seqlen_k_buffer_);
  std::optional<se::DeviceMemoryBase> cu_seqlens_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, cu_seqlens_q_buffer_);
  std::optional<se::DeviceMemoryBase> max_seqlen_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, max_seqlen_q_buffer_);

  TF_RETURN_IF_ERROR(RunFlashAttnVarlenForward(
      params.stream, config_, query_buffer, key_buffer, value_buffer,
      key_padding_mask_buffer, query_padding_mask_buffer, alibi_slopes_buffer,
      output_buffer, query_unpad_buffer, key_unpad_buffer, value_unpad_buffer,
      output_unpad_buffer, softmax_lse_buffer, indices_k_buffer,
      indices_len_k_buffer, batch_seqlens_k_buffer, indices_q_buffer,
      indices_len_q_buffer, batch_seqlens_q_buffer, rng_state_buffer,
      s_dmask_buffer, cu_seqlens_k_buffer, max_seqlen_k_buffer,
      cu_seqlens_q_buffer, max_seqlen_q_buffer));

  if (!params.stream->ok()) {
    return Internal("FlashAttnVarlenForwardThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

FlashAttnBackwardThunk::FlashAttnBackwardThunk(
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
    BufferAllocation::Slice grad_key_expanded_slice,
    BufferAllocation::Slice grad_value_expanded_slice)
    : Thunk(Kind::kFlashAttn, thunk_info),
      config_(std::move(config)),
      grad_output_buffer_(grad_output_slice),
      query_buffer_(query_slice),
      key_buffer_(key_slice),
      value_buffer_(value_slice),
      output_buffer_(output_slice),
      softmax_lse_buffer_(softmax_lse_slice),
      rng_state_buffer_(rng_state_slice),
      alibi_slopes_buffer_(alibi_slopes_slice),
      grad_query_buffer_(grad_query_slice),
      grad_key_buffer_(grad_key_slice),
      grad_value_buffer_(grad_value_slice),
      grad_softmax_buffer_(grad_softmax_slice),
      grad_query_accum_buffer_(grad_query_accum_slice),
      grad_key_expanded_buffer_(grad_key_expanded_slice),
      grad_value_expanded_buffer_(grad_value_expanded_slice) {}

absl::Status FlashAttnBackwardThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  // ======================== Inputs ========================

  se::DeviceMemoryBase grad_output_buffer =
      buffer_allocations.GetDeviceAddress(grad_output_buffer_);
  se::DeviceMemoryBase query_buffer =
      buffer_allocations.GetDeviceAddress(query_buffer_);
  se::DeviceMemoryBase key_buffer =
      buffer_allocations.GetDeviceAddress(key_buffer_);
  se::DeviceMemoryBase value_buffer =
      buffer_allocations.GetDeviceAddress(value_buffer_);
  se::DeviceMemoryBase output_buffer =
      buffer_allocations.GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase softmax_lse_buffer =
      buffer_allocations.GetDeviceAddress(softmax_lse_buffer_);
  se::DeviceMemoryBase rng_state_buffer =
      buffer_allocations.GetDeviceAddress(rng_state_buffer_);
  std::optional<se::DeviceMemoryBase> alibi_slopes_buffer =
      AssignBufferIfNotNull(buffer_allocations, alibi_slopes_buffer_);

  // ======================== Outputs ========================

  se::DeviceMemoryBase grad_query_buffer =
      buffer_allocations.GetDeviceAddress(grad_query_buffer_);
  se::DeviceMemoryBase grad_key_buffer =
      buffer_allocations.GetDeviceAddress(grad_key_buffer_);
  se::DeviceMemoryBase grad_value_buffer =
      buffer_allocations.GetDeviceAddress(grad_value_buffer_);
  se::DeviceMemoryBase grad_softmax_buffer =
      buffer_allocations.GetDeviceAddress(grad_softmax_buffer_);
  se::DeviceMemoryBase grad_query_accum_buffer =
      buffer_allocations.GetDeviceAddress(grad_query_accum_buffer_);
  std::optional<se::DeviceMemoryBase> grad_key_expanded_buffer =
      AssignBufferIfNotNull(buffer_allocations, grad_key_expanded_buffer_);
  std::optional<se::DeviceMemoryBase> grad_value_expanded_buffer =
      AssignBufferIfNotNull(buffer_allocations, grad_value_expanded_buffer_);

  TF_RETURN_IF_ERROR(RunFlashAttnBackward(
      params.stream, config_, grad_output_buffer, query_buffer, key_buffer,
      value_buffer, output_buffer, softmax_lse_buffer, rng_state_buffer,
      alibi_slopes_buffer, grad_query_buffer, grad_key_buffer,
      grad_value_buffer, grad_softmax_buffer, grad_query_accum_buffer,
      grad_key_expanded_buffer, grad_value_expanded_buffer));

  if (!params.stream->ok()) {
    return Internal("FlashAttnBackwardThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

FlashAttnVarlenBackwardThunk::FlashAttnVarlenBackwardThunk(
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
    BufferAllocation::Slice indices_q_slice,       /* may be null */
    BufferAllocation::Slice indices_len_q_slice,   /* may be null */
    BufferAllocation::Slice batch_seqlens_q_slice, /* may be null */
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
    BufferAllocation::Slice grad_key_expanded_slice,
    BufferAllocation::Slice grad_value_expanded_slice)
    : Thunk(Kind::kFlashAttn, thunk_info),
      config_(std::move(config)),
      grad_output_buffer_(grad_output_slice),
      query_unpad_buffer_(query_unpad_slice),
      key_unpad_buffer_(key_unpad_slice),
      value_unpad_buffer_(value_unpad_slice),
      output_unpad_buffer_(output_unpad_slice),
      softmax_lse_buffer_(softmax_lse_slice),
      indices_k_buffer_(indices_k_slice),
      indices_len_k_buffer_(indices_len_k_slice),
      batch_seqlens_k_buffer_(batch_seqlens_k_slice),
      indices_q_buffer_(indices_q_slice),
      indices_len_q_buffer_(indices_len_q_slice),
      batch_seqlens_q_buffer_(batch_seqlens_q_slice),
      rng_state_buffer_(rng_state_slice),
      alibi_slopes_buffer_(alibi_slopes_slice),
      grad_query_buffer_(grad_query_slice),
      grad_key_buffer_(grad_key_slice),
      grad_value_buffer_(grad_value_slice),
      grad_softmax_buffer_(grad_softmax_slice),
      cu_seqlens_k_buffer_(cu_seqlens_k_slice),
      max_seqlen_k_buffer_(max_seqlen_k_slice),
      cu_seqlens_q_buffer_(cu_seqlens_q_slice),
      max_seqlen_q_buffer_(max_seqlen_q_slice),
      grad_query_unpad_buffer_(grad_query_unpad_slice),
      grad_key_unpad_buffer_(grad_key_unpad_slice),
      grad_value_unpad_buffer_(grad_value_unpad_slice),
      grad_output_unpad_buffer_(grad_output_unpad_slice),
      grad_query_accum_buffer_(grad_query_accum_slice),
      grad_key_expanded_buffer_(grad_key_expanded_slice),
      grad_value_expanded_buffer_(grad_value_expanded_slice) {}

absl::Status FlashAttnVarlenBackwardThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  // ======================== Inputs ========================

  se::DeviceMemoryBase grad_output_buffer =
      buffer_allocations.GetDeviceAddress(grad_output_buffer_);
  se::DeviceMemoryBase query_unpad_buffer =
      buffer_allocations.GetDeviceAddress(query_unpad_buffer_);
  se::DeviceMemoryBase key_unpad_buffer =
      buffer_allocations.GetDeviceAddress(key_unpad_buffer_);
  se::DeviceMemoryBase value_unpad_buffer =
      buffer_allocations.GetDeviceAddress(value_unpad_buffer_);
  se::DeviceMemoryBase output_unpad_buffer =
      buffer_allocations.GetDeviceAddress(output_unpad_buffer_);
  se::DeviceMemoryBase softmax_lse_buffer =
      buffer_allocations.GetDeviceAddress(softmax_lse_buffer_);
  se::DeviceMemoryBase indices_k_buffer =
      buffer_allocations.GetDeviceAddress(indices_k_buffer_);
  se::DeviceMemoryBase indices_len_k_buffer =
      buffer_allocations.GetDeviceAddress(indices_len_k_buffer_);
  se::DeviceMemoryBase batch_seqlens_k_buffer =
      buffer_allocations.GetDeviceAddress(batch_seqlens_k_buffer_);
  std::optional<se::DeviceMemoryBase> indices_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, indices_q_buffer_);
  std::optional<se::DeviceMemoryBase> indices_len_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, indices_len_q_buffer_);
  std::optional<se::DeviceMemoryBase> batch_seqlens_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, batch_seqlens_q_buffer_);
  se::DeviceMemoryBase rng_state_buffer =
      buffer_allocations.GetDeviceAddress(rng_state_buffer_);
  std::optional<se::DeviceMemoryBase> alibi_slopes_buffer =
      AssignBufferIfNotNull(buffer_allocations, alibi_slopes_buffer_);

  // ======================== Outputs ========================

  se::DeviceMemoryBase grad_query_buffer =
      buffer_allocations.GetDeviceAddress(grad_query_buffer_);
  se::DeviceMemoryBase grad_key_buffer =
      buffer_allocations.GetDeviceAddress(grad_key_buffer_);
  se::DeviceMemoryBase grad_value_buffer =
      buffer_allocations.GetDeviceAddress(grad_value_buffer_);
  se::DeviceMemoryBase grad_softmax_buffer =
      buffer_allocations.GetDeviceAddress(grad_softmax_buffer_);
  se::DeviceMemoryBase cu_seqlens_k_buffer =
      buffer_allocations.GetDeviceAddress(cu_seqlens_k_buffer_);
  se::DeviceMemoryBase max_seqlen_k_buffer =
      buffer_allocations.GetDeviceAddress(max_seqlen_k_buffer_);
  std::optional<se::DeviceMemoryBase> cu_seqlens_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, cu_seqlens_q_buffer_);
  std::optional<se::DeviceMemoryBase> max_seqlen_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, max_seqlen_q_buffer_);
  se::DeviceMemoryBase grad_query_unpad_buffer =
      buffer_allocations.GetDeviceAddress(grad_query_unpad_buffer_);
  se::DeviceMemoryBase grad_key_unpad_buffer =
      buffer_allocations.GetDeviceAddress(grad_key_unpad_buffer_);
  se::DeviceMemoryBase grad_value_unpad_buffer =
      buffer_allocations.GetDeviceAddress(grad_value_unpad_buffer_);
  se::DeviceMemoryBase grad_output_unpad_buffer =
      buffer_allocations.GetDeviceAddress(grad_output_unpad_buffer_);
  se::DeviceMemoryBase grad_query_accum_buffer =
      buffer_allocations.GetDeviceAddress(grad_query_accum_buffer_);
  std::optional<se::DeviceMemoryBase> grad_key_expanded_buffer =
      AssignBufferIfNotNull(buffer_allocations, grad_key_expanded_buffer_);
  std::optional<se::DeviceMemoryBase> grad_value_expanded_buffer =
      AssignBufferIfNotNull(buffer_allocations, grad_value_expanded_buffer_);

  TF_RETURN_IF_ERROR(RunFlashAttnVarlenBackward(
      params.stream, config_, grad_output_buffer, query_unpad_buffer,
      key_unpad_buffer, value_unpad_buffer, output_unpad_buffer,
      softmax_lse_buffer, indices_k_buffer, indices_len_k_buffer,
      batch_seqlens_k_buffer, indices_q_buffer, indices_len_q_buffer,
      batch_seqlens_q_buffer, rng_state_buffer, alibi_slopes_buffer,
      grad_query_buffer, grad_key_buffer, grad_value_buffer,
      grad_softmax_buffer, cu_seqlens_k_buffer, max_seqlen_k_buffer,
      cu_seqlens_q_buffer, max_seqlen_q_buffer, grad_query_unpad_buffer,
      grad_key_unpad_buffer, grad_value_unpad_buffer, grad_output_unpad_buffer,
      grad_query_accum_buffer, grad_key_expanded_buffer,
      grad_value_expanded_buffer));

  if (!params.stream->ok()) {
    return Internal("FlashAttnVarlenBackwardThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

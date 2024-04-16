/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/flash_attn_thunk.h"

namespace xla {
namespace gpu {

FlashAttnFwdThunk::FlashAttnFwdThunk(
    ThunkInfo thunk_info, FlashAttnFwdConfig config,
    BufferAllocation::Slice query_slice, BufferAllocation::Slice key_slice,
    BufferAllocation::Slice value_slice,
    BufferAllocation::Slice cu_seqlens_query_slice,  /* may be null */
    BufferAllocation::Slice cu_seqlens_key_slice,    /* may be null */
    BufferAllocation::Slice alibi_slopes_slice,      /* may be null */
    BufferAllocation::Slice output_accum_slice,      /* may be null */
    BufferAllocation::Slice softmax_lse_accum_slice, /* may be null */
    BufferAllocation::Slice output_slice,
    BufferAllocation::Slice softmax_lse_slice,
    BufferAllocation::Slice rng_state_slice,
    BufferAllocation::Slice s_dmask_slice /* may be null */
    )
    : Thunk(Kind::kFlashAttn, thunk_info),
      config_(std::move(config)),
      query_buffer_(query_slice),
      key_buffer_(key_slice),
      value_buffer_(value_slice),
      cu_seqlens_query_buffer_(cu_seqlens_query_slice),
      cu_seqlens_key_buffer_(cu_seqlens_key_slice),
      alibi_slopes_buffer_(alibi_slopes_slice),
      output_accum_buffer_(output_accum_slice),
      softmax_lse_accum_buffer_(softmax_lse_accum_slice),
      output_buffer_(output_slice),
      softmax_lse_buffer_(softmax_lse_slice),
      rng_state_buffer_(rng_state_slice),
      s_dmask_buffer_(s_dmask_slice) {}

static std::optional<se::DeviceMemoryBase> AssignBufferIfNotNull(
    const BufferAllocations& buffer_allocations,
    BufferAllocation::Slice& slice) {
  return slice.allocation() != nullptr
             ? std::optional<se::DeviceMemoryBase>{buffer_allocations
                                                       .GetDeviceAddress(slice)}
             : std::nullopt;
}

absl::Status FlashAttnFwdThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  se::DeviceMemoryBase query_buffer =
      buffer_allocations.GetDeviceAddress(query_buffer_);
  se::DeviceMemoryBase key_buffer =
      buffer_allocations.GetDeviceAddress(key_buffer_);
  se::DeviceMemoryBase value_buffer =
      buffer_allocations.GetDeviceAddress(value_buffer_);
  std::optional<se::DeviceMemoryBase> cu_seqlens_query_buffer =
      AssignBufferIfNotNull(buffer_allocations, cu_seqlens_query_buffer_);
  std::optional<se::DeviceMemoryBase> cu_seqlens_key_buffer =
      AssignBufferIfNotNull(buffer_allocations, cu_seqlens_key_buffer_);
  std::optional<se::DeviceMemoryBase> alibi_slopes_buffer =
      AssignBufferIfNotNull(buffer_allocations, alibi_slopes_buffer_);
  std::optional<se::DeviceMemoryBase> output_accum_buffer =
      AssignBufferIfNotNull(buffer_allocations, output_accum_buffer_);
  std::optional<se::DeviceMemoryBase> softmax_lse_accum_buffer =
      AssignBufferIfNotNull(buffer_allocations, softmax_lse_accum_buffer_);

  se::DeviceMemoryBase output_buffer =
      buffer_allocations.GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase softmax_lse_buffer =
      buffer_allocations.GetDeviceAddress(softmax_lse_buffer_);
  se::DeviceMemoryBase rng_state_buffer =
      buffer_allocations.GetDeviceAddress(rng_state_buffer_);
  std::optional<se::DeviceMemoryBase> s_dmask_buffer =
      AssignBufferIfNotNull(buffer_allocations, s_dmask_buffer_);

  TF_RETURN_IF_ERROR(RunFlashAttnFwd(
      params.stream, config_, query_buffer, key_buffer, value_buffer,
      cu_seqlens_query_buffer, cu_seqlens_key_buffer, alibi_slopes_buffer,
      output_accum_buffer, softmax_lse_accum_buffer, output_buffer,
      softmax_lse_buffer, rng_state_buffer, s_dmask_buffer, -1, -1));

  if (!params.stream->ok()) {
    return Internal("FlashAttnFwdThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

FlashAttnBwdThunk::FlashAttnBwdThunk(
    ThunkInfo thunk_info, FlashAttnBwdConfig config,
    BufferAllocation::Slice grad_output_slice,
    BufferAllocation::Slice query_slice, BufferAllocation::Slice key_slice,
    BufferAllocation::Slice value_slice, BufferAllocation::Slice output_slice,
    BufferAllocation::Slice softmax_lse_slice,
    BufferAllocation::Slice rng_state_slice,
    BufferAllocation::Slice cu_seqlens_query_slice, /* may be null */
    BufferAllocation::Slice cu_seqlens_key_slice,   /* may be null */
    BufferAllocation::Slice alibi_slopes_slice,     /* may be null */
    BufferAllocation::Slice grad_query_accum_slice,
    BufferAllocation::Slice grad_query_slice,
    BufferAllocation::Slice grad_key_slice,
    BufferAllocation::Slice grad_value_slice,
    BufferAllocation::Slice grad_softmax_slice)
    : Thunk(Kind::kFlashAttn, thunk_info),
      config_(std::move(config)),
      grad_output_buffer_(grad_output_slice),
      query_buffer_(query_slice),
      key_buffer_(key_slice),
      value_buffer_(value_slice),
      output_buffer_(output_slice),
      softmax_lse_buffer_(softmax_lse_slice),
      rng_state_buffer_(rng_state_slice),
      cu_seqlens_query_buffer_(cu_seqlens_query_slice),
      cu_seqlens_key_buffer_(cu_seqlens_key_slice),
      alibi_slopes_buffer_(alibi_slopes_slice),
      grad_query_accum_buffer_(grad_query_accum_slice),
      grad_query_buffer_(grad_query_slice),
      grad_key_buffer_(grad_key_slice),
      grad_value_buffer_(grad_value_slice),
      grad_softmax_buffer_(grad_softmax_slice) {}

absl::Status FlashAttnBwdThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

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
  std::optional<se::DeviceMemoryBase> cu_seqlens_query_buffer =
      AssignBufferIfNotNull(buffer_allocations, cu_seqlens_query_buffer_);
  std::optional<se::DeviceMemoryBase> cu_seqlens_key_buffer =
      AssignBufferIfNotNull(buffer_allocations, cu_seqlens_key_buffer_);
  std::optional<se::DeviceMemoryBase> alibi_slopes_buffer =
      AssignBufferIfNotNull(buffer_allocations, alibi_slopes_buffer_);
  se::DeviceMemoryBase grad_query_accum_buffer =
      buffer_allocations.GetDeviceAddress(grad_query_accum_buffer_);

  se::DeviceMemoryBase grad_query_buffer =
      buffer_allocations.GetDeviceAddress(grad_query_buffer_);
  se::DeviceMemoryBase grad_key_buffer =
      buffer_allocations.GetDeviceAddress(grad_key_buffer_);
  se::DeviceMemoryBase grad_value_buffer =
      buffer_allocations.GetDeviceAddress(grad_value_buffer_);
  se::DeviceMemoryBase grad_softmax_buffer =
      buffer_allocations.GetDeviceAddress(grad_softmax_buffer_);

  TF_RETURN_IF_ERROR(RunFlashAttnBwd(
      params.stream, config_, grad_output_buffer, query_buffer, key_buffer,
      value_buffer, output_buffer, softmax_lse_buffer, rng_state_buffer,
      cu_seqlens_query_buffer, cu_seqlens_key_buffer, alibi_slopes_buffer,
      grad_query_accum_buffer, grad_query_buffer, grad_key_buffer,
      grad_value_buffer, grad_softmax_buffer, -1, -1));

  if (!params.stream->ok()) {
    return Internal("FlashAttnBwdThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

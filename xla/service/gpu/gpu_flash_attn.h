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

#ifndef XLA_SERVICE_GPU_GPU_FLASH_ATTN_H_
#define XLA_SERVICE_GPU_GPU_FLASH_ATTN_H_

#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

extern const absl::string_view kGpuFlashAttnForwardCallTarget;
extern const absl::string_view kGpuFlashAttnBackwardCallTarget;
extern const absl::string_view kGpuFlashAttnVarlenForwardCallTarget;
extern const absl::string_view kGpuFlashAttnVarlenBackwardCallTarget;

bool IsCustomCallToFlashAttnForward(const HloInstruction &hlo);
bool IsCustomCallToFlashAttnVarlenForward(const HloInstruction &hlo);
bool IsCustomCallToFlashAttnBackward(const HloInstruction &hlo);
bool IsCustomCallToFlashAttnVarlenBackward(const HloInstruction &hlo);
bool IsCustomCallToFlashAttn(const HloInstruction &hlo);

enum class FlashAttnKind {
  kForward,
  kVarlenForward,
  kBackward,
  kVarlenBackward,
};

absl::StatusOr<FlashAttnKind> GetFlashAttnKind(
    const HloCustomCallInstruction *instr);

struct FlashAttnConfig {
  PrimitiveType dtype;

  int64_t batch_size;
  int64_t seqlen_q;
  int64_t seqlen_k;
  int64_t num_heads;
  int64_t num_heads_k;
  int64_t head_size_og;
  int64_t head_size;
  int64_t head_size_rounded;
  int64_t alibi_slopes_batch_stride;

  float dropout_p;
  double softmax_scale;
  bool causal;
  int window_size_left;
  int window_size_right;

  // only used in varlen forward
  bool has_query_padding_mask;
  // only used in forward
  bool return_softmax;
  // only used in backward(inference dq_accum_split_stride  kernel)
  bool deterministic;
  // only used in non-varlen forward
  int64_t num_splits;
  // only used in non-varlen backward
  int64_t dq_accum_split_stride;
};

namespace se = ::stream_executor;

absl::Status RunFlashAttnForward(
    se::Stream *stream, const FlashAttnConfig &config,
    se::DeviceMemoryBase query_buffer, se::DeviceMemoryBase key_buffer,
    se::DeviceMemoryBase value_buffer,
    std::optional<se::DeviceMemoryBase> alibi_slopes_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase softmax_lse_buffer,
    se::DeviceMemoryBase rng_state_buffer,
    std::optional<se::DeviceMemoryBase> s_dmask_buffer,
    std::optional<se::DeviceMemoryBase> output_accum_buffer,
    std::optional<se::DeviceMemoryBase> softmax_lse_accum_buffer);

absl::Status RunFlashAttnVarlenForward(
    se::Stream *stream, const FlashAttnConfig &config,
    se::DeviceMemoryBase query_buffer, se::DeviceMemoryBase key_buffer,
    se::DeviceMemoryBase value_buffer,
    se::DeviceMemoryBase key_padding_mask_buffer,
    std::optional<se::DeviceMemoryBase> query_padding_mask_buffer,
    std::optional<se::DeviceMemoryBase> alibi_slopes_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase query_unpad_buffer,
    se::DeviceMemoryBase key_unpad_buffer,
    se::DeviceMemoryBase value_unpad_buffer,
    se::DeviceMemoryBase output_unpad_buffer,
    se::DeviceMemoryBase softmax_lse_buffer,
    se::DeviceMemoryBase indices_k_buffer,
    se::DeviceMemoryBase indices_len_k_buffer,
    se::DeviceMemoryBase batch_seqlens_k_buffer,
    std::optional<se::DeviceMemoryBase> indices_q_buffer,
    std::optional<se::DeviceMemoryBase> indices_len_q_buffer,
    std::optional<se::DeviceMemoryBase> batch_seqlens_q_buffer,
    se::DeviceMemoryBase rng_state_buffer,
    std::optional<se::DeviceMemoryBase> s_dmask_buffer,
    se::DeviceMemoryBase cu_seqlens_k_buffer,
    se::DeviceMemoryBase max_seqlen_k_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_q_buffer,
    std::optional<se::DeviceMemoryBase> max_seqlen_q_buffer);

absl::Status RunFlashAttnBackward(
    se::Stream *stream, const FlashAttnConfig &config,
    se::DeviceMemoryBase grad_output_buffer, se::DeviceMemoryBase query_buffer,
    se::DeviceMemoryBase key_buffer, se::DeviceMemoryBase value_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase softmax_lse_buffer,
    se::DeviceMemoryBase rng_state_buffer,
    std::optional<se::DeviceMemoryBase> alibi_slopes_buffer,
    se::DeviceMemoryBase grad_query_buffer,
    se::DeviceMemoryBase grad_key_buffer,
    se::DeviceMemoryBase grad_value_buffer,
    se::DeviceMemoryBase grad_softmax_buffer,
    se::DeviceMemoryBase grad_query_accum_buffer,
    std::optional<se::DeviceMemoryBase> grad_key_expanded_buffer,
    std::optional<se::DeviceMemoryBase> grad_value_expanded_buffer);

absl::Status RunFlashAttnVarlenBackward(
    se::Stream *stream, const FlashAttnConfig &config,
    se::DeviceMemoryBase grad_output_buffer,
    se::DeviceMemoryBase query_unpad_buffer,
    se::DeviceMemoryBase key_unpad_buffer,
    se::DeviceMemoryBase value_unpad_buffer,
    se::DeviceMemoryBase output_unpad_buffer,
    se::DeviceMemoryBase softmax_lse_buffer,
    se::DeviceMemoryBase indices_k_buffer,
    se::DeviceMemoryBase indices_len_k_buffer,
    se::DeviceMemoryBase batch_seqlens_k_buffer,
    std::optional<se::DeviceMemoryBase> indices_q_buffer,
    std::optional<se::DeviceMemoryBase> indices_len_q_buffer,
    std::optional<se::DeviceMemoryBase> batch_seqlens_q_buffer,
    se::DeviceMemoryBase rng_state_buffer,
    std::optional<se::DeviceMemoryBase> alibi_slopes_buffer,
    se::DeviceMemoryBase grad_query_buffer,
    se::DeviceMemoryBase grad_key_buffer,
    se::DeviceMemoryBase grad_value_buffer,
    se::DeviceMemoryBase grad_softmax_buffer,
    se::DeviceMemoryBase cu_seqlens_k_buffer,
    se::DeviceMemoryBase max_seqlen_k_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_q_buffer,
    std::optional<se::DeviceMemoryBase> max_seqlen_q_buffer,
    se::DeviceMemoryBase grad_query_unpad_buffer,
    se::DeviceMemoryBase grad_key_unpad_buffer,
    se::DeviceMemoryBase grad_value_unpad_buffer,
    se::DeviceMemoryBase grad_output_unpad_buffer,
    se::DeviceMemoryBase grad_query_accum_buffer,
    std::optional<se::DeviceMemoryBase> grad_key_expanded_buffer,
    std::optional<se::DeviceMemoryBase> grad_value_expanded_buffer);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_FLASH_ATTN_H_

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

#ifndef XLA_SERVICE_GPU_GPU_FLASH_ATTN_H_
#define XLA_SERVICE_GPU_GPU_FLASH_ATTN_H_

#include <optional>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/types.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

extern const absl::string_view kGpuFlashAttnFwdCallTarget;
extern const absl::string_view kGpuFlashAttnBwdCallTarget;
extern const absl::string_view kGpuFlashAttnVarLenFwdCallTarget;
extern const absl::string_view kGpuFlashAttnVarLenBwdCallTarget;

bool IsCustomCallToFlashAttn(const HloInstruction &hlo);

enum class FlashAttnKind {
  kForward,
  kVarLenForward,
  kBackward,
  kVarLenBackward,
};

absl::StatusOr<FlashAttnKind> GetFlashAttnKind(
    const HloCustomCallInstruction *instr);

struct FlashAttnConfig {
  static absl::StatusOr<FlashAttnConfig> For(
      const Shape &query_shape, const Shape &key_shape,
      const Shape &value_shape,
      const std::optional<Shape> &cu_seqlens_query_shape,
      const std::optional<Shape> &cu_seqlens_key_shape,
      const Shape &output_shape, const Shape &softmax_lse_shape,
      const std::optional<Shape> &alibi_slopes_shape, float dropout_rate,
      float scale, bool is_causal, const std::optional<int> &max_seqlen_q,
      const std::optional<int> &max_seqlen_k);
  PrimitiveType type;

  se::dnn::TensorDescriptor query_desc;        // input
  se::dnn::TensorDescriptor key_desc;          // input
  se::dnn::TensorDescriptor value_desc;        // input
  se::dnn::TensorDescriptor output_desc;       // output(fwd), input(bwd)
  se::dnn::TensorDescriptor softmax_lse_desc;  // output(fwd), input(bwd)
  std::optional<se::dnn::TensorDescriptor> alibi_slopes_desc;  // input

  // These four fields are only used in the variable-length flash-attention
  std::optional<se::dnn::TensorDescriptor> cu_seqlens_query_desc;  // input
  std::optional<se::dnn::TensorDescriptor> cu_seqlens_key_desc;    // input
  std::optional<int> max_seqlen_q;
  std::optional<int> max_seqlen_k;

  float dropout_rate;
  float scale;
  bool is_causal;
};

struct FlashAttnFwdConfig : public FlashAttnConfig {
  static absl::StatusOr<FlashAttnFwdConfig> For(
      const Shape &query_shape, const Shape &key_shape,
      const Shape &value_shape,
      const std::optional<Shape> &cu_seqlens_query_shape,
      const std::optional<Shape> &cu_seqlens_key_shape,
      const Shape &output_shape, const Shape &softmax_lse_shape,
      const std::optional<Shape> &s_dmask_shape,
      const std::optional<Shape> &alibi_slopes_shape, float dropout_rate,
      float scale, bool is_causal, const std::optional<int> &max_seqlen_q,
      const std::optional<int> &max_seqlen_k);

  FlashAttnFwdConfig(const FlashAttnConfig &config) : FlashAttnConfig(config) {}

  std::optional<se::dnn::TensorDescriptor> s_dmask_desc;  // output
};

struct FlashAttnBwdConfig : public FlashAttnConfig {
  static absl::StatusOr<FlashAttnBwdConfig> For(
      const Shape &grad_output_shape, const Shape &query_shape,
      const Shape &key_shape, const Shape &value_shape,
      const std::optional<Shape> &cu_seqlens_query_shape,
      const std::optional<Shape> &cu_seqlens_key_shape,
      const Shape &output_shape, const Shape &softmax_lse_shape,
      const Shape &grad_query_shape, const Shape &grad_key_shape,
      const Shape &grad_value_shape, const Shape &grad_softmax_shape,
      const std::optional<Shape> &alibi_slopes_shape, float dropout_rate,
      float scale, bool is_causal, bool deterministic,
      const std::optional<int> &max_seqlen_q,
      const std::optional<int> &max_seqlen_k);

  FlashAttnBwdConfig(const FlashAttnConfig &config) : FlashAttnConfig(config) {}

  se::dnn::TensorDescriptor grad_output_desc;   // input
  se::dnn::TensorDescriptor grad_query_desc;    // output
  se::dnn::TensorDescriptor grad_key_desc;      // output
  se::dnn::TensorDescriptor grad_value_desc;    // output
  se::dnn::TensorDescriptor grad_softmax_desc;  // output

  bool deterministic;
};

int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs,
                         int num_n_blocks, int max_splits);

absl::Status RunFlashAttnFwd(
    se::Stream *stream, const FlashAttnFwdConfig &config,
    se::DeviceMemoryBase query_buffer, se::DeviceMemoryBase key_buffer,
    se::DeviceMemoryBase value_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_query_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_key_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase softmax_lse_buffer,
    std::optional<se::DeviceMemoryBase> s_dmask_buffer,
    se::DeviceMemoryBase rng_state_buffer,
    std::optional<se::DeviceMemoryBase> alibi_slopes_buffer,
    std::optional<se::DeviceMemoryBase> softmax_lse_accum_buffer,
    std::optional<se::DeviceMemoryBase> output_accum_buffer,
    int window_size_left, int window_size_right);

absl::Status RunFlashAttnBwd(
    se::Stream *stream, const FlashAttnBwdConfig &config,
    se::DeviceMemoryBase grad_output_buffer, se::DeviceMemoryBase query_buffer,
    se::DeviceMemoryBase key_buffer, se::DeviceMemoryBase value_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_query_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_key_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase softmax_lse_buffer,
    se::DeviceMemoryBase rng_state_buffer,
    se::DeviceMemoryBase grad_query_buffer,
    se::DeviceMemoryBase grad_key_buffer,
    se::DeviceMemoryBase grad_value_buffer,
    se::DeviceMemoryBase grad_softmax_buffer,
    std::optional<se::DeviceMemoryBase> alibi_slopes_buffer,
    se::DeviceMemoryBase grad_query_accum_buffer, int window_size_left,
    int window_size_right);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_FLASH_ATTN_H_

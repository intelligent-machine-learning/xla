/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.1
==============================================================================*/

#include "xla/service/gpu/runtime/flash_attn.h"

#include <string>

#include "tsl/platform/human_readable_json.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

absl::Status FlashAttnFwd::Run(const ServiceExecutableRunOptions* run_options,
                               const DebugOptions* debug_options,
                               std::string_view call_target_name,
                               runtime::CustomCall::RemainingArgs args,
                               std::string_view backend_config,
                               bool is_varlen) {
  int64_t args_size = args.size();
  if (args_size < 6) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected at least 6 arguments, got %d", args_size));
  }

  auto query = args.get<runtime::StridedMemrefView>(0);
  auto key = args.get<runtime::StridedMemrefView>(1);
  auto value = args.get<runtime::StridedMemrefView>(2);
  if (failed(query) || failed(key) || failed(value)) {
    return absl::InvalidArgumentError("Incorrect argument types");
  }

  int64_t cur_arg_idx = 3;

  std::optional<runtime::StridedMemrefView> cu_seqlens_query;
  std::optional<runtime::StridedMemrefView> cu_seqlens_key;
  std::optional<runtime::StridedMemrefView> alibi_slopes;
  std::optional<runtime::StridedMemrefView> output_accum;
  std::optional<runtime::StridedMemrefView> softmax_lse_accum;

  FlashAttnBackendConfig config;
  TF_RETURN_IF_ERROR(
      tsl::HumanReadableJsonToProto(std::string(backend_config), &config));

  if (is_varlen) {
    auto cu_seqlens_query_ =
        args.get<runtime::StridedMemrefView>(cur_arg_idx++);
    auto cu_seqlens_key_ = args.get<runtime::StridedMemrefView>(cur_arg_idx++);
    if (failed(cu_seqlens_query_) || failed(cu_seqlens_key_)) {
      return absl::InvalidArgumentError("Incorrect argument types");
    }
    cu_seqlens_query = *cu_seqlens_query_;
    cu_seqlens_key = *cu_seqlens_key_;
    CHECK(config.has_max_seqlen_q() && config.has_max_seqlen_k());
  }

  if (config.has_alibi_slopes()) {
    auto alibi_slopes_ = args.get<runtime::StridedMemrefView>(cur_arg_idx++);
    if (failed(alibi_slopes_)) {
      return absl::InvalidArgumentError("Incorrect argument types");
    }
    alibi_slopes = *alibi_slopes_;
  }

  int64_t r_cur_arg_idx = args_size - 1;
  std::optional<runtime::StridedMemrefView> s_dmask;
  CHECK(config.has_return_softmax());
  bool return_softmax = config.return_softmax();
  if (return_softmax) {
    auto s_dmask_ = args.get<runtime::StridedMemrefView>(r_cur_arg_idx--);
    if (failed(s_dmask_)) {
      return absl::InvalidArgumentError("Incorrect argument types");
    }
    s_dmask = *s_dmask_;
  }

  auto rng_state = args.get<runtime::StridedMemrefView>(r_cur_arg_idx--);
  auto softmax_lse = args.get<runtime::StridedMemrefView>(r_cur_arg_idx--);
  auto output = args.get<runtime::StridedMemrefView>(r_cur_arg_idx--);

  CHECK(std::abs(cur_arg_idx - r_cur_arg_idx) == 1);

  if (cur_arg_idx < r_cur_arg_idx) {
    auto output_accum_ = args.get<runtime::StridedMemrefView>(cur_arg_idx++);
    auto softmax_lse_accum_ =
        args.get<runtime::StridedMemrefView>(cur_arg_idx++);
    if (failed(output_accum_) || failed(softmax_lse_accum_)) {
      return absl::InvalidArgumentError("Incorrect argument types");
    }
    output_accum = *output_accum_;
    softmax_lse_accum = *softmax_lse_accum_;
  }

  FlashAttnFwd handler = FlashAttnFwd::Handler();
  return handler(run_options, debug_options, config, *query, *key, *value,
                 cu_seqlens_query, cu_seqlens_key, alibi_slopes, output_accum,
                 softmax_lse_accum, *output, *softmax_lse, *rng_state, s_dmask);
}

static std::optional<se::DeviceMemoryBase> GetDeviceAddressIfNotNull(
    const std::optional<runtime::StridedMemrefView>& memref) {
  if (!memref.has_value()) {
    return std::nullopt;
  }
  return GetDeviceAddress(*memref);
}

static std::optional<Shape> GetShapeIfNotNull(
    const std::optional<runtime::StridedMemrefView>& memref) {
  if (!memref.has_value()) {
    return std::nullopt;
  }
  return ToShape(*memref);
}

absl::Status FlashAttnFwd::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options,
    const FlashAttnBackendConfig& backend_config,
    // Inputs
    runtime::StridedMemrefView query, runtime::StridedMemrefView key,
    runtime::StridedMemrefView value,
    std::optional<runtime::StridedMemrefView> cu_seqlens_query,
    std::optional<runtime::StridedMemrefView> cu_seqlens_key,
    std::optional<runtime::StridedMemrefView> alibi_slopes,
    std::optional<runtime::StridedMemrefView> output_accum,
    std::optional<runtime::StridedMemrefView> softmax_lse_accum,
    // Outputs
    runtime::StridedMemrefView output, runtime::StridedMemrefView softmax_lse,
    runtime::StridedMemrefView rng_state,
    std::optional<runtime::StridedMemrefView> s_dmask) const {
  const Shape& query_shape = ToShape(query);
  const Shape& key_shape = ToShape(key);
  const Shape& value_shape = ToShape(value);
  const std::optional<Shape>& cu_seqlens_query_shape =
      GetShapeIfNotNull(cu_seqlens_query);
  const std::optional<Shape>& cu_seqlens_key_shape =
      GetShapeIfNotNull(cu_seqlens_key);
  const std::optional<Shape>& alibi_slopes_shape =
      GetShapeIfNotNull(alibi_slopes);
  const Shape& output_shape = ToShape(output);
  const Shape& softmax_lse_shape = ToShape(softmax_lse);
  const std::optional<Shape>& s_dmask_shape = GetShapeIfNotNull(s_dmask);

  // Ignore output_accum shape, softmax_lse_accum shape, and rng_state shape,
  // because we only need their pointers

  bool is_varlen = backend_config.has_max_seqlen_q();
  CHECK(is_varlen == backend_config.has_max_seqlen_k());
  std::optional<int> max_seqlen_q, max_seqlen_k;
  if (is_varlen) {
    max_seqlen_q = backend_config.max_seqlen_q();
    max_seqlen_k = backend_config.max_seqlen_k();
  }

  TF_ASSIGN_OR_RETURN(
      FlashAttnFwdConfig config,
      FlashAttnFwdConfig::For(
          query_shape, key_shape, value_shape, cu_seqlens_query_shape,
          cu_seqlens_key_shape, alibi_slopes_shape, output_shape,
          softmax_lse_shape, s_dmask_shape, backend_config.dropout_rate(),
          backend_config.scale(), backend_config.is_causal(), max_seqlen_q,
          max_seqlen_k));

  se::DeviceMemoryBase query_buffer = GetDeviceAddress(query);
  se::DeviceMemoryBase key_buffer = GetDeviceAddress(key);
  se::DeviceMemoryBase value_buffer = GetDeviceAddress(value);
  std::optional<se::DeviceMemoryBase> cu_seqlens_query_buffer =
      GetDeviceAddressIfNotNull(cu_seqlens_query);
  std::optional<se::DeviceMemoryBase> cu_seqlens_key_buffer =
      GetDeviceAddressIfNotNull(cu_seqlens_key);
  std::optional<se::DeviceMemoryBase> alibi_slopes_buffer =
      GetDeviceAddressIfNotNull(alibi_slopes);
  std::optional<se::DeviceMemoryBase> output_accum_buffer =
      GetDeviceAddressIfNotNull(output_accum);
  std::optional<se::DeviceMemoryBase> softmax_lse_accum_buffer =
      GetDeviceAddressIfNotNull(softmax_lse_accum);

  se::DeviceMemoryBase output_buffer = GetDeviceAddress(output);
  se::DeviceMemoryBase softmax_lse_buffer = GetDeviceAddress(softmax_lse);
  se::DeviceMemoryBase rng_state_buffer = GetDeviceAddress(rng_state);
  std::optional<se::DeviceMemoryBase> s_dmask_buffer =
      GetDeviceAddressIfNotNull(s_dmask);

  se::Stream* stream = run_options->stream();
  return RunFlashAttnFwd(
      stream, config, query_buffer, key_buffer, value_buffer,
      cu_seqlens_query_buffer, cu_seqlens_key_buffer, alibi_slopes_buffer,
      output_accum_buffer, softmax_lse_accum_buffer, output_buffer,
      softmax_lse_buffer, rng_state_buffer, s_dmask_buffer, -1, -1);
}

absl::Status FlashAttnBwd::Run(const ServiceExecutableRunOptions* run_options,
                               const DebugOptions* debug_options,
                               std::string_view call_target_name,
                               runtime::CustomCall::RemainingArgs args,
                               std::string_view backend_config,
                               bool is_varlen) {
  FlashAttnBackendConfig config;
  TF_RETURN_IF_ERROR(
      tsl::HumanReadableJsonToProto(std::string(backend_config), &config));

  int64_t expected_args_size =
      12 + (is_varlen ? 2 : 0) + config.has_alibi_slopes();
  if (args.size() != expected_args_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected %d arguments, got %d", expected_args_size, args.size()));
  }

  auto grad_output = args.get<runtime::StridedMemrefView>(0);
  auto query = args.get<runtime::StridedMemrefView>(1);
  auto key = args.get<runtime::StridedMemrefView>(2);
  auto value = args.get<runtime::StridedMemrefView>(3);
  auto output = args.get<runtime::StridedMemrefView>(4);
  auto softmax_lse = args.get<runtime::StridedMemrefView>(5);
  auto rng_state = args.get<runtime::StridedMemrefView>(6);
  if (failed(grad_output) || failed(query) || failed(key) || failed(value) ||
      failed(output) || failed(softmax_lse) || failed(rng_state)) {
    return absl::InvalidArgumentError("Incorrect argument types");
  }

  int64_t cur_arg_idx = 7;

  std::optional<runtime::StridedMemrefView> cu_seqlens_query;
  std::optional<runtime::StridedMemrefView> cu_seqlens_key;
  std::optional<runtime::StridedMemrefView> alibi_slopes;

  if (is_varlen) {
    auto cu_seqlens_query_ =
        args.get<runtime::StridedMemrefView>(cur_arg_idx++);
    auto cu_seqlens_key_ = args.get<runtime::StridedMemrefView>(cur_arg_idx++);
    if (failed(cu_seqlens_query_) || failed(cu_seqlens_key_)) {
      return absl::InvalidArgumentError("Incorrect argument types");
    }
    cu_seqlens_query = *cu_seqlens_query_;
    cu_seqlens_key = *cu_seqlens_key_;
  }

  if (config.has_alibi_slopes()) {
    auto alibi_slopes_ = args.get<runtime::StridedMemrefView>(cur_arg_idx++);
    if (failed(alibi_slopes_)) {
      return absl::InvalidArgumentError("Incorrect argument types");
    }
    alibi_slopes = *alibi_slopes_;
  }

  auto grad_query_accum = args.get<runtime::StridedMemrefView>(cur_arg_idx++);
  auto grad_query = args.get<runtime::StridedMemrefView>(cur_arg_idx++);
  auto grad_key = args.get<runtime::StridedMemrefView>(cur_arg_idx++);
  auto grad_value = args.get<runtime::StridedMemrefView>(cur_arg_idx++);
  auto grad_softmax = args.get<runtime::StridedMemrefView>(cur_arg_idx++);

  CHECK(cur_arg_idx == expected_args_size);

  FlashAttnBwd handler = FlashAttnBwd::Handler();
  return handler(run_options, debug_options, config, *grad_output, *query, *key,
                 *value, *output, *softmax_lse, *rng_state, cu_seqlens_query,
                 cu_seqlens_key, alibi_slopes, *grad_query_accum, *grad_query,
                 *grad_key, *grad_value, *grad_softmax);
}

absl::Status FlashAttnBwd::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options,
    const FlashAttnBackendConfig& backend_config,
    // Inputs
    runtime::StridedMemrefView grad_output, runtime::StridedMemrefView query,
    runtime::StridedMemrefView key, runtime::StridedMemrefView value,
    runtime::StridedMemrefView output, runtime::StridedMemrefView softmax_lse,
    runtime::StridedMemrefView rng_state,
    std::optional<runtime::StridedMemrefView> cu_seqlens_query,
    std::optional<runtime::StridedMemrefView> cu_seqlens_key,
    std::optional<runtime::StridedMemrefView> alibi_slopes,
    runtime::StridedMemrefView grad_query_accum,
    // Outputs
    runtime::StridedMemrefView grad_query, runtime::StridedMemrefView grad_key,
    runtime::StridedMemrefView grad_value,
    runtime::StridedMemrefView grad_softmax) const {
  const Shape& grad_output_shape = ToShape(grad_output);
  const Shape& query_shape = ToShape(query);
  const Shape& key_shape = ToShape(key);
  const Shape& value_shape = ToShape(value);
  const Shape& output_shape = ToShape(output);
  const Shape& softmax_lse_shape = ToShape(softmax_lse);
  const std::optional<Shape>& cu_seqlens_query_shape =
      GetShapeIfNotNull(cu_seqlens_query);
  const std::optional<Shape>& cu_seqlens_key_shape =
      GetShapeIfNotNull(cu_seqlens_key);
  const std::optional<Shape>& alibi_slopes_shape =
      GetShapeIfNotNull(alibi_slopes);
  const Shape& grad_query_shape = ToShape(grad_query);
  const Shape& grad_key_shape = ToShape(grad_key);
  const Shape& grad_value_shape = ToShape(grad_value);
  const Shape& grad_softmax_shape = ToShape(grad_softmax);

  // Ignore rng_state shape and grad_query_accum shape because we only need its
  // pointer

  bool is_varlen = backend_config.has_max_seqlen_q();
  CHECK(is_varlen == backend_config.has_max_seqlen_k());
  std::optional<int> max_seqlen_q, max_seqlen_k;
  if (is_varlen) {
    max_seqlen_q = backend_config.max_seqlen_q();
    max_seqlen_k = backend_config.max_seqlen_k();
  }

  TF_ASSIGN_OR_RETURN(
      FlashAttnBwdConfig config,
      FlashAttnBwdConfig::For(
          grad_output_shape, query_shape, key_shape, value_shape, output_shape,
          softmax_lse_shape, cu_seqlens_query_shape, cu_seqlens_key_shape,
          alibi_slopes_shape, grad_query_shape, grad_key_shape,
          grad_value_shape, grad_softmax_shape, backend_config.dropout_rate(),
          backend_config.scale(), backend_config.is_causal(),
          backend_config.deterministic(), max_seqlen_q, max_seqlen_k));

  se::DeviceMemoryBase grad_output_buffer = GetDeviceAddress(grad_output);
  se::DeviceMemoryBase query_buffer = GetDeviceAddress(query);
  se::DeviceMemoryBase key_buffer = GetDeviceAddress(key);
  se::DeviceMemoryBase value_buffer = GetDeviceAddress(value);
  se::DeviceMemoryBase output_buffer = GetDeviceAddress(output);
  se::DeviceMemoryBase softmax_lse_buffer = GetDeviceAddress(softmax_lse);
  se::DeviceMemoryBase rng_state_buffer = GetDeviceAddress(rng_state);
  std::optional<se::DeviceMemoryBase> cu_seqlens_query_buffer =
      GetDeviceAddressIfNotNull(cu_seqlens_query);
  std::optional<se::DeviceMemoryBase> cu_seqlens_key_buffer =
      GetDeviceAddressIfNotNull(cu_seqlens_key);
  std::optional<se::DeviceMemoryBase> alibi_slopes_buffer =
      GetDeviceAddressIfNotNull(alibi_slopes);
  se::DeviceMemoryBase grad_query_accum_buffer =
      GetDeviceAddress(grad_query_accum);

  se::DeviceMemoryBase grad_query_buffer = GetDeviceAddress(grad_query);
  se::DeviceMemoryBase grad_key_buffer = GetDeviceAddress(grad_key);
  se::DeviceMemoryBase grad_value_buffer = GetDeviceAddress(grad_value);
  se::DeviceMemoryBase grad_softmax_buffer = GetDeviceAddress(grad_softmax);

  se::Stream* stream = run_options->stream();
  return RunFlashAttnBwd(
      stream, config, grad_output_buffer, query_buffer, key_buffer,
      value_buffer, output_buffer, softmax_lse_buffer, rng_state_buffer,
      cu_seqlens_query_buffer, cu_seqlens_key_buffer, alibi_slopes_buffer,
      grad_query_accum_buffer, grad_query_buffer, grad_key_buffer,
      grad_value_buffer, grad_softmax_buffer);
}

}  // namespace gpu
}  // namespace xla

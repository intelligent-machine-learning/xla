/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_FLASH_ATTENTION_H_
#define XLA_SERVICE_GPU_RUNTIME_FLASH_ATTENTION_H_

#include <optional>
#include <string_view>

#include "xla/runtime/custom_call.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_flash_attn.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

struct FlashAttnFwd {
  // Adaptor from XlaCustomCall API to properly typed FlashAttnFwd handler.
  static absl::Status Run(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          std::string_view call_target_name,
                          runtime::CustomCall::RemainingArgs args,
                          std::string_view backend_config, bool is_varlen);

  absl::Status operator()(
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
      std::optional<runtime::StridedMemrefView> s_dmask) const;

  static FlashAttnFwd Handler() { return FlashAttnFwd(); }
};

struct FlashAttnBwd {
  // Adaptor from XlaCustomCall API to properly typed FlashAttnBwd handler.
  static absl::Status Run(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          std::string_view call_target_name,
                          runtime::CustomCall::RemainingArgs args,
                          std::string_view backend_config, bool is_varlen);

  absl::Status operator()(
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
      runtime::StridedMemrefView grad_query,
      runtime::StridedMemrefView grad_key,
      runtime::StridedMemrefView grad_value,
      runtime::StridedMemrefView grad_softmax) const;

  static FlashAttnBwd Handler() { return FlashAttnBwd(); }
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_FLASH_ATTENTION_H_

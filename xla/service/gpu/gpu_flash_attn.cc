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

#include "xla/service/gpu/gpu_flash_attn.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "absl/strings/string_view.h"
#include "cutlass/numeric_types.h"
#include "flash_attn/flash.h"
#include "flash_attn/static_switch.h"
#include "flash_attn/utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/gpu/gpu_stream.h"

namespace xla {
namespace gpu {

const absl::string_view kGpuFlashAttnFwdCallTarget = "__gpu$flash_attn_fwd";
const absl::string_view kGpuFlashAttnBwdCallTarget = "__gpu$flash_attn_bwd";
const absl::string_view kGpuFlashAttnVarLenFwdCallTarget =
    "__gpu$flash_attn_varlen_fwd";
const absl::string_view kGpuFlashAttnVarLenBwdCallTarget =
    "__gpu$flash_attn_varlen_bwd";

bool IsCustomCallToFlashAttn(const HloInstruction &hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const std::string &target = hlo.custom_call_target();
  return target == kGpuFlashAttnFwdCallTarget ||
         target == kGpuFlashAttnVarLenFwdCallTarget ||
         target == kGpuFlashAttnBwdCallTarget ||
         target == kGpuFlashAttnVarLenBwdCallTarget;
}

absl::StatusOr<FlashAttnKind> GetFlashAttnKind(
    const HloCustomCallInstruction *instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kGpuFlashAttnFwdCallTarget) {
    return FlashAttnKind::kForward;
  }
  if (target == kGpuFlashAttnVarLenFwdCallTarget) {
    return FlashAttnKind::kVarLenForward;
  }
  if (target == kGpuFlashAttnBwdCallTarget) {
    return FlashAttnKind::kBackward;
  }
  if (target == kGpuFlashAttnVarLenBwdCallTarget) {
    return FlashAttnKind::kVarLenBackward;
  }
  return Internal("Unexpected call target: %s", target);
}

absl::StatusOr<FlashAttnConfig> FlashAttnConfig::For(
    const Shape &query_shape, const Shape &key_shape, const Shape &value_shape,
    const std::optional<Shape> &cu_seqlens_query_shape,
    const std::optional<Shape> &cu_seqlens_key_shape,
    const std::optional<Shape> &alibi_slopes_shape, const Shape &output_shape,
    const Shape &softmax_lse_shape, float dropout_rate, float scale,
    bool is_causal, const std::optional<int> &max_seqlen_q,
    const std::optional<int> &max_seqlen_k) {
  PrimitiveType type = query_shape.element_type();

  CHECK(type == PrimitiveType::F16 || type == PrimitiveType::BF16);
  CHECK(type == key_shape.element_type() &&
        type == value_shape.element_type() &&
        type == output_shape.element_type());

  FlashAttnConfig config;
  config.type = type;

  TF_ASSIGN_OR_RETURN(se::dnn::DataType elem_type,
                      GetDNNDataTypeFromPrimitiveType(type));
  TF_ASSIGN_OR_RETURN(se::dnn::DataType f32_type,
                      GetDNNDataTypeFromPrimitiveType(PrimitiveType::F32));

  config.query_desc =
      se::dnn::TensorDescriptor::For(elem_type, query_shape.dimensions(),
                                     query_shape.layout().minor_to_major());
  config.key_desc = se::dnn::TensorDescriptor::For(
      elem_type, key_shape.dimensions(), key_shape.layout().minor_to_major());
  config.value_desc =
      se::dnn::TensorDescriptor::For(elem_type, value_shape.dimensions(),
                                     value_shape.layout().minor_to_major());
  bool is_varlen = cu_seqlens_query_shape.has_value();
  CHECK(is_varlen == cu_seqlens_key_shape.has_value() &&
        is_varlen == max_seqlen_q.has_value() &&
        is_varlen == max_seqlen_k.has_value());
  if (is_varlen) {
    CHECK(cu_seqlens_query_shape->element_type() == PrimitiveType::S32);
    CHECK(cu_seqlens_key_shape->element_type() == PrimitiveType::S32);
    TF_ASSIGN_OR_RETURN(se::dnn::DataType cu_type,
                        GetDNNDataTypeFromPrimitiveType(PrimitiveType::S32));
    config.cu_seqlens_query_desc = se::dnn::TensorDescriptor::For(
        cu_type, cu_seqlens_query_shape->dimensions(),
        cu_seqlens_query_shape->layout().minor_to_major());
    config.cu_seqlens_key_desc = se::dnn::TensorDescriptor::For(
        cu_type, cu_seqlens_key_shape->dimensions(),
        cu_seqlens_key_shape->layout().minor_to_major());
    config.max_seqlen_q = max_seqlen_q;
    config.max_seqlen_k = max_seqlen_k;
  }

  if (alibi_slopes_shape.has_value()) {
    config.alibi_slopes_desc = se::dnn::TensorDescriptor::For(
        f32_type, alibi_slopes_shape.value().dimensions(),
        alibi_slopes_shape.value().layout().minor_to_major());
  }

  config.output_desc =
      se::dnn::TensorDescriptor::For(elem_type, output_shape.dimensions(),
                                     output_shape.layout().minor_to_major());
  config.softmax_lse_desc = se::dnn::TensorDescriptor::For(
      f32_type, softmax_lse_shape.dimensions(),
      softmax_lse_shape.layout().minor_to_major());

  config.dropout_rate = dropout_rate;
  config.scale = scale;
  config.is_causal = is_causal;
  return config;
}

absl::StatusOr<FlashAttnFwdConfig> FlashAttnFwdConfig::For(
    const Shape &query_shape, const Shape &key_shape, const Shape &value_shape,
    const std::optional<Shape> &cu_seqlens_query_shape,
    const std::optional<Shape> &cu_seqlens_key_shape,
    const std::optional<Shape> &alibi_slopes_shape, const Shape &output_shape,
    const Shape &softmax_lse_shape, const std::optional<Shape> &s_dmask_shape,
    float dropout_rate, float scale, bool is_causal,
    const std::optional<int> &max_seqlen_q,
    const std::optional<int> &max_seqlen_k) {
  TF_ASSIGN_OR_RETURN(
      FlashAttnFwdConfig config,
      FlashAttnConfig::For(query_shape, key_shape, value_shape,
                           cu_seqlens_query_shape, cu_seqlens_key_shape,
                           alibi_slopes_shape, output_shape, softmax_lse_shape,
                           dropout_rate, scale, is_causal, max_seqlen_q,
                           max_seqlen_k));

  if (s_dmask_shape.has_value()) {
    TF_ASSIGN_OR_RETURN(se::dnn::DataType elem_type,
                        GetDNNDataTypeFromPrimitiveType(config.type));

    config.s_dmask_desc = se::dnn::TensorDescriptor::For(
        elem_type, s_dmask_shape->dimensions(),
        s_dmask_shape->layout().minor_to_major());
  }

  return config;
}

absl::StatusOr<FlashAttnBwdConfig> FlashAttnBwdConfig::For(
    const Shape &grad_output_shape, const Shape &query_shape,
    const Shape &key_shape, const Shape &value_shape, const Shape &output_shape,
    const Shape &softmax_lse_shape,
    const std::optional<Shape> &cu_seqlens_query_shape,
    const std::optional<Shape> &cu_seqlens_key_shape,
    const std::optional<Shape> &alibi_slopes_shape,
    const Shape &grad_query_shape, const Shape &grad_key_shape,
    const Shape &grad_value_shape, const Shape &grad_softmax_shape,
    float dropout_rate, float scale, bool is_causal, bool deterministic,
    const std::optional<int> &max_seqlen_q,
    const std::optional<int> &max_seqlen_k) {
  TF_ASSIGN_OR_RETURN(
      FlashAttnBwdConfig config,
      FlashAttnConfig::For(query_shape, key_shape, value_shape,
                           cu_seqlens_query_shape, cu_seqlens_key_shape,
                           alibi_slopes_shape, output_shape, softmax_lse_shape,
                           dropout_rate, scale, is_causal, max_seqlen_q,
                           max_seqlen_k));

  TF_ASSIGN_OR_RETURN(se::dnn::DataType elem_type,
                      GetDNNDataTypeFromPrimitiveType(config.type));

  config.grad_output_desc = se::dnn::TensorDescriptor::For(
      elem_type, grad_output_shape.dimensions(),
      grad_output_shape.layout().minor_to_major());

  config.grad_query_desc = se::dnn::TensorDescriptor::For(
      elem_type, grad_query_shape.dimensions(),
      grad_query_shape.layout().minor_to_major());
  config.grad_key_desc =
      se::dnn::TensorDescriptor::For(elem_type, grad_key_shape.dimensions(),
                                     grad_key_shape.layout().minor_to_major());
  config.grad_value_desc = se::dnn::TensorDescriptor::For(
      elem_type, grad_value_shape.dimensions(),
      grad_value_shape.layout().minor_to_major());
  config.grad_softmax_desc = se::dnn::TensorDescriptor::For(
      se::dnn::ToDataType<float>::value, grad_softmax_shape.dimensions(),
      grad_softmax_shape.layout().minor_to_major());

  config.deterministic = deterministic;

  return config;
}

static void set_params_fprop(
    Flash_fwd_params &params, const FlashAttnConfig &config,
    se::DeviceMemoryBase query_buffer, se::DeviceMemoryBase key_buffer,
    se::DeviceMemoryBase value_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase softmax_lse_buffer,
    std::optional<se::DeviceMemoryBase> s_dmask_buffer,
    // sizes
    const size_t batch_size, const size_t seqlen_q, const size_t seqlen_k,
    const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
    const size_t num_heads, const size_t num_heads_k, const size_t head_size,
    const size_t head_size_rounded, void *cu_seqlens_q_d, void *cu_seqlens_k_d,
    void *seqused_k, float p_dropout, float softmax_scale, int window_size_left,
    int window_size_right, bool seqlenq_ngroups_swapped = false) {
  // Reset the parameters
  params = {};

  params.is_bf16 = config.type == PrimitiveType::BF16;

  // Set the pointers and strides.
  params.q_ptr = query_buffer.opaque();
  params.k_ptr = key_buffer.opaque();
  params.v_ptr = value_buffer.opaque();
  params.o_ptr = output_buffer.opaque();

  // All stride are in elements, not bytes.
  const auto &q_strides = config.query_desc.GetLogicalStrides();
  const auto &k_strides = config.key_desc.GetLogicalStrides();
  const auto &v_strides = config.value_desc.GetLogicalStrides();
  const auto &o_strides = config.output_desc.GetLogicalStrides();

  // sequence length
  params.q_row_stride = q_strides[q_strides.size() - 3];
  params.k_row_stride = k_strides[k_strides.size() - 3];
  params.v_row_stride = v_strides[v_strides.size() - 3];
  params.o_row_stride = o_strides[o_strides.size() - 3];

  // head number
  params.q_head_stride = q_strides[q_strides.size() - 2];
  params.k_head_stride = k_strides[k_strides.size() - 2];
  params.v_head_stride = v_strides[v_strides.size() - 2];
  params.o_head_stride = o_strides[o_strides.size() - 2];

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = q_strides[0];
    params.k_batch_stride = k_strides[0];
    params.v_batch_stride = v_strides[0];
    params.o_batch_stride = o_strides[0];
    if (seqlenq_ngroups_swapped) {
      params.q_batch_stride *= seqlen_q;
      params.o_batch_stride *= seqlen_q;
    }
  }

  params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int *>(seqused_k);

  // P = softmax(QK^T)
  params.p_ptr =
      s_dmask_buffer.has_value() ? s_dmask_buffer->opaque() : nullptr;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_buffer.opaque();

  // Set the dimensions.
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.h_h_k_ratio = num_heads / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = head_size;
  params.d_rounded = head_size_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to
  // float to compare. [Minor] We want to round down since when we do the
  // comparison we use <= instead of < params.p_dropout_in_uint =
  // uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout *
  // 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  CHECK(p_dropout < 1.f);
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  CHECK(p_dropout == 0.0f)
      << "This flash attention build does not support dropout.";
#endif

  // Causal is the special case where window_size_right == 0 and
  // window_size_left < 0. Local is the more general case where
  // window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

#ifdef FLASHATTENTION_DISABLE_LOCAL
  CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0))
      << "This flash attention build does not support local attention.";
#endif

  params.is_seqlens_k_cumulative = true;

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  CHECK(head_size == head_size_rounded)
      << "This flash attention build does not support "
         "headdim not being a multiple of 32.";
#endif
}

static void set_params_dgrad(
    Flash_bwd_params &params, const FlashAttnBwdConfig &config,
    se::DeviceMemoryBase grad_output_buffer, se::DeviceMemoryBase query_buffer,
    se::DeviceMemoryBase key_buffer, se::DeviceMemoryBase value_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase softmax_lse_buffer,
    se::DeviceMemoryBase grad_query_buffer,
    se::DeviceMemoryBase grad_key_buffer,
    se::DeviceMemoryBase grad_value_buffer,
    se::DeviceMemoryBase grad_softmax_buffer,
    // sizes
    const size_t batch_size, const size_t seqlen_q, const size_t seqlen_k,
    const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
    const size_t num_heads, const size_t num_heads_k, const size_t head_size,
    const size_t head_size_rounded, void *cu_seqlens_q_d, void *cu_seqlens_k_d,
    void *dq_accum_d, void *dk_accum_d, void *dv_accum_d, float p_dropout,
    float softmax_scale, int window_size_left, int window_size_right,
    bool deterministic) {
  set_params_fprop(params, config, query_buffer, key_buffer, value_buffer,
                   output_buffer, softmax_lse_buffer, std::nullopt, batch_size,
                   seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded,
                   num_heads, num_heads_k, head_size, head_size_rounded,
                   cu_seqlens_q_d, cu_seqlens_k_d,
                   /*seqused_k=*/nullptr, p_dropout, softmax_scale,
                   window_size_left, window_size_right);

  // Set the pointers and strides.
  params.do_ptr = grad_output_buffer.opaque();
  params.dq_ptr = grad_query_buffer.opaque();
  params.dk_ptr = grad_key_buffer.opaque();
  params.dv_ptr = grad_value_buffer.opaque();

  // All stride are in elements, not bytes.
  const auto &grad_output_strides = config.grad_output_desc.GetLogicalStrides();
  const auto &grad_query_strides = config.grad_query_desc.GetLogicalStrides();
  const auto &grad_key_strides = config.grad_key_desc.GetLogicalStrides();
  const auto &grad_value_strides = config.grad_value_desc.GetLogicalStrides();

  // sequence length
  params.do_row_stride = grad_output_strides[grad_output_strides.size() - 3];
  params.dq_row_stride = grad_query_strides[grad_query_strides.size() - 3];
  params.dk_row_stride = grad_key_strides[grad_key_strides.size() - 3];
  params.dv_row_stride = grad_value_strides[grad_value_strides.size() - 3];

  // head number
  params.do_head_stride = grad_output_strides[grad_output_strides.size() - 2];
  params.dq_head_stride = grad_query_strides[grad_query_strides.size() - 2];
  params.dk_head_stride = grad_key_strides[grad_key_strides.size() - 2];
  params.dv_head_stride = grad_value_strides[grad_value_strides.size() - 2];

  if (cu_seqlens_q_d == nullptr) {
    params.do_batch_stride = grad_output_strides[0];
    params.dq_batch_stride = grad_query_strides[0];
    params.dk_batch_stride = grad_key_strides[0];
    params.dv_batch_stride = grad_value_strides[0];
  }

  params.dq_accum_ptr = dq_accum_d;
  params.dk_accum_ptr = dk_accum_d;
  params.dv_accum_ptr = dv_accum_d;

  // Softmax sum
  params.dsoftmax_sum = grad_softmax_buffer.opaque();

  params.deterministic = deterministic;
}

// Find the number of splits that maximizes the occupancy. For example, if we
// have batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency =
// 0.89) is better than having 3 splits (efficiency = 0.67). However, we also
// don't want too many splits as that would incur more HBM reads/writes. So we
// find the best efficiency, then find the smallest number of splits that gets
// 85% of the best efficiency.
int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs,
                         int num_n_blocks, int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) {
    return 1;
  }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose
  // 11 splits, we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have
  // 6 * 11 + (-2) blocks (i.e. it's 11 splits anyway). So we check if the
  // number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1 || ceildiv(num_n_blocks, num_splits) !=
                                  ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      // printf("num_splits = %d, eff = %f\n", num_splits, eff);
      if (eff > max_efficiency) {
        max_efficiency = eff;
      }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

static void set_params_splitkv(
    Flash_fwd_params &params, const int batch_size, const int num_heads,
    const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout, const int num_splits,
    std::optional<se::DeviceMemoryBase> output_accum_buffer,
    std::optional<se::DeviceMemoryBase> softmax_lse_accum_buffer,
    const cudaDeviceProp *dprops) {
  params.num_splits = num_splits;
  if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
    if (num_splits < 1) {
      // This needs to match with run_mha_fwd_splitkv_dispatch
      const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
      const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
      // Technically kBlockM = 64 only for the splitKV kernels, not the standard
      // kernel. In any case we don't expect seqlen_q to be larger than 64 for
      // inference.
      const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
      params.num_splits = num_splits_heuristic(
          batch_size * num_heads * num_m_blocks,
          dprops->multiProcessorCount * 2, num_n_blocks, 128);
    }
    bool splitkv = params.num_splits > 1;
    CHECK(splitkv == output_accum_buffer.has_value() &&
          splitkv == softmax_lse_accum_buffer.has_value());
    if (splitkv) {
      params.oaccum_ptr = output_accum_buffer->opaque();
      params.softmax_lseaccum_ptr = softmax_lse_accum_buffer->opaque();
    }
    CHECK(params.num_splits <= 128) << "num_splits > 128 not supported";
  }
}

static void set_params_alibi(
    Flash_fwd_params &params,
    std::optional<se::DeviceMemoryBase> &alibi_slopes_buffer,
    std::optional<se::dnn::TensorDescriptor> alibi_slopes_desc, int batch_size,
    int num_heads) {
#ifdef FLASHATTENTION_DISABLE_ALIBI
  TORCH_CHECK(!alibi_slopes_buffer.has_value(),
              "This flash attention build does not support alibi.");
  params.alibi_slopes_ptr = nullptr;
#else
  if (alibi_slopes_buffer.has_value()) {
    CHECK(alibi_slopes_desc->type() == se::dnn::ToDataType<float>::value)
        << "ALiBi slopes must have dtype fp32";
    const auto &alibi_slopes_strides = alibi_slopes_desc->GetLogicalStrides();
    CHECK(alibi_slopes_strides.back() == 1)
        << "ALiBi slopes tensor must have contiguous last dimension";
    CHECK((alibi_slopes_desc->dimensions() == std::vector<int64_t>{num_heads} ||
           alibi_slopes_desc->dimensions() ==
               std::vector<int64_t>{batch_size, num_heads}));
    params.alibi_slopes_ptr = alibi_slopes_buffer->opaque();
    params.alibi_slopes_batch_stride =
        alibi_slopes_desc->ndims() == 2 ? alibi_slopes_strides.front() : 0;
  } else {
    params.alibi_slopes_ptr = nullptr;
  }
#endif
}

static void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream,
                        bool force_split_kernel = false) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d, [&] {
      if (params.num_splits <= 1 &&
          !force_split_kernel) {  // If we don't set it num_splits == 0
        run_mha_fwd_<elem_type, kHeadDim>(params, stream);
      } else {
        run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
      }
    });
  });
}

static void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d,
                   [&] { run_mha_bwd_<elem_type, kHeadDim>(params, stream); });
  });
}

static int64_t GetNumElements(const std::vector<int64_t> &dims) {
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<int64_t>());
}

absl::Status RunFlashAttnFwd(
    se::Stream *stream, const FlashAttnFwdConfig &config,
    se::DeviceMemoryBase query_buffer, se::DeviceMemoryBase key_buffer,
    se::DeviceMemoryBase value_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_query_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_key_buffer,
    std::optional<se::DeviceMemoryBase> alibi_slopes_buffer,
    std::optional<se::DeviceMemoryBase> output_accum_buffer,
    std::optional<se::DeviceMemoryBase> softmax_lse_accum_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase softmax_lse_buffer,
    se::DeviceMemoryBase rng_state_buffer,
    std::optional<se::DeviceMemoryBase> s_dmask_buffer, int window_size_left,
    int window_size_right) {
  const float p_dropout = config.dropout_rate;
  const float softmax_scale = config.scale;
  bool is_causal = config.is_causal;

  const cudaDeviceProp *dprops = flash::cuda::getCurrentDeviceProperties();
  const bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
  const bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  CHECK(is_sm8x || is_sm90)
      << "FlashAttention only supports Ampere GPUs or newer.";

  bool is_varlen = cu_seqlens_query_buffer.has_value();
  CHECK(is_varlen == cu_seqlens_key_buffer.has_value() &&
        is_varlen == config.cu_seqlens_query_desc.has_value() &&
        is_varlen == config.cu_seqlens_key_desc.has_value() &&
        is_varlen == config.max_seqlen_q.has_value() &&
        is_varlen == config.max_seqlen_k.has_value())
      << "cu_seqlens_query_buffer, cu_seqlens_key_buffer, max_seqlen_q, and "
         "max_seqlen_k must be all set or all unset.";

  const auto &q_strides = config.query_desc.GetLogicalStrides();
  const auto &k_strides = config.key_desc.GetLogicalStrides();
  const auto &v_strides = config.value_desc.GetLogicalStrides();

  CHECK(q_strides.back() == 1 && k_strides.back() == 1 && v_strides.back() == 1)
      << "Input tensor must have contiguous last dimension in FlashAttention.";

  if (is_varlen) {
    CHECK(config.cu_seqlens_query_desc->ndims() == 1 &&
          config.cu_seqlens_key_desc->ndims() == 1);
  }

  const auto &o_strides = config.output_desc.GetLogicalStrides();
  CHECK(o_strides.back() == 1)
      << "Output tensor must have contiguous last dimension in FlashAttention.";

  const auto &q_sizes = config.query_desc.dimensions();
  const auto &k_sizes = config.key_desc.dimensions();

  int batch_size, num_heads, head_size_og, num_heads_k;
  int seqlen_q, seqlen_k;

  if (is_varlen) {
    batch_size = GetNumElements(config.cu_seqlens_query_desc->dimensions()) - 1;
    num_heads = q_sizes[1];
    head_size_og = q_sizes[2];
    num_heads_k = k_sizes[1];

    seqlen_q = config.max_seqlen_q.value();
    seqlen_k = config.max_seqlen_k.value();
  } else {
    batch_size = q_sizes[0];
    num_heads = q_sizes[2];
    head_size_og = q_sizes[3];
    num_heads_k = k_sizes[2];

    seqlen_q = q_sizes[1];
    seqlen_k = k_sizes[1];
  }

  CHECK(batch_size > 0) << "Batch size must be positive in FlashAttention.";
  // TODO: more loose check for head_size?
  CHECK(head_size_og % 8 == 0)
      << "Head size must be a multiple of 8 in FlashAttention.";
  CHECK(head_size_og <= 256)
      << "FlashAttention forward only supports head dimension at most 256";
  // TODO: num_heads % num_heads_k == 0
  CHECK(num_heads == num_heads_k) << "Number of heads in key/value must be "
                                     "equal to number of heads in query";

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }
  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && !alibi_slopes_buffer.has_value()) {
    is_causal = false;
  }
  if (is_causal) {
    window_size_right = 0;
  }

  // TODO: seqlenq_ngroups_swapped

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  Flash_fwd_params params;
  set_params_fprop(params, config, query_buffer, key_buffer, value_buffer,
                   output_buffer, softmax_lse_buffer, s_dmask_buffer,
                   batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, num_heads, num_heads_k, head_size,
                   head_size_rounded,
                   is_varlen ? cu_seqlens_query_buffer->opaque() : nullptr,
                   is_varlen ? cu_seqlens_key_buffer->opaque() : nullptr,
                   /*seqused_k=*/nullptr, p_dropout, softmax_scale,
                   window_size_left, window_size_right);

  // TODO: seqlenq_ngroups_swapped
  if (!is_varlen) {
    set_params_splitkv(params, batch_size, num_heads, head_size, seqlen_k,
                       seqlen_q, head_size_rounded, p_dropout,
                       /*num_splits*/ 0, output_accum_buffer,
                       softmax_lse_accum_buffer, dprops);
  }

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state We use a custom RNG that increases the offset
  // by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h * 32;
  // Forward kernel will populate memory with the seed and offset.
  params.rng_state = reinterpret_cast<uint64_t *>(rng_state_buffer.opaque());

  if (p_dropout > 0.0) {
    int cur_stream_device = stream->parent()->device_ordinal();
    auto &gen = flash::cuda::getDefaultCUDAGenerator(cur_stream_device);
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex_);
    params.philox_args = gen.philox_cuda_state(counter_offset);
  }

  set_params_alibi(params, alibi_slopes_buffer, config.alibi_slopes_desc,
                   batch_size, num_heads);

  if (seqlen_k > 0) {
    run_mha_fwd(params, se::gpu::AsGpuStreamValue(stream));
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    stream->ThenMemZero(&output_buffer, output_buffer.size());
    static uint32_t inf_pattern = []() {
      float value = std::numeric_limits<float>::infinity();
      uint32_t pattern;
      std::memcpy(&pattern, &value, sizeof(pattern));
      return pattern;
    }();
    stream->ThenMemset32(&softmax_lse_buffer, inf_pattern,
                         softmax_lse_buffer.size());
  }

  return absl::OkStatus();
}

absl::Status RunFlashAttnBwd(
    se::Stream *stream, const FlashAttnBwdConfig &config,
    se::DeviceMemoryBase grad_output_buffer, se::DeviceMemoryBase query_buffer,
    se::DeviceMemoryBase key_buffer, se::DeviceMemoryBase value_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase softmax_lse_buffer,
    se::DeviceMemoryBase rng_state_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_query_buffer,
    std::optional<se::DeviceMemoryBase> cu_seqlens_key_buffer,
    std::optional<se::DeviceMemoryBase> alibi_slopes_buffer,
    se::DeviceMemoryBase grad_query_accum_buffer,
    se::DeviceMemoryBase grad_query_buffer,
    se::DeviceMemoryBase grad_key_buffer,
    se::DeviceMemoryBase grad_value_buffer,
    se::DeviceMemoryBase grad_softmax_buffer, int window_size_left,
    int window_size_right) {
#ifdef FLASHATTENTION_DISABLE_BACKWARD
  CHECK(false) << "This flash attention build does not support backward.";
#endif

  const float p_dropout = config.dropout_rate;
  bool is_dropout = p_dropout > 0.0;
  bool is_causal = config.is_causal;
  if (is_causal) {
    window_size_right = 0;
  }

  const cudaDeviceProp *dprops = flash::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
  bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
  bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  CHECK(is_sm8x || is_sm90)
      << "FlashAttention only supports Ampere GPUs or newer.";

  bool is_varlen = cu_seqlens_query_buffer.has_value();
  CHECK(is_varlen == cu_seqlens_key_buffer.has_value() &&
        is_varlen == config.cu_seqlens_query_desc.has_value() &&
        is_varlen == config.cu_seqlens_key_desc.has_value() &&
        is_varlen == config.max_seqlen_q.has_value() &&
        is_varlen == config.max_seqlen_k.has_value())
      << "cu_seqlens_query_buffer, cu_seqlens_key_buffer, max_seqlen_q, and "
         "max_seqlen_k must be all set or all unset.";

  const auto &query_strides = config.query_desc.GetLogicalStrides();
  const auto &key_strides = config.key_desc.GetLogicalStrides();
  const auto &value_strides = config.value_desc.GetLogicalStrides();
  const auto &output_strides = config.output_desc.GetLogicalStrides();
  const auto &grad_output_strides = config.grad_output_desc.GetLogicalStrides();
  CHECK(query_strides.back() == 1 && key_strides.back() == 1 &&
        value_strides.back() == 1)
      << "Input tensor must have contiguous last dimension in FlashAttention.";
  CHECK(output_strides.back() == 1)
      << "Output tensor must have contiguous last dimension in FlashAttention.";
  CHECK(grad_output_strides.back() == 1)
      << "Gradient output tensor must have contiguous last dimension in "
         "FlashAttention.";

  if (is_varlen) {
    CHECK(config.cu_seqlens_query_desc->ndims() == 1 &&
          config.cu_seqlens_key_desc->ndims() == 1);
  }

  const auto &q_sizes = config.query_desc.dimensions();
  const auto &k_sizes = config.key_desc.dimensions();
  const auto &dout_sizes = config.grad_output_desc.dimensions();

  int batch_size, num_heads, head_size_og, head_size, num_heads_k;
  int seqlen_q, seqlen_k;
  int total_q, total_k;

  if (is_varlen) {
    batch_size = GetNumElements(config.cu_seqlens_query_desc->dimensions()) - 1;
    num_heads = q_sizes[1];
    head_size_og = dout_sizes[2];
    head_size = q_sizes[2];
    num_heads_k = k_sizes[1];

    seqlen_q = config.max_seqlen_q.value();
    seqlen_k = config.max_seqlen_k.value();

    total_q = q_sizes[0];
    total_k = k_sizes[0];
  } else {
    batch_size = q_sizes[0];
    num_heads = q_sizes[2];
    head_size_og = dout_sizes[3];
    head_size = q_sizes[3];
    num_heads_k = k_sizes[2];

    seqlen_q = q_sizes[1];
    seqlen_k = k_sizes[1];
  }

  CHECK(batch_size > 0) << "Batch size must be positive in FlashAttention.";
  CHECK(head_size % 8 == 0)
      << "Head size must be a multiple of 8 in FlashAttention.";
  CHECK(head_size <= 256)
      << "FlashAttention forward only supports head dimension at most 256";
  if (head_size > 192 && (head_size <= 224 || is_dropout)) {
    CHECK(is_sm80 || is_sm90)
        << "FlashAttention backward for head dim 256 with dropout, or head dim "
           "224 with/without dropout requires A100/A800 or H100/H800";
  }
  CHECK(num_heads % num_heads_k == 0)
      << "Number of heads in key/value must divide number of heads in query";

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  CHECK(head_size == round_multiple(head_size_og, 8))
      << "head_size must be head_size_og rounded to a multiple of 8";

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }

  bool deterministic = config.deterministic;

  bool loop = true;

  // at::Tensor dk_expanded, dv_expanded;
  // if (num_heads_k != num_heads) {  // MQA / GQA
  //     dk_expanded = torch::empty({batch_size, seqlen_k, num_heads,
  //     head_size}, opts); dv_expanded = torch::empty({batch_size, seqlen_k,
  //     num_heads, head_size}, opts);
  // } else {
  //     dk_expanded = dk;
  //     dv_expanded = dv;
  // }

  Flash_bwd_params params;
  set_params_dgrad(params, config, grad_output_buffer, query_buffer, key_buffer,
                   value_buffer, output_buffer, softmax_lse_buffer,
                   grad_query_buffer, grad_key_buffer, grad_value_buffer,
                   grad_softmax_buffer, batch_size, seqlen_q, seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded, num_heads, num_heads_k,
                   head_size, head_size_rounded,
                   is_varlen ? cu_seqlens_query_buffer->opaque() : nullptr,
                   is_varlen ? cu_seqlens_key_buffer->opaque() : nullptr,
                   loop ? grad_query_accum_buffer.opaque() : nullptr,
                   /*dk_accum_d=*/nullptr,
                   /*dv_accum_d=*/nullptr, p_dropout, config.scale,
                   window_size_left, window_size_right, deterministic);

  if (deterministic) {
    int64_t hidden_size = num_heads * head_size_rounded;
    if (is_varlen) {
      params.dq_accum_split_stride = (total_q + 128 * batch_size) * hidden_size;
    } else {
      params.dq_accum_split_stride =
          batch_size * seqlen_q_rounded * hidden_size;
    }
  } else {
    params.dq_accum_split_stride = 0;
  }

  int64_t counter_offset = params.b * params.h * 32;

  params.rng_state = reinterpret_cast<uint64_t *>(rng_state_buffer.opaque());

  set_params_alibi(params, alibi_slopes_buffer, config.alibi_slopes_desc,
                   batch_size, num_heads);

  if (seqlen_q > 0) {
    run_mha_bwd(params, se::gpu::AsGpuStreamValue(stream));
  } else {
    // If seqlen_q == 0, then we have an empty tensor. We need to set the output
    // to 0.
    stream->ThenMemZero(&grad_key_buffer, grad_key_buffer.size());
    stream->ThenMemZero(&grad_value_buffer, grad_value_buffer.size());
    stream->ThenMemZero(&grad_softmax_buffer, grad_softmax_buffer.size());
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

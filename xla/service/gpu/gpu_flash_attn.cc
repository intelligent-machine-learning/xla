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

#include "xla/service/gpu/gpu_flash_attn.h"

#include <cmath>
#include <mutex>
#include <limits>

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cutlass/numeric_types.h>

#include "flash_attn/flash_api.h"
#include "xla/service/gpu/gpu_flash_attn_helper_kernel.h"
#include "xla/stream_executor/gpu/gpu_stream.h"

namespace xla {
namespace gpu {

const absl::string_view kGpuFlashAttnForwardCallTarget =
    "__gpu$flash_attn_forward";
const absl::string_view kGpuFlashAttnVarlenForwardCallTarget =
    "__gpu$flash_attn_varlen_forward";
const absl::string_view kGpuFlashAttnBackwardCallTarget =
    "__gpu$flash_attn_backward";
const absl::string_view kGpuFlashAttnVarlenBackwardCallTarget =
    "__gpu$flash_attn_varlen_backward";

bool IsCustomCallToFlashAttnForward(const HloInstruction &hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  return hlo.custom_call_target() == kGpuFlashAttnForwardCallTarget;
}

bool IsCustomCallToFlashAttnVarlenForward(const HloInstruction &hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  return hlo.custom_call_target() == kGpuFlashAttnVarlenForwardCallTarget;
}

bool IsCustomCallToFlashAttnBackward(const HloInstruction &hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  return hlo.custom_call_target() == kGpuFlashAttnBackwardCallTarget;
}

bool IsCustomCallToFlashAttnVarlenBackward(const HloInstruction &hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  return hlo.custom_call_target() == kGpuFlashAttnVarlenBackwardCallTarget;
}

bool IsCustomCallToFlashAttn(const HloInstruction &hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const std::string &target = hlo.custom_call_target();
  return target == kGpuFlashAttnForwardCallTarget ||
         target == kGpuFlashAttnVarlenForwardCallTarget ||
         target == kGpuFlashAttnBackwardCallTarget ||
         target == kGpuFlashAttnVarlenBackwardCallTarget;
}

absl::StatusOr<FlashAttnKind> GetFlashAttnKind(
    const HloCustomCallInstruction *instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kGpuFlashAttnForwardCallTarget) {
    return FlashAttnKind::kForward;
  }
  if (target == kGpuFlashAttnVarlenForwardCallTarget) {
    return FlashAttnKind::kVarlenForward;
  }
  if (target == kGpuFlashAttnBackwardCallTarget) {
    return FlashAttnKind::kBackward;
  }
  if (target == kGpuFlashAttnVarlenBackwardCallTarget) {
    return FlashAttnKind::kVarlenBackward;
  }
  return Internal("Unexpected call target: %s", target);
}

static void set_params_fprop(
    flash_attn::Flash_fwd_params &params, const bool is_bf16,
    // sizes
    const size_t b, const size_t seqlen_q, const size_t seqlen_k,
    const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
    const size_t h, const size_t h_k, const size_t d, const size_t d_rounded,
    // device pointers
    void *q, void *k, void *v, void *out, void *cu_seqlens_q_d,
    void *cu_seqlens_k_d, void *seqused_k, void *p_d, void *softmax_lse_d,
    float p_dropout, float softmax_scale, int window_size_left,
    int window_size_right, bool seqlenq_ngroups_swapped = false) {
  // Reset the parameters
  params = {};

  params.is_bf16 = is_bf16;

  // Set the pointers and strides.
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  // All stride are in elements, not bytes.
  params.q_row_stride = h * d;
  params.k_row_stride = h_k * d;
  params.v_row_stride = h_k * d;
  params.q_head_stride = d;
  params.k_head_stride = d;
  params.v_head_stride = d;
  params.o_ptr = out;
  params.o_row_stride = h * d;
  params.o_head_stride = d;

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = seqlen_q * h * d;
    params.k_batch_stride = seqlen_k * h_k * d;
    params.v_batch_stride = seqlen_k * h_k * d;
    params.o_batch_stride = seqlen_q * h * d;
    if (seqlenq_ngroups_swapped) {
      CHECK(false) << "Not support seqlenq_ngroups_swapped yet";
      params.q_batch_stride *= seqlen_q;
      params.o_batch_stride *= seqlen_q;
    }
  }

  params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int *>(seqused_k);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

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
  TORCH_CHECK(p_dropout < 1.f);
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  CHECK(p_dropout == 0.0f)
      << "This flash attention build does not support dropout.";
#endif

  // Causal is the special case where window_size_right == 0 and
  // window_size_left < 0. Local is the more general case where
  // window_size_right >= 0 or window_size_left >= 0.
  // If cu_seqlens_q_d is not nullptr, which is in the varlen case, we set these
  // fields in the kernel.
  if (cu_seqlens_q_d == nullptr) {
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) {
      window_size_left = seqlen_k;
    }
    if (window_size_left >= 0 && window_size_right < 0) {
      window_size_right = seqlen_k;
    }
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

#ifdef FLASHATTENTION_DISABLE_LOCAL
  CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0))
      << "This flash attention build does not support local attention.";
#endif

  params.is_seqlens_k_cumulative = true;

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  CHECK(d == d_rounded) << "This flash attention build does not support "
                           "headdim not being a multiple of 32.";
#endif
}

static void set_params_dgrad(
    flash_attn::Flash_bwd_params &params, const bool is_bf16,
    // sizes
    const size_t b, const size_t seqlen_q, const size_t seqlen_k,
    const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
    const size_t h, const size_t h_k, const size_t d, const size_t d_rounded,
    // device pointers
    void *q, void *k, void *v, void *out, void *dout, void *dq, void *dk,
    void *dv, void *cu_seqlens_q_d, void *cu_seqlens_k_d, void *dq_accum_d,
    void *dk_accum_d, void *dv_accum_d, void *softmax_lse_d,
    void *dsoftmax_sum_d, float p_dropout, float softmax_scale,
    int window_size_left, int window_size_right, bool deterministic) {
  set_params_fprop(params, is_bf16, b, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, h, h_k, d, d_rounded, q, k, v, out,
                   cu_seqlens_q_d, cu_seqlens_k_d, nullptr, nullptr,
                   softmax_lse_d, p_dropout, softmax_scale, window_size_left,
                   window_size_right);

  // Set the pointers and strides.
  params.do_ptr = dout;
  params.do_row_stride = h * d;
  params.do_head_stride = d;
  params.dq_ptr = dq;
  params.dk_ptr = dk;
  params.dv_ptr = dv;
  params.dq_row_stride = h * d;
  params.dk_row_stride = h * d;  // use dk_expanded's num_heads
  params.dv_row_stride = h * d;  // use dv_expanded's num_heads
  params.dq_head_stride = d;
  params.dk_head_stride = d;
  params.dv_head_stride = d;

  if (cu_seqlens_q_d == nullptr) {
    params.do_batch_stride = seqlen_q * h * d;
    params.dq_batch_stride = seqlen_q * h * d;
    params.dk_batch_stride = seqlen_k * h * d;  // use dk_expanded's num_heads
    params.dv_batch_stride = seqlen_k * h * d;  // use dv_expanded's num_heads
  }

  params.dq_accum_ptr = dq_accum_d;
  params.dk_accum_ptr = dk_accum_d;
  params.dv_accum_ptr = dv_accum_d;

  // Softmax sum
  params.dsoftmax_sum = dsoftmax_sum_d;

  params.deterministic = deterministic;
}

#define RETURN_STATUS_IF_CUDA_ERROR(expr)                          \
  do {                                                             \
    cudaError_t _res = (expr);                                     \
    if (ABSL_PREDICT_FALSE(_res != cudaSuccess)) {                 \
      return absl::InternalError(                                  \
          absl::StrCat("CUDA Error: ", cudaGetErrorString(_res))); \
    }                                                              \
  } while (0)

#define RETURN_STATUS_IF_LAST_CUDA_ERROR() \
  RETURN_STATUS_IF_CUDA_ERROR(cudaGetLastError())

static inline int64_t RoundMultiple(int64_t x, int64_t m) {
  return (x + m - 1) / m * m;
}

absl::Status RunFlashAttnForward(
    se::Stream *stream, const FlashAttnConfig &config,
    se::DeviceMemoryBase query_buffer, se::DeviceMemoryBase key_buffer,
    se::DeviceMemoryBase value_buffer,
    std::optional<se::DeviceMemoryBase> alibi_slopes_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase softmax_lse_buffer,
    se::DeviceMemoryBase rng_state_buffer,
    std::optional<se::DeviceMemoryBase> s_dmask_buffer,
    std::optional<se::DeviceMemoryBase> output_accum_buffer,
    std::optional<se::DeviceMemoryBase> softmax_lse_accum_buffer) {
  cudaStream_t stream_ = se::gpu::AsGpuStreamValue(stream);

  const bool is_bf16 = config.dtype == PrimitiveType::BF16;
  CHECK(is_bf16 || config.dtype == PrimitiveType::F16)
      << "Flash Attention only supports FP16 and BF16";

  const int64_t batch_size = config.batch_size;
  const int64_t seqlen_q = config.seqlen_q;
  const int64_t seqlen_k = config.seqlen_k;
  const int64_t num_heads = config.num_heads;
  const int64_t num_heads_k = config.num_heads_k;
  const int64_t head_size_og = config.head_size_og;
  const int64_t head_size = config.head_size;
  const int64_t head_size_rounded = config.head_size_rounded;
  const int64_t seqlen_q_rounded = RoundMultiple(seqlen_q, 128);
  const int64_t seqlen_k_rounded = RoundMultiple(seqlen_k, 128);

  int window_size_left = config.window_size_left;
  int window_size_right = config.window_size_right;
  bool is_causal = config.causal;

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

  flash_attn::Flash_fwd_params params;
  set_params_fprop(params, is_bf16, batch_size, seqlen_q, seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded, num_heads, num_heads_k,
                   head_size, head_size_rounded, query_buffer.opaque(),
                   key_buffer.opaque(), value_buffer.opaque(),
                   output_buffer.opaque(), /*cu_seqlens_q_d=*/nullptr,
                   /*cu_seqlens_k_d=*/nullptr, /*seqused_k=*/nullptr,
                   config.return_softmax ? s_dmask_buffer->opaque() : nullptr,
                   softmax_lse_buffer.opaque(), config.dropout_p,
                   config.softmax_scale, window_size_left, window_size_right);

  params.num_splits = static_cast<int>(config.num_splits);
  if (params.num_splits > 1) {
    CHECK(output_accum_buffer.has_value() &&
          softmax_lse_accum_buffer.has_value());
    params.oaccum_ptr = output_accum_buffer->opaque();
    params.softmax_lseaccum_ptr = softmax_lse_accum_buffer->opaque();
  } else {
    CHECK(!output_accum_buffer.has_value() &&
          !softmax_lse_accum_buffer.has_value());
  }

  params.rng_state = reinterpret_cast<uint64_t *>(rng_state_buffer.opaque());

  if (config.dropout_p > 0.0) {
    int device_ordinal = stream->parent()->device_ordinal();
    int64_t counter_offset = params.b * params.h * 32;
    auto gen = at::check_generator<at::CUDAGeneratorImpl>(
        at::cuda::detail::getDefaultCUDAGenerator(device_ordinal));
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
  }

  if (alibi_slopes_buffer.has_value()) {
    params.alibi_slopes_ptr = alibi_slopes_buffer->opaque();
    params.alibi_slopes_batch_stride = config.alibi_slopes_batch_stride;
  } else {
    params.alibi_slopes_ptr = nullptr;
  }

  if (seqlen_k > 0) {
    flash_attn::run_mha_fwd(params, stream_);
    RETURN_STATUS_IF_LAST_CUDA_ERROR();
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    TF_RETURN_IF_ERROR(stream->MemZero(&output_buffer, output_buffer.size()));
    static uint32_t inf_pattern = []() {
      float value = std::numeric_limits<float>::infinity();
      uint32_t pattern;
      std::memcpy(&pattern, &value, sizeof(pattern));
      return pattern;
    }();
    TF_RETURN_IF_ERROR(stream->Memset32(&softmax_lse_buffer, inf_pattern,
                                        softmax_lse_buffer.size()));
  }

  return absl::OkStatus();
}

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
    std::optional<se::DeviceMemoryBase> max_seqlen_q_buffer) {
  cudaStream_t stream_ = se::gpu::AsGpuStreamValue(stream);

  const bool is_bf16 = config.dtype == PrimitiveType::BF16;
  CHECK(is_bf16 || config.dtype == PrimitiveType::F16)
      << "Flash Attention only supports FP16 and BF16";

  const int64_t batch_size = config.batch_size;
  const int64_t seqlen_q = config.seqlen_q;
  const int64_t seqlen_k = config.seqlen_k;
  const int64_t num_heads = config.num_heads;
  const int64_t num_heads_k = config.num_heads_k;
  const int64_t head_size_og = config.head_size_og;
  const int64_t head_size = config.head_size;
  const int64_t head_size_rounded = config.head_size_rounded;

  const int64_t max_seqlen_q = seqlen_q;
  const int64_t max_seqlen_k = seqlen_k;
  const int64_t seqlen_q_rounded = RoundMultiple(max_seqlen_q, 128);
  const int64_t seqlen_k_rounded = RoundMultiple(max_seqlen_k, 128);

  const int window_size_left = config.window_size_left;
  int window_size_right = config.window_size_right;
  if (config.causal) {
    window_size_right = 0;
  }

  // Use output_buffer as temporary storage in cub library.
  void *temp_storage_ptr = output_buffer.opaque();
  size_t temp_storage_bytes = output_buffer.size();

  RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnRowSum(
      stream_, static_cast<const bool *>(key_padding_mask_buffer.opaque()),
      static_cast<int *>(batch_seqlens_k_buffer.opaque()),
      static_cast<int>(batch_size), static_cast<int>(seqlen_k),
      temp_storage_ptr, temp_storage_bytes));

  RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnCumSum(
      stream_, static_cast<const int *>(batch_seqlens_k_buffer.opaque()),
      static_cast<int *>(cu_seqlens_k_buffer.opaque()),
      static_cast<int>(batch_size + 1), temp_storage_ptr, temp_storage_bytes));

  RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnMax(
      stream_, static_cast<const int *>(batch_seqlens_k_buffer.opaque()),
      static_cast<int *>(max_seqlen_k_buffer.opaque()),
      static_cast<int>(batch_size), temp_storage_ptr, temp_storage_bytes));

  RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnNonzero(
      stream_, static_cast<const bool *>(key_padding_mask_buffer.opaque()),
      static_cast<int *>(indices_k_buffer.opaque()),
      static_cast<int *>(indices_len_k_buffer.opaque()),
      static_cast<int>(batch_size * seqlen_k), temp_storage_ptr,
      temp_storage_bytes));

  const bool has_query_padding_mask = config.has_query_padding_mask;
  if (has_query_padding_mask) {
    CHECK(query_padding_mask_buffer.has_value() &&
          indices_q_buffer.has_value() && indices_len_q_buffer.has_value() &&
          batch_seqlens_q_buffer.has_value() &&
          cu_seqlens_q_buffer.has_value() && max_seqlen_q_buffer.has_value());
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnRowSum(
        stream_, static_cast<const bool *>(query_padding_mask_buffer->opaque()),
        static_cast<int *>(batch_seqlens_q_buffer->opaque()),
        static_cast<int>(batch_size), static_cast<int>(seqlen_q),
        temp_storage_ptr, temp_storage_bytes));

    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnCumSum(
        stream_, static_cast<const int *>(batch_seqlens_q_buffer->opaque()),
        static_cast<int *>(cu_seqlens_q_buffer->opaque()),
        static_cast<int>(batch_size + 1), temp_storage_ptr,
        temp_storage_bytes));

    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnMax(
        stream_, static_cast<const int *>(batch_seqlens_q_buffer->opaque()),
        static_cast<int *>(max_seqlen_q_buffer->opaque()),
        static_cast<int>(batch_size), temp_storage_ptr, temp_storage_bytes));

    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnNonzero(
        stream_, static_cast<const bool *>(query_padding_mask_buffer->opaque()),
        static_cast<int *>(indices_q_buffer->opaque()),
        static_cast<int *>(indices_len_q_buffer->opaque()),
        static_cast<int>(batch_size * seqlen_q), temp_storage_ptr,
        temp_storage_bytes));
  }

  void *indices_k_d = indices_k_buffer.opaque();
  void *indices_q_d =
      has_query_padding_mask ? indices_q_buffer->opaque() : indices_k_d;
  void *indices_len_k_d = indices_len_k_buffer.opaque();
  void *indices_len_q_d =
      has_query_padding_mask ? indices_len_q_buffer->opaque() : indices_len_k_d;

  if (!has_query_padding_mask && num_heads == num_heads_k) {
    // Gather/Scatter function only copies data, ignores f16/bf16 type (no
    // computation)
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnGather3(
        stream_, static_cast<const cutlass::half_t *>(query_buffer.opaque()),
        static_cast<cutlass::half_t *>(query_unpad_buffer.opaque()),
        static_cast<const cutlass::half_t *>(key_buffer.opaque()),
        static_cast<cutlass::half_t *>(key_unpad_buffer.opaque()),
        static_cast<const cutlass::half_t *>(value_buffer.opaque()),
        static_cast<cutlass::half_t *>(value_unpad_buffer.opaque()),
        static_cast<const int *>(indices_k_d),
        static_cast<const int *>(indices_len_k_d),
        static_cast<int>(batch_size * seqlen_k),
        static_cast<int>(num_heads * head_size_og)));
  } else {
    // Gather/Scatter function only copies data, ignores f16/bf16 type (no
    // computation)
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnGather(
        stream_, static_cast<const cutlass::half_t *>(query_buffer.opaque()),
        static_cast<cutlass::half_t *>(query_unpad_buffer.opaque()),
        static_cast<const int *>(indices_q_d),
        static_cast<const int *>(indices_len_q_d),
        static_cast<int>(batch_size * seqlen_q),
        static_cast<int>(num_heads * head_size_og)));
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnGather(
        stream_, static_cast<const cutlass::half_t *>(key_buffer.opaque()),
        static_cast<cutlass::half_t *>(key_unpad_buffer.opaque()),
        static_cast<const int *>(indices_k_d),
        static_cast<const int *>(indices_len_k_d),
        static_cast<int>(batch_size * seqlen_k),
        static_cast<int>(num_heads_k * head_size_og)));
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnGather(
        stream_, static_cast<const cutlass::half_t *>(value_buffer.opaque()),
        static_cast<cutlass::half_t *>(value_unpad_buffer.opaque()),
        static_cast<const int *>(indices_k_d),
        static_cast<const int *>(indices_len_k_d),
        static_cast<int>(batch_size * seqlen_k),
        static_cast<int>(num_heads_k * head_size_og)));
  }

  void *cu_seqlens_k_d = cu_seqlens_k_buffer.opaque();
  void *cu_seqlens_q_d =
      has_query_padding_mask ? cu_seqlens_q_buffer->opaque() : cu_seqlens_k_d;
  void *max_seqlen_k_d = max_seqlen_k_buffer.opaque();
  void *max_seqlen_q_d =
      has_query_padding_mask ? max_seqlen_q_buffer->opaque() : max_seqlen_k_d;

  flash_attn::Flash_fwd_params params;
  set_params_fprop(params, is_bf16, batch_size, max_seqlen_q, max_seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded, num_heads, num_heads_k,
                   head_size, head_size_rounded, query_unpad_buffer.opaque(),
                   key_unpad_buffer.opaque(), value_unpad_buffer.opaque(),
                   output_unpad_buffer.opaque(), cu_seqlens_q_d, cu_seqlens_k_d,
                   nullptr /*=seqused_k*/,
                   config.return_softmax ? s_dmask_buffer->opaque() : nullptr,
                   softmax_lse_buffer.opaque(), config.dropout_p,
                   config.softmax_scale, window_size_left, window_size_right);
  params.seqlen_q_ptr = static_cast<int *>(max_seqlen_q_d);
  params.seqlen_k_ptr = static_cast<int *>(max_seqlen_k_d);
  params.page_block_size = 1;
  params.rng_state = reinterpret_cast<uint64_t *>(rng_state_buffer.opaque());

  if (config.dropout_p > 0.0) {
    int device_ordinal = stream->parent()->device_ordinal();
    int64_t counter_offset = params.b * params.h * 32;
    auto gen = at::check_generator<at::CUDAGeneratorImpl>(
        at::cuda::detail::getDefaultCUDAGenerator(device_ordinal));
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
  }

  if (alibi_slopes_buffer.has_value()) {
    params.alibi_slopes_ptr = alibi_slopes_buffer->opaque();
    params.alibi_slopes_batch_stride = config.alibi_slopes_batch_stride;
  } else {
    params.alibi_slopes_ptr = nullptr;
  }

  flash_attn::run_mha_fwd(params, stream_);
  RETURN_STATUS_IF_LAST_CUDA_ERROR();

  // Gather/Scatter function only copies data, ignores f16/bf16 type (no
  // computation)
  RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnScatter(
      stream_,
      static_cast<const cutlass::half_t *>(output_unpad_buffer.opaque()),
      static_cast<cutlass::half_t *>(output_buffer.opaque()),
      static_cast<const int *>(indices_q_d),
      static_cast<const int *>(indices_len_q_d),
      static_cast<int>(batch_size * seqlen_q),
      static_cast<int>(num_heads * head_size_og)));

  return absl::OkStatus();
}

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
    std::optional<se::DeviceMemoryBase> grad_value_expanded_buffer) {
  cudaStream_t stream_ = se::gpu::AsGpuStreamValue(stream);

  const bool is_bf16 = config.dtype == PrimitiveType::BF16;
  CHECK(is_bf16 || config.dtype == PrimitiveType::F16)
      << "Flash Attention only supports FP16 and BF16";

  const int64_t batch_size = config.batch_size;
  const int64_t seqlen_q = config.seqlen_q;
  const int64_t seqlen_k = config.seqlen_k;
  const int64_t num_heads = config.num_heads;
  const int64_t num_heads_k = config.num_heads_k;
  const int64_t head_size_og = config.head_size_og;
  const int64_t head_size = config.head_size;
  const int64_t head_size_rounded = config.head_size_rounded;
  const int64_t seqlen_q_rounded = RoundMultiple(seqlen_q, 128);
  const int64_t seqlen_k_rounded = RoundMultiple(seqlen_k, 128);

  int window_size_left = config.window_size_left;
  int window_size_right = config.window_size_right;
  if (config.causal) {
    window_size_right = 0;
  }

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }

  se::DeviceMemoryBase dk_expanded_buffer, dv_expanded_buffer;
  if (num_heads != num_heads_k) {  // MQA / GQA
    CHECK(grad_key_expanded_buffer.has_value() &&
          grad_value_expanded_buffer.has_value());
    dk_expanded_buffer = grad_key_expanded_buffer.value();
    dv_expanded_buffer = grad_value_expanded_buffer.value();
  } else {
    dk_expanded_buffer = grad_key_buffer;
    dv_expanded_buffer = grad_value_buffer;
  }

  flash_attn::Flash_bwd_params params;
  set_params_dgrad(params, is_bf16, batch_size, seqlen_q, seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded, num_heads, num_heads_k,
                   head_size, head_size_rounded, query_buffer.opaque(),
                   key_buffer.opaque(), value_buffer.opaque(),
                   output_buffer.opaque(), grad_output_buffer.opaque(),
                   grad_query_buffer.opaque(), dk_expanded_buffer.opaque(),
                   dv_expanded_buffer.opaque(), nullptr, nullptr,
                   grad_query_accum_buffer.opaque(), nullptr, /*=dk_accum_d*/
                   nullptr,                                   /*=dv_accum_d*/
                   softmax_lse_buffer.opaque(), grad_softmax_buffer.opaque(),
                   config.dropout_p, config.softmax_scale, window_size_left,
                   window_size_right, config.deterministic);
  params.dq_accum_split_stride =
      !config.deterministic ? 0 : config.dq_accum_split_stride;
  params.rng_state = reinterpret_cast<uint64_t *>(rng_state_buffer.opaque());
  if (alibi_slopes_buffer.has_value()) {
    params.alibi_slopes_ptr = alibi_slopes_buffer->opaque();
    params.alibi_slopes_batch_stride = config.alibi_slopes_batch_stride;
  } else {
    params.alibi_slopes_ptr = nullptr;
  }

  if (config.deterministic) {
    TF_RETURN_IF_ERROR(stream->MemZero(&grad_query_accum_buffer,
                                       grad_query_accum_buffer.size()));
  }

  if (seqlen_q > 0) {
    flash_attn::run_mha_bwd(params, stream_);
    RETURN_STATUS_IF_LAST_CUDA_ERROR();
  } else {
    TF_RETURN_IF_ERROR(
        stream->MemZero(&dk_expanded_buffer, dk_expanded_buffer.size()));
    TF_RETURN_IF_ERROR(
        stream->MemZero(&dk_expanded_buffer, dk_expanded_buffer.size()));
    TF_RETURN_IF_ERROR(
        stream->MemZero(&grad_softmax_buffer, grad_softmax_buffer.size()));
  }

  // For MQA/GQA we need to sum dK and dV across the groups
  if (num_heads_k != num_heads) {
    if (is_bf16) {
      RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnMqaGqaSum(
          stream_,
          static_cast<const cutlass::bfloat16_t *>(dk_expanded_buffer.opaque()),
          static_cast<cutlass::bfloat16_t *>(grad_key_buffer.opaque()),
          static_cast<const cutlass::bfloat16_t *>(dv_expanded_buffer.opaque()),
          static_cast<cutlass::bfloat16_t *>(grad_value_buffer.opaque()),
          static_cast<const int *>(nullptr), /*nnz*/
          static_cast<int>(batch_size * seqlen_k),
          static_cast<int>(num_heads_k),
          static_cast<int>(num_heads / num_heads_k),
          static_cast<int>(head_size)));
    } else {
      RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnMqaGqaSum(
          stream_,
          static_cast<const cutlass::half_t *>(dk_expanded_buffer.opaque()),
          static_cast<cutlass::half_t *>(grad_key_buffer.opaque()),
          static_cast<const cutlass::half_t *>(dv_expanded_buffer.opaque()),
          static_cast<cutlass::half_t *>(grad_value_buffer.opaque()),
          static_cast<const int *>(nullptr), /*nnz*/
          static_cast<int>(batch_size * seqlen_k),
          static_cast<int>(num_heads_k),
          static_cast<int>(num_heads / num_heads_k),
          static_cast<int>(head_size)));
    }
  }
  return absl::OkStatus();
}

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
    std::optional<se::DeviceMemoryBase> grad_value_expanded_buffer) {
  cudaStream_t stream_ = se::gpu::AsGpuStreamValue(stream);

  const bool is_bf16 = config.dtype == PrimitiveType::BF16;
  CHECK(is_bf16 || config.dtype == PrimitiveType::F16)
      << "Flash Attention only supports FP16 and BF16";

  const int64_t batch_size = config.batch_size;
  const int64_t seqlen_q = config.seqlen_q;
  const int64_t seqlen_k = config.seqlen_k;
  const int64_t num_heads = config.num_heads;
  const int64_t num_heads_k = config.num_heads_k;
  const int64_t head_size_og = config.head_size_og;
  const int64_t head_size = config.head_size;
  const int64_t head_size_rounded = config.head_size_rounded;

  const int64_t max_seqlen_q = seqlen_q;
  const int64_t max_seqlen_k = seqlen_k;
  const int64_t seqlen_q_rounded = RoundMultiple(max_seqlen_q, 128);
  const int64_t seqlen_k_rounded = RoundMultiple(max_seqlen_k, 128);

  const int window_size_left = config.window_size_left;
  int window_size_right = config.window_size_right;
  if (config.causal) {
    window_size_right = 0;
  }

  // Use grad_softmax_buffer as temporary storage in cub library.
  void *temp_storage_ptr = grad_softmax_buffer.opaque();
  size_t temp_storage_bytes = grad_softmax_buffer.size();

  RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnCumSum(
      stream_, static_cast<const int *>(batch_seqlens_k_buffer.opaque()),
      static_cast<int *>(cu_seqlens_k_buffer.opaque()),
      static_cast<int>(batch_size + 1), temp_storage_ptr, temp_storage_bytes));

  RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnMax(
      stream_, static_cast<const int *>(batch_seqlens_k_buffer.opaque()),
      static_cast<int *>(max_seqlen_k_buffer.opaque()),
      static_cast<int>(batch_size), temp_storage_ptr, temp_storage_bytes));

  const bool has_query_padding_mask = config.has_query_padding_mask;
  if (has_query_padding_mask) {
    CHECK(cu_seqlens_q_buffer.has_value() && max_seqlen_q_buffer.has_value());
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnCumSum(
        stream_, static_cast<const int *>(batch_seqlens_q_buffer->opaque()),
        static_cast<int *>(cu_seqlens_q_buffer->opaque()),
        static_cast<int>(batch_size + 1), temp_storage_ptr,
        temp_storage_bytes));

    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnMax(
        stream_, static_cast<const int *>(batch_seqlens_q_buffer->opaque()),
        static_cast<int *>(max_seqlen_q_buffer->opaque()),
        static_cast<int>(batch_size), temp_storage_ptr, temp_storage_bytes));
  }

  void *indices_k_d = indices_k_buffer.opaque();
  void *indices_q_d =
      has_query_padding_mask ? indices_q_buffer->opaque() : indices_k_d;
  void *indices_len_k_d = indices_len_k_buffer.opaque();
  void *indices_len_q_d =
      has_query_padding_mask ? indices_len_q_buffer->opaque() : indices_len_k_d;

  // Gather function only copies data, ignores f16/bf16 type (no computation)
  RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnGather(
      stream_,
      static_cast<const cutlass::half_t *>(grad_output_buffer.opaque()),
      static_cast<cutlass::half_t *>(grad_output_unpad_buffer.opaque()),
      static_cast<const int *>(indices_q_d),
      static_cast<const int *>(indices_len_q_d),
      static_cast<int>(batch_size * seqlen_q),
      static_cast<int>(num_heads * head_size_og)));

  se::DeviceMemoryBase dk_expanded_buffer, dv_expanded_buffer;
  if (num_heads != num_heads_k) {  // MQA / GQA
    CHECK(grad_key_expanded_buffer.has_value() &&
          grad_value_expanded_buffer.has_value());
    dk_expanded_buffer = grad_key_expanded_buffer.value();
    dv_expanded_buffer = grad_value_expanded_buffer.value();
  } else {
    dk_expanded_buffer = grad_key_unpad_buffer;
    dv_expanded_buffer = grad_value_unpad_buffer;
  }

  void *cu_seqlens_k_d = cu_seqlens_k_buffer.opaque();
  void *cu_seqlens_q_d =
      has_query_padding_mask ? cu_seqlens_q_buffer->opaque() : cu_seqlens_k_d;
  void *max_seqlen_k_d = max_seqlen_k_buffer.opaque();
  void *max_seqlen_q_d =
      has_query_padding_mask ? max_seqlen_q_buffer->opaque() : max_seqlen_k_d;

  flash_attn::Flash_bwd_params params;
  set_params_dgrad(
      params, is_bf16, batch_size, max_seqlen_q, max_seqlen_k, seqlen_q_rounded,
      seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded,
      query_unpad_buffer.opaque(), key_unpad_buffer.opaque(),
      value_unpad_buffer.opaque(), output_unpad_buffer.opaque(),
      grad_output_unpad_buffer.opaque(), grad_query_unpad_buffer.opaque(),
      dk_expanded_buffer.opaque(), dv_expanded_buffer.opaque(), cu_seqlens_q_d,
      cu_seqlens_k_d, grad_query_accum_buffer.opaque(), nullptr, /*=dk_accum_d*/
      nullptr,                                                   /*=dv_accum_d*/
      softmax_lse_buffer.opaque(), grad_softmax_buffer.opaque(),
      config.dropout_p, config.softmax_scale, window_size_left,
      window_size_right, config.deterministic);
  params.seqlen_q_ptr = static_cast<int *>(max_seqlen_q_d);
  params.seqlen_k_ptr = static_cast<int *>(max_seqlen_k_d);
  params.total_q_ptr = static_cast<int *>(indices_len_q_d);
  // Set it in the kernel
  params.dq_accum_split_stride = 0;
  params.rng_state = reinterpret_cast<uint64_t *>(rng_state_buffer.opaque());
  if (alibi_slopes_buffer.has_value()) {
    params.alibi_slopes_ptr = alibi_slopes_buffer->opaque();
    params.alibi_slopes_batch_stride = config.alibi_slopes_batch_stride;
  } else {
    params.alibi_slopes_ptr = nullptr;
  }

  if (config.deterministic) {
    TF_RETURN_IF_ERROR(stream->MemZero(&grad_query_accum_buffer,
                                       grad_query_accum_buffer.size()));
  }

  flash_attn::run_mha_bwd(params, stream_);
  RETURN_STATUS_IF_LAST_CUDA_ERROR();

  // For MQA/GQA we need to sum dK and dV across the groups
  if (num_heads_k != num_heads) {
    if (is_bf16) {
      RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnMqaGqaSum(
          stream_,
          static_cast<const cutlass::bfloat16_t *>(dk_expanded_buffer.opaque()),
          static_cast<cutlass::bfloat16_t *>(grad_key_unpad_buffer.opaque()),
          static_cast<const cutlass::bfloat16_t *>(dv_expanded_buffer.opaque()),
          static_cast<cutlass::bfloat16_t *>(grad_value_unpad_buffer.opaque()),
          static_cast<const int *>(indices_len_k_d),
          static_cast<int>(batch_size * seqlen_k),
          static_cast<int>(num_heads_k),
          static_cast<int>(num_heads / num_heads_k),
          static_cast<int>(head_size)));
    } else {
      RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnMqaGqaSum(
          stream_,
          static_cast<const cutlass::half_t *>(dk_expanded_buffer.opaque()),
          static_cast<cutlass::half_t *>(grad_key_unpad_buffer.opaque()),
          static_cast<const cutlass::half_t *>(dv_expanded_buffer.opaque()),
          static_cast<cutlass::half_t *>(grad_value_unpad_buffer.opaque()),
          static_cast<const int *>(indices_len_k_d),
          static_cast<int>(batch_size * seqlen_k),
          static_cast<int>(num_heads_k),
          static_cast<int>(num_heads / num_heads_k),
          static_cast<int>(head_size)));
    }
  }

  if (!has_query_padding_mask && num_heads == num_heads_k) {
    // Gather/Scatter function only copies data, ignores f16/bf16 type (no
    // computation)
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnScatter3(
        stream_,
        static_cast<const cutlass::half_t *>(grad_query_unpad_buffer.opaque()),
        static_cast<cutlass::half_t *>(grad_query_buffer.opaque()),
        static_cast<const cutlass::half_t *>(grad_key_unpad_buffer.opaque()),
        static_cast<cutlass::half_t *>(grad_key_buffer.opaque()),
        static_cast<const cutlass::half_t *>(grad_value_unpad_buffer.opaque()),
        static_cast<cutlass::half_t *>(grad_value_buffer.opaque()),
        static_cast<const int *>(indices_k_d),
        static_cast<const int *>(indices_len_k_d),
        static_cast<int>(batch_size * seqlen_k),
        static_cast<int>(num_heads * head_size_og)));
  } else {
    // Gather/Scatter function only copies data, ignores f16/bf16 type (no
    // computation)
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnScatter(
        stream_,
        static_cast<const cutlass::half_t *>(grad_query_unpad_buffer.opaque()),
        static_cast<cutlass::half_t *>(grad_query_buffer.opaque()),
        static_cast<const int *>(indices_q_d),
        static_cast<const int *>(indices_len_q_d),
        static_cast<int>(batch_size * seqlen_q),
        static_cast<int>(num_heads * head_size_og)));
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnScatter(
        stream_,
        static_cast<const cutlass::half_t *>(grad_key_unpad_buffer.opaque()),
        static_cast<cutlass::half_t *>(grad_key_buffer.opaque()),
        static_cast<const int *>(indices_k_d),
        static_cast<const int *>(indices_len_k_d),
        static_cast<int>(batch_size * seqlen_k),
        static_cast<int>(num_heads_k * head_size_og)));
    RETURN_STATUS_IF_CUDA_ERROR(RunFlashAttnScatter(
        stream_,
        static_cast<const cutlass::half_t *>(grad_value_unpad_buffer.opaque()),
        static_cast<cutlass::half_t *>(grad_value_buffer.opaque()),
        static_cast<const int *>(indices_k_d),
        static_cast<const int *>(indices_len_k_d),
        static_cast<int>(batch_size * seqlen_k),
        static_cast<int>(num_heads_k * head_size_og)));
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

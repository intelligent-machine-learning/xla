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

#include "xla/service/gpu/gpu_flash_attn_rewriter.h"

#include <vector>

#include <ATen/cuda/CUDAContextLight.h>

#include "flash_attn/flash_api.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_flash_attn.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

StatusOr<bool> GpuFlashAttnRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kCustomCall) {
        continue;
      }
      auto kind_status =
          GetFlashAttnKind(Cast<HloCustomCallInstruction>(instr));
      if (!kind_status.ok()) {
        continue;
      }
      FlashAttnKind kind = kind_status.value();
      bool attn_changed = false;
      switch (kind) {
        case FlashAttnKind::kForward: {
          TF_ASSIGN_OR_RETURN(
              attn_changed,
              RunOnFlashAttnForward(computation, instr, /*is_varlen=*/false));
          break;
        }
        case FlashAttnKind::kVarlenForward: {
          TF_ASSIGN_OR_RETURN(
              attn_changed,
              RunOnFlashAttnForward(computation, instr, /*is_varlen=*/true));
          break;
        }
        case FlashAttnKind::kBackward: {
          TF_ASSIGN_OR_RETURN(
              attn_changed,
              RunOnFlashAttnBackward(computation, instr, /*is_varlen=*/false));
          break;
        }
        case FlashAttnKind::kVarlenBackward: {
          TF_ASSIGN_OR_RETURN(
              attn_changed,
              RunOnFlashAttnBackward(computation, instr, /*is_varlen=*/true));
          break;
        }
      }

      changed |= attn_changed;
    }
  }

  return changed;
}

static inline int64_t RoundMultiple(int64_t num, int64_t alignment) {
  return (num + alignment - 1) & (~(alignment - 1));
}

absl::StatusOr<bool> GpuFlashAttnRewriter::RunOnFlashAttnForward(
    HloComputation* computation, HloInstruction* instr, bool is_varlen) {
  bool changed = false;

  const HloInstruction* query = instr->operand(0);
  const HloInstruction* key = instr->operand(1);
  const Shape& q_shape = query->shape();
  const Shape& k_shape = key->shape();
  const int64_t batch_size = q_shape.dimensions(0);
  const int64_t seqlen_q = q_shape.dimensions(1);
  const int64_t seqlen_k = k_shape.dimensions(1);

  std::vector<Shape> return_shapes = instr->shape().tuple_shapes();

  if (is_varlen) {
    TF_ASSIGN_OR_RETURN(const auto gpu_config,
                        instr->backend_config<xla::gpu::GpuBackendConfig>());
    const auto& config = gpu_config.flash_attn_backend_config();
    const Shape& cu_seqlens_shape =
        ShapeUtil::MakeShape(PrimitiveType::S32, {batch_size + 1});
    const Shape& max_seqlen_shape =
        ShapeUtil::MakeScalarShape(PrimitiveType::S32);
    // cu_seqlens_k
    return_shapes.push_back(cu_seqlens_shape);
    // max_seqlen_k
    return_shapes.push_back(max_seqlen_shape);
    CHECK(config.has_has_query_padding_mask());
    if (config.has_query_padding_mask()) {
      // cu_seqlens_q
      return_shapes.push_back(cu_seqlens_shape);
      // max_seqlen_q
      return_shapes.push_back(max_seqlen_shape);
    }
    changed = true;
  } else {
    TF_ASSIGN_OR_RETURN(const auto gpu_config,
                        instr->backend_config<xla::gpu::GpuBackendConfig>());
    const auto& config = gpu_config.flash_attn_backend_config();
    if (config.dropout_p() == 0.0f) {
      const int64_t num_heads = q_shape.dimensions(2);
      const int64_t head_size_og = q_shape.dimensions(3);

      const int64_t head_size = RoundMultiple(head_size_og, 8);
      const int64_t head_size_rounded = RoundMultiple(head_size, 32);

      const int64_t max_seqlen_q = seqlen_q;
      const int64_t max_seqlen_k = seqlen_k;

      // https://github.com/Dao-AILab/flash-attention/blob/v2.5.9.post1/csrc/flash_attn/flash_api.cpp#L276-L281
      const int64_t block_n =
          head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
      const int64_t num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
      const int64_t num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
      const cudaDeviceProp* dprops = at::cuda::getCurrentDeviceProperties();
      int64_t num_splits = 0;  // always 0 for non-kvcache
      if (num_splits < 1) {
        num_splits = flash_attn::num_splits_heuristic(
            batch_size * num_heads * num_m_blocks,
            dprops->multiProcessorCount * 2, num_n_blocks, 128);
      }
      CHECK(num_splits <= 128) << "num_splits > 128 not supported";

      if (num_splits > 1) {
        const Shape& output_accum_shape = ShapeUtil::MakeShape(
            PrimitiveType::F32, {num_splits, batch_size, num_heads,
                                 max_seqlen_q, head_size_rounded});
        const Shape& softmax_lse_accum_shape = ShapeUtil::MakeShape(
            PrimitiveType::F32,
            {num_splits, batch_size, num_heads, max_seqlen_q});
        return_shapes.push_back(output_accum_shape);
        return_shapes.push_back(softmax_lse_accum_shape);
        changed = true;
      }

      FrontendAttributes attrs;
      (*attrs.mutable_map())["num_splits"] = std::to_string(num_splits);
      instr->add_frontend_attributes(std::move(attrs));
    }
  }

  if (changed) {
    HloInstruction* new_instr = instr->AddInstruction(
        instr->CloneWithNewShape(ShapeUtil::MakeTupleShape(return_shapes)));
    TF_RETURN_IF_ERROR(
        computation->ReplaceInstructionWithDifferentShape(instr, new_instr));
  }

  return changed;
}

absl::StatusOr<bool> GpuFlashAttnRewriter::RunOnFlashAttnBackward(
    HloComputation* computation, HloInstruction* instr, bool is_varlen) {
  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const auto& config = gpu_config.flash_attn_backend_config();

  const bool deterministic = config.deterministic();

  const HloInstruction* query = instr->operand(1);
  const HloInstruction* key = instr->operand(2);
  const Shape& q_shape = query->shape();
  const Shape& k_shape = key->shape();

  PrimitiveType dtype = q_shape.element_type();

  const int64_t batch_size = q_shape.dimensions(0);
  const int64_t seqlen_q = q_shape.dimensions(1);
  const int64_t seqlen_k = k_shape.dimensions(1);
  const int64_t num_heads = q_shape.dimensions(2);
  const int64_t num_heads_k = k_shape.dimensions(2);
  const int64_t head_size_og = q_shape.dimensions(3);
  const int64_t head_size = RoundMultiple(head_size_og, 8);
  const int64_t head_size_rounded = RoundMultiple(head_size, 32);
  const int64_t seqlen_q_rounded = RoundMultiple(seqlen_q, 128);

  std::vector<Shape> return_shapes = instr->shape().tuple_shapes();

  if (is_varlen) {
    TF_ASSIGN_OR_RETURN(const auto gpu_config,
                        instr->backend_config<xla::gpu::GpuBackendConfig>());
    const auto& config = gpu_config.flash_attn_backend_config();
    const Shape& max_seqlen_shape =
        ShapeUtil::MakeScalarShape(PrimitiveType::S32);
    const Shape& cu_seqlens_shape =
        ShapeUtil::MakeShape(PrimitiveType::S32, {batch_size + 1});
    // cu_seqlens_k
    return_shapes.push_back(max_seqlen_shape);
    // max_seqlen_k
    return_shapes.push_back(cu_seqlens_shape);
    CHECK(config.has_has_query_padding_mask());
    if (config.has_query_padding_mask()) {
      // cu_seqlens_q
      return_shapes.push_back(cu_seqlens_shape);
      // max_seqlen_q
      return_shapes.push_back(max_seqlen_shape);
    }

    int64_t q_sizes[] = {batch_size, seqlen_q, num_heads, head_size_og};
    int64_t k_sizes[] = {batch_size, seqlen_k, num_heads_k, head_size_og};

    // dq_unpad
    return_shapes.push_back(ShapeUtil::MakeShape(dtype, q_sizes));
    // dk_unpad
    return_shapes.push_back(ShapeUtil::MakeShape(dtype, k_sizes));
    // dv_unpad
    return_shapes.push_back(ShapeUtil::MakeShape(dtype, k_sizes));
    // do_unpad
    return_shapes.push_back(ShapeUtil::MakeShape(dtype, q_sizes));
  }

  std::vector<int64_t> dq_accum_sizes;
  if (!deterministic) {
    if (is_varlen) {
      dq_accum_sizes = {
          batch_size * seqlen_q + 128 * batch_size,
          num_heads,
          head_size_rounded,
      };
    } else {
      dq_accum_sizes = {
          batch_size,
          seqlen_q_rounded,
          num_heads,
          head_size_rounded,
      };
    }
  } else {
    const cudaDeviceProp* dprops = at::cuda::getCurrentDeviceProperties();
    const int nsplits =
        (dprops->multiProcessorCount + batch_size * num_heads - 1) /
        (batch_size * num_heads);
    if (is_varlen) {
      dq_accum_sizes = {
          nsplits,
          batch_size * seqlen_q + 128 * batch_size,
          num_heads,
          head_size_rounded,
      };
    } else {
      dq_accum_sizes = {
          nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded,
      };
    }
  }
  const Shape& dq_accum_shape =
      ShapeUtil::MakeShape(PrimitiveType::F32, dq_accum_sizes);
  return_shapes.push_back(dq_accum_shape);

  if (num_heads != num_heads_k) {
    const Shape& dkv_expanded_shape = ShapeUtil::MakeShape(
        dtype, {batch_size, seqlen_k, num_heads, head_size});
    // dk_expanded
    return_shapes.push_back(dkv_expanded_shape);
    // dv_expanded
    return_shapes.push_back(dkv_expanded_shape);
  }

  HloInstruction* new_instr = instr->AddInstruction(
      instr->CloneWithNewShape(ShapeUtil::MakeTupleShape(return_shapes)));
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstructionWithDifferentShape(instr, new_instr));

  return true;
}

}  // namespace gpu
}  // namespace xla

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

#include "xla/service/gpu/gpu_flash_attn_normalization.h"

#include <algorithm>
#include <vector>

#include "flash_attn/utils.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_flash_attn.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

StatusOr<bool> GpuFlashAttnNormalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  VLOG(2) << "Before flash attention normalization:";
  XLA_VLOG_LINES(2, module->ToString());

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
              RunOnFwdFlashAttn(computation, instr, /*is_varlen=*/false));
          break;
        }
        case FlashAttnKind::kVarLenForward: {
          TF_ASSIGN_OR_RETURN(
              attn_changed,
              RunOnFwdFlashAttn(computation, instr, /*is_varlen=*/true));
          break;
        }
        case FlashAttnKind::kBackward: {
          TF_ASSIGN_OR_RETURN(
              attn_changed,
              RunOnBwdFlashAttn(computation, instr, /*is_varlen=*/false));
          break;
        }
        case FlashAttnKind::kVarLenBackward: {
          TF_ASSIGN_OR_RETURN(
              attn_changed,
              RunOnBwdFlashAttn(computation, instr, /*is_varlen=*/true));
          break;
        }
      }

      changed |= attn_changed;
    }
  }

  VLOG(2) << "After flash attention normalization:";
  XLA_VLOG_LINES(2, module->ToString());

  return changed;
}

static int RoundMultiple(int x, int m) { return (x + m - 1) / m * m; }

static HloInstruction* CreateZeroTensor(HloComputation* computation,
                                        PrimitiveType element_type,
                                        absl::Span<const int64_t> dimensions) {
  HloInstruction* const_zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(element_type)));
  const Shape& shape = ShapeUtil::MakeShape(element_type, dimensions);
  return computation->AddInstruction(
      HloInstruction::CreateBroadcast(shape, const_zero, {}));
}

absl::StatusOr<bool> GpuFlashAttnNormalization::RunOnFwdFlashAttn(
    HloComputation* computation, HloInstruction* instr, bool is_varlen) {
  // flash_attn_varlen does not support splitkv
  if (is_varlen) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(const auto config,
                      instr->backend_config<FlashAttnBackendConfig>());

  float p_dropout = config.dropout_rate();
  if (p_dropout == 0.0f) {
    const HloInstruction* query = instr->operand(0);
    const HloInstruction* key = instr->operand(1);

    const Shape& q_shape = query->shape();
    const Shape& k_shape = key->shape();

    int batch_size = q_shape.dimensions(0);
    int num_heads = q_shape.dimensions(2);
    int head_size = RoundMultiple(q_shape.dimensions(3), 8);
    int head_size_rounded = RoundMultiple(head_size, 32);

    int max_seqlen_q = q_shape.dimensions(1);
    int max_seqlen_k = k_shape.dimensions(1);

    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    const cudaDeviceProp* dprops = flash::cuda::getCurrentDeviceProperties();
    int num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks,
                                          dprops->multiProcessorCount * 2,
                                          num_n_blocks, 128);
    if (num_splits > 1) {
      HloInstruction* softmax_lse_accum =
          CreateZeroTensor(computation, PrimitiveType::F32,
                           {num_splits, batch_size, num_heads, max_seqlen_q});
      HloInstruction* output_accum = CreateZeroTensor(
          computation, PrimitiveType::F32,
          {num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded});
      instr->AppendOperand(softmax_lse_accum);
      instr->AppendOperand(output_accum);
      return true;
    }
  }

  return false;
}

absl::StatusOr<bool> GpuFlashAttnNormalization::RunOnBwdFlashAttn(
    HloComputation* computation, HloInstruction* instr, bool is_varlen) {
  TF_ASSIGN_OR_RETURN(const auto config,
                      instr->backend_config<FlashAttnBackendConfig>());

  bool deterministic = config.deterministic();

  const HloInstruction* query = instr->operand(1);
  const Shape& q_shape = query->shape();

  int batch_size, num_heads, head_size;
  int max_seqlen_q;
  int total_q;

  if (is_varlen) {
    const HloInstruction* cu_seqlens_query = instr->operand(7);
    const auto cu_seqlens_query_dims = cu_seqlens_query->shape().dimensions();
    batch_size = std::accumulate(cu_seqlens_query_dims.begin(),
                                 cu_seqlens_query_dims.end(), 1,
                                 std::multiplies<int64_t>()) -
                 1;
    total_q = q_shape.dimensions(0);
    num_heads = q_shape.dimensions(1);
    head_size = q_shape.dimensions(2);
    max_seqlen_q = config.max_seqlen_q();
  } else {
    batch_size = q_shape.dimensions(0);
    max_seqlen_q = q_shape.dimensions(1);
    num_heads = q_shape.dimensions(2);
    head_size = q_shape.dimensions(3);
  }

  int seqlen_q_rounded = RoundMultiple(max_seqlen_q, 128);
  int head_size_rounded = RoundMultiple(head_size, 32);

  std::vector<int64_t> dq_accum_dims;

  if (!deterministic) {
    if (is_varlen) {
      dq_accum_dims = {
          total_q + 128 * batch_size,
          num_heads,
          head_size_rounded,
      };
    } else {
      dq_accum_dims = {
          batch_size,
          seqlen_q_rounded,
          num_heads,
          head_size_rounded,
      };
    }
  } else {
    const cudaDeviceProp* dprops = flash::cuda::getCurrentDeviceProperties();
    const int nsplits =
        (dprops->multiProcessorCount + batch_size * num_heads - 1) /
        (batch_size * num_heads);
    if (is_varlen) {
      dq_accum_dims = {
          nsplits,
          total_q + 128 * batch_size,
          num_heads,
          head_size_rounded,
      };
    } else {
      dq_accum_dims = {
          nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded,
      };
    }
  }

  HloInstruction* grad_query_accum =
      CreateZeroTensor(computation, PrimitiveType::F32, dq_accum_dims);

  instr->AppendOperand(grad_query_accum);

  return true;
}

}  // namespace gpu
}  // namespace xla

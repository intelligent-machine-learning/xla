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

#include "xla/service/spmd/gpu_flash_attn_handler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_flash_attn.h"

namespace xla {
namespace gpu {

std::optional<HloSharding>
GpuFlashAttnCustomCallPartitioner::InferShardingFromOperands(
    const HloInstruction* instr) const {
  const HloInstruction* operand = instr->operand(0);
  if (!operand->has_sharding()) {
    LOG(WARNING) << "Failed to partition flash-attention because operand has "
                    "no sharding: "
                 << instr->ToString();
    return std::nullopt;
  }

  const HloSharding& operand_sharding = operand->sharding();
  if (operand_sharding.IsTileMaximal() || operand_sharding.IsReplicated() ||
      operand_sharding.IsManual() || operand_sharding.IsUnknown()) {
    LOG(WARNING) << "Failed to partition flash-attention because operand "
                    "sharding is not tile: "
                 << instr->ToString();
    return std::nullopt;
  }

  const auto& tile = operand_sharding.tile_assignment();
  if (tile.num_dimensions() != 4) {
    LOG(WARNING) << "Failed to partition flash-attention because "
                    "flash-attention only supports 4D sharding for Q/K/V: "
                 << instr->ToString();
    return std::nullopt;
  }

  auto backend_config = instr->backend_config<xla::gpu::GpuBackendConfig>();
  if (TF_PREDICT_FALSE(!backend_config.ok())) {
    return std::nullopt;
  }
  const auto& config = backend_config->flash_attn_backend_config();

  constexpr int64_t batch_dim = 0;
  constexpr int64_t seqlen_dim = 1;
  constexpr int64_t num_heads_dim = 2;
  constexpr int64_t head_dim_dim = 3;
  const int64_t batch_dim_partition = tile.dim(batch_dim);
  const int64_t seqlen_dim_partition = tile.dim(seqlen_dim);
  const int64_t num_heads_dim_partition = tile.dim(num_heads_dim);
  const int64_t head_dim_dim_partition = tile.dim(head_dim_dim);

  if (seqlen_dim_partition != 1 || head_dim_dim_partition != 1 ||
      num_heads_dim_partition != 1) {
    LOG(WARNING)
        << "Failed to partition flash-attention because flash-attention only "
           "supports partitioning on the batch_size dimension: "
        << instr->ToString();
    return std::nullopt;
  }

  const Shape& operand_shape = operand->shape();
  const int64_t batch_size = operand_shape.dimensions(batch_dim);
  if (batch_size % batch_dim_partition != 0) {
    LOG(WARNING) << "Failed to partition flash-attention because batch size "
                    "must be divisible by batch dimension partition: "
                 << instr->ToString();
    return std::nullopt;
  }

  // Used for softmax_lse and grad_softmax
  Array3D<int64_t> softmax_tile(batch_dim_partition, 1, 1);
  // Used for S_dmask
  Array4D<int64_t> dmask_tile(batch_dim_partition, 1, 1, 1);
  // Used for indices and batch_seqlens
  Array<int64_t> indices_tile({batch_dim_partition});
  int64_t cur_device_id = 0;
  for (int64_t i = 0; i < batch_dim_partition; ++i) {
    softmax_tile(i, 0, 0) = cur_device_id;
    dmask_tile(i, 0, 0, 0) = cur_device_id;
    indices_tile(i) = cur_device_id++;
  }

  std::vector<HloSharding> subshardings;
  if (IsCustomCallToFlashAttnForward(*instr)) {
    subshardings = {
        // output
        operand_sharding,
        // softmax lse
        HloSharding::Tile(softmax_tile),
        // rng state
        HloSharding::Replicate(),
    };

    CHECK(config.has_return_softmax());
    bool return_softmax = config.return_softmax();
    if (return_softmax) {
      subshardings.push_back(HloSharding::Tile(dmask_tile));
    }
  } else if (IsCustomCallToFlashAttnVarlenForward(*instr)) {
    const Shape& q_shape = operand_shape;
    const Shape& k_shape = instr->operand(1)->shape();
    const int64_t seqlen_q = q_shape.dimensions(seqlen_dim);
    const int64_t seqlen_k = k_shape.dimensions(seqlen_dim);

    subshardings = {
        // output
        operand_sharding,
        // query_unpad
        operand_sharding,
        // key_unpad
        operand_sharding,
        // value_unpad
        operand_sharding,
        // output_unpad
        operand_sharding,
        // softmax lse
        HloSharding::Tile(softmax_tile),
        // indices_k
        HloSharding::Tile(indices_tile),
        // indices_len_k
        HloSharding::Replicate(),
        // batch_seqlens_k(tile same as indices_k)
        HloSharding::Tile(indices_tile),
    };

    CHECK(config.has_has_query_padding_mask());
    if (config.has_query_padding_mask()) {
      // indices_q
      subshardings.push_back(HloSharding::Tile(indices_tile));
      // indices_len_q
      subshardings.push_back(HloSharding::Replicate());
      // batch_seqlens_q(tile same as indices_q)
      subshardings.push_back(HloSharding::Tile(indices_tile));
    }

    // rng state
    subshardings.push_back(HloSharding::Replicate());

    CHECK(config.has_return_softmax());
    bool return_softmax = config.return_softmax();
    if (return_softmax) {
      // S_dmask
      subshardings.push_back(HloSharding::Tile(dmask_tile));
    }
  } else if (IsCustomCallToFlashAttnBackward(*instr) ||
             IsCustomCallToFlashAttnVarlenBackward(*instr)) {
    subshardings = {
        operand_sharding,                 // grad_query
        operand_sharding,                 // grad_key
        operand_sharding,                 // grad_value
        HloSharding::Tile(softmax_tile),  // grad_softmax
    };
  } else {
    LOG(FATAL) << "This should be unreachable for :" << instr->ToString();
    return std::nullopt;
  }

  return HloSharding::Tuple(std::move(subshardings));
}

absl::Status GpuFlashAttnCustomCallPartitioner::Partition(
    spmd::SpmdPartitioningVisitor* partitioner, HloInstruction* instr) const {
  const HloInstruction* operand = instr->operand(0);
  if (!operand->has_sharding()) {
    LOG(WARNING) << "Failed to partition flash-attention because operand has "
                    "no sharding: "
                 << instr->ToString();
    return partitioner->DefaultAction(instr);
  }

  const HloSharding& operand_sharding = operand->sharding();
  if (operand_sharding.IsTileMaximal() || operand_sharding.IsReplicated() ||
      operand_sharding.IsManual() || operand_sharding.IsUnknown()) {
    LOG(WARNING) << "Failed to partition flash-attention because operand "
                    "sharding is not tile: "
                 << instr->ToString();
    return partitioner->DefaultAction(instr);
  }

  const auto& tile = operand_sharding.tile_assignment();
  if (tile.num_dimensions() != 4) {
    LOG(WARNING) << "Failed to partition flash-attention because "
                    "flash-attention only supports 4D sharding for Q/K/V: "
                 << instr->ToString();
    return partitioner->DefaultAction(instr);
  }

  TF_ASSIGN_OR_RETURN(const auto gpu_config,
                      instr->backend_config<xla::gpu::GpuBackendConfig>());
  const auto& config = gpu_config.flash_attn_backend_config();

  constexpr int64_t batch_dim = 0;
  constexpr int64_t seqlen_dim = 1;
  constexpr int64_t num_heads_dim = 2;
  constexpr int64_t head_dim_dim = 3;
  const int64_t batch_dim_partition = tile.dim(batch_dim);
  const int64_t seqlen_dim_partition = tile.dim(seqlen_dim);
  const int64_t num_heads_dim_partition = tile.dim(num_heads_dim);
  const int64_t head_dim_dim_partition = tile.dim(head_dim_dim);

  if (seqlen_dim_partition != 1 || head_dim_dim_partition != 1 ||
      num_heads_dim_partition != 1) {
    LOG(WARNING)
        << "Failed to partition flash-attention because flash-attention only "
           "supports partitioning on batch_size dimensions: "
        << instr->ToString();
    return partitioner->DefaultAction(instr);
  }

  const Shape& operand_shape = operand->shape();
  const int64_t batch_size = operand_shape.dimensions(batch_dim);
  if (batch_size % batch_dim_partition != 0) {
    LOG(WARNING) << "Failed to partition flash-attention because batch size "
                    "must be divisible by batch dimension partition: "
                 << instr->ToString();
    return partitioner->DefaultAction(instr);
  }
  const int64_t new_batch_size = batch_size / batch_dim_partition;

  Shape new_return_tuple_shape;
  if (IsCustomCallToFlashAttnForward(*instr)) {
    Shape output_shape = instr->shape().tuple_shapes(0);
    Shape softmax_lse_shape = instr->shape().tuple_shapes(1);
    output_shape.set_dimensions(batch_dim, new_batch_size);
    softmax_lse_shape.set_dimensions(batch_dim, new_batch_size);
    std::vector<xla::Shape> return_shapes = {
        // output
        output_shape,
        // softmax lse
        softmax_lse_shape,
        // rng state
        instr->shape().tuple_shapes(2),
    };

    CHECK(config.has_return_softmax());
    bool return_softmax = config.return_softmax();
    if (return_softmax) {
      CHECK(instr->shape().tuple_shapes_size() == 4);
      // S_dmask
      Shape dmask_shape = instr->shape().tuple_shapes(3);
      dmask_shape.set_dimensions(batch_dim, new_batch_size);
      return_shapes.push_back(dmask_shape);
    }

    new_return_tuple_shape = ShapeUtil::MakeTupleShape(return_shapes);
  } else if (IsCustomCallToFlashAttnVarlenForward(*instr)) {
    Shape q_shape = operand_shape;
    Shape k_shape = instr->operand(1)->shape();
    const int64_t seqlen_q = q_shape.dimensions(seqlen_dim);
    const int64_t seqlen_k = k_shape.dimensions(seqlen_dim);
    q_shape.set_dimensions(batch_dim, new_batch_size);
    k_shape.set_dimensions(batch_dim, new_batch_size);

    Shape output_shape = instr->shape().tuple_shapes(0);
    Shape softmax_lse_shape = instr->shape().tuple_shapes(5);
    Shape indices_k_shape = instr->shape().tuple_shapes(6);
    Shape batch_seqlens_k_shape = instr->shape().tuple_shapes(8);
    output_shape.set_dimensions(batch_dim, new_batch_size);
    softmax_lse_shape.set_dimensions(batch_dim, new_batch_size);
    // 0 is the total seqlen_k
    indices_k_shape.set_dimensions(0, new_batch_size * seqlen_k);
    batch_seqlens_k_shape.set_dimensions(batch_dim, new_batch_size);

    std::vector<xla::Shape> return_shapes = {
        // output
        output_shape,
        // query_unpad
        q_shape,
        // key_unpad
        k_shape,
        // value_unpad
        k_shape,
        // output_unpad
        output_shape,
        // softmax lse
        softmax_lse_shape,
        // indices_k
        indices_k_shape,
        // indices_len_k
        instr->shape().tuple_shapes(7),
        // batch_seqlens_k
        batch_seqlens_k_shape,
    };

    size_t result_idx = 9;
    CHECK(config.has_has_query_padding_mask());
    if (config.has_query_padding_mask()) {
      CHECK(result_idx + 3 <= instr->shape().tuple_shapes_size());
      Shape indices_q_shape = instr->shape().tuple_shapes(result_idx++);
      Shape indices_len_q_shape = instr->shape().tuple_shapes(result_idx++);
      Shape batch_seqlens_q_shape = instr->shape().tuple_shapes(result_idx++);
      // 0 is the total seqlen_q
      indices_q_shape.set_dimensions(0, new_batch_size * seqlen_q);
      batch_seqlens_q_shape.set_dimensions(batch_dim, new_batch_size);
      return_shapes.push_back(indices_q_shape);
      return_shapes.push_back(indices_len_q_shape);
      return_shapes.push_back(batch_seqlens_q_shape);
    }

    // rng state
    CHECK(result_idx + 1 == instr->shape().tuple_shapes_size());
    return_shapes.push_back(instr->shape().tuple_shapes(result_idx++));

    CHECK(config.has_return_softmax());
    bool return_softmax = config.return_softmax();
    if (return_softmax) {
      // S_dmask
      CHECK(result_idx + 1 == instr->shape().tuple_shapes_size());
      Shape dmask_shape = instr->shape().tuple_shapes(result_idx++);
      dmask_shape.set_dimensions(batch_dim, new_batch_size);
      return_shapes.push_back(dmask_shape);
    } else {
      CHECK(result_idx == instr->shape().tuple_shapes_size());
    }

    new_return_tuple_shape = ShapeUtil::MakeTupleShape(return_shapes);
  } else if (IsCustomCallToFlashAttnBackward(*instr) ||
             IsCustomCallToFlashAttnVarlenBackward(*instr)) {
    Shape dq_shape = instr->shape().tuple_shapes(0);
    Shape dk_shape = instr->shape().tuple_shapes(1);
    Shape softmax_d_shape = instr->shape().tuple_shapes(3);
    dq_shape.set_dimensions(batch_dim, new_batch_size);
    dk_shape.set_dimensions(batch_dim, new_batch_size);
    softmax_d_shape.set_dimensions(batch_dim, new_batch_size);
    new_return_tuple_shape = ShapeUtil::MakeTupleShape({
        dq_shape,         // grad_query
        dk_shape,         // grad_key
        dk_shape,         // grad_value
        softmax_d_shape,  // grad_softmax
    });
  } else {
    return Internal("This should be unreachable for %s", instr->ToString());
  }

  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(instr->operands().size());
  for (HloInstruction* operand : instr->operands()) {
    new_operands.push_back(partitioner->GetPartitionedHlo(operand).hlo());
  }
  HloInstruction* new_instr = partitioner->builder()->AddInstruction(
      instr->CloneWithNewOperands(new_return_tuple_shape, new_operands));
  new_instr->copy_sharding(instr);

  partitioner->SetPartitionedHlo(
      instr, spmd::PartitionedHlo(new_instr, instr->shape(),
                                  partitioner->MakePartitioningState()));

  return absl::OkStatus();
}

namespace {
struct Registerer {
  Registerer() {
    RegisterCustomCallPartitioner(
        std::string(kGpuFlashAttnForwardCallTarget),
        std::make_unique<GpuFlashAttnCustomCallPartitioner>());
    RegisterCustomCallPartitioner(
        std::string(kGpuFlashAttnBackwardCallTarget),
        std::make_unique<GpuFlashAttnCustomCallPartitioner>());
    RegisterCustomCallPartitioner(
        std::string(kGpuFlashAttnVarlenForwardCallTarget),
        std::make_unique<GpuFlashAttnCustomCallPartitioner>());
    RegisterCustomCallPartitioner(
        std::string(kGpuFlashAttnVarlenBackwardCallTarget),
        std::make_unique<GpuFlashAttnCustomCallPartitioner>());
  }
};
Registerer gpu_flash_attn_custom_call_registerer;
}  // namespace

}  // namespace gpu
}  // namespace xla
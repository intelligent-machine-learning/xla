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

#ifndef XLA_SERVICE_SPMD_GPU_FLASH_ATTN_HANDLER_H_
#define XLA_SERVICE_SPMD_GPU_FLASH_ATTN_HANDLER_H_

#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/spmd_partitioner.h"

namespace xla {
namespace gpu {

class GpuFlashAttnCustomCallPartitioner : public CustomCallPartitioner {
 public:
  HloSharding PropagateUserSharding(
      const HloInstruction* instr, const HloInstruction* user,
      const HloSharding& sharding) const override {
    return sharding;
  }

  std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* instr) const override;

  bool IsCustomCallShardable(const HloInstruction* instr) const override {
    return true;
  }

  absl::Status Partition(spmd::SpmdPartitioningVisitor* partitioner,
                         HloInstruction* instr) const override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_GPU_FLASH_ATTN_HANDLER_H_

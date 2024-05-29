#ifndef XLA_HLO_EXPERIMENTAL_AUTO_REORDER_COMMON_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_REORDER_COMMON_H_
#include "xla/service/gpu/model/analytical_latency_estimator.h"
#include "xla/service/gpu/cublas_cudnn.h"

namespace xla {
constexpr int kMaxConcurrentAsyncCollectivePermutes = 5;
SchedulerConfig GetSchedulerConfig(int64_t memory_limit) {
  SchedulerConfig config;
  config.all_reduce_overlap_limit = 1;
  config.collective_permute_overlap_limit = 1;
  config.use_real_cost_model = false;
  config.aggressive_scheduling_policies = true;
  config.schedule_send_recvs = true;
  config.memory_limit = memory_limit;
  return config;
}
SchedulerConfig GetDefaultSchedConfig() {
  SchedulerConfig sched_cfg;
  sched_cfg.collective_permute_overlap_limit =
      kMaxConcurrentAsyncCollectivePermutes;
  sched_cfg.send_recv_overlap_limit = INT32_MAX;
  return sched_cfg;
}

CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kSend:
      return {HloOpcode::kAsyncStart, HloOpcode::kSend};
    case HloOpcode::kSendDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kSend};
    case HloOpcode::kRecv:
      return {HloOpcode::kAsyncStart, HloOpcode::kRecv};
    case HloOpcode::kRecvDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kRecv};
    default:
      return DefaultGetCanonicalAsyncOp(hlo);
  }
}
class GpuLatencyEstimator : public ApproximateLatencyEstimator {
 public:
  explicit GpuLatencyEstimator(
      GetCanonicalAsyncOpFunc func = GpuGetCanonicalAsyncOp)
      : ApproximateLatencyEstimator(func) {}
  TimeCost NodeCost(const HloInstruction* instr) const override {
    HloOpcode op = instr->opcode();
    if (op == HloOpcode::kGetTupleElement || op == HloOpcode::kBitcast ||
        op == HloOpcode::kConstant || op == HloOpcode::kParameter ||
        instr->IsEffectiveBitcast()) {
      return 0.0;
    }
    // Consider cublas/cuddn/softmax custom calls as medium cost. Since the
    // latency between async-start and async-done is 5000 and cost of each
    // custom call is 1000, the LHS will try to schedule approximately 5 of
    // these in between each start/end pair.
    if (instr->opcode() == HloOpcode::kCustomCall) {
      if (gpu::IsCublasGemm(*instr) ||
          gpu::IsCustomCallToDnnConvolution(*instr)) {
        return ApproximateLatencyEstimator::kMediumCost;
      }
      // consider other custom calls as medium cost for now. Keeping the case
      // explicitly separate for further tuning.
      return ApproximateLatencyEstimator::kMediumCost;
    }
    return ApproximateLatencyEstimator::NodeCost(instr);
  }

  LatencyEstimator::TimeCost GetLatencyBetween(
      const HloGraphNode& from, const HloGraphNode& target) const override {
    if (IsAsyncPair(from, target)) {
      if (from.GetInstr().opcode() == HloOpcode::kRecv) {
        // Recv -> RecvDone has a low latency.
        return ApproximateLatencyEstimator::kLowLatency;
      } else if (from.GetInstr().opcode() == HloOpcode::kSend) {
        // Send -> SendDone has a very high latency.
        return ApproximateLatencyEstimator::kHighLatency * 10;
      }

      return ApproximateLatencyEstimator::kHighLatency;
    }
    // Every other instruction we consider synchronous, which means the
    // latency between each of them is always one unit.
    return ApproximateLatencyEstimator::kLowLatency;
  }
};
}  // namespace xla
#endif
#ifndef XLA_AUTO_REORDER_H_
#define XLA_AUTO_REORDER_H_
#include "absl/strings/string_view.h"
#include "xla/hlo/experimental/auto_reorder/auto_reorder_solver.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/backend.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/model/analytical_latency_estimator.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/hlo/experimental/auto_reorder/common.h"
#include <functional>
#include <queue>
#include <deque>  //c++11
// #include "xla/statusor.h"
namespace xla {
inline NodeType GetEdgeTypeOfInst(const HloInstruction& hlo) {
  auto op = GpuGetCanonicalAsyncOp(hlo);
  if (op.outer == HloOpcode::kAsyncDone) {
    if (hlo.IsAsynchronous() &&
        hlo.async_execution_thread() != hlo.parent()->execution_thread()) {
      return NodeType::kAsynchronous;
    }
    switch (op.inner) {
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kCollectivePermute:
      {
        //check replica_groups,{{0,8},{1,9}} means inter-replica communication
        // {0,1,2,3,4,5,6,7} means intra-replica communication
        // TODO: 8 is hard code?
        //this hlo is done,we need to check it's operand is done
        bool in_same_replica = true;
        for (auto i = 0; i < hlo.operand_count(); i++) {
          auto src_inst = hlo.operand(i);
          auto replica_groups = src_inst->replica_groups();
          if(replica_groups.empty()){
            return NodeType::kCommunication;
          }
          for (auto replica_group : replica_groups)
          {
            // if device_id//8 all equal, it is intra-replica communication
            const auto& ids = replica_group.replica_ids();
            if (ids.empty()) {
              continue;
            }
            int init_device_divide = ids.Get(0) / 8;
            for (int i = 0; i < ids.size(); i++) {
              if (ids.Get(i) / 8 != init_device_divide) {
                in_same_replica = false;
                break;
              }
            }
          }
        }
        if(in_same_replica){
          return NodeType::kInnerCommunication;
        }
        else{
          return NodeType::kCommunication;
        }
      }
      case HloOpcode::kCopy:
        return NodeType::kCommunication;
      default:
        return NodeType::kAsynchronous;
    }
  }
  // If there are parallel thread computations, always schedule.

  VLOG(2) << "edge from:" << hlo.ToString() << "," << hlo.IsAsynchronous()
          << " thread=" << hlo.async_execution_thread()
          << " parent=" << hlo.parent()->execution_thread();
  return NodeType::kCommunication;
}
class AutoReorderPass : public HloModulePass {
 public:
  AutoReorderPass(){};
  AutoReorderPass(std::unique_ptr<LatencyEstimator> latency_estimator,
                  std::unique_ptr<AsyncTracker> async_tracker,
                  std::unique_ptr<SchedulerCore> scheduler_core,
                  HloCostAnalysis::ShapeSizeFunction shape_size_bytes)
      : async_tracker_(std::move(async_tracker)),
        scheduler_core_(std::move(scheduler_core)),
        latency_estimator_(std::move(latency_estimator)),
        shape_size_bytes_(shape_size_bytes){};
  absl::string_view name() const override { return "auto-reorder"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
  // when computation is changed, we need to rebuild the hlo ordering
  tsl::Status RebuildHloOrdering(HloSchedule& module_schedule,
                                 HloComputation* entry_computation);
  tsl::Status MoveInstruction(HloComputation* src_computation,
                              absl::string_view src_name,
                              HloComputation* dst_computation);
  int64_t OriginalInstructionPosition(const HloInstruction* instr) const {
    auto it = instr_order_map_.find(instr);
    CHECK(it != instr_order_map_.end());
    return it->second;
  }
  tsl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      HloComputation* computation);
  CostType GetInstructionStart(const HloInstruction* instr) const {
    auto it = instr_order_map_.find(instr);
    CHECK(it != instr_order_map_.end());
    return it->second;
  }
  void LogScheduleStatistics(const HloComputation* computation) {
    XLA_VLOG_LINES(1, LatencyHidingScheduler::SchedulerStatisticsString(
                          LatencyHidingScheduler::LatencyHidingStatistics(
                              computation, latency_estimator_.get(),
                              async_tracker_.get(), shape_size_bytes_)));
  }
  static int64_t GetSpaceCost(const HloInstruction* instr);

 private:
  std::unique_ptr<AsyncTracker> async_tracker_;
  std::unique_ptr<SchedulerCore> scheduler_core_;
  std::unique_ptr<LatencyEstimator> latency_estimator_;
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloGraphNode>>
      nodes_;
  absl::flat_hash_map<const HloInstruction*, int64_t> instr_order_map_;
  // std::unique_ptr<LinearProgramScheduler> solver_;
  int64_t move_cost_threshold_in_bytes_;
  HloCostAnalysis::ShapeSizeFunction shape_size_bytes_;
  std::vector<LPContainer<const HloInstruction*>*> FunetuneNodeOrder(
      std::vector<LPContainer<const HloInstruction*>*> sorted_nodes);
  void SaveOrder(std::vector<LPContainer<const HloInstruction*>*> sorted_nodes,
                 std::string filename);
  // right node's deps include left;
  bool NodeHaveDep(LPContainer<const HloInstruction*>* node,
                   LPContainer<const HloInstruction*>* maybe_deps_node);
};

// CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo);

}  // namespace xla

#endif
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

#include "xla/hlo/experimental/auto_reorder/common.h"

// #include "xla/statusor.h"
namespace xla {
NodeType GetEdgeTypeOfInst(const HloInstruction& hlo) {
  auto op = GpuGetCanonicalAsyncOp(hlo);
  if (op.outer == HloOpcode::kAsyncDone){
    if (hlo.IsAsynchronous() &&
        hlo.async_execution_thread() != hlo.parent()->execution_thread()) {
      return NodeType::kAsynchronous;
    }
    switch (op.inner) {
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kCopy:
        return NodeType::kCommunication;
      default:
        return NodeType::kAsynchronous;
        
    }
  }
  // If there are parallel thread computations, always schedule.
  
  VLOG(2)<<"edge from:"<<hlo.ToString()<<","<<hlo.IsAsynchronous()<<" thread="<<hlo.async_execution_thread()<<" parent="<<hlo.parent()->execution_thread();
  return NodeType::kCommunication;
}

class AutoReorderSchedulerCore : public SchedulerCore {
 public:
  using ReadyQueueSet = std::vector<HloGraphNode*>;
  using ResourceMap = absl::flat_hash_map<int64_t, int64_t>;
  using ShouldSkipNodeFunction = std::function<bool(const HloGraphNode*)>;

  // Class used to cache expensive information. Currently memory pressure
  // changes are cached. The caching is invalidated at the end of the scheduling
  // process for this next candidate. The information shouldn't survive across
  // scheduling two different instructions.
  struct ScheduleCandidate {
    HloGraphNode* node = nullptr;
    std::optional<std::pair<int64_t, int64_t>> pressure_change;
    std::optional<HloGraphNode::TimeCost> estimated_connected_send_ready_time;
    std::optional<bool> resource_constrained;
  };

  struct CandidateResult {
    ScheduleCandidate result;
    const char* reason;
  };

  using TargetSchedulingRule = std::function<std::optional<CandidateResult>(
      ScheduleCandidate&, ScheduleCandidate&)>;

  // Returns nullopt if both parameters are equal, otherwise true if the first
  // parameter is true and false if the second is true
  static std::optional<bool> TrueForOneOnly(bool first, bool second) {
    if (first == second) {
      return std::nullopt;
    }
    return first;
  }

  static std::optional<CandidateResult> ChooseBestCandidate(
      bool first_cond, const ScheduleCandidate& first_candidate,
      bool second_cond, const ScheduleCandidate& second_candidate,
      const char* reason) {
    if (auto cond = TrueForOneOnly(first_cond, second_cond)) {
      return CandidateResult{*cond ? first_candidate : second_candidate,
                             reason};
    }
    return std::nullopt;
  }

  // The scheduling state contains everything that is required for the
  // bookkeeping of the scheduling algorithm. Functions that perform operations
  // over the scheduling state can directly operate on the state contained into
  // this struct instead of having to pass many individual pointers to elements
  // of the state.
  struct SchedulingState {
    HloScheduleGraph sched_graph;
    // Ready set for the nodes. Its ordered by our heuristic defined in
    // ReadySetLt.
    ReadyQueueSet ready_set;
    // Maximum allowed number of overlapping instructions using the key resource
    // type.
    ResourceMap max_concurrent_resource;
    // New scheduling sequence produced by the scheduler. This is in reversed
    // order (because we schedule bottom up). This will be required to be
    // reversed before assigning to the HloSchedule.
    std::vector<HloInstruction*> new_sequence_reversed;
    // Units of time passed in the schedule. To keep track of latency hiding.
    HloGraphNode::TimeCost current_time = 0;
    // Number of resources in flight.
    ResourceMap resources_in_flight;
    // Number of instructions using the key resource type in the set waiting to
    // be scheduled.
    ResourceMap resource_users_in_queue;
    // Number of nodes scheduled.
    int64_t scheduled_count = 0;
    // Class returning information about instruction cost and latency between
    // instructions.
    const LatencyEstimator* latency_estimator;
    // Class used to track which instructions are async instructions and which
    // async instructions computations contain.
    const AsyncTracker* async_tracker;
    // Tracker of memory pressure for the computation.
    MemoryPressureTracker* memory_pressure_tracker;
    // Vector containing a list of nodes that aren't ready to schedule yet in
    // order of time when they are going to become ready.
    std::vector<const HloGraphNode*> next_ready_stack;
    // List of the graph edges currently occupying the key shareable resource
    // with projected finish times.
    absl::flat_hash_map<
        int64_t, std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>>
        shareable_resource_occupiers;
    // Reference to this scheduler run configuration.
    const SchedulerConfig& config;
    SchedulingState(const HloInstructionSequence* instr_sequence,
                    HloAliasAnalysis* alias_analysis,
                    const LatencyEstimator* latency_estimator,
                    const AsyncTracker* async_tracker,
                    MemoryPressureTracker* memory_pressure_tracker,
                    const SchedulerConfig& config)
        : sched_graph(&instr_sequence->instructions(), alias_analysis,
                      latency_estimator, async_tracker),
          latency_estimator(latency_estimator),
          async_tracker(async_tracker),
          memory_pressure_tracker(memory_pressure_tracker),
          config(config) {}
  };

  using PostProcessingFn = std::function<void(SchedulingState&)>;

  AutoReorderSchedulerCore(
      HloCostAnalysis::ShapeSizeFunction shape_size_bytes,
      const AsyncTracker* async_tracker,
      const LatencyEstimator* latency_estimator, const SchedulerConfig& config,
      TargetSchedulingRule target_scheduling_rule = nullptr,
      TargetSchedulingRule early_target_scheduling_rule = nullptr,
      PostProcessingFn post_processing_fn = nullptr)
      : shape_size_bytes_(shape_size_bytes),
        async_tracker_(async_tracker),
        latency_estimator_(latency_estimator),
        config_(config),
        target_scheduling_rule_(target_scheduling_rule),
        early_target_scheduling_rule_(early_target_scheduling_rule),
        post_processing_fn_(post_processing_fn) {}
  
  // TODO: implement these functions
  Status InitializeScheduler(const HloModule* module) override;
  
  absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation) override;
  
  // static bool AddOccupierToResource(
  //     HloGraphNode::TimeCost current_time, HloEdge& new_edge,
  //     std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers);
  
  // static bool DeleteOccupierFromResource(
  //     HloGraphNode::TimeCost current_time, HloEdge& edge,
  //     std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers);
  
  int64_t GetMemoryPeak() override {
    return module_pressure_state_->GetMemoryPeak();
  }
  uint64_t GetMemoryLimit() override { return config_.memory_limit; }
  void SetMemoryLimit(uint64_t new_limit) override {
    this->config_.memory_limit = new_limit;
  }
  int64_t GetRerunTimes() override { return config_.rerun; }

 protected:
  // virtual void LogInstruction(const HloInstruction* instr) const;
  // // Update node that has been scheduled.
  // virtual absl::StatusOr<HloGraphNode::TimeCost> ScheduleNode(
  //     HloGraphNode* n, SchedulingState* sched_state) const;
  // // Perform the scheduling of one or more instructions. Called every time the
  // // ready set is not empty.
  // virtual Status SchedulingStep(SchedulingState* sched_state);
  // Pick a node to schedule according to cost model.
  // virtual HloGraphNode* FindAndExtractBestNodeAvailable(
  //     SchedulingState& sched_state,
  //     AutoReorderSchedulerCore::ShouldSkipNodeFunction should_skip_node);
  // void DumpAutoReorderSchedule(
  //     const HloComputation* computation, const HloScheduleGraph& schedule_graph,
  //     const std::vector<HloInstruction*>& instructions,
  //     int cycles_per_microsecond, const DebugOptions& debug_options);

  HloCostAnalysis::ShapeSizeFunction shape_size_bytes_;
  std::unique_ptr<ModulePressureState> module_pressure_state_;
  std::unique_ptr<HloAliasAnalysis> alias_analysis_;
  const AsyncTracker* async_tracker_;
  const LatencyEstimator* latency_estimator_;
  SchedulerConfig config_;
  TargetSchedulingRule target_scheduling_rule_ = nullptr;
  TargetSchedulingRule early_target_scheduling_rule_ = nullptr;
  PostProcessingFn post_processing_fn_ = nullptr;
};


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
};

// CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo);

}  // namespace xla

#endif
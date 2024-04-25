

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "xla/hlo/experimental/auto_reorder/auto_reorder.h"
#include "xla/service/async_collective_creator.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/tests/hlo_test_base.h"

#define debug_log(x)                           \
  {                                            \
    if (is_debug) std::cout << x << std::endl; \
  }
namespace xla {
uint32_t kRandomSeed = 1243;
struct ScheduleStatus {
  bool success;
  uint32_t exec_time;
  uint32_t memory_usage;
};
class CostGenerator {
 public:
  CostGenerator(int mean, int std, int seed) {
    gen_ = std::mt19937(seed);
    dist_ = std::normal_distribution<float>(static_cast<float>(mean),
                                            static_cast<float>(std));
  };
  int operator()() { return std::max(1, static_cast<int>(dist_(gen_))); }

 private:
  std::mt19937 gen_;
  std::normal_distribution<float> dist_;
};
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

class SavedInstLatencyEstimator : public GpuLatencyEstimator {
  // make random inst cost
  // usage:
  // 1. create instruction
  // 2. using SetInstructionCost;
  // 3. using this estimator in scheduler

 public:
  explicit SavedInstLatencyEstimator(
      GetCanonicalAsyncOpFunc func = GpuGetCanonicalAsyncOp)
      : GpuLatencyEstimator(func) {}
  TimeCost NodeCost(const HloInstruction* instr) const override {
    auto cost = GpuLatencyEstimator::NodeCost(instr);
    if (inst_cost_.find(instr->unique_id()) != inst_cost_.end()) {
      cost = inst_cost_.at(instr->unique_id());
    }
    return cost;
  }
  LatencyEstimator::TimeCost GetLatencyBetween(
      const HloGraphNode& from, const HloGraphNode& target) const override {
    if (IsAsyncPair(from, target)) {
      if (edge_cost_.find(target.GetInstr().unique_id()) != edge_cost_.end()) {
        auto cost = edge_cost_.at(target.GetInstr().unique_id());
        return cost;
      }
      if (edge_cost_.find(from.GetInstr().unique_id()) != edge_cost_.end()) {
        auto cost = edge_cost_.at(from.GetInstr().unique_id());
        return cost;
      }
      return ApproximateLatencyEstimator::kLowLatency;
    }
    // Every other instruction we consider synchronous, which means the
    // latency between each of them is always one unit.
    return ApproximateLatencyEstimator::kLowLatency;
  }

  void SetInstructionCost(const HloInstruction* instr, TimeCost cost) {
    inst_cost_.emplace(instr->unique_id(), cost);
  }
  void SetInstructionBetween(const HloInstruction* target, TimeCost cost) {
    // let all node link to target have cost
    if (target->unique_id() == -1) {
      // raise exception?
      std::cout << "SetInstructionBetween fail" << target->ToShortString()
                << std::endl;
      ASSERT_ANY_THROW(target->unique_id() != -1);
    }
    edge_cost_.emplace(target->unique_id(), cost);
  }
  void CloneCost(std::unordered_map<int, TimeCost> input_costs,
                 std::unordered_map<int, TimeCost> edge_cost) {
    inst_cost_ = input_costs;
    edge_cost_ = edge_cost;
  }
  std::unordered_map<int, TimeCost> GetCosts() { return inst_cost_; }
  std::unique_ptr<SavedInstLatencyEstimator> clone() {
    auto estimator = std::make_unique<SavedInstLatencyEstimator>();
    estimator->CloneCost(inst_cost_, edge_cost_);
    return estimator;
  }

 private:
  std::unordered_map<int, TimeCost> inst_cost_;
  std::unordered_map<int, TimeCost> edge_cost_;
};
using namespace xla::gpu;
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

class AutoReorderingTest : public HloTestBase {
 protected:
  const char* const add_hlo_string_ = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[16,32,64]{2,1,0} parameter(0)
  %param1 = f32[16,32,64]{2,1,0} parameter(1)
  ROOT root = f32[16,32,64]{2,1,0} add(%param0, %param1)
})";
  void RunDemoAutoReorderWithOptions(size_t expected_num_tiles,
                                     size_t expected_sharded_dimensions = 1) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(add_hlo_string_));
    auto* instruction = FindInstruction(module.get(), "param0");
  }

 public:
  HloComputation* MakeReduction(const HloOpcode type, HloModule* module) {
    HloComputation::Builder sum_builder(HloOpcodeString(type));
    auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "x"));
    auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "y"));
    sum_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), type, x, y));
    HloComputation* reduction =
        module->AddEmbeddedComputation(sum_builder.Build());
    return reduction;
  }
  HloComputation* MakeReduceScatter(const HloOpcode type, Shape input_shape,
                                    Shape output_shape, HloModule* module) {
    HloComputation::Builder async_builder("AsyncOp");
    HloInstruction* param = async_builder.AddInstruction(
        HloInstruction::CreateParameter(0, input_shape, "pasync"));
    async_builder.AddInstruction(HloInstruction::CreateReduceScatter(
        output_shape, {param}, MakeReduction(type, module), CreateReplicaGroups({{0, 1}}), false,
        std::nullopt, false, 0));
    HloComputation* reduction =
        module->AddEmbeddedComputation(async_builder.Build());

    return reduction;
  }
  HloComputation* MakeAll2All(Shape input_shape,HloModule* module){
    HloComputation::Builder async_builder("AsyncOp");
    HloInstruction* param = async_builder.AddInstruction(
        HloInstruction::CreateParameter(0, input_shape, "pasync"));
    HloInstruction* param1 = async_builder.AddInstruction(
        HloInstruction::CreateParameter(0, input_shape, "pasync"));
    async_builder.AddInstruction(HloInstruction::CreateAllToAll(
          input_shape, {param,param1},
          /*replica_groups=*/CreateReplicaGroups({{0, 1}}),
          /*constrain_layout=*/false, /*channel_id=*/std::nullopt,
          /*split_dimension*/ 0
      ));
    HloComputation* reduction =
        module->AddEmbeddedComputation(async_builder.Build());

    return reduction;
  }
  std::string GetInstructionsOrderString(HloModule* hlo_module) {
    auto insts = hlo_module->schedule()
                     .sequence(hlo_module->entry_computation())
                     .instructions();
    auto ret = std::string("");
    // hard to keep all instruction order,only keep communicate order
    for (auto inst : insts) {
      switch (inst->opcode()) {
        case HloOpcode::kAllToAll:
        case HloOpcode::kAllGather:
        case HloOpcode::kAllGatherDone:
        case HloOpcode::kAllReduce:
        case HloOpcode::kAllReduceDone:
        case HloOpcode::kCollectivePermute:
        case HloOpcode::kCollectivePermuteDone:
        case HloOpcode::kReduceScatter:
        case HloOpcode::kSend:
        case HloOpcode::kSendDone:
        case HloOpcode::kRecv:
        case HloOpcode::kRecvDone: {
          ret = ret + "\n" + inst->ToString();
        }
        default: {
          continue;
        }
      }
    }
    bool isStart = false;
    // check start-done pair, there is no start-start
    for (auto inst : insts) {
      switch (inst->opcode()) {
        case HloOpcode::kAllGatherStart:
        case HloOpcode::kAllReduceStart:
        case HloOpcode::kCollectivePermuteStart:
        case HloOpcode::kSend:
        case HloOpcode::kRecv: {
          CHECK_NE(isStart, true);
          isStart = true;
          continue;
        }
        case HloOpcode::kAllGatherDone:
        case HloOpcode::kAllReduceDone:
        case HloOpcode::kCollectivePermuteDone:
        case HloOpcode::kSendDone:
        case HloOpcode::kRecvDone: {
          isStart = false;
          continue;
        }
        case HloOpcode::kReduceScatter: {
          ret = ret + "\n" + inst->ToString();
        }
        default: {
          continue;
        }
      }
    }
    return ret;
  }
  bool CheckParameterInst(HloModule* hlo_module) {
    auto insts = hlo_module->schedule()
                     .sequence(hlo_module->entry_computation())
                     .instructions();
    bool isParameter = true;
    // check parameter must at head
    for (auto inst : insts) {
      if (inst->opcode() == HloOpcode::kParameter) {
        if (isParameter) {
          continue;
        } else {
          return false;
        }
      } else {
        isParameter = false;
      }
    }
    return true;
  }
  StatusOr<LatencyHidingScheduler::SchedulerStatistics> GetModuleCost(
      HloModule* module, SchedulerConfig sched_config,
      const xla::LatencyEstimator* latency_estimator

  ) {
    // we should implement method independent of scheduler to get module time
    // cost

    // ASSERT_ANY_THROW(latency_estimator!=nullptr);
    auto computation = module->entry_computation();
    auto schedule = module->schedule();
    auto seq = schedule.sequence(computation);

    HloCostAnalysis::ShapeSizeFunction shape_size_bytes =
        [&shape_size_bytes](const Shape& shape) -> int64_t {
      int64_t shape_size = 0;
      if (shape.IsTuple()) {
        for (auto& sub_shape : shape.tuple_shapes()) {
          shape_size += shape_size_bytes(sub_shape);
        }
        return shape_size;
      }
      return ShapeUtil::ByteSizeOfElements(shape);
    };
    auto async_tracker = std::make_unique<AsyncTracker>(sched_config);
    // copy some code from LatencyHidingStatistics
    if (latency_estimator == nullptr) {
      return absl::InvalidArgumentError("latency_estimator is nullptr!");
    }
    // here latency_estimator make segmant fault?
    auto ret = LatencyHidingScheduler::LatencyHidingStatistics(
        computation, latency_estimator, async_tracker.get(), shape_size_bytes);
    return ret;
  }
  std::vector<ReplicaGroup> CreateReplicaGroups(
      absl::Span<const std::vector<int64_t>> groups) {
    std::vector<ReplicaGroup> replica_groups(groups.size());
    for (int64_t i = 0; i < groups.size(); ++i) {
      *replica_groups[i].mutable_replica_ids() = {groups[i].begin(),
                                                  groups[i].end()};
    }
    return replica_groups;
  }
  StatusOr<HloComputation*> MakeCommunicateComputation(
      HloModule* module, SavedInstLatencyEstimator* estimator) {
    /*
     p0->allreduce-100->allreduce_done->p0.1
     p1->allreduce.1-10->allreduce_done.1->p0.2
     dot(p0,p1):cost=10
     dot(p0.1,p0.2):cost=10

    p0

    */
    HloComputation::Builder builder("test");
    auto add_reducer = MakeReduction(HloOpcode::kAdd, module);
    Shape shape = ShapeUtil::MakeShape(F32, {4, 256, 256});
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    int64_t channel_id = 0;
    auto precision_config = DefaultPrecisionConfig(2);

    auto p0 = builder.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0, shape, "p0"));
    auto p1 = builder.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/1, shape, "p1"));
    HloInstruction* ar_start0 =
        builder.AddInstruction(HloInstruction::CreateAllReduceStart(
            shape, {p0}, add_reducer,
            /*replica_groups=*/CreateReplicaGroups({{0, 1}}),
            /*constrain_layout=*/false, /*channel_id=*/std::nullopt,
            /*use_global_device_ids=*/false));
    HloInstruction* ar_done0 =
        builder.AddInstruction(HloInstruction::CreateUnary(
            shape, HloOpcode::kAllReduceDone, ar_start0));

    auto dot0 = builder.AddInstruction(
        HloInstruction::CreateDot(shape, p0, p1, dot_dnums, precision_config));

    HloInstruction* ar_start1 =
        builder.AddInstruction(HloInstruction::CreateAllReduceStart(
            shape, {p1}, add_reducer,
            /*replica_groups=*/CreateReplicaGroups({{0, 1}}),
            /*constrain_layout=*/false, /*channel_id=*/std::nullopt,
            /*use_global_device_ids=*/false));
    HloInstruction* ar_done1 =
        builder.AddInstruction(HloInstruction::CreateUnary(
            shape, HloOpcode::kAllReduceDone, ar_start1));

    auto dot1 = builder.AddInstruction(HloInstruction::CreateDot(
        shape, ar_done0, ar_done1, dot_dnums, precision_config));

    auto ret =
        builder.AddInstruction(HloInstruction::CreateTuple({dot0, dot1}));
    auto computation = builder.Build();
    computation->set_root_instruction(ret);
    auto entry_computation =
        module->AddEntryComputation(std::move(computation));
    estimator->SetInstructionCost(ar_start0, 1);
    estimator->SetInstructionCost(ar_done0, 1);
    estimator->SetInstructionCost(dot0, 10);
    estimator->SetInstructionCost(ar_start1, 1);
    estimator->SetInstructionCost(ar_done1, 1);
    estimator->SetInstructionCost(dot1, 10);
    estimator->SetInstructionBetween(ar_done0, 100);
    estimator->SetInstructionBetween(ar_done1, 10);

    VLOG(2) << "finish creating instruction now scheduling"
            << module->has_schedule();

    // let module have one schedule
    TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                        ScheduleModule(module, [](const BufferValue& buffer) {
                          return ShapeUtil::ByteSizeOf(
                              buffer.shape(),
                              /*pointer_size=*/sizeof(void*));
                        }));

    TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

    return entry_computation;
  }
  StatusOr<HloComputation*> MakeTestComputation(HloModule* module) {
    // param: p0,p1,p2,p3
    // d01 = dot(p0,p1)
    // d23 = dot(p2,p3)
    //
    auto add_reducer = MakeReduction(HloOpcode::kAdd, module);
    HloComputation::Builder builder(TestName());
    Shape shape = ShapeUtil::MakeShape(F32, {4, 256, 256});
    Shape reduce_shape = ShapeUtil::MakeShape(F32, {2, 256, 256});
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    int64_t channel_id = 0;
    auto precision_config = DefaultPrecisionConfig(2);
    auto p0 = builder.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0, shape, "p0"));
    auto p1 = builder.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/1, shape, "p1"));
    auto p2 = builder.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/2, shape, "p2"));
    auto p3 = builder.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/3, shape, "p3"));
    auto d01 = builder.AddInstruction(
        HloInstruction::CreateDot(shape, p0, p1, dot_dnums, precision_config));
    auto d23 = builder.AddInstruction(
        HloInstruction::CreateDot(shape, p2, p3, dot_dnums, precision_config));
    HloInstruction* all_reduce_start =
        builder.AddInstruction(HloInstruction::CreateAllReduceStart(
            shape, {d01}, add_reducer,
            /*replica_groups=*/CreateReplicaGroups({{0, 1}}),
            /*constrain_layout=*/false, /*channel_id=*/1,
            /*use_global_device_ids=*/true), "all_reduce_start");
    HloInstruction* ar_done0 =
        builder.AddInstruction(HloInstruction::CreateUnary(
            shape, HloOpcode::kAllReduceDone, all_reduce_start),"ar_done0");

    HloInstruction* all_reduce_start1 =
        builder.AddInstruction(HloInstruction::CreateAllReduceStart(
            shape, {d23}, add_reducer,
            /*replica_groups=*/CreateReplicaGroups({{0, 1}}),
            /*constrain_layout=*/false, /*channel_id=*/std::nullopt,
            /*use_global_device_ids=*/false));
    HloInstruction* ar_done1 =
        builder.AddInstruction(HloInstruction::CreateUnary(
            shape, HloOpcode::kAllReduceDone, all_reduce_start1));

    // d01 dot with p0
    auto d01_dot_p0 = builder.AddInstruction(HloInstruction::CreateDot(
        shape, ar_done0, p0, dot_dnums, precision_config));
    // d23 dot with p1
    auto d23_dot_p1 = builder.AddInstruction(HloInstruction::CreateDot(
        shape, ar_done1, p1, dot_dnums, precision_config));
    // d01_dot_p0 add p2
    auto d01_dot_p0_add_p2 = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, d01_dot_p0, p2));
    // d23_dot_p1 add p3
    auto d23_dot_p1_add_p3 = builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, d23_dot_p1, p3));
    auto rs_computation =
        MakeReduceScatter(HloOpcode::kAdd, shape, reduce_shape, module);

    const Shape async_start_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeTupleShape({shape}), reduce_shape});
    HloInstruction* async_start0 =
        builder.AddInstruction(HloInstruction::CreateAsyncStart(
            async_start_shape, {d01_dot_p0_add_p2}, rs_computation,
            /*async_execution_thread=*/"parallel_thread"));
    HloInstruction* async_update = builder.AddInstruction(
      HloInstruction::CreateAsyncUpdate(async_start_shape, async_start0));
    auto reducescater_ret0 = builder.AddInstruction(
        HloInstruction::CreateAsyncDone(reduce_shape, async_update));
    //new version:need create other computation
     auto rs_computation2 =
        MakeReduceScatter(HloOpcode::kAdd, shape, reduce_shape, module);
    HloInstruction* async_start1 =
        builder.AddInstruction(HloInstruction::CreateAsyncStart(
            async_start_shape, {d23_dot_p1_add_p3}, rs_computation2,
            /*async_execution_thread=*/"parallel_thread"));
    HloInstruction* async_update1 = builder.AddInstruction(
      HloInstruction::CreateAsyncUpdate(async_start_shape, async_start1));
    auto reducescater_ret1 = builder.AddInstruction(
        HloInstruction::CreateAsyncDone(reduce_shape, async_update1));

    auto add_reduce_scatter =
        builder.AddInstruction(HloInstruction::CreateBinary(
            reduce_shape, HloOpcode::kAdd, reducescater_ret1, reducescater_ret0));
    auto all2all_ret = builder.AddInstruction(HloInstruction::CreateAllToAll(
          reduce_shape, {add_reduce_scatter},
          /*replica_groups=*/CreateReplicaGroups({{0, 1}}),
          /*constrain_layout=*/false, /*channel_id=*/std::nullopt,
          /*split_dimension*/ 0
      ));
    std::vector<HloInstruction*> compute_vec = {d01_dot_p0_add_p2, d23_dot_p1_add_p3, all2all_ret};
    auto ret = builder.AddInstruction(HloInstruction::CreateTuple(
        compute_vec));
    auto computation = builder.Build();
    computation->set_root_instruction(ret);
    auto entry_computation =
        module->AddEntryComputation(std::move(computation));
    VLOG(2) << "finish creating instruction now scheduling"
            << module->has_schedule();

    // let module have one schedule
    TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                        ScheduleModule(module, [](const BufferValue& buffer) {
                          return ShapeUtil::ByteSizeOf(
                              buffer.shape(),
                              /*pointer_size=*/sizeof(void*));
                        }));

    TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

    return entry_computation;
  }
  StatusOr<HloComputation*> MakeRandomComputation(
      HloModule* module, SavedInstLatencyEstimator* estimator,
      uint32_t inst_nums = 100, uint8_t max_deps = 5,
      double communication_rate = 0.1f,
      std::mt19937 gen = std::mt19937{kRandomSeed},
      CostGenerator cost_gen = CostGenerator(50, 5, kRandomSeed)) {
    /* create instruction list with inst_nums instructions
     every inst be used by output
    */
    VLOG(2) << "create computation begin,test name: " << TestName()
            << ",inst_nums=" << inst_nums << ",max_deps=" << max_deps
            << ",communication_rate=" << communication_rate;
    HloComputation::Builder builder(TestName());
    Shape shape = ShapeUtil::MakeShape(F32, {4, 256, 256});

    // insts_list: store instruction list,which have one result
    std::vector<HloInstruction*> insts_list;

    uint32_t communication_count = std::floor(communication_rate * inst_nums);
    uint32_t insert_comm_every = inst_nums / communication_count;

    // Node cost must add after AddEntryComputation,so that instruction have
    // unique_id
    std::vector<std::tuple<HloInstruction*, int>> insts2cost;
    std::vector<std::tuple<HloInstruction*, int>> edge2cost;

    std::set<HloInstruction*> used_insts;
    std::set<HloInstruction*> not_used_insts;

    CostGenerator random_gen = CostGenerator(50, 5, kRandomSeed);

    for (size_t i = 0; i < inst_nums; i++) {
      // random deps from 1~5
      if (i < 2) {
        auto inst = builder.AddInstruction(HloInstruction::CreateParameter(
            /*parameter_number=*/i, shape, "p" + std::to_string(i)));
        insts_list.push_back(inst);
        insts2cost.push_back(std::make_tuple(inst, 0));
        not_used_insts.insert(inst);
        continue;
      }
      uint32_t deps_count = 2;
      std::vector<HloInstruction*> deps;
      // from 0~i, pick deps_count insts as deps
      if (not_used_insts.size() >= deps_count) {
        // first pick not used insts
        std::sample(not_used_insts.begin(), not_used_insts.end(),
                    std::back_inserter(deps), deps_count, gen);
        // remove deps from not_used_insts
        for (auto& dep : deps) {
          not_used_insts.erase(dep);
        }
      } else {
        std::sample(insts_list.begin(), insts_list.end(),
                    std::back_inserter(deps), deps_count, gen);
      }

      if (deps.size() != 2) {
        return absl::InvalidArgumentError("deps size not equal 2");
      }

      auto inst = builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, deps.at(0), deps.at(1)));
      // we can add control dependency to inst
      // random add control dep, test control_predecessors issue
      if (random_gen() > 60) {
        std::vector<HloInstruction*> control_deps;
        std::sample(insts_list.begin(), insts_list.end(),
                    std::back_inserter(control_deps), 2, gen);
        for (auto control_dep : control_deps) {
          auto status = control_dep->AddControlDependencyTo(inst);
          if (!status.ok()) {
            return absl::InvalidArgumentError("AddControlDependencyTo error");
          }
        }
      }

      insts_list.push_back(inst);
      insts2cost.push_back(std::make_tuple(inst, cost_gen()));
      not_used_insts.insert(inst);

      if (i % insert_comm_every == 0) {
        uint8_t comm_deps_count = 2;
        // from 0~i, pick deps_count insts
        std::vector<HloInstruction*> comm_deps;
        if (not_used_insts.size() >= comm_deps_count) {
          // first pick not used insts
          std::sample(not_used_insts.begin(), not_used_insts.end(),
                      std::back_inserter(comm_deps), comm_deps_count, gen);
          // remove deps from not_used_insts
          for (auto& dep : comm_deps) {
            not_used_insts.erase(dep);
          }
        } else {
          // pick from all insts
          std::sample(insts_list.begin(), insts_list.end(),
                      std::back_inserter(comm_deps), comm_deps_count, gen);
        }
        if (comm_deps.size() != comm_deps_count) {
          return absl::InvalidArgumentError("comm_deps size not equal 2");
        }
        for (auto& dep : comm_deps) {
          auto all_reduce_start =
              builder.AddInstruction(HloInstruction::CreateAllReduceStart(
                  shape, {dep}, MakeReduction(HloOpcode::kAdd, module),
                  /*replica_groups=*/CreateReplicaGroups({{0, 1}}),
                  /*constrain_layout=*/false, /*channel_id=*/std::nullopt,
                  /*use_global_device_ids=*/false));
          // estimator->SetInstructionCost(all_reduce_start, 1);
          insts2cost.push_back(std::make_tuple(all_reduce_start, 1));
          auto ar_done = builder.AddInstruction(HloInstruction::CreateUnary(
              shape, HloOpcode::kAllReduceDone, all_reduce_start));
          // estimator->SetInstructionCost(ar_done, 1);
          insts2cost.push_back(std::make_tuple(ar_done, 1));

          insts_list.push_back(ar_done);
          edge2cost.push_back(std::make_tuple(ar_done, cost_gen() + 50));
          not_used_insts.insert(ar_done);
        }
      }
    }
    // get no use insts,let them sum and return,avoid graph optimizer delete
    // them

    auto reduce_sum_func = [&](HloInstruction* left, HloInstruction* right) {
      return builder.AddInstruction(
          HloInstruction::CreateBinary(shape, HloOpcode::kAdd, left, right));
    };
    auto sum_op = std::reduce(
        not_used_insts.begin(), not_used_insts.end(),
        builder.AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f))),
        reduce_sum_func);

    std::vector<HloInstruction*> last_n_insts;
    uint32_t last_n = 3;
    for (size_t i = 0; i < last_n; i++) {
      last_n_insts.push_back(insts_list.at(inst_nums - 1 - i));
    }
    last_n_insts.push_back(sum_op);
    auto last_ret = builder.AddInstruction(
        HloInstruction::CreateTuple(absl::MakeSpan(last_n_insts)));
    estimator->SetInstructionCost(last_ret, 0);
    auto computation = builder.Build();
    VLOG(2) << "create computation success,test name" << TestName();
    computation->set_root_instruction(last_ret);
    // Node cost must after AddEntryComputation,so that instruction have
    // unique_id
    auto computation_ptr = module->AddEntryComputation(std::move(computation));
    for (auto& inst_cost : insts2cost) {
      estimator->SetInstructionCost(std::get<0>(inst_cost),
                                    std::get<1>(inst_cost));
    }
    for (auto& edge_cost : edge2cost) {
      estimator->SetInstructionBetween(std::get<0>(edge_cost),
                                       std::get<1>(edge_cost));
    }
    // let module have one schedule
    TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                        ScheduleModule(module, [](const BufferValue& buffer) {
                          return ShapeUtil::ByteSizeOf(
                              buffer.shape(),
                              /*pointer_size=*/sizeof(void*));
                        }));

    TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
    VLOG(2) << "setting default schedule finish." << TestName();

    return computation_ptr;
  }
  StatusOr<ScheduleStatus> RunLatencyHidingScheduler(
      HloModule* module, SchedulerConfig sched_config = GetDefaultSchedConfig(),
      std::unique_ptr<LatencyEstimator> latency_estimator =
          std::make_unique<ApproximateLatencyEstimator>()) {
    AsyncCollectiveCreator::CollectiveCreatorConfig config{
        /*convert_all_reduce=*/HloPredicateTrue,
        /*convert_all_gather=*/HloPredicateTrue,
        /*convert_collective_permute=*/HloPredicateTrue};
    TF_ASSIGN_OR_RETURN(bool value,
                        AsyncCollectiveCreator(std::move(config)).Run(module));
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes =
        [&shape_size_bytes](const Shape& shape) -> int64_t {
      int64_t shape_size = 0;
      if (shape.IsTuple()) {
        for (auto& sub_shape : shape.tuple_shapes()) {
          shape_size += shape_size_bytes(sub_shape);
        }
        return shape_size;
      }
      return ShapeUtil::ByteSizeOfElements(shape);
    };
    auto async_tracker = std::make_unique<AsyncTracker>(sched_config);
    auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
        shape_size_bytes, async_tracker.get(), latency_estimator.get(),
        sched_config);
    TF_ASSIGN_OR_RETURN(
        bool is_success,
        LatencyHidingScheduler(std::move(latency_estimator),
                               std::move(async_tracker),
                               std::move(scheduler_core), shape_size_bytes)
            .Run(module));

    return ScheduleStatus{is_success, 0, 0};
  }
  StatusOr<ScheduleStatus> RunScheduler(
      HloModule* module, SchedulerConfig sched_config = GetDefaultSchedConfig(),
      std::unique_ptr<LatencyEstimator> latency_estimator =
          std::make_unique<ApproximateLatencyEstimator>()) {
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes =
        [&shape_size_bytes](const Shape& shape) -> int64_t {
      int64_t shape_size = 0;
      if (shape.IsTuple()) {
        for (auto& sub_shape : shape.tuple_shapes()) {
          shape_size += shape_size_bytes(sub_shape);
        }
        return shape_size;
      }
      return ShapeUtil::ByteSizeOfElements(shape);
    };
    auto async_tracker = std::make_unique<AsyncTracker>(sched_config);
    auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
        shape_size_bytes, async_tracker.get(), latency_estimator.get(),
        sched_config);
    auto test_pass =
        AutoReorderPass(std::move(latency_estimator), std::move(async_tracker),
                        std::move(scheduler_core), shape_size_bytes);
    TF_ASSIGN_OR_RETURN(bool is_success, test_pass.Run(module));

    return ScheduleStatus{is_success, 0, 0};
  }
  StatusOr<std::unique_ptr<HloModule>> ParseHloText(
      absl::string_view hlo_string) {
    TF_ASSIGN_OR_RETURN(
        auto hlo_module,
        ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest()));
    return StatusOr<std::unique_ptr<HloModule>>(std::move(hlo_module));
  }
  tsl::Status RebuildHloOrdering(HloSchedule& module_schedule,
                                 HloComputation* entry_computation) {
    bool is_debug = false;
    // module_schedule.remove_computation(entry_computation);
    // module_schedule.GetOrCreateSequence(entry_computation);
    auto status = module_schedule.UpdateComputationSchedule(entry_computation);
    debug_log("UpdateComputationSchedule");

    if (!status.ok()) {
      debug_log("UpdateComputationSchedule error:" << status.message());
      return status;
    } else {
      debug_log(
          "UpdateComputationSchedule success:"
          << module_schedule.sequence(entry_computation).instructions().size());
    }
    status = module_schedule.Update({});
    if (!status.ok()) {
      std::cout << "Update error:" << status.message() << std::endl;
      return status;
    }
    // SequentialHloOrdering seq_ordering(module_schedule);
    // auto seqs = seq_ordering.SequentialOrder(*entry_computation);
    // module_schedule.set_sequence(entry_computation, *seqs);
    // debug_log("seqs length" << seqs.size());

    auto new_instruction_sequence =
        module_schedule.sequence(entry_computation).instructions();
    debug_log("new_instruction_sequence length"
              << new_instruction_sequence.size());
    for (auto i = 0; i < new_instruction_sequence.size(); i++) {
      auto inst = new_instruction_sequence.at(i);
      debug_log("rebuild idx=" << i << "=" << inst->ToString());
    }
    status = module_schedule.Verify();
    if (!status.ok()) {
      debug_log("Verify error:" << status.message());
      return status;
    } else {
      debug_log(
          "Verify success:"
          << module_schedule.sequence(entry_computation).instructions().size());
    }
  }

  void MoveInstruction(HloComputation* src_computation,
                       absl::string_view src_name,
                       HloComputation* dst_computation) {
    bool is_debug = true;

    // Move instruction from src_computation to dst_computation.
    auto src_instruction = src_computation->GetInstructionWithName(src_name);
    // step 1: found src_instruction input args and output args
    std::vector<HloInstruction*>
        src_inputs;  // instruction which outputs is needed by src_instruction
    std::vector<HloInstruction*>
        src_outputs;  // instruction which input is src_instruction's output
    for (auto i = 0; i < src_instruction->operand_count(); i++) {
      auto src_input = src_instruction->mutable_operand(i);
      src_inputs.push_back(src_input);
    }
    std::vector<xla::HloInstruction*> user_insts = src_instruction->users();
    for (auto i = 0; i < src_instruction->user_count(); i++) {
      src_outputs.push_back(user_insts.at(i));
    }
    // step 2: create Send Instruction for input args, create Recv Instruction
    // for output args
    int64_t channel_id = 0;
    std::vector<HloInstruction*> dst_inputs;
    std::vector<HloInstruction*> send_params;
    dst_inputs.reserve(src_inputs.size());
    send_params.reserve(src_inputs.size());
    for (size_t i = 0; i < src_inputs.size(); i++) {
      channel_id++;
      auto src_input = src_inputs.at(i);
      auto src_input_shape = src_input->shape();
      // src_instruction
      auto token =
          src_computation->AddInstruction(HloInstruction::CreateToken());

      auto send_inst =
          src_computation->AddInstruction(HloInstruction::CreateSend(
              src_input, token, channel_id, false /*is_host_transfer*/));
      auto send_done = src_computation->AddInstruction(
          HloInstruction::CreateSendDone(send_inst));
      token = dst_computation->AddInstruction(HloInstruction::CreateToken());
      auto recv_inst = dst_computation->AddInstruction(
          HloInstruction::CreateRecv(src_input_shape, token, channel_id,
                                     false /*is_host_transfer*/),
          "dst_recv" + std::to_string(i));
      auto recv_done = dst_computation->AddInstruction(
          HloInstruction::CreateRecvDone(recv_inst));
      HloInstruction* recv_parameter = dst_computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(recv_done, 0));

      dst_inputs.push_back(recv_parameter);
    }
    channel_id++;
    // step3: clone same instruction to dst_computation
    auto dst_inst =
        dst_computation->AddInstruction(src_instruction->CloneWithNewOperands(
            src_instruction->shape(), dst_inputs));

    // step4 :create Send Instruction from dst_compuation, create Recv
    // Instruction in src_computation
    auto token = dst_computation->AddInstruction(HloInstruction::CreateToken());

    auto ret_send_inst =
        dst_computation->AddInstruction(HloInstruction::CreateSend(
            dst_inst, token, channel_id, false /*is_host_transfer*/));
    auto send_done = dst_computation->AddInstruction(
        HloInstruction::CreateSendDone(ret_send_inst));

    // create recv in src_computation, create token node,so recv_inst will be
    // executed by scheduler
    token = src_computation->AddInstruction(HloInstruction::CreateToken());

    auto recv_inst = src_computation->AddInstruction(
        HloInstruction::CreateRecv(dst_inst->shape(), token, channel_id,
                                   false /*is_host_transfer*/),
        "src_recv_ret");
    auto recv_done = src_computation->AddInstruction(
        HloInstruction::CreateRecvDone(recv_inst));
    HloInstruction* recv_parameter = src_computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(recv_done, 0));

    // step5: replace instruction which use src_instruction's output with Recv
    // Instruction
    for (size_t i = 0; i < src_outputs.size(); i++) {
      /* code */
      auto src_output = src_outputs.at(i);
      // add dependency
      auto status = src_instruction->ReplaceUseWith(src_output, recv_parameter);
      if (!status.ok()) {
        std::cout << "ReplaceUseWith error:" << status.message() << std::endl;
      }
      absl::flat_hash_map<int, HloInstruction*> new_instruction_uses;
      int operand_num = 0;
      for (const HloInstruction* operand : src_output->operands()) {
        if (operand->unique_id() == src_instruction->unique_id()) {
          new_instruction_uses[operand_num] = recv_parameter;
        }
        operand_num++;
      }
      for (auto it = new_instruction_uses.begin();
           it != new_instruction_uses.end(); ++it) {
        status = src_output->ReplaceOperandWith(it->first, it->second);
        if (!status.ok()) {
          std::cout << "ReplaceOperandWith error:" << status.message()
                    << std::endl;
        }
      }
    }
    // step6: remove src_instruction
    src_instruction->DetachFromOperandsAndUsers();
    auto status = src_computation->RemoveInstruction(src_instruction);
    if (!status.ok()) {
      std::cout << "RemoveInstruction error:" << status.message() << std::endl;
    } else {
      std::cout << "RemoveInstruction success"
                << src_computation->instruction_count() << std::endl;
    }
  }
};
TEST_F(AutoReorderingTest, CommOpCostTest){
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction = [&](const Shape& shape) {
    constexpr int64_t kPointerSize = 8;
    return ShapeUtil::ByteSizeOf(shape, kPointerSize);
  };
  

  HloCostAnalysis::Options options_{ShapeSizeBytesFunction,
                                    /*per_second_rates=*/{},
                                    /*count_multiple_input_accesses=*/true};
  GpuHloCostAnalysis analysis_(options_);

  auto hlo_module = CreateNewVerifiedModule(TestName(),/*replica_count*/2);
  auto st = MakeTestComputation(hlo_module.get());
  HloComputation* comp = st.value();
  ASSERT_IS_OK(hlo_module->entry_computation()->Accept(&analysis_));
  
  const HloModuleConfig& config = hlo_module->config();
  int64_t num_devices = config.num_partitions();
  int64_t num_replicas = config.replica_count();
  const HloInstruction* all_reduce_start = comp->GetInstructionWithName("all_reduce_start");

  // CollectiveOpGroupMode group_mode = 
  //     analysis_.GetCollectiveOpGroupMode(
  //         all_reduce_start->channel_id().has_value(),
  //         Cast<HloAllReduceInstruction>(all_reduce_start)->use_global_device_ids()));

  EXPECT_EQ(analysis_.NumOfDevices(*all_reduce_start),2);
};
TEST_F(AutoReorderingTest, SPMDAutoReorder) {
  // GTEST_SKIP()<<"new version xla can'parse this old hlo";
  absl::string_view hlo_string = R"(
HloModule SyncTensorsGraph.23, is_scheduled=true, entry_computation_layout={(f32[9600]{0}, f32[2400,12800]{1,0}, f32[400,12800]{1,0}, f32[320,9600]{1,0}, f32[1280]{0})->(f32[400,9600]{1,0}, f32[9600,320]{1,0}, f32[400,1280]{1,0})}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=4, frontend_attributes={fingerprint_before_lhs="45ccf07b9a0d113ede565121a6790508"}

fused_add {
  param_0 = f32[400,9600]{1,0} parameter(0)
  param_1.1 = f32[9600]{0} parameter(1)
  broadcast.6.1 = f32[400,9600]{1,0} broadcast(param_1.1), dimensions={1}, metadata={op_type="aten__addmm" op_name="aten__addmm"}
  ROOT add.4.1 = f32[400,9600]{1,0} add(param_0, broadcast.6.1), metadata={op_type="aten__addmm" op_name="aten__addmm"}
}

fused_add.1 {
  param_0.1 = f32[400,1280]{1,0} parameter(0)
  param_1.3 = f32[1280]{0} parameter(1)
  broadcast.8.1 = f32[400,1280]{1,0} broadcast(param_1.3), dimensions={1}, metadata={op_type="aten__addmm" op_name="aten__addmm"}
  ROOT add.5.1 = f32[400,1280]{1,0} add(param_0.1, broadcast.8.1), metadata={op_type="aten__addmm" op_name="aten__addmm"}
}

wrapped_transpose_computation {
  param_0.2 = f32[320,9600]{1,0} parameter(0)
  ROOT transpose.4.1 = f32[9600,320]{1,0} transpose(param_0.2), dimensions={1,0}
}

ENTRY SyncTensorsGraph.23_spmd {
  param.5 = f32[400,12800]{1,0} parameter(2), sharding={devices=[4,1]0,1,2,3}, metadata={op_type="xla__device_data" op_name="xla__device_data"}
  param.1.0 = f32[2400,12800]{1,0} parameter(1), sharding={devices=[4,1]0,1,2,3}, metadata={op_type="xla__device_data" op_name="xla__device_data"}
  param.2.0 = f32[9600]{0} parameter(0), sharding={replicated}, metadata={op_type="xla__device_data" op_name="xla__device_data"}
  param.3.0 = f32[320,9600]{1,0} parameter(3), sharding={devices=[4,1]0,1,2,3}, metadata={op_type="xla__device_data" op_name="xla__device_data"}
  param.4.0 = f32[1280]{0} parameter(4), sharding={replicated}, metadata={op_type="xla__device_data" op_name="xla__device_data"}
  bitcast.44.0 = f32[12800,2400]{0,1} bitcast(param.1.0)
  wrapped_transpose = f32[9600,320]{1,0} fusion(param.3.0), kind=kInput, calls=wrapped_transpose_computation
  bitcast.61.0 = f32[9600,320]{0,1} bitcast(param.3.0)
  all-gather-start = (f32[12800,2400]{0,1}, f32[12800,9600]{0,1}) all-gather-start(bitcast.44.0), channel_id=1, replica_groups={{0,1,2,3}}, dimensions={1}, use_global_device_ids=true, metadata={op_type="aten__addmm" op_name="aten__addmm"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":false,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  all-gather-done = f32[12800,9600]{0,1} all-gather-done(all-gather-start), metadata={op_type="aten__addmm" op_name="aten__addmm"}
  all-gather-start.1 = (f32[9600,320]{0,1}, f32[9600,1280]{0,1}) all-gather-start(bitcast.61.0), channel_id=2, replica_groups={{0,1,2,3}}, dimensions={1}, use_global_device_ids=true, metadata={op_type="aten__addmm" op_name="aten__addmm"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":false,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  all-gather-done.1 = f32[9600,1280]{0,1} all-gather-done(all-gather-start.1), metadata={op_type="aten__addmm" op_name="aten__addmm"}
  custom-call.2.0 = (f32[400,9600]{1,0}, s8[4194304]{0}) custom-call(param.5, all-gather-done), custom_call_target="__cublas$gemm", metadata={op_type="aten__addmm" op_name="aten__addmm"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","lhs_stride":"5120000","rhs_stride":"122880000","grad_x":false,"grad_y":false},"force_earliest_schedule":false}
  get-tuple-element.3.0 = f32[400,9600]{1,0} get-tuple-element(custom-call.2.0), index=0, metadata={op_type="aten__addmm" op_name="aten__addmm"}
  loop_add_fusion = f32[400,9600]{1,0} fusion(get-tuple-element.3.0, param.2.0), kind=kLoop, calls=fused_add, metadata={op_type="aten__addmm" op_name="aten__addmm"}
  custom-call.3.0 = (f32[400,1280]{1,0}, s8[4194304]{0}) custom-call(loop_add_fusion, all-gather-done.1), custom_call_target="__cublas$gemm", metadata={op_type="aten__addmm" op_name="aten__addmm"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","lhs_stride":"3840000","rhs_stride":"12288000","grad_x":false,"grad_y":false},"force_earliest_schedule":false}
  get-tuple-element.4.0 = f32[400,1280]{1,0} get-tuple-element(custom-call.3.0), index=0, metadata={op_type="aten__addmm" op_name="aten__addmm"}
  loop_add_fusion.1 = f32[400,1280]{1,0} fusion(get-tuple-element.4.0, param.4.0), kind=kLoop, calls=fused_add.1, metadata={op_type="aten__addmm" op_name="aten__addmm"}
  ROOT tuple.1.0 = (f32[400,9600]{1,0}, f32[9600,320]{1,0}, f32[400,1280]{1,0}) tuple(loop_add_fusion, wrapped_transpose, loop_add_fusion.1)
} // SyncTensorsGraph.23_spmd
)";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  auto gpu_latency_estimator = std::make_unique<GpuLatencyEstimator>();
  std::unique_ptr<LatencyEstimator> latency_estimator;
  int pointer_size_ = 4;
  Backend& test_backend = backend();
  auto gpu_device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  VLOG(2) << "threads_per_block_limit:"
          << gpu_device_info.threads_per_block_limit() << " threads_per_warp"
          << gpu_device_info.threads_per_warp();
  const int64_t scheduler_mem_limit = xla::gpu::GetSchedulerMemoryLimit(
      hlo_module.get(), gpu_device_info, pointer_size_);
  SchedulerConfig config = GetSchedulerConfig(scheduler_mem_limit);
  SchedulerConfig sched_config = GetDefaultSchedConfig();
  HloCostAnalysis::ShapeSizeFunction shape_size_bytes =
      [&shape_size_bytes](const Shape& shape) -> int64_t {
    int64_t shape_size = 0;
    if (shape.IsTuple()) {
      for (auto& sub_shape : shape.tuple_shapes()) {
        shape_size += shape_size_bytes(sub_shape);
      }
      return shape_size;
    }
    return ShapeUtil::ByteSizeOfElements(shape);
  };

  auto async_tracker = std::make_unique<AsyncTracker>(sched_config);
  latency_estimator = std::make_unique<AnalyticalLatencyEstimator>(
      config, std::move(gpu_latency_estimator), gpu_device_info,
      [input_pointer_size = pointer_size_](const Shape& shape) {
        return GetSizeOfShape(shape, input_pointer_size);
      },
      hlo_module->entry_computation());
  auto entry_computation = hlo_module->entry_computation();
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_bytes, async_tracker.get(), latency_estimator.get(),
      sched_config);
  auto test_pass =
      AutoReorderPass(std::move(latency_estimator), std::move(async_tracker),
                      std::move(scheduler_core), shape_size_bytes);

  for (HloComputation* computation :
       hlo_module->MakeNonfusionComputations({})) {
    auto status = test_pass.ScheduleComputation(computation);
    if (!status.ok()) {
      std::cout << "NonfusionComputations src_module fail" << std::endl;
    }
    EXPECT_TRUE(status.ok());
  }
}
TEST_F(AutoReorderingTest, DemoAutoReorder) {
  GTEST_SKIP() << "Skipping DemoAutoReorder";

  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

%add {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %a = f32[] add(p0, p1)
}

ENTRY %module {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[2,8,256,256]{3,2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[2,8,256,256]{3,2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %ar-start = f32[2,8,256,256] all-reduce-start(
    f32[2,8,256,256] %color_operand.1), replica_groups={{0,1}}, to_apply=%add,
    metadata={op_type="AllReduce" op_name="ar0"}
  %ar-start.2 = f32[2,8,256,256] all-reduce-start(
    f32[2,8,256,256] %color_operand.2), replica_groups={{0,1}}, to_apply=%add,
    metadata={op_type="AllReduce" op_name="ar1"}
  %ar-done = f32[2,8,256,256] all-reduce-done(
    f32[2,8,256,256] %ar-start),
    metadata={op_type="AllReduce" op_name="ar0"}
  %ar-done-bc = f32[16,256,256] bitcast(f32[2,8,256,256] %ar-done),
    metadata={op_type="Bitcast" op_name="ar0"}
  %ar-done.2 = f32[2,8,256,256] all-reduce-done(
    f32[2,8,256,256] %ar-start.2),
    metadata={op_type="AllReduce" op_name="ar1"}
  %ar-done-bc.2 = f32[16,256,256] bitcast(f32[2,8,256,256] %ar-done.2),
    metadata={op_type="Bitcast" op_name="ar1"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllReduce" op_name="c0"}
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllReduce" op_name="c1"}
  a2 = f32[16,256,256]{2,1,0} add(c1, c0)
  a3 = f32[16,256,256]{2,1,0} add(a2, c0)
  a4 = f32[16,256,256]{2,1,0} add(a3, c1)
  ROOT t = (f32[16,256,256], f32[16,256,256], f32[16,256,256]) tuple(a4, %ar-done-bc.2, %ar-done-bc)
}
)";
  absl::string_view hlo_string_cpu = R"(
HloModule module, is_scheduled=true

%add {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %a = f32[] add(p0, p1)
}

ENTRY %module {
  p2 = f32[16,256,256]{2,1,0} parameter(0)
  p3 = f32[16,256,256]{2,1,0} parameter(1)
  c0 = f32[16,256,256]{2,1,0} multiply(p2, p3)
  a2 = f32[16,256,256]{2,1,0} add(p2, p3)
  a3 = f32[16,256,256]{2,1,0} add(a2, c0)
  ROOT t = (f32[16,256,256]) tuple(a3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module_cpu, ParseHloText(hlo_string_cpu));
  // VLOG(10) << module->ToString();
  auto* instruction = FindInstruction(hlo_module.get(), "param0");
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  HloComputation* entry_computation_cpu = hlo_module_cpu->entry_computation();

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  for (auto i = 0; i < new_instruction_sequence.size(); i++) {
    auto inst = new_instruction_sequence.at(i);
    std::cout << "idx=" << i << "=" << inst->ToString() << std::endl;
  }
  // test create H2D
  // entry_computation
  // idx=17=%a2 = f32[16,256,256]{2,1,0} add(f32[16,256,256]{2,1,0} %c1,
  // f32[16,256,256]{2,1,0} %c0)
  auto inst_a2 = new_instruction_sequence.at(17);
  std::cout << "Before Move,there are:" << new_instruction_sequence.size()
            << " instructions" << std::endl;
  std::cout << "Before Move,there are:"
            << hlo_module_cpu->schedule()
                   .sequence(hlo_module_cpu->entry_computation())
                   .instructions()
                   .size()
            << " instructions" << std::endl;
  auto test_pass = AutoReorderPass();
  auto status = test_pass.MoveInstruction(hlo_module->entry_computation(), "a3",
                                          hlo_module_cpu->entry_computation());
  if (!status.ok()) {
    std::cout << "MoveInstruction src_module fail" << status.message()
              << std::endl;
    EXPECT_TRUE(status.ok());
  }
  std::cout << "after Move" << std::endl;
  status =
      test_pass.RebuildHloOrdering(hlo_module->schedule(), entry_computation);
  if (!status.ok()) {
    std::cout << "RebuildHloOrdering src_module fail" << status.message()
              << std::endl;
    EXPECT_TRUE(status.ok());
  }
  // std::cout << "after rebuild ordering src module="<< std::endl;
  // new_instruction_sequence =
  //     hlo_module->schedule().sequence(entry_computation).instructions();
  // for (auto i = 0; i < new_instruction_sequence.size(); i++) {
  //   auto inst = new_instruction_sequence.at(i);
  //   std::cout << "idx=" << i << "=" << inst->ToString() << std::endl;
  // }

  status = test_pass.RebuildHloOrdering(hlo_module_cpu->schedule(),
                                        entry_computation_cpu);

  if (!status.ok()) {
    std::cout << "RebuildHloOrdering hlo_module_cpu fail" << status.message()
              << std::endl;
    EXPECT_TRUE(status.ok());
  }
}
TEST_F(AutoReorderingTest, ReorderScheduleComputation) {
  auto hlo_module = CreateNewUnverifiedModule();
  auto st = MakeTestComputation(hlo_module.get());
  EXPECT_TRUE(st.ok());
  auto gpu_latency_estimator = std::make_unique<GpuLatencyEstimator>();
  std::unique_ptr<LatencyEstimator> latency_estimator;
  int pointer_size_ = 4;
  Backend& test_backend = backend();
  auto gpu_device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  VLOG(2) << "threads_per_block_limit:"
          << gpu_device_info.threads_per_block_limit() << " threads_per_warp"
          << gpu_device_info.threads_per_warp();
  const int64_t scheduler_mem_limit = xla::gpu::GetSchedulerMemoryLimit(
      hlo_module.get(), gpu_device_info, pointer_size_);
  SchedulerConfig config = GetSchedulerConfig(scheduler_mem_limit);
  SchedulerConfig sched_config = GetDefaultSchedConfig();
  HloCostAnalysis::ShapeSizeFunction shape_size_bytes =
      [&shape_size_bytes](const Shape& shape) -> int64_t {
    int64_t shape_size = 0;
    if (shape.IsTuple()) {
      for (auto& sub_shape : shape.tuple_shapes()) {
        shape_size += shape_size_bytes(sub_shape);
      }
      return shape_size;
    }
    return ShapeUtil::ByteSizeOfElements(shape);
  };

  auto async_tracker = std::make_unique<AsyncTracker>(sched_config);
  latency_estimator = std::make_unique<AnalyticalLatencyEstimator>(
      config, std::move(gpu_latency_estimator), gpu_device_info,
      [input_pointer_size = pointer_size_](const Shape& shape) {
        return GetSizeOfShape(shape, input_pointer_size);
      },
      hlo_module->entry_computation());
  auto entry_computation = hlo_module->entry_computation();
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_bytes, async_tracker.get(), latency_estimator.get(),
      sched_config);
  auto test_pass =
      AutoReorderPass(std::move(latency_estimator), std::move(async_tracker),
                      std::move(scheduler_core), shape_size_bytes);

  for (HloComputation* computation :
       hlo_module->MakeNonfusionComputations({})) {
    auto status = test_pass.ScheduleComputation(computation);
    if (!status.ok()) {
      std::cout << "NonfusionComputations src_module fail" << std::endl;
    }
    EXPECT_TRUE(status.ok());
  }
}
TEST_F(AutoReorderingTest, ReorderPass) {
  auto hlo_module = CreateNewUnverifiedModule();
  auto st = MakeTestComputation(hlo_module.get());
  EXPECT_TRUE(st.ok());
  int pointer_size_ = 4;
  Backend& test_backend = backend();
  auto gpu_device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  const int64_t scheduler_mem_limit = xla::gpu::GetSchedulerMemoryLimit(
      hlo_module.get(), gpu_device_info, pointer_size_);
  SchedulerConfig config = GetSchedulerConfig(scheduler_mem_limit);
  auto gpu_latency_estimator = std::make_unique<GpuLatencyEstimator>();
  std::unique_ptr<LatencyEstimator> latency_estimator =
      std::make_unique<AnalyticalLatencyEstimator>(
          config, std::move(gpu_latency_estimator), gpu_device_info,
          [input_pointer_size = pointer_size_](const Shape& shape) {
            return GetSizeOfShape(shape, input_pointer_size);
          },
          hlo_module->entry_computation());
  // we should create other estimator,otherwise it's nullptr(move to other
  // place)
  auto gpu_latency_estimator2 = std::make_unique<GpuLatencyEstimator>();

  auto latency_estimator2 = std::make_unique<AnalyticalLatencyEstimator>(
      config, std::move(gpu_latency_estimator2), gpu_device_info,
      [input_pointer_size = pointer_size_](const Shape& shape) {
        return GetSizeOfShape(shape, input_pointer_size);
      },
      hlo_module->entry_computation());
  SchedulerConfig sched_config = GetDefaultSchedConfig();
  auto status = RunScheduler(hlo_module.get(), sched_config,
                             std::move(latency_estimator));
  EXPECT_TRUE(status.ok());
  auto statics_or_status =
      GetModuleCost(hlo_module.get(), sched_config, latency_estimator2.get());

  EXPECT_TRUE(statics_or_status.ok());
  auto statics = statics_or_status.value();
  // statics.
};
TEST_F(AutoReorderingTest, ReorderPassWithDefaultEstimator) {
  auto hlo_module = CreateNewUnverifiedModule();
  auto st = MakeTestComputation(hlo_module.get());
  EXPECT_TRUE(st.ok());
  int pointer_size_ = 4;
  Backend& test_backend = backend();
  auto gpu_device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  const int64_t scheduler_mem_limit = xla::gpu::GetSchedulerMemoryLimit(
      hlo_module.get(), gpu_device_info, pointer_size_);
  SchedulerConfig config = GetSchedulerConfig(scheduler_mem_limit);
  auto gpu_latency_estimator = std::make_unique<GpuLatencyEstimator>();
  std::unique_ptr<LatencyEstimator> latency_estimator;
  latency_estimator = std::make_unique<AnalyticalLatencyEstimator>(
      config, std::move(gpu_latency_estimator), gpu_device_info,
      [input_pointer_size = pointer_size_](const Shape& shape) {
        return GetSizeOfShape(shape, input_pointer_size);
      },
      hlo_module->entry_computation());
  SchedulerConfig sched_config = GetDefaultSchedConfig();
  auto status = RunScheduler(hlo_module.get(), sched_config);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(CheckParameterInst(hlo_module.get()));
}
TEST_F(AutoReorderingTest, ReorderPassCommunicateComputation) {
  // GTEST_SKIP() << "Skipping single test";
  std::srand(kRandomSeed);
  auto hlo_module = CreateNewUnverifiedModule();
  auto gpu_latency_estimator = std::make_unique<SavedInstLatencyEstimator>();
  SchedulerConfig sched_config = GetDefaultSchedConfig();
  auto st =
      MakeCommunicateComputation(hlo_module.get(), gpu_latency_estimator.get());
  auto gpu_latency_estimator2 = gpu_latency_estimator->clone();
  auto gpu_latency_estimator3 = gpu_latency_estimator->clone();

  auto status = RunScheduler(hlo_module.get(), sched_config,
                             std::move(gpu_latency_estimator));
  EXPECT_TRUE(status.ok());
  auto insts = hlo_module->schedule()
                   .sequence(hlo_module->entry_computation())
                   .instructions();
  for (auto inst : insts) {
    std::cout << inst->ToString() << std::endl;
  }
}

TEST_F(AutoReorderingTest, ReorderPassStableOrder) {
  // GTEST_SKIP() << "Skipping single test";
  std::srand(kRandomSeed);
  auto hlo_module = CreateNewUnverifiedModule();
  auto gpu_latency_estimator = std::make_unique<SavedInstLatencyEstimator>();
  SchedulerConfig sched_config = GetDefaultSchedConfig();
  auto st = MakeRandomComputation(hlo_module.get(), gpu_latency_estimator.get(),
                                  /*inst num*/ 200,
                                  /*max deps*/ 5,
                                  /*communication rate*/ 0.2);
  auto gpu_latency_estimator2 = gpu_latency_estimator->clone();
  auto gpu_latency_estimator3 = gpu_latency_estimator->clone();

  auto status = RunScheduler(hlo_module.get(), sched_config,
                             std::move(gpu_latency_estimator));
  EXPECT_TRUE(status.ok());
  auto comm_op_1 = GetInstructionsOrderString(hlo_module.get());

  // get hlo_module instruction order,and compute a hash
  status = RunScheduler(hlo_module.get(), sched_config,
                        std::move(gpu_latency_estimator2));
  auto comm_op_2 = GetInstructionsOrderString(hlo_module.get());

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(comm_op_1, comm_op_2);
  EXPECT_TRUE(CheckParameterInst(hlo_module.get()));
}
TEST_F(AutoReorderingTest, ReorderPassWithRandom) {
  // GTEST_SKIP() << "Skipping single test";
  std::srand(kRandomSeed);
  auto hlo_module = CreateNewUnverifiedModule();
  auto gpu_latency_estimator = std::make_unique<SavedInstLatencyEstimator>();
  SchedulerConfig sched_config = GetDefaultSchedConfig();
  auto st = MakeRandomComputation(hlo_module.get(), gpu_latency_estimator.get(),
                                  /*inst num*/ 100,
                                  /*max deps*/ 5,
                                  /*communication rate*/ 0.2);
  // std::cout<<hlo_module->ToString()<<std::endl;
  EXPECT_TRUE(st.ok());
  VLOG(2) << "finish random computation make" << std::endl;
  auto gpu_latency_estimator2 = gpu_latency_estimator->clone();
  auto gpu_latency_estimator3 = gpu_latency_estimator->clone();
  // run AutoReorder for compare

  auto status = RunScheduler(hlo_module.get(), sched_config,
                             std::move(gpu_latency_estimator));
  EXPECT_TRUE(status.ok());

  auto statics = GetModuleCost(hlo_module.get(), sched_config,
                               gpu_latency_estimator2.get());
  EXPECT_TRUE(statics.ok());
  EXPECT_TRUE(CheckParameterInst(hlo_module.get()));
  auto auto_reorder_cost = statics.value().total_cycles;
  std::cout << "ReorderPassWithRandom:" << auto_reorder_cost << std::endl;

  // compare post order vs reorder
  auto post_insts_order =
      hlo_module->entry_computation()->MakeInstructionPostOrder();
  hlo_module->schedule().set_sequence(hlo_module->entry_computation(),
                                      post_insts_order);

  statics = GetModuleCost(hlo_module.get(), sched_config,
                          gpu_latency_estimator2.get());
  EXPECT_TRUE(statics.ok());
  auto post_order_cost = statics.value().total_cycles;
  EXPECT_TRUE(CheckParameterInst(hlo_module.get()));
  std::cout << "MakeInstructionPostOrder:" << post_order_cost << std::endl;

  // run LatencyHidingScheduler for compare
  // NOTICE:  DO NOT using gpu_latency_estimator after
  // std::move(gpu_latency_estimator)
  auto lhs_status = RunLatencyHidingScheduler(
      hlo_module.get(), sched_config, std::move(gpu_latency_estimator3));
  EXPECT_TRUE(lhs_status.ok());
  EXPECT_TRUE(CheckParameterInst(hlo_module.get()));
  statics = GetModuleCost(hlo_module.get(), sched_config,
                          gpu_latency_estimator2.get());
  EXPECT_TRUE(statics.ok());
  auto xla_lhs_cost = statics.value().total_cycles;
  EXPECT_LE(auto_reorder_cost, post_order_cost);
  EXPECT_LE(auto_reorder_cost, xla_lhs_cost);
}
// skip this test
TEST_F(AutoReorderingTest, ReorderPassDataAnalyse) {
  GTEST_SKIP() << "Skipping single test";
  std::srand(kRandomSeed);
  auto gen = std::mt19937{kRandomSeed};
  int repeat_time = 1;
  uint32_t nnodes = 100;
  std::vector<float> communication_rates;
  //  = {
  //   0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
  // };
  for (float current = 0.1; current < 0.9; current += 0.05) {
    communication_rates.push_back(current);
  }

  // communication rate from 0.05 to 0.95,step is 0.05
  std::ofstream csv_out("/tmp/test_ret.csv");
  csv_out
      << "exp_id,nnodes,communication_rate,auto_reorder_cost,post_order_cost,"
         "xla_hiding_order_cost,xla_hiding_solve_time,auto_reorder_solve_time"
      << std::endl;
  for (auto communication_rate : communication_rates) {
    for (size_t i = 0; i < repeat_time; i++) {
      std::cout << TestName() << " repeat time:" << i << std::endl;
      auto hlo_module = CreateNewUnverifiedModule();
      auto gpu_latency_estimator =
          std::make_unique<SavedInstLatencyEstimator>();
      // float communication_rate = 0.2;
      SchedulerConfig sched_config = GetDefaultSchedConfig();
      auto st =
          MakeRandomComputation(hlo_module.get(), gpu_latency_estimator.get(),
                                /*inst num*/ nnodes,
                                /*max deps*/ 5,
                                /*communication rate*/ communication_rate,
                                /* gen */ gen);
      EXPECT_TRUE(st.ok());
      // auto latency_estimator = create_latency_estimator();

      auto gpu_latency_estimator2 = gpu_latency_estimator->clone();
      auto gpu_latency_estimator3 = gpu_latency_estimator->clone();
      // run AutoReorder for compare
      // get running time cost
      auto start = std::chrono::steady_clock::now();
      auto status = RunScheduler(hlo_module.get(), sched_config,
                                 std::move(gpu_latency_estimator));
      EXPECT_TRUE(status.ok());
      auto end = std::chrono::steady_clock::now();
      auto auto_reorder_solve_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();

      auto statics = GetModuleCost(hlo_module.get(), sched_config,
                                   gpu_latency_estimator2.get());
      EXPECT_TRUE(statics.ok());
      auto auto_reorder_cost = statics.value().total_cycles;
      // compare post order vs reorder
      auto post_insts_order =
          hlo_module->entry_computation()->MakeInstructionPostOrder();
      hlo_module->schedule().set_sequence(hlo_module->entry_computation(),
                                          post_insts_order);

      statics = GetModuleCost(hlo_module.get(), sched_config,
                              gpu_latency_estimator2.get());
      EXPECT_TRUE(statics.ok());
      auto post_order_cost = statics.value().total_cycles;

      // run LatencyHidingScheduler for compare
      // NOTICE:  DO NOT using gpu_latency_estimator after
      // std::move(gpu_latency_estimator)
      start = std::chrono::steady_clock::now();
      auto lhs_status = RunLatencyHidingScheduler(
          hlo_module.get(), sched_config, std::move(gpu_latency_estimator3));
      EXPECT_TRUE(lhs_status.ok());
      end = std::chrono::steady_clock::now();
      auto xla_hiding_solve_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      statics = GetModuleCost(hlo_module.get(), sched_config,
                              gpu_latency_estimator2.get());
      EXPECT_TRUE(statics.ok());
      auto xla_hiding_order_cost = statics.value().total_cycles;
      csv_out << i << "," << nnodes << "," << communication_rate << ","
              << auto_reorder_cost << "," << post_order_cost << ","
              << xla_hiding_order_cost << "," << xla_hiding_solve_time << ","
              << auto_reorder_solve_time << std::endl;
    }
  }
}

}  // namespace xla

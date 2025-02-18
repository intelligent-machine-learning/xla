#include "xla/hlo/experimental/auto_reorder/auto_reorder.h"
namespace xla {
constexpr int64_t kPointerSize = 8;
// get shape byte size, f32 have 4 bytes;
int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

tsl::Status AutoReorderPass::RebuildHloOrdering(
    HloSchedule& module_schedule, HloComputation* entry_computation) {
  bool is_debug = false;
  // module_schedule.remove_computation(entry_computation);
  // module_schedule.GetOrCreateSequence(entry_computation);
  auto status = module_schedule.UpdateComputationSchedule(entry_computation);

  if (!status.ok()) {
    return status;
  } else {
  }
  status = module_schedule.Update({});
  if (!status.ok()) {
    VLOG(2) << "Update error:" << status.message() << std::endl;
    return status;
  }
  // SequentialHloOrdering seq_ordering(module_schedule);
  // auto seqs = seq_ordering.SequentialOrder(*entry_computation);
  // module_schedule.set_sequence(entry_computation, *seqs);

  auto new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  for (auto i = 0; i < new_instruction_sequence.size(); i++) {
    auto inst = new_instruction_sequence.at(i);
  }
  status = module_schedule.Verify();
  if (!status.ok()) {
    return status;
  }
  return absl::OkStatus();
}

int64_t AutoReorderPass::GetSpaceCost(const HloInstruction* instr) {
  // if instr is param, the space will keep on memory,so it's cost is shape
  // dont't consider buffer size
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
  int64_t space_cost = 0;
  //add output_size
  ShapeUtil::ForEachSubshape(instr->shape(), [&](const Shape& subshape,
                                                 const ShapeIndex& index) {
    // SpaceCost = output_size - input_size
    space_cost += shape_size_bytes(subshape);
  });
  return space_cost;
}

tsl::StatusOr<std::vector<HloInstruction*>>
AutoReorderPass::ScheduleComputation(HloComputation* computation) {
  int64_t current_pos = 0;
  auto post_order_instructions = computation->MakeInstructionPostOrder();
  HloScheduleGraph schedule_graph(&post_order_instructions,
                                  /*alias_analysis=*/nullptr,
                                  latency_estimator_.get(),
                                  async_tracker_.get());
  async_tracker_->PostProcessScheduleGraph(&schedule_graph,
                                           latency_estimator_.get());
  // we don't need InitializeGraphAnalysis for init node status;

  auto solver_ = absl::make_unique<LinearProgramScheduler<
      LPContainer<const HloInstruction*>, const HloInstruction*>>();
  std::vector<LPContainer<const HloInstruction*>*> comm_lp_nodes;

  // scan instructions, get every instruction cost and deps
  // post order,every inst will iter before it's operators
  for (HloInstruction* instr : post_order_instructions) {
    // AddHint

    const HloGraphNode& instr_node = schedule_graph.GetNode(instr);
    VLOG(2) << instr->ToShortString() << "flops cost :" << instr_node.GetCost();
    auto addEdge = [&](const xla::HloInstruction* from_inst,
                       LPContainer<const HloInstruction*>* dst_node,
                       NodeType edge_type) {
      auto operand_lp_node = solver_->FindInstructionLPNode(from_inst);
      if (!operand_lp_node.ok()) {
        VLOG(2) << "operand_lp_node not found:" << from_inst->ToShortString();
        return false;
      }
      //lp_node space_cost;
      
      auto operand_node = schedule_graph.GetNode(from_inst);
      CostType edge_cost =
          latency_estimator_->GetLatencyBetween(operand_node, instr_node);
      VLOG(2) << from_inst->ToShortString() + " should execute before " +
                     instr->ToShortString();

      dst_node->AddDep(operand_lp_node.value(), edge_cost, edge_type);

      return true;
    };

    CostType cost = std::ceil(instr_node.GetCost());
    // there are 3 type node: 1. compute 2. communication 3. copy
    // 3 type edges:  1. compute 2. communication 3. async
    if (async_tracker_->IsSupportedAsyncStart(*instr) ||
        async_tracker_->IsSupportedAsyncDone(*instr)) {
      // communication
      // GetCost return float, floor to int
      NodeType node_type;
      if (instr->IsAsynchronous()) {
        node_type = NodeType::kAsynchronous;
      } else {
        node_type = NodeType::kCommunication;
      }
      auto current_inst_lp_node =
          solver_->FindLPNodeOrCreate(instr, cost, node_type);
      // add current node as constraint
      current_inst_lp_node->SetSpaceCost(GetSpaceCost(instr));

      if (async_tracker_->IsSupportedAsyncDone(*instr)) {
        // create a edge, which is communication
        auto operand_inst = instr->operand(0);

        auto is_success = addEdge(operand_inst, current_inst_lp_node,
                                  GetEdgeTypeOfInst(*instr));
        TF_RET_CHECK(is_success)
            << "operand_lp_node not found:" << operand_inst->ToShortString();
      } else {
        // add it's operands to his deps
        for (auto i = 0; i < instr->operand_count(); i++) {
          auto operand_inst = instr->operand(i);
          auto is_success =
              addEdge(operand_inst, current_inst_lp_node, NodeType::kCompute);
          TF_RET_CHECK(is_success)
              << "operand_lp_node not found:" << operand_inst->ToShortString();
        }
        for (auto control_inst : instr->control_predecessors()) {
          // if it's communication, if control_inst is communicate op,this type
          // should be kCommunication?
          auto is_success = addEdge(control_inst, current_inst_lp_node,
                                    NodeType::kCompute);  // which type?
          TF_RET_CHECK(is_success)
              << "operand_lp_node not found:" << control_inst->ToShortString();
        }
      }

      TF_CHECK_OK(solver_->AddConstraint(current_inst_lp_node));
      if (reorder::is_keep_communicate_order()) {
        comm_lp_nodes.push_back(current_inst_lp_node);
      }
    } else {
      // TODO: copy_fusion need more work, analyse copy in fusion
      // NodeType node_type;
      // NodeType edge_type = NodeType::kCompute;
      // if(instr->opcode() == HloOpcode::kCopy){
      //   node_type=NodeType::kCopy;
      // }
      // else if(instr->opcode() == HloOpcode::kCopyStart || instr->opcode() ==
      // HloOpcode::kCopyDone){
      //   node_type=NodeType::kCopy;
      //   if(instr->opcode() == HloOpcode::kCopyDone){
      //     edge_type = NodeType::kCopy;
      //   }
      // }
      // else if(instr->opcode() == HloOpcode::kFusion){
      //   //only when fusion is all copy
      // }
      // else{
      //   node_type=NodeType::kCompute;
      // }
      auto current_inst_lp_node =
          solver_->FindLPNodeOrCreate(instr, cost, NodeType::kCompute);
      current_inst_lp_node->SetSpaceCost(GetSpaceCost(instr));
      // when adding edge node, current node have no add to Constraint?
      for (auto i = 0; i < instr->operand_count(); i++) {
        auto operand_inst = instr->operand(i);
        auto is_success =
            addEdge(operand_inst, current_inst_lp_node, NodeType::kCompute);
        TF_RET_CHECK(is_success)
            << "operand_lp_node not found:" << operand_inst->ToShortString();
      }
      for (auto control_inst : instr->control_predecessors()) {
        // if it's
        auto is_success = addEdge(control_inst, current_inst_lp_node,
                                  NodeType::kCompute);  // which type?
        TF_RET_CHECK(is_success)
            << "operand_lp_node not found:" << control_inst->ToShortString();
      }

      TF_CHECK_OK(solver_->AddConstraint(current_inst_lp_node));
    }
  }

  // set hint, using post order
  std::reverse(post_order_instructions.begin(), post_order_instructions.end());
  // for debug,show which type cost
  for (HloInstruction* instr : post_order_instructions) {
    auto lp_node = solver_->FindInstructionLPNode(instr);
    if (!lp_node.ok()) {
      VLOG(2) << "operand_lp_node not found:" << instr->ToShortString();
      continue;
    }
    auto operand_lp_node = lp_node.value();
    operand_lp_node->SetOpcode(instr->opcode());
    CostType start_at = -1;
    for (auto dep_pair : operand_lp_node->GetDeps()) {
      CostType cost = std::get<1>(dep_pair);
      auto dep_node = std::get<0>(dep_pair);
      if (dep_node->GetHintStart() > -1) {
        start_at = std::max(start_at, dep_node->GetHintStart() + cost);
      }
    }
    if (start_at > -1) {
      operand_lp_node->SetHintStart(start_at);
    }
  }
  if (reorder::solve_debug) {
    // save to pid related file
    solver_->SaveGraphviz(absl::StrCat("gantt_before_", computation->name()));
    solver_->SaveJSON(absl::StrCat("gantt_before_", computation->name()));
  }
  auto solve_method = xla::reorder::get_solve_method();
  auto status = solver_->Solve(solve_method);
  if (reorder::solve_debug) {
    // save to pid related file
    solver_->SaveGantt(absl::StrCat("gantt_",solve_method, computation->name()));
    solver_->SaveGraphviz(absl::StrCat("gantt_",solve_method, computation->name()));
  }

  if (status.ok()) {
    // return instruction order by solver
    std::vector<HloInstruction*> new_schedule_params;
    std::vector<HloInstruction*> new_schedule;
    auto sorted_nodes = solver_->GetSortedNodes();
    SaveOrder(sorted_nodes, absl::StrCat("origin_order_", computation->name()));

    std::vector<LPContainer<const HloInstruction*>*> finetune_ordered_queue =
        FunetuneNodeOrder(sorted_nodes);
    SaveOrder(finetune_ordered_queue,
              absl::StrCat("finetune_order_", computation->name()));
    for (auto node : finetune_ordered_queue) {
      auto insts = node->GetValues();
      for (auto inst : insts) {
        // extra check: param inst must move to head;
        if (inst->opcode() == HloOpcode::kParameter) {
          new_schedule_params.insert(new_schedule_params.begin(),
                                     const_cast<xla::HloInstruction*>(inst));
        } else {
          new_schedule.push_back(const_cast<xla::HloInstruction*>(inst));
        }
      }
    }
    std::sort(new_schedule_params.begin(), new_schedule_params.end(),
              [](const HloInstruction* a, const HloInstruction* b) {
                return a->unique_id() < b->unique_id();
              });
    new_schedule_params.insert(new_schedule_params.end(), new_schedule.begin(),
                               new_schedule.end());
    return new_schedule_params;
  }
  TF_RET_CHECK(status.ok()) << "Solver error:" << status.message();
  return status;
}
bool AutoReorderPass::NodeHaveDep(
    LPContainer<const HloInstruction*>* node,
    LPContainer<const HloInstruction*>* maybe_deps_node) {
  // node's deps have left
  for (auto dep_pair : node->GetDeps()) {
    auto dep_node = std::get<0>(dep_pair);
    if (maybe_deps_node->UUID() == dep_node->UUID()) {
      return true;
    }
  }
  return false;
}
std::vector<LPContainer<const HloInstruction*>*>
AutoReorderPass::FunetuneNodeOrder(
    std::vector<LPContainer<const HloInstruction*>*> sorted_nodes) {
  /*
  Why need this function, LP will solve multi inst queue, but xla accept one
inst queue, we must adapt, so inst will execute at two queue too. there are
rule:
  1. if `.start` node start time less than compute node finish time, it will
schedule before compute
  2. if start and compute node's deps in post_schedule_queue,then clear
post_schedule_queue.
  3.  `.done` put into post_schedule_queue
*/
  std::deque<LPContainer<const HloInstruction*>*> compute_queue;
  std::deque<LPContainer<const HloInstruction*>*> communicate_queue;
  std::vector<LPContainer<const HloInstruction*>*> return_inst_queue;

  //  std::vector<LPContainer<const HloInstruction*>*> pre_schedule_queue;
  std::deque<LPContainer<const HloInstruction*>*> post_schedule_queue;

  std::set<int> scheduled_node_set;
  return_inst_queue.reserve(sorted_nodes.size());

  auto ensure_node_be_schedule = [&scheduled_node_set, &return_inst_queue](
                                     LPContainer<const HloInstruction*>* node) {
    if (scheduled_node_set.count(node->UUID()) == 0) {
      // where dep nodes is not schedule
      scheduled_node_set.insert(node->UUID());
      return_inst_queue.push_back(node);
    }
  };
  auto clean_post_queue = [&post_schedule_queue, &return_inst_queue,
                           ensure_node_be_schedule]() {
    // let current holded_compute_node push into queue
    while (!post_schedule_queue.empty()) {
      ensure_node_be_schedule(post_schedule_queue.front());
      post_schedule_queue.pop_front();
    }
  };
  auto let_deps_node_schduler_soon =
      [&post_schedule_queue, &return_inst_queue, ensure_node_be_schedule](
          std::deque<LPContainer<const HloInstruction*>*> dep_done_nodes) {
        // O(m+n)
        std::unordered_set<int> toRemoveSet;
        for (auto dep_node : dep_done_nodes) {
          toRemoveSet.insert(dep_node->UUID());
        }
        // 1. from post_schedule_queue clear deps node
        post_schedule_queue.erase(
            std::remove_if(post_schedule_queue.begin(),
                           post_schedule_queue.end(),
                           [&](LPContainer<const HloInstruction*>* node) {
                             bool removed = false;
                             if (toRemoveSet.count(node->UUID()) > 0) {
                               ensure_node_be_schedule(node);
                               removed = true;
                             }
                             return removed;
                           }),
            post_schedule_queue.end());
        // 2. ensure all deps node have scheduled
        for (auto dep_node : dep_done_nodes) {
          ensure_node_be_schedule(dep_node);
        }
      };
  // step 1:O(n),init compute_queue and communicate_queue FIFO, head will first
  // execute.
  for (auto node : sorted_nodes) {
    // insert and pop_back

    if (node->IsComputation()) {
      compute_queue.push_back(node);
    } else {
      communicate_queue.push_back(node);
    }
  }
  CostType current_compute_time = 0;
  CostType current_communicate_time = 0;
  while (!communicate_queue.empty() || !compute_queue.empty()) {
    if (communicate_queue.empty()) {  // all other is compute queue
      clean_post_queue();
      while (!compute_queue.empty()) {
        ensure_node_be_schedule(compute_queue.front());
        compute_queue.pop_front();
      }
      break;
    }
    if (compute_queue.empty()) {
      clean_post_queue();
      while (!communicate_queue.empty()) {
        ensure_node_be_schedule(communicate_queue.front());
        communicate_queue.pop_front();
      }
      break;
    }

    auto next_communicate_node = communicate_queue.front();
    auto next_compute_node = compute_queue.front();
    // case 1: let communicate.start go before compute.
    bool all_inst_is_start = true;
    bool all_inst_is_done = true;
    auto insts = next_communicate_node->GetValues();

    const int MAX_HOLDED_SIZE = 20;

    for (auto instr : insts) {
      if (!async_tracker_->IsSupportedAsyncStart(*instr)) {
        all_inst_is_start = false;
      }
      if (!async_tracker_->IsSupportedAsyncDone(*instr)) {
        all_inst_is_done = false;
      }
    }
    if (NodeHaveDep(next_communicate_node, next_compute_node)) {
      // next_compute_node should be scheduler first
      std::deque<LPContainer<const HloInstruction*>*> deps_queue;
      for (auto dep_pair : next_compute_node->GetDeps()) {
        LPContainer<const HloInstruction*>* dep_node =
            dynamic_cast<LPContainer<const HloInstruction*>*>(
                std::get<0>(dep_pair));
        deps_queue.push_back(dep_node);
      }
      let_deps_node_schduler_soon(deps_queue);
      ensure_node_be_schedule(next_compute_node);

      compute_queue.pop_front();
      continue;
    }
    if (NodeHaveDep(next_compute_node, next_communicate_node)) {
      // next_communicate_node should be scheduler first
      std::deque<LPContainer<const HloInstruction*>*> deps_queue;
      for (auto dep_pair : next_communicate_node->GetDeps()) {
        LPContainer<const HloInstruction*>* dep_node =
            dynamic_cast<LPContainer<const HloInstruction*>*>(
                std::get<0>(dep_pair));
        deps_queue.push_back(dep_node);
      }
      let_deps_node_schduler_soon(deps_queue);
      ensure_node_be_schedule(next_communicate_node);
      communicate_queue.pop_front();
      continue;
    }

    if (all_inst_is_start) {
      // if start
      bool all_deps_finish = true;
      std::deque<LPContainer<const HloInstruction*>*> deps_queue;
      for (auto dep_pair : next_communicate_node->GetDeps()) {
        LPContainer<const HloInstruction*>* dep_node =
            dynamic_cast<LPContainer<const HloInstruction*>*>(
                std::get<0>(dep_pair));
        deps_queue.push_back(dep_node);
        if (dep_node->GetStart() + dep_node->GetCost() >
            next_compute_node->GetStart()) {
          all_deps_finish = false;
        }
      }

      // start need before compute
      if (all_deps_finish &&
          next_communicate_node->GetStart() > next_compute_node->GetStart() &&
          next_communicate_node->GetStart() <
              next_compute_node->GetStart() + next_compute_node->GetCost()) {
        let_deps_node_schduler_soon(deps_queue);
        ensure_node_be_schedule(next_communicate_node);
        communicate_queue.pop_front();
        continue;
      }
      // goto commoncase;
    } else if (all_inst_is_done) {
      // done schedule until next_compute_node need it
      // otherwise put into post_schedule_queue
      // let all done node put into post_schedule_queue
      post_schedule_queue.push_back(next_communicate_node);
      communicate_queue.pop_front();
      continue;
    }
    // commoncase:
    if (next_compute_node->GetStart() < next_communicate_node->GetStart()) {
      std::deque<LPContainer<const HloInstruction*>*> deps_queue;
      for (auto dep_pair : next_compute_node->GetDeps()) {
        LPContainer<const HloInstruction*>* dep_node =
            dynamic_cast<LPContainer<const HloInstruction*>*>(
                std::get<0>(dep_pair));
        deps_queue.push_back(dep_node);
      }
      let_deps_node_schduler_soon(deps_queue);
      ensure_node_be_schedule(next_compute_node);
      compute_queue.pop_front();
    } else {
      std::deque<LPContainer<const HloInstruction*>*> deps_queue;

      for (auto dep_pair : next_communicate_node->GetDeps()) {
        LPContainer<const HloInstruction*>* dep_node =
            dynamic_cast<LPContainer<const HloInstruction*>*>(
                std::get<0>(dep_pair));
        deps_queue.push_back(dep_node);
      }
      let_deps_node_schduler_soon(deps_queue);
      ensure_node_be_schedule(next_communicate_node);
      communicate_queue.pop_front();
    }
  }  // end for
  clean_post_queue();

  return return_inst_queue;
}
void AutoReorderPass::SaveOrder(
    std::vector<LPContainer<const HloInstruction*>*> sorted_nodes,
    std::string filename) {
  std::string json_file = absl::StrCat("/tmp/", filename, ".json");
  std::ofstream json_out(json_file);
  for (auto node : sorted_nodes) {
    json_out << node->GetName() << std::endl;
  }
  json_out.close();
}
tsl::Status AutoReorderPass::MoveInstruction(HloComputation* src_computation,
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
  // step 2: create Send Instruction for input args, create Recv Instruction for
  // output args
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
    auto token = src_computation->AddInstruction(HloInstruction::CreateToken());

    auto send_inst = src_computation->AddInstruction(HloInstruction::CreateSend(
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

  // step4 :create Send Instruction from dst_compuation, create Recv Instruction
  // in src_computation
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
      VLOG(2) << "ReplaceUseWith error:" << status.message() << std::endl;
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
        VLOG(2) << "ReplaceOperandWith error:" << status.message() << std::endl;
      }
    }
  }
  // step6: remove src_instruction
  src_instruction->DetachFromOperandsAndUsers();
  auto status = src_computation->RemoveInstruction(src_instruction);
  if (!status.ok()) {
    VLOG(2) << "RemoveInstruction error:" << status.message() << std::endl;
    return status;
  } else {
    VLOG(3) << "RemoveInstruction success"
            << src_computation->instruction_count() << std::endl;
    return absl::OkStatus();
  }
}
StatusOr<bool> AutoReorderPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // about reorder: be careful about RNG, such as dropout, random_shuffle,
  // random_uniform;
  // HloCostAnalysis, get instruction cost
  HloComputation* entry_computation = module->entry_computation();

  // Currently we expect that a schedule that minimizes memory pressure is
  // provided as a base. It's not necessary for the algorithm itself but it
  // allows us to not having to think for now about memory pressure.
  std::vector<HloComputation*> computations_to_schedule;
  computations_to_schedule.reserve(module->computation_count());
  // Collect which computations have latency hiding opportunities.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instr : computation->instructions()) {
      if (async_tracker_->IsSupportedAsyncStart(*instr) ||
          async_tracker_->IsSupportedAsyncDone(*instr)) {
        computations_to_schedule.push_back(computation);
        break;
      }
    }
  }
  if (computations_to_schedule.empty()) {
    return false;
  }

  absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>
      saved_schedules;
  // TF_RETURN_IF_ERROR(scheduler_core_->InitializeScheduler(module)); //TODO:
  // we don't limit memory usage
  for (HloComputation* computation : computations_to_schedule) {
    TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> new_schedule,
                        ScheduleComputation(computation));
    VLOG(2) << "new_schedule length:" << new_schedule.size()
            << " computation instruction_count:"
            << computation->instruction_count();

    saved_schedules[computation] = std::move(new_schedule);
  }

  // TODO: now memory is not in constraction
  // LOG(INFO) << "AutoReorderPass current memory usage: "
  //           << scheduler_core_->GetMemoryPeak() << " bytes.";
  for (HloComputation* computation : computations_to_schedule) {
    // VLOG(1) << "Statistics before scheduling:";
    VLOG(4) << "sequences length:" << module->schedule().sequences().size()
            << std::endl;
    module->schedule().set_sequence(
        computation, absl::MakeConstSpan(saved_schedules[computation]));
    VLOG(1) << "Statistics after scheduling:";
    // LogScheduleStatistics(computation);
  }
  return true;

}  // AutoReorderPass::Run
// CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo) {
//   switch (hlo.opcode()) {
//     case HloOpcode::kSend:
//       return {HloOpcode::kAsyncStart, HloOpcode::kSend};
//     case HloOpcode::kSendDone:
//       return {HloOpcode::kAsyncDone, HloOpcode::kSend};
//     case HloOpcode::kRecv:
//       return {HloOpcode::kAsyncStart, HloOpcode::kRecv};
//     case HloOpcode::kRecvDone:
//       return {HloOpcode::kAsyncDone, HloOpcode::kRecv};
//     default:
//       return DefaultGetCanonicalAsyncOp(hlo);
//   }
// }

}  // namespace xla

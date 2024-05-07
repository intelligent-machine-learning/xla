#include "xla/hlo/experimental/auto_reorder/auto_reorder_solver.h"
#include <fstream>
#include <iostream>

#ifndef LPSchedulerFunc(return_type)
#define LPSchedulerFunc(return_type)                      \
  template <typename ContainerType, typename ElementType> \
  return_type LinearProgramScheduler<ContainerType, ElementType>
#endif

#ifndef LPContainerDAGFunc(return_type)
#define LPContainerDAGFunc(return_type) \
  template <typename ElementType>       \
  return_type LPContainerDAG<ElementType>
#endif

namespace xla {
using IntVar = operations_research::sat::IntVar;
using CpModelBuilder = operations_research::sat::CpModelBuilder;
using IntervalVar = operations_research::sat::IntervalVar;
// namespace ORTools = operations_research::sat;
using Task =
    std::tuple<int8_t, CostType>;  // (channel, processing_time), we have two
                                   // channel now:communication and computation
using Job = std::vector<Task>;

template <typename ContainerType, typename ElementType>
LinearProgramScheduler<ContainerType, ElementType>::~LinearProgramScheduler() {
  uuid2container.clear();
  node_to_task_.clear();
  channel_to_intervals_.clear();
  // destroy nodes
  for (auto node : nodes_) {
    delete node;
  }
  nodes_.clear();
};
template <class T>
void LPContainer<T>::AddDep(LPContainer<T>* dep, CostType cost,
                            NodeType edgetype) {
  if (frozen_) {
    LOG(FATAL) << "Can not add dep to a frozen node";
    // raise exception
    return;
  }
  // every node should start after dep+cost
  deps_.push_back(std::make_tuple(dep, cost, edgetype));
};

LPSchedulerFunc(StatusOr<ContainerType*>)::FindInstructionLPNode(
    ElementType instruction) {
  auto it = uuid2container.find(instruction->unique_id());

  if (it != uuid2container.end()) {
    return it->second;
  }
  TF_RET_CHECK(false) << "Can not find the node:" << instruction->ToString();
}
LPSchedulerFunc(ContainerType*)::FindLPNodeOrCreate(ElementType element,
                                                    CostType cost,
                                                    NodeType type) {
  auto it = uuid2container.find(element->unique_id());
  if (it != uuid2container.end()) {
    return it->second;
  }
  auto node = new ContainerType(element, cost, type);
  nodes_.push_back(node);
  uuid2container.emplace(element->unique_id(), node);
  return node;
};
LPSchedulerFunc(bool)::NodeHasAddTasks(ContainerType* node) {
  auto it = node_to_task_.find(node->UUID());
  return it != node_to_task_.end();
};
LPSchedulerFunc(void)::AddNodeToTask(ContainerType* node, TaskType task) {}

LPSchedulerFunc(StatusOr<TaskType>)::FindTask(ContainerType* node) {
  auto it = node_to_task_.find(node->UUID());
  if (it != node_to_task_.end()) {
    VLOG(3) << "Find task for node:" << node->GetName() << " success";
    return std::get<1>(it->second);
  } else {
    TF_RET_CHECK(false) << "Can not find the task for node:" << node->GetName();
  }
};
LPSchedulerFunc(Status)::AddConstraint(ContainerType* node) {
  if (NodeHasAddTasks(node)) {
    return OkStatus();
  }
  // XD can't frozen node here, we will add other constraint after that
  return OkStatus();
};
LPSchedulerFunc(StatusOr<TaskType>)::AddNodeToTask(ContainerType* node) {
  IntVar start = cp_model_.NewIntVar({0, horizon_});
  IntVar end = cp_model_.NewIntVar({0, horizon_});
  IntervalVar interval = cp_model_.NewIntervalVar(start, node->GetCost(), end);
  TaskType task{start, end, interval};
  if (node->GetHintStart() != -1) {
    cp_model_.AddHint(start, node->GetHintStart());
  }
  // AddNodeToTask(node, task);
  node_to_task_.emplace(node->UUID(), std::make_tuple(node, task));
  return task;
};
LPSchedulerFunc(tsl::Status)::Solve() {
  uint32_t max_execution_time = 0;
  for (auto node : nodes_) {
    node->Freeze();
    max_execution_time += node->GetCost();
    for (auto dep_pair : node->GetDeps()) {
      auto cost = std::get<1>(dep_pair);
      max_execution_time += cost;
    };
  }
  SetHorizon(reorder::get_horizon(max_execution_time));
  // nodes_ is added by post order,so we should add it before its deps;
  for (auto node : nodes_) {
    VLOG(3) << "Add to scheduler" << node->GetName();
    TF_ASSIGN_OR_RETURN(TaskType node_task, AddNodeToTask(node));
  }
  for (auto node : nodes_) {
    auto node_task = std::get<1>(node_to_task_.at(node->UUID()));

    channel_to_intervals_[node->GetType()].push_back(node_task.interval);
    for (auto dep_pair : node->GetDeps()) {
      auto dep_node = std::get<0>(dep_pair);
      auto cost = std::get<1>(dep_pair);
      TaskType dep_task;
      VLOG(3) << node->GetName() << "should start after" << dep_node->GetName()
              << "+" << cost;
      TF_ASSIGN_OR_RETURN(dep_task, FindTask(dep_node));

      cp_model_.AddGreaterOrEqual(node_task.start, dep_task.end + cost);
    }
  }
  // add constraint, channels can be overlap each other
  for (auto it = channel_to_intervals_.begin();
       it != channel_to_intervals_.end(); it++) {
    cp_model_.AddNoOverlap(it->second);
  }
  // for communicate stream, edge also should no overlap
  std::vector<IntervalVar> no_overlap_edges;
  for (auto node : nodes_) {
    if (!node->IsCommunication()) {
      continue;
    }
    // simple method to create 01 program
    auto node_task = std::get<1>(node_to_task_.at(node->UUID()));
    for (auto dep_tuple : node->GetDeps()) {
      auto dep_node = std::get<0>(dep_tuple);
      auto cost = std::get<1>(dep_tuple);
      auto dep_type = std::get<2>(dep_tuple);

      if (IsSingleChannel(dep_type)) {
        auto dep_task = std::get<1>(node_to_task_.at(dep_node->UUID()));
        // interval
        IntervalVar interval = cp_model_.NewIntervalVar(
            dep_task.end, cost,
            node_task.start);
        no_overlap_edges.push_back(interval);
      }
    }
  }
  cp_model_.AddNoOverlap(no_overlap_edges);

  //  objective.
  IntVar obj_var = cp_model_.NewIntVar({0, horizon_}).WithName("makespan");
  std::vector<IntVar> ends;
  for (auto it = node_to_task_.begin(); it != node_to_task_.end(); it++) {
    ends.push_back(std::get<1>(it->second).end);
  }
  cp_model_.AddMaxEquality(obj_var, ends);
  cp_model_.Minimize(obj_var);

  // cp_model_.
  // VLOG(2)<<"Number of variables:"<<cp_model_.NumVariables()<<" Number of
  // constraint:"<<cp_model_.NumConstraints();
  VLOG(1) << "Solving:" << node_to_task_.size() << " nodes";
  operations_research::sat::SatParameters parameters;
  parameters.set_max_time_in_seconds(reorder::get_autoreorder_timeout());
  parameters.set_random_seed(19260817);
  // Currently, at level 1 we detect them in presolve and try
  // to fix Booleans. At level 2, we also do some form of dynamic symmetry
  // breaking during search.(default=2)
  parameters.set_symmetry_level(1);
  if (reorder::solve_debug) {
    parameters.set_log_to_stdout(true);
    // parameters.set_log_search_progress(true);
  }
  parameters.set_num_search_workers(1);
  const operations_research::sat::CpSolverResponse response =
      operations_research::sat::SolveWithParameters(cp_model_.Build(),
                                                    parameters);
  uint64_t solve_time = response.wall_time();
  VLOG(1) << "Solve finish:" << response.status()
          << " solve time:" << solve_time;

  if (response.status() == operations_research::sat::CpSolverStatus::OPTIMAL ||
      response.status() == operations_research::sat::CpSolverStatus::FEASIBLE) {
    VLOG(2) << "Optimal objective value:" << response.objective_value()
            << " status:" << response.status();
    for (auto kv : node_to_task_) {
      auto node_task_tuple = std::get<1>(kv);
      auto node = std::get<0>(node_task_tuple);
      auto task = std::get<1>(node_task_tuple);
      CostType start =
          operations_research::sat::SolutionIntegerValue(response, task.start);
      node->SetStart(start);
      VLOG(2) << node->GetName() << "should start at" << start << std::endl;
      node_starttime_.emplace(node->UUID(), start);
    }

    return OkStatus();
  } else {
    VLOG(2) << "Solve failed:" << response.status();
    return tsl::errors::NotFound("Linear Programming solve failed");
  }
};
std::string ReplaceUnusedChar(const std::string str,
                              const std::string need_move_str) {
  std::string result = str;
  for (auto c : need_move_str) {
    result.erase(std::remove(result.begin(), result.end(), c), result.end());
  }
  return result;
}
LPSchedulerFunc(std::vector<ContainerType*>)::GetSortedNodes() const{
  std::vector<ContainerType*> sorted_nodes;
  sorted_nodes.reserve(nodes_.size());
  for (auto node : nodes_) {
    sorted_nodes.push_back(node);
  }
  // we need stable_sort,let same graph on diffence device have same computation
  std::stable_sort(
      // std::sort(
      sorted_nodes.begin(), sorted_nodes.end(),
      [this](ContainerType* a, ContainerType* b) {
        return a->GetStart() < b->GetStart();
      });
  return sorted_nodes;
}
LPSchedulerFunc(void)::RenderGraphviz(std::string filename) const{
  // write a dot file
  std::string dot_file = absl::StrCat("/tmp/", filename, ".dot");
  std::ofstream out(dot_file);
  out << "digraph G {\n";
  VLOG(4) << "write node number:" << nodes_.size() << " to /tmp/" << filename
          << ".dot" << std::endl;
  auto get_node_name = [](const ContainerType* node) {
    return "\"" + ReplaceUnusedChar(node->GetName(), "%") + "\"";
  };
  bool draw_start_time = (node_starttime_.size() > 0);
  for (auto node : nodes_) {
    std::string color;
    if (node->IsCommunication()) {
      color = "orange";
    } else {
      color = "green";
    }
    if (draw_start_time) {
      out << get_node_name(node) << "[label=\""
          << ReplaceUnusedChar(node->GetName(), "") << "\\n"
          << "cost=" << node->GetCost()
          << "\nstart=" << node_starttime_.at(node->UUID())
          << "\",shape=box,color=" << color << "];\n";
    } else {
      out << get_node_name(node) << "[label=\""
          << ReplaceUnusedChar(node->GetName(), "") << "\\n"
          << "cost=" << node->GetCost() << "\",shape=box,color=" << color
          << "];\n";
    }

    for (auto dep_pair : node->GetDeps()) {
      auto dep_node = std::get<0>(dep_pair);
      auto dep_cost = std::get<1>(dep_pair);
      // draw edge
      out << get_node_name(dep_node) << "->" << get_node_name(node)
          << "[label=\"" << dep_cost << "\"];\n";
    }
  }
  out << "}\n";

  out.close();
  // convert dot file to png
  std::string png_file = absl::StrCat("/tmp/", filename, ".png");
  std::string cmd = absl::StrCat("dot -Tpng ", dot_file, " -o ", png_file);
  auto status = system(cmd.c_str());
  VLOG(4) << cmd << " execute status:" << status << std::endl;
}
LPSchedulerFunc(void)::RenderGantt(std::string filename) const{
  // https://g2.antv.antgroup.com/en/examples/storytelling/storytelling/#gantt
  // { name: 'compute',label:'kernel name1', startTime: 1, endTime: 4 },
  VLOG(4) << "write node number:" << nodes_.size() << " to /tmp/" << filename
          << ".js" << std::endl;
  auto get_node_name = [](const ContainerType* node) {
    return ReplaceUnusedChar(node->GetName(), "'");
  };
  bool draw_start_time = (node_starttime_.size() > 0);
  std::string csv_file = absl::StrCat("/tmp/", filename, ".js");
  std::ofstream csv_out(csv_file);
  csv_out << R"(import { Chart } from '@antv/g2'; 
  const events = [ )";
  for (auto node : this->GetSortedNodes()) {
    std::string name;
    if (node->IsCommunication()) {
      name = "communication";
    } else {
      name = "compute";
    }
    if (draw_start_time) {
      csv_out << "{ name: \"" << name << "\",label:'"
              << ReplaceUnusedChar(node->GetName(), "'")
              << "', startTime: " << node_starttime_.at(node->UUID())
              << ", endTime: "
              << node_starttime_.at(node->UUID()) + node->GetCost() << " },\n";
    }
  }
  csv_out << "];";

  csv_out << R"(
  const chart = new Chart({
    container: 'container',
    autoFit: true,
  });

  chart.coordinate({ transform: [{ type: 'transpose' }] });

  chart
    .interval()
    .data(events)
    .encode('x', 'name')
    .encode('y', ['endTime', 'startTime'])
    .encode('color', 'name')
    .label({
      text: 'label',
      position: 'inside',
      transform: [{ type: 'overflowHide' }],
    })
    .encode('enterDuration', (d) => d.endTime - d.startTime)
    .encode('enterDelay', 'startTime')
    .scale('enterDuration', {
      zero: true,
      range: [0, 3000],
    });

  chart.render();)";
}


LPContainerDAGFunc(bool)::IsIn(LPContainer<ElementType>* a) {
  return operands_.find(a) != operands_.end();
};
LPContainerDAGFunc(void)::AddToDAG(LPContainer<ElementType>* child) {
  inner_elements.push_back(child);
  if (IsIn(child)) {
    operands_.erase(child);
  }
  for (auto dep_pair : child->GetDeps()) {
    auto dep = std::get<0>(dep_pair);
    auto cost = std::get<1>(dep_pair);  // if cost  need store ?
    operands_.insert(dep);
  }
}
LPContainerDAGFunc(Status)::MergeFrom(LPContainerDAG<ElementType>* other) {
  /*
   step 1: this inner_elements must have dep to other's inner_elements. so that
   link to other's inner_elements change to inner edges
  */

  // maintain this LPContainerDAG inner_elements's deps,so that can create inner
  // edge after merge {dep: [<element1, cost>,<element2, cost>]}
  std::unordered_map<
      int, std::vector<std::tuple<LPContainer<ElementType>*, CostType>>>
      dep_operands2element;

  for (LPContainer<ElementType>* element : GetInnerElements()) {
    // from operate to element, there are outer edge,maybe convert to inner edge
    for (auto dep_pair : element->GetDeps()) {
      auto dep = std::get<0>(dep_pair);
      auto cost = std::get<1>(dep_pair);
      if (dep_operands2element.find(dep->UUID()) ==
          dep_operands2element.end()) {
        dep_operands2element[dep->UUID()] =
            std::vector<std::tuple<LPContainer<ElementType>*, CostType>>();
      }
      dep_operands2element[dep->UUID()].push_back(
          std::make_tuple(element, cost));
    }
  }
  // other
  for (auto child : other->GetInnerElements()) {
    // there child must in inner_elements_deps
    TF_RET_CHECK(dep_operands2element.find(child->UUID()) ==
                 dep_operands2element.end())
        << "child is not in dep_operands2element";
    for (auto dep_pair : dep_operands2element[child->UUID()]) {
      auto dep = std::get<0>(dep_pair);
      auto cost = std::get<1>(dep_pair);
      if (dep_operands2element.find(dep->UUID()) !=
          dep_operands2element.end()) {
        for (auto element_pair : dep_operands2element[dep->UUID()]) {
          auto element = std::get<0>(element_pair);
          auto cost = std::get<1>(element_pair);
          // create edge between element and child
          DAGEdge edge;
          edge.from = element;
          edge.to = child;
          edge.cost = cost;
          edges_.push_back(edge);
        }
      }
    }

    AddToDAG(child);
  };
}
template class LPContainer<const HloInstruction*>;
template class LinearProgramScheduler<LPContainer<const HloInstruction*>,
                                      const HloInstruction*>;

template class LPContainerDAG<const HloInstruction*>;
// template class LinearProgramScheduler<LPContainerDAG<const HloInstruction*>,
// const HloInstruction*>;

}  // namespace xla

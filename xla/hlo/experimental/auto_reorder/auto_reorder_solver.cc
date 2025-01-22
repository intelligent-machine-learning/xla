#include "xla/hlo/experimental/auto_reorder/auto_reorder_solver.h"
#include <fstream>
#include <iostream>
using bazel::tools::cpp::runfiles::Runfiles;
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
using Status=absl::Status;

// namespace ORTools = operations_research::sat;
using Task =
    std::tuple<int8_t, CostType>;  // (channel, processing_time), we have two
                                   // channel now:communication and computation
using Job = std::vector<Task>;
namespace reorder {
uint32_t get_autoreorder_timeout() {
  const char* env = std::getenv("XLA_AUTOREORDER_TIMEOUT");
  if (env == nullptr) {
    return ksolveTimeout;
  }
  return std::atoi(env);
};
uint32_t get_autoreorder_worker() {
  const char* env = std::getenv("XLA_AUTOREORDER_WORKER");
  if (env == nullptr) {
    return 1;
  }
  return std::atoi(env);
};
xla::SolveMethod get_solve_method() {
  const char* env = std::getenv("XLA_AUTOREORDER_METHOD");
  if (env == nullptr) {
    VLOG(2) << "using SolveMethod: ortools";
    return SolveMethod::kORTools;
  }
  // kORTools
  // kAntOpt
  if (std::strcmp(env, "antoptremote") == 0) {
    VLOG(2) << "using SolveMethod: antoptremote";
    return SolveMethod::kAntOptRemote;
  } else if (std::strcmp(env, "antopt") == 0) {
    VLOG(2) << "using SolveMethod: antopt";
    return SolveMethod::kAntOpt;
  }
  VLOG(2) << "using SolveMethod: ortools";
  return SolveMethod::kORTools;
}
int get_horizon(int max_time) {
  // scale should be fit with module?
  return max_time * 2;
}
const bool is_keep_communicate_order() {
  const char* env = std::getenv("XLA_KEEP_COMMUNICATE_ORDER");
  if (env == nullptr) {
    return false;
  }
  return std::strcmp(env, "true") == 0;
};
int get_cpu_number() {
  // return 8;
  return std::thread::hardware_concurrency();
}

}  // namespace reorder
template <typename ContainerType, typename ElementType>
LinearProgramScheduler<ContainerType, ElementType>::~LinearProgramScheduler() {
  uuid2container.clear();
  node_to_task_.clear();
  container_users_.clear();
  channel_to_intervals_.clear();
  // destroy nodes
  for (auto node : nodes_) {
    delete node;
  }
  nodes_.clear();
};
void LPNode::AddDep(LPNode* dep, CostType cost, NodeType edgetype) {
  if (frozen_) {
    LOG(FATAL) << "Can not add dep to a frozen node";
    // raise exception
    return;
  }
  // every node should start after dep+cost
  deps_.push_back(std::make_tuple(dep, cost, edgetype));
};
bool LPNode::ReplaceDep(LPNode* old_dep, LPNode* new_dep, CostType cost,
                        NodeType nodetype) {
  if (frozen_) {
    LOG(FATAL) << "Can not change dep to a frozen node";
    // raise exception
    return false;
  }
  for (size_t i = 0; i < deps_.size(); ++i) {
    auto dep_tuple = deps_[i];
    auto dep_node = std::get<0>(dep_tuple);
    if (dep_node == old_dep) {
      deps_[i] = std::make_tuple(new_dep, cost, nodetype);
      return true;
    }
  }
  return false;
}
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
    return absl::OkStatus();
  }
  // XD can't frozen node here, we will add other constraint after that
  return absl::OkStatus();
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
LPSchedulerFunc(uint32_t)::GetNodesCount() const { return nodes_.size(); }
LPSchedulerFunc(tsl::Status)::SolveUsingAntOPT(SolveMethod solve_method) {
  /* 1. export to json
     2. send to server
     3. query until get result
     4. return
  */
  std::string json_filename = "AntOPT.req";
  std::string input_filename = SaveJSON(json_filename);
  std::string output_filename = absl::StrCat("/tmp/", "AntOPT.res.json");

  // if it's in torch_xla env, we need using module;otherwise this script point
  // to workdir
  std::vector<std::string> args = {
      "python3",
      "-m",
  };
  if (solve_method == SolveMethod::kAntOptRemote) {
    args.push_back("lynx.tools.solve_remote");
  } else {
    args.push_back("lynx.tools.solve_local");
  }
  args.push_back(input_filename);
  args.push_back(output_filename);
  // absl::Cleanup file_cleanup = [&]() {
  //   if(std::filesystem::exists(input_filename)){
  //     std::filesystem::remove(input_filename);
  //   }
  //   if(std::filesystem::exists(output_filename)){
  //     std::filesystem::remove(output_filename);
  //   }
  // };
  // SubProcess can't set env
  tsl::SubProcess proc;
  proc.SetProgram("python3", args);
  proc.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  proc.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);

  bool start = proc.Start();
  std::string in = "\n";
  std::string out, err;

  int process_ret = proc.Communicate(&in, &out, &err);

  if (process_ret != 0) {
    LOG(ERROR) << "Error running python script, input=" << input_filename
               << " output:" << output_filename << err;
    return tsl::errors::InvalidArgument(
        absl::StrCat("Error running python script:", input_filename,
                     " output:", output_filename));
  }
  VLOG(2) << "python script stdout:";
  VLOG(2) << out;
  VLOG(2) << "python script stderr:";
  VLOG(2) << err;

  xla::auto_reorder::AntOPTSolverResponse solve_response;
  std::ifstream jsonfile_in(output_filename);
  if (!jsonfile_in.is_open()) {
    LOG(ERROR) << "Could not open the file - '" << output_filename << "'"
               << std::endl;
    return tsl::errors::InvalidArgument("Could not open the result file");
  }
  // read all json
  jsonfile_in.seekg(0, std::ios::end);
  std::streampos fileSize = jsonfile_in.tellg();
  jsonfile_in.seekg(0, std::ios::beg);
  std::string content;
  content.resize(fileSize);
  jsonfile_in.read(&content[0], fileSize);

  auto status = tsl::HumanReadableJsonToProto(content, &solve_response);
  if (!status.ok()) {
    LOG(ERROR) << "parse json fail:" << status.message();
    return status;
  }
  // std::vector<ContainerType*> nodes_;
  std::map<int, ContainerType*> uuid2node;
  for (auto node : nodes_) {
    uuid2node[node->UUID()] = node;
  }
  uint32_t solved_node_count = 0;
  uint32_t solved_edge_count = 0;
  for (auto node_item : solve_response.finalized_node_start_end_time()) {
    std::string node_uuid = node_item.node_uuid();
    int32_t find_idx = node_uuid.find("__");
    if (find_idx > 0) {  // it's edge
      solved_edge_count += 1;
      continue;
    }
    // convert uuid to int
    solved_node_count += 1;
    int uuid_int = std::stoi(node_uuid);
    int start_time = node_item.start_time();
    uuid2node[uuid_int]->SetStart(start_time);
    node_starttime_.emplace(uuid_int, start_time);
  }
  if (solved_node_count != GetNodesCount()) {
    return tsl::errors::InvalidArgument(
        absl::StrCat("Not all nodes are solved, solved:", solved_node_count,
                     " total:", GetNodesCount(), " edges:", solved_edge_count));
  }
  return absl::OkStatus();
}
LPSchedulerFunc(void)::IninContainerUsers() {
  // who is user of this node
  for (auto node : nodes_) {
    for (auto dep_pair : node->GetDeps()) {
      auto dep_node = std::get<0>(dep_pair);
      container_users_[dep_node->UUID()].emplace_back(node);
    }
  }
}
LPSchedulerFunc(tsl::Status)::MergeIfPossible() {
  // for(auto node : nodes_){

  //   std::vector<ContainerType*> merged_nodes =
  //   node->MergeUntilImpossible(ReplaceNodeUser);
  // }
  return absl::OkStatus();
}

LPSchedulerFunc(tsl::Status)::Solve(SolveMethod solve_method) {
  if (ContainerType::SupportMerge()) {
    // 1. maintain container_users_
    IninContainerUsers();
    auto status = MergeIfPossible();
    if (!status.ok()) {
      return status;
    }
  }
  if (solve_method == SolveMethod::kORTools) {
    return SolveUsingORTools();
  } else {
    return SolveUsingAntOPT(solve_method);
  }
}
LPSchedulerFunc(tsl::Status)::SolveUsingORTools() {
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
    if (node->GetCost() >= 1) {  // side effect
      channel_to_intervals_[node->GetType()].push_back(node_task.interval);
    }

    for (auto dep_pair : node->GetDeps()) {
      ContainerType* dep_node =
          dynamic_cast<ContainerType*>(std::get<0>(dep_pair));
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
  std::map<NodeType, std::vector<IntervalVar>> no_overlap_edges;
  // std::vector<IntervalVar> no_overlap_edges;
  for (auto node : nodes_) {
    // async and communication edge can't overlap
    if (!node->IsCommunication()) {
      continue;
    }
    // simple method to create 01 program
    auto node_task = std::get<1>(node_to_task_.at(node->UUID()));
    for (auto dep_tuple : node->GetDeps()) {
      auto dep_node = std::get<0>(dep_tuple);
      auto edge_cost = std::get<1>(dep_tuple);
      auto dep_type = std::get<2>(dep_tuple);

      if (IsSingleChannel(dep_type) && edge_cost > 1) {
        auto dep_task = std::get<1>(node_to_task_.at(dep_node->UUID()));
        // interval
        IntervalVar interval =
            cp_model_.NewIntervalVar(dep_task.end, edge_cost, node_task.start);
        no_overlap_edges[dep_type].push_back(interval);
      }
    }
  }
  for (auto it = no_overlap_edges.begin(); it != no_overlap_edges.end(); it++) {
    if (!it->second.empty()) {
      cp_model_.AddNoOverlap(it->second);
    }
  }

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
  parameters.set_symmetry_level(2);
  if (reorder::solve_debug) {
    parameters.set_log_to_stdout(true);
    parameters.set_log_search_progress(true);
  }
  parameters.set_num_search_workers(xla::reorder::get_autoreorder_worker());
  operations_research::sat::Model sat_model;
  // auto model = cp_model_.Build();
  // model is operations_research::sat::CpModelProto type
  // need operations_research::MPModelProto& type, so we need to convert it
  // model
  
  sat_model.Add(operations_research::sat::NewFeasibleSolutionObserver([&](const operations_research::sat::CpSolverResponse& r) {
    // 获得当前解和理论最优解之间的距离
    CostType current_makespan = r.objective_value();
    std::cout<<"current_makespan:"<<current_makespan<<" best_bound:"<<r.best_objective_bound()<<std::endl;
  }));
  sat_model.Add(operations_research::sat::NewSatParameters(parameters));
  const operations_research::sat::CpSolverResponse response = SolveCpModel(cp_model_.Build(), &sat_model);

  uint64_t solve_time = response.wall_time();
  VLOG(1) << "Solve finish:" << response.status()
          << " solve time:" << solve_time<<" best_objective_bound:"<<response.best_objective_bound();

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

    return absl::OkStatus();
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
LPSchedulerFunc(std::vector<ContainerType*>)::GetSortedNodes() const {
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
        if(a->GetStart() == b->GetStart()){
          std::unordered_set<int> a_deps;
          a_deps.reserve(a->GetDeps().size());
          for(auto dep : a->GetDeps()){
            a_deps.insert(std::get<0>(dep)->UUID());
          }
          if(a_deps.count(b->UUID()) > 0){
            return false; // b need before a
          }
          std::unordered_set<int> b_deps;
          b_deps.reserve(b->GetDeps().size());
          for(auto dep : b->GetDeps()){
            b_deps.insert(std::get<0>(dep)->UUID());
          }
          if(b_deps.count(a->UUID()) > 0){
            return false;
          }
          //else, any order is ok
          return true;
        }
        return a->GetStart() < b->GetStart();
      });
  return sorted_nodes;
}
LPSchedulerFunc(std::string)::SaveJSON(std::string filename) const {
  /* save json, this always use before solve, to define graph */
  std::string json_file = absl::StrCat("/tmp/", filename, ".json");
  std::ofstream json_out(json_file);
  json_out << "{" << std::endl;
  json_out << "\"nodes\": [" << std::endl;
  int32_t node_count = 0;
  int32_t edge_count = 0;

  for (auto node : this->GetSortedNodes()) {
    std::string name = NodeTypeToString(node->GetType());
    if (node_count > 0) {
      json_out << ",\n{ \"uuid\": \"" << node->UUID() << "\",\"typename\": \""
               << name << "\", \"name\": " << GetNodeName(node)
               << ", \"cost\": " << node->GetCost() << ", \"opcode\": \""
               << node->GetOpcode() << "\", \"space_cost\": "
               << node->GetSpaceCost() << "}";
    } else {
      json_out << "{ \"uuid\": \"" << node->UUID() << "\",\"typename\": \""
               << name << "\", \"name\": " << GetNodeName(node)
               << ", \"cost\": " << node->GetCost() << ", \"opcode\": \""
               << node->GetOpcode() << "\", \"space_cost\": "
               << node->GetSpaceCost() << " }";
    }
    node_count++;
  }
  json_out << "]," << std::endl;
  json_out << "\"edges\": [" << std::endl;
  for (auto node : this->GetSortedNodes()) {
    for (auto dep_pair : node->GetDeps()) {
      auto dep_node = std::get<0>(dep_pair);
      auto dep_cost = std::get<1>(dep_pair);
      NodeType dep_type = std::get<2>(dep_pair);
      std::string name = NodeTypeToString(dep_type);

      // draw edge
      if (edge_count > 0) {
        json_out << ",\n{ \"from\": \"" << dep_node->UUID() << "\", \"to\": \""
                 << node->UUID() << "\", \"typename\": \"" << name
                 << "\", \"cost\": " << dep_cost << " }";
      } else {
        json_out << "{ \"from\": \"" << dep_node->UUID() << "\", \"to\": \""
                 << node->UUID() << "\", \"typename\": \"" << name
                 << "\", \"cost\": " << dep_cost << " }";
      }
      edge_count++;
    }
  }
  json_out << "]" << std::endl;
  json_out << "}" << std::endl;
  json_out.flush();
  json_out.close();
  return json_file;
}
LPSchedulerFunc(std::string)::GetNodeName(ContainerType* node) {
  return "\"" + ReplaceUnusedChar(node->GetName(), "%") + "\"";
}
LPSchedulerFunc(void)::SaveGraphviz(std::string filename) const {
  // write a dot file
  std::string dot_file = absl::StrCat("/tmp/", filename, ".dot");
  std::ofstream out(dot_file);
  out << "digraph G {\n";
  VLOG(4) << "write node number:" << nodes_.size() << " to /tmp/" << filename
          << ".dot" << std::endl;
  bool draw_start_time = (node_starttime_.size() > 0);
  for (auto node : nodes_) {
    std::string color;
    if (node->IsCommunication()) {
      color = "orange";
    } else {
      color = "green";
    }
    if (draw_start_time) {
      out << GetNodeName(node) << "[label=\""
          << ReplaceUnusedChar(node->GetName(), "") << "\\n"
          << "cost=" << node->GetCost()
          << "\nstart=" << node_starttime_.at(node->UUID())
          << "\",shape=box,color=" << color << "];\n";
    } else {
      out << GetNodeName(node) << "[label=\""
          << ReplaceUnusedChar(node->GetName(), "") << "\\n"
          << "cost=" << node->GetCost() << "\",shape=box,color=" << color
          << "];\n";
    }

    for (auto dep_pair : node->GetDeps()) {
      ContainerType* dep_node =
          dynamic_cast<ContainerType*>(std::get<0>(dep_pair));

      auto dep_cost = std::get<1>(dep_pair);
      // draw edge
      out << GetNodeName(dep_node) << "->" << GetNodeName(node) << "[label=\""
          << dep_cost << "\"];\n";
    }
  }
  out << "}\n";

  out.close();
  // convert dot file to png,do not use this,will cause long time
  // std::string png_file = absl::StrCat("/tmp/", filename, ".png");
  // std::string cmd = absl::StrCat("dot -Tpng ", dot_file, " -o ", png_file);
  // auto status = system(cmd.c_str());
  // VLOG(4) << cmd << " execute status:" << status << std::endl;
}
LPSchedulerFunc(void)::SaveGantt(std::string filename) const {
  // https://g2.antv.antgroup.com/en/examples/storytelling/storytelling/#gantt
  /* save json, this always use after solve, to show solve result, use
   * lynx.tools.convert_xla2chrome_trace, change to chrome format */
  // { "typename": "compute","label":"kernel name1", startTime: 1, endTime: 4
  // ,"uuid":"1234"},
  VLOG(4) << "write node number:" << nodes_.size() << " to /tmp/" << filename
          << ".json" << std::endl;
  auto get_node_name = [](const ContainerType* node) {
    return ReplaceUnusedChar(ReplaceUnusedChar(node->GetName(), "'"), "\"");
  };
  bool first_line = true;
  std::string csv_file = absl::StrCat("/tmp/", filename, ".json");
  std::ofstream csv_out(csv_file);
  csv_out << "{\"nodes\":[";
  for (auto node : this->GetSortedNodes()) {
    std::string name = NodeTypeToString(node->GetType());
    if (first_line) {
      csv_out << "{ \"typename\": \"" << name << "\", \"name\":\""
              << get_node_name(node) << "\", \"uuid\": \"" << node->UUID()
              << "\", \"startTime\": " << node_starttime_.at(node->UUID())
              << ", \"endTime\": "
              << node_starttime_.at(node->UUID()) + node->GetCost() << " }";
      first_line = false;
    } else {
      csv_out << ",\n{ \"typename\": \"" << name << "\", \"name\":\""
              << get_node_name(node) << "\", \"uuid\": \"" << node->UUID()
              << "\", \"startTime\": " << node_starttime_.at(node->UUID())
              << ", \"endTime\": "
              << node_starttime_.at(node->UUID()) + node->GetCost() << " }";
    }
  }
  csv_out << "]}";
}
LPContainerDAGFunc(std::vector<LPContainerDAG<ElementType>*>)::
    MergeUntilImpossible(std::function<Status(LPContainerDAG<ElementType>*,
                                              LPContainerDAG<ElementType>*)>
                             replace_callback) {
  while (CanMerge()) {
    //  merge its' deps
    // 1. let outer edge move to inner
    // 2. move dep node to inner elements
    // 3. make dep's dep become node's outer edge
    auto origin_deps = GetDeps();
    MoveOuterEdgeIn();
    // set origin deps' deps to this node's deps
    for (auto dep_pair : origin_deps) {
      LPNode* dep_node = std::get<0>(dep_pair);
      auto dep_cost = std::get<1>(dep_pair);
      auto dep_type = std::get<2>(dep_pair);

      AddInnerElements(dep_node);
      MergeEdge(dep_node);
      // replace_callback(dep_node, this);
      // let a node which dep dep_node,move to this node
      // dep_node->ReplaceDep(dep_node,this,dep_cost,dep_type);
    }
  }
}
LPContainerDAGFunc(void)::MergeEdge(LPNode* dep) {
  // make dep's dep become node's outer edge. dep's inner_edges_ to this
  // inner_edge
  LPContainerDAG<ElementType>* dep_dag_container =
      dynamic_cast<LPContainerDAG<ElementType>*>(dep);
  for (auto dep_outer_edge : dep_dag_container->GetOuterEdges()) {
    outer_edges_.push_back(dep_outer_edge);
  };
  for (auto dep_inner_edge : dep_dag_container->GetInnerEdges()) {
    inner_edges_.push_back(dep_inner_edge);
  };
}

LPContainerDAGFunc(void)::AddInnerElements(LPNode* dep) {
  LPContainerDAG<ElementType>* dep_dag_container =
      dynamic_cast<LPContainerDAG<ElementType>*>(dep);
  for (auto dep_inner_ele : dep_dag_container->GetInnerElements()) {
    inner_elements.push_back(dep_inner_ele);
  }
}
LPContainerDAGFunc(void)::MoveOuterEdgeIn() {
  while (outer_edges_.size() > 0) {
    auto edge = outer_edges_.back();
    inner_edges_.push_back(edge);
    outer_edges_.pop_back();
  }
}
LPContainerDAGFunc(void)::AddDep(LPNode* dep, CostType cost,
                                 NodeType nodetype) {
  LPNode::AddDep(dep, cost, nodetype);
  if (inner_elements.size() != 1) {
    LOG(ERROR) << "can't add dep when this dag is merged";
    return;
  }
  LPContainerDAG* dep_dag_container = dynamic_cast<LPContainerDAG*>(dep);
  if (dep_dag_container->GetValues().size() != 1) {
    LOG(ERROR) << "can't add dep when this depency dag is merged";
    return;
  }
  // first call, now this is only one element
  LPContainerDAG::DAGEdge edge = {
      inner_elements[0], dep_dag_container->GetValues()[0], cost, nodetype};
  outer_edges_.push_back(edge);
}
LPContainerDAGFunc(bool)::CanMerge() {
  if (!IsComputation()) {
    return false;
  }
  // only merge computation graph
  for (auto dep_pair : GetDeps()) {
    LPContainerDAG* dep_node =
        dynamic_cast<LPContainerDAG*>(std::get<0>(dep_pair));
    // only merge computation graph
    if (!dep_node->IsComputation()) {
      return false;
    }
  }
  return true;
}
LPContainerDAGFunc(Status)::MergeFrom(LPContainerDAG<ElementType>* other) {
  /*
   step 1: this inner_elements must have dep to other's inner_elements. so that
   link to other's inner_elements change to inner edges
  */

  // maintain this LPContainerDAG inner_elements's deps,so that can create inner
  // edge after merge {dep: [<element1, cost>,<element2, cost>]}

  other->GetDeps();
  return absl::OkStatus();
}
template class LPContainer<const HloInstruction*>;
template class LinearProgramScheduler<LPContainer<const HloInstruction*>,
                                      const HloInstruction*>;

template class LPContainerDAG<const HloInstruction*>;
template class LinearProgramScheduler<LPContainerDAG<const HloInstruction*>,
                                      const HloInstruction*>;

}  // namespace xla

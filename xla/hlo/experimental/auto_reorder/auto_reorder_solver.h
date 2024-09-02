#ifndef XLA_AUTO_REORDER_SOLVER_H_
#define XLA_AUTO_REORDER_SOLVER_H_
#include <limits>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unistd.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/cleanup/cleanup.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/common_ortools_deps.h"
#include "tsl/platform/human_readable_json.h"
#include "tsl/platform/subprocess.h"
#include "absl/status/status.h"

#include "xla/hlo/experimental/auto_reorder/instr_profile_info.pb.h"
#include "tools/cpp/runfiles/runfiles.h"
#include <functional>
namespace xla {
using IntVar = operations_research::sat::IntVar;
using CpModelBuilder = operations_research::sat::CpModelBuilder;
using IntervalVar = operations_research::sat::IntervalVar;
// using SearchMonitor=operations_research::SearchMonitor;
enum class SolveMethod {
  kORTools = 0,
  kAntOpt = 1,
  kAntOptRemote = 2,
};

namespace reorder {
const uint32_t ksolveTimeout = 180;  // 3min
uint32_t get_autoreorder_timeout();
uint32_t get_autoreorder_worker();
constexpr const int kChannelNumber = 2;
int get_horizon(int max_time);
constexpr bool solve_debug = true;
// TODO: no keep order will cause hung on multi processing, we should consider
// how to resolve it
// get cpu number of current machine
const bool is_keep_communicate_order();
int get_cpu_number();
xla::SolveMethod get_solve_method();
}  // namespace reorder
enum class NodeType {
  kCompute = 0,
  kCommunication = 1,
  kAsynchronous = 2,
  kCopy = 3,
  kUnknown = 99,
};

inline std::string NodeTypeToString(NodeType nodetype) {
  switch (nodetype) {
    case NodeType::kCompute:
      return "compute";
    case NodeType::kCommunication:
      return "communication";
    case NodeType::kAsynchronous:
      return "asynchronous";
    //memory copy, there is no d2h and h2d
    case NodeType::kCopy:
      return "copy";
    default:
      return "unknown";
  }
}
inline NodeType StringToNodeType(std::string s) {
  if (s == "compute") {
    return NodeType::kCompute;
  } else if (s == "communication") {
    return NodeType::kCommunication;
  } else if (s == "asynchronous") {
    return NodeType::kAsynchronous;
  }
  else if (s == "copy") {
    return NodeType::kCopy;
  }
  return NodeType::kUnknown;
}

static bool IsSingleChannel(NodeType nodetype) {
  return nodetype == NodeType::kCommunication ||
         nodetype == NodeType::kAsynchronous;
}

struct TaskType {
  IntVar start;
  IntVar end;
  IntervalVar interval;
};
using CostType = int64_t;  // we can change it to double?

// TODO: using LPNode to abstract LPContainer and LPContainerDAG
class LPNode {
 public:
  LPNode(CostType cost, NodeType type):cost_(cost), type_(type){

  }
  ~LPNode() { deps_.clear(); };
  virtual const std::string GetName() const = 0;
  virtual const int UUID() = 0;
  void AddDep(LPNode* dep, CostType cost,NodeType nodetype);
  const std::vector<std::tuple<LPNode*, CostType, NodeType>> GetDeps()
      const {
    return deps_;
  }
  bool CanMerge(){return false;};
  CostType GetCost() { return cost_; }
  bool HasDep(LPNode* dep) {
    for (auto d : deps_) {
      if (std::get<0>(d) == dep) {
        return true;
      }
    }
    return false;
  }
  bool ReplaceDep(LPNode* old_dep, LPNode* new_dep, CostType cost, NodeType nodetype);
  void Freeze() { frozen_ = true; }
  // Get the type of the container: compute or communication
  bool IsComputation() const{ return type_ == NodeType::kCompute; }
  bool IsCommunication() const{ return type_ == NodeType::kCommunication; }
  CostType GetStart() { return startat_; }
  void SetStart(CostType start) { startat_ = start; }
  CostType GetHintStart() { return hint_start_; }
  void SetHintStart(CostType start) { hint_start_ = start; }
  HloOpcode GetOpcode() {return opcode_;};
  void SetOpcode(HloOpcode opcode) { opcode_ = opcode; }

  NodeType GetType() const{ return type_; }
 protected:
  std::vector<std::tuple<LPNode*, CostType, NodeType>> deps_;
  bool frozen_ = false;
  CostType cost_;
  NodeType type_;
  CostType startat_;
  HloOpcode opcode_;
  CostType hint_start_ = -1;

};

// LPContainer is a template class, it can be used to store any type of data
// 1. LPContainer<const HloInstruction*>; using to store one instruction
// 2. LPContainer<const LPContainerDAG>; using to store a graph of
// instructions,decrese lp hard
// 3. LPContainer<const Stage>; maybe we can use it to store a pipeline stage
template <typename ElementType>
class LPContainer: public LPNode {
 public:
  // create a LPContainer with inner_element, cost and type
  LPContainer(ElementType inner_element, CostType cost, NodeType type)
      : inner_element_(inner_element), LPNode(cost, type) {
        
    uuid_ = reinterpret_cast<uintptr_t>(this);
  };
  const std::string GetName() const { return inner_element_->ToShortString(); }
  const int UUID(){ return inner_element_->unique_id(); }
  static bool SupportMerge(){return false;}
  // speed up reorder, we can set a hint start time
  std::vector<LPContainer<ElementType>* > MergeUntilImpossible(std::function<Status(LPContainer*,LPContainer*)> callback
  ){return std::vector<LPContainer* >{};}
  
  const bool HasValue(){ return inner_element_ != nullptr; }
  const std::vector<ElementType> GetValues(){
    return std::vector<ElementType>{inner_element_};
  }
  

 private:
  ElementType inner_element_;
  // deps store the edge
  uintptr_t uuid_;
  std::string name_;  // edge need a name
};
// LPContainerDAG is a graph of container, it can be used to store the DAG of
// container be used as a atomic unit of LPContainer
template <typename ElementType>
class LPContainerDAG : public LPNode {
  // we can use InstructionDAG to get memory effect order
 public:
  // maintain a DAG of deps elements
  struct DAGEdge {
    ElementType from;
    ElementType to;
    CostType cost;
    NodeType edgetype;
  };
  // create a  LPContainerDAG with one element
  LPContainerDAG(ElementType inner_element, CostType cost, NodeType type)
      : LPNode(cost, type){
    
    inner_elements.push_back(inner_element);
  };
  //if 
  bool IsIn(ElementType a);
  // which container can be put together:1. they have the same type 2. they have
  // dep between them
  // static bool CanFused(LPContainerDAG<ElementType>* a,
  // LPContainerDAG<ElementType>* b);

  // override LPContainer
  const std::string GetName() const{
    std::string name = "LPContainerDAG{";
    for (auto ele : inner_elements) {
      name += ele->ToShortString();
      name += "\n";
    }
    name += "}";
    return name;
  }
  const int UUID() { return inner_elements[0]->unique_id(); }
  const bool HasValue() { return inner_elements.size() > 0; }
  static bool SupportMerge(){return true;}
  bool CanMerge();
  const std::vector<ElementType> GetValues() {
    return inner_elements;
  }
  void AddDep(LPNode* dep, CostType cost,NodeType nodetype);
  std::vector<LPContainerDAG<ElementType>* > MergeUntilImpossible(std::function<Status(LPContainerDAG*,LPContainerDAG*)> callback);
  

  const std::vector<ElementType> GetInnerElements() const {
    return inner_elements;
  }
  // merge other LPContainerDAG to this LPContainerDAG,then destroy other
  // LPContainerDAG
  Status MergeFrom(LPContainerDAG<ElementType>* other);
  
 private:
 //Merge request three function:
  //1. let outer edge move to inner
  void MoveOuterEdgeIn();
  // 2. move dep node to inner elements
  void AddInnerElements(LPNode* dep);
  // 3. make dep's dep become node's outer edge. dep's inner_edges_ to this inner_edge
  void MergeEdge(LPNode* dep);
  std::vector<DAGEdge> GetOuterEdges(){return outer_edges_;}
  std::vector<DAGEdge> GetInnerEdges(){return inner_edges_;};

  bool ChangeDepEdge();
  //nodes: inner_elements and deps(outer_elements)
  //edge:inner_edges and outer_edges
  std::set<ElementType> operands_;
  std::vector<ElementType> inner_elements;
  // maintain edges between inner_elements
  std::vector<DAGEdge> inner_edges_;
  std::vector<DAGEdge> outer_edges_;
  CostType cost_;
  CostType startat_;
  NodeType type_;
};

// we only define node, edge is express by deps;
// edge is use to express the dependency between two nodes ，it have no effect
// constraint

// ContainerType is a template class, it can be used to store ElementType of
// data example: LPContainer<const HloInstruction*>; using to store one
// instruction, ElementType is const HloInstruction*, ContainerType is
// LPContainer<const HloInstruction*>
template <typename ContainerType, typename ElementType>
class LinearProgramScheduler {
  // https://developers.google.com/optimization/scheduling/job_shop?hl=zh-cn
  // be a linear programming problem or a integer programming problem,that's a
  // problem
 public:
  explicit LinearProgramScheduler(bool verbose = false) {
    cp_model_ = CpModelBuilder();
    verbose_ = verbose;
  };
  ~LinearProgramScheduler();
  // add Node to scheduler, its deps will execute before it
  Status AddConstraint(ContainerType* node);
  // solve the LP problem
  // Status Solve();
  Status Solve(SolveMethod solve_method);
  // find instruction,if not exist, return error
  StatusOr<ContainerType*> FindInstructionLPNode(ElementType instruction);
  // find LPNode by instruction,if not exist,create it
  ContainerType* FindLPNodeOrCreate(ElementType instruction, CostType cost,
                                    NodeType type);
  // ContainerType*
  std::vector<ContainerType*> GetSortedNodes() const;
  uint32_t GetNodesCount() const;
  // for debug: save graph viz file
  void SaveGraphviz(std::string filename) const;
  // for debug: render gantt chart
  void SaveGantt(std::string filename) const;
  std::string SaveJSON(std::string filename) const;
  // set max start time as horizon
  void SetHorizon(uint32_t horizon) { horizon_ = horizon; }
  StatusOr<TaskType> FindTask(ContainerType* node);
  bool NodeHasAddTasks(ContainerType* node);
  CostType GetNodeStartTime(ContainerType* node);
  void AddNodeToTask(ContainerType* node, TaskType task);
  StatusOr<TaskType> AddNodeToTask(ContainerType* node);
  Status MergeGraph();
  static std::string GetNodeName(ContainerType* node);
 private:
  Status SolveUsingORTools();
  Status SolveUsingAntOPT(SolveMethod solve_method);
  StatusOr<bool> AddEdgesNoOverlap(ContainerType* node);
  void IninContainerUsers();
  Status MergeIfPossible();
  //make it callback by MergeIfPossible
  Status ReplaceNodeUser(ContainerType* old_node,ContainerType* new_node);
  CpModelBuilder cp_model_;
  bool verbose_ = false;
  std::unordered_map<int, ContainerType*> uuid2container;
  // this container is used by other users
  std::unordered_map<int, std::vector<ContainerType*>> container_users_;
  std::vector<ContainerType*> nodes_;
  uint32_t horizon_ = std::numeric_limits<uint32_t>::max();
  absl::flat_hash_map<int, std::tuple<ContainerType*, TaskType>>
      node_to_task_;  // every node hold interval_var,show what time it start
                      // and end
  // channels can be overlap each other
  std::map<NodeType, std::vector<IntervalVar>> channel_to_intervals_;
  std::map<int, int64_t> node_starttime_;
};
// class BoundMonitor:public SearchMonitor{

//   /// This method is called when a valid solution is found. If the
//   /// return value is true, then search will resume after. If the result
//   /// is false, then search will stop there.
//   bool AtSolution(){
//       return true;
//   };
// }

}  // namespace xla
#endif  // XLA_AUTO_REORDER_H_
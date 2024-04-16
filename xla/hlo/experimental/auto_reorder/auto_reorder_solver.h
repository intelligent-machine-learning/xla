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
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/common_ortools_deps.h"

namespace xla {
using IntVar = operations_research::sat::IntVar;
using CpModelBuilder = operations_research::sat::CpModelBuilder;
using IntervalVar = operations_research::sat::IntervalVar;
namespace reorder {
const uint32_t ksolveTimeout = 180;  // 30s

static const int kChannelNumber = 2;
int get_horizon(int max_time) {
  // scale should be fit with module?
  return max_time * 2;
}
bool solve_debug = true;
// TODO: no keep order will cause hung on multi processing, we should consider
// how to resolve it
// get cpu number of current machine
const bool is_keep_communicate_order() {
  const char* env = std::getenv("XLA_KEEP_COMMUNICATE_ORDER");
  if (env == nullptr) {
    return false;
  }
  return std::strcmp(env, "true") == 0;
};
void save_to_cache(const std::string& content) {
  const char* cache_filename = std::getenv("XLA_REORDER_CACHE_FILE");
  if (cache_filename == nullptr) {
    cache_filename = "reorder.cache";
  }
  std::ofstream file(cache_filename);
  file << content;
  file.close();
};
bool is_cache_enable() {
  const char* cache_filename = std::getenv("XLA_REORDER_CACHE_FILE");
  if (cache_filename == nullptr) {
    cache_filename = "reorder.cache";
  }
  // check file exists
  return std::filesystem::exists(cache_filename);
};
std::string load_from_cache() {
  const char* cache_filename = std::getenv("XLA_REORDER_CACHE_FILE");
  if (cache_filename == nullptr) {
    cache_filename = "reorder.cache";
  }

  std::ifstream file(cache_filename);
  std::string content;
  std::string line;
  while (std::getline(file, line)) {
    content += line;
  }
  file.close();
  return content;
};
bool accuired_reorder_lock() {
  const char* lock_filename = std::getenv("XLA_REORDER_LOCK_FILE");
  if (lock_filename == nullptr) {
    lock_filename = "/tmp/reorder.lock";
  }
  mode_t m = umask(0);
  int fd = open(lock_filename, O_RDWR | O_CREAT, 0666);
  umask(m);
  if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) < 0) {
    close(fd);
    fd = -1;
  }
  return fd >= 0;
};
void release_reorder_lock() {
  const char* lock_filename = std::getenv("XLA_REORDER_LOCK_FILE");
  if (lock_filename == nullptr) {
    lock_filename = "/tmp/reorder.lock";
  }
  mode_t m = umask(0);
  int fd = open(lock_filename, O_RDWR | O_CREAT, 0666);
  umask(m);
  if (fd >= 0 && flock(fd, LOCK_UN) < 0) {
    close(fd);
    fd = -1;
  }
};
int get_cpu_number() {
  // return 8;
  return std::thread::hardware_concurrency();
}
}  // namespace reorder
enum class NodeType {
  kCompute = 0,
  kCommunication = 1

};
static bool IsSingleChannel(NodeType nodetype) {
  return nodetype == NodeType::kCommunication;
}

struct TaskType {
  IntVar start;
  IntVar end;
  IntervalVar interval;
};
using CostType = int64_t;  // we can change it to double?

// TODO: using LPNode to abstract LPContainer and LPContainerDAG
template <typename ElementType>
class LPNode {
 public:
  virtual const std::string GetName() const = 0;
  virtual const int UUID() = 0;
  virtual CostType GetCost() const = 0;
  virtual void SetStart(CostType start) = 0;
  virtual CostType GetStart() = 0;
  virtual bool IsComputation() const = 0;
  virtual bool IsCommunication() const = 0;
  virtual NodeType GetType() const = 0;
  virtual bool HasValue() const = 0;
  virtual ElementType GetValue() const = 0;
  virtual void AddDep(LPNode* dep, CostType cost) = 0;
  virtual const std::vector<std::tuple<LPNode*, CostType>> GetDeps() const = 0;
  virtual void Freeze() = 0;

 private:
  std::vector<std::tuple<LPNode*, CostType>> deps_;
};

// LPContainer is a template class, it can be used to store any type of data
// 1. LPContainer<const HloInstruction*>; using to store one instruction
// 2. LPContainer<const LPContainerDAG>; using to store a graph of
// instructions,decrese lp hard
// 3. LPContainer<const Stage>; maybe we can use it to store a pipeline stage
template <typename ElementType>
class LPContainer {
 public:
  // create a LPContainer with inner_element, cost and type
  LPContainer(ElementType inner_element, CostType cost, NodeType type)
      : inner_element_(inner_element), cost_(cost), type_(type) {
    uuid_ = reinterpret_cast<uintptr_t>(this);
  };
  ~LPContainer() { deps_.clear(); };
  const std::string GetName() const { return inner_element_->ToShortString(); }
  const int UUID() { return inner_element_->unique_id(); }

  CostType GetCost() const { return cost_; }
  void SetStart(CostType start) { startat_ = start; }
  CostType GetStart() { return startat_; }
  // speed up reorder, we can set a hint start time
  CostType GetHintStart() { return hint_start_; }
  void SetHintStart(CostType start) { hint_start_ = start; }

  // Get the type of the container: compute or communication
  bool IsComputation() const { return type_ == NodeType::kCompute; }
  bool IsCommunication() const { return type_ == NodeType::kCommunication; }

  NodeType GetType() const { return type_; }

  const bool HasValue() { return inner_element_ != nullptr; }
  const std::vector<ElementType> GetValues() {
    return std::vector<ElementType>{inner_element_};
  }
  // Add a dep of this container, cost is the cost of the edge; this Container
  // will be executed after dep
  void AddDep(LPContainer* dep, CostType cost, NodeType nodetype);
  // Get all deps of the container
  const std::vector<std::tuple<LPContainer*, CostType, NodeType>> GetDeps()
      const {
    return deps_;
  }
  /**
   * Checks if the given dependency in this container.
   *
   * @param dep The dependency to check.
   * @return True if the dependency in this container, false otherwise.
   */
  bool HasDep(LPContainer* dep) {
    for (auto d : deps_) {
      if (std::get<0>(d) == dep) {
        return true;
      }
    }
    return false;
  }
  // when a container is frozen, it can not be add deps
  void Freeze() { frozen_ = true; }

 private:
  CostType cost_;
  CostType startat_;
  CostType hint_start_ = -1;
  NodeType type_;
  ElementType inner_element_;
  // deps store the edge
  std::vector<std::tuple<LPContainer*, CostType, NodeType>> deps_;
  bool frozen_ =
      false;  // if it is frozen, it can not be changed,such as add deps
  uintptr_t uuid_;
  std::string name_;  // edge need a name
};
// LPContainerDAG is a graph of container, it can be used to store the DAG of
// container be used as a atomic unit of LPContainer
template <typename ElementType>
class LPContainerDAG : public LPContainer<ElementType> {
  // we can use InstructionDAG to get memory effect order
 public:
  // maintain a DAG of inner elements
  struct DAGEdge {
    LPContainer<ElementType>* from;
    LPContainer<ElementType>* to;
    CostType cost;
  };
  // create a  LPContainerDAG with one element
  LPContainerDAG(ElementType inner_element, CostType cost, NodeType type)
      : LPContainer<ElementType>(inner_element, cost, type) {
    // TODO: there should not create element?
    auto ele = new LPContainer<ElementType>(inner_element, cost, type);
    inner_elements.push_back(ele);
  };
  bool IsIn(LPContainer<ElementType>* a);
  // which container can be put together:1. they have the same type 2. they have
  // dep between them
  // static bool CanFused(LPContainerDAG<ElementType>* a,
  // LPContainerDAG<ElementType>* b);

  // override LPContainer
  const std::string GetName() {
    std::string name = "LPContainerDAG{";
    for (auto ele : inner_elements) {
      name += ele->GetName();
      name += "\n";
    }
    name += "}";
    return name;
  }
  const int UUID() { return inner_elements[0]->UUID(); }
  const bool HasValue() { return inner_elements.size() > 0; }
  const std::vector<ElementType> GetValues() {
    std::vector<ElementType> values;
    for (auto ele : inner_elements) {
      for (auto inst : ele->GetValues()) {
        values.push_back(inst);
      }
    }
    return values;
  }
  // AddChild, child should maintain the deps before
  void AddToDAG(LPContainer<ElementType>* child);
  const std::vector<LPContainer<ElementType>*> GetInnerElements() const {
    return inner_elements;
  }
  // merge other LPContainerDAG to this LPContainerDAG,then destroy other
  // LPContainerDAG
  Status MergeFrom(LPContainerDAG<ElementType>* other);

 private:
  std::set<LPContainer<ElementType>*> operands_;
  std::vector<LPContainer<ElementType>*> inner_elements;
  // maintain edges between inner_elements
  std::vector<DAGEdge> edges_;
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
  Status Solve();
  // find instruction,if not exist, return error
  StatusOr<ContainerType*> FindInstructionLPNode(ElementType instruction);
  // find LPNode by instruction,if not exist,create it
  ContainerType* FindLPNodeOrCreate(ElementType instruction, CostType cost,
                                    NodeType type);
  // ContainerType*
  std::vector<ContainerType*> GetSortedNodes();
  // for debug: render graph viz
  void RenderGraphviz(std::string filename) const;
  // for debug: render gantt chart
  void RenderGantt(std::string filename) const;
  // set max start time as horizon
  void SetHorizon(uint32_t horizon) { horizon_ = horizon; }
  StatusOr<TaskType> FindTask(ContainerType* node);
  bool NodeHasAddTasks(ContainerType* node);
  CostType GetNodeStartTime(ContainerType* node);
  void AddNodeToTask(ContainerType* node, TaskType task);
  StatusOr<TaskType> AddNodeToTask(ContainerType* node);

 private:
  StatusOr<bool> AddEdgesNoOverlap(ContainerType* node);
  CpModelBuilder cp_model_;
  bool verbose_ = false;
  std::unordered_map<int, ContainerType*> uuid2container;
  std::vector<ContainerType*> nodes_;
  uint32_t horizon_ = std::numeric_limits<uint32_t>::max();
  absl::flat_hash_map<int, std::tuple<ContainerType*, TaskType>>
      node_to_task_;  // every node hold interval_var,show what time it start
                      // and end
  // channels can be overlap each other
  std::map<NodeType, std::vector<IntervalVar>> channel_to_intervals_;
  std::map<int, int64_t> node_starttime_;
};
}  // namespace xla
#endif  // XLA_AUTO_REORDER_H_
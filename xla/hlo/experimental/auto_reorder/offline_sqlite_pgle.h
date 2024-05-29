/*


*/
#ifndef XLA_AUTO_REORDER_OFFLINE_SQLITE_PGLE_H_
#define XLA_AUTO_REORDER_OFFLINE_SQLITE_PGLE_H_
#include <string>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "llvm/Support/MD5.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>
#include <utility>
#include <vector>

#include <iostream>
#include <fstream>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "xla/hlo/experimental/auto_reorder/instr_profile_info.pb.h"
#include "xla/hlo/experimental/auto_reorder/common.h"
#include "xla/hlo/utils/hlo_query.h"

#include "sqlite3.h"
namespace xla {
using InstructionVector = absl::InlinedVector<HloInstruction*, 2>;
namespace auto_reorder {
struct HloLatencyStats {
  uint32_t hits = 0;
  uint32_t misses = 0;
};
std::string kSQLCreate = R"(create table inst_profiler (
            name string,
            operandCount real,
            opcode real,
            version real,
            operandTypes string,

            resultTypes string,
            operand_hash string,
            operandSizes_str string,
            replicaGroupSize real,
            customCallTarget string,

            cost real,
            hwinfo string
        ) )";
std::vector<std::string> kSQLCreateIndexes = {
    "create index idx_opcode on inst_profiler(opcode, operand_hash);",
    "create index idx_customcall on inst_profiler(opcode, customCallTarget, "
    "operand_hash);",
    "create index idx_comm on inst_profiler(opcode, replicaGroupSize, "
    "operand_hash);",
};
std::string kSQLInsert = R"(insert into inst_profiler values)";
size_t kParamsCount = 12;
std::string BatchInsertSQL(size_t size);
/*
Using DB(sqlite) for Profile Guided Latency Estimator
    1. compute the hash of instruction operands using InstOperandHash
    2. save instruction profiler info to database.
    3. lookup database when using pgle, if instruction have save opcode and
operand_hash, then use the latency
*/
class OfflineSQLitePgle : public LatencyEstimator {
 public:
  // create a memory db;
  OfflineSQLitePgle(const SchedulerConfig& config,
                    std::unique_ptr<LatencyEstimator> latency_estimator,
                    const std::string& db_path);
  ~OfflineSQLitePgle();
  //   static std::string InstOperandHash(const xla::HloInstruction* inst);
  static std::string InstOperandHash(const xla::HloInstruction& inst);
  // Create a new table and create it's indexes
  absl::Status CreateDB();
  // Open a db for read/write
  absl::Status OpenDB(const std::string& db_path);
  absl::Status SaveMemoryDBToFile(const std::string& db_path);
  // insert a vector of InstrProfileInfo into database
  absl::Status BatchInsertInstrProfileInfo(
      std::vector<xla::auto_reorder::InstrProfileInfo>& infos);

  // HloGraphNode GetInstr() return const xla::HloInstruction
  absl::StatusOr<double> QueryInstCost(const xla::HloInstruction& inst) const;

  // parse computation instructions, put it into hlo_module_info,but no cost;
  static absl::Status ParseToInstProfileInfo(
      HloComputation* computation,
      absl::flat_hash_map<std::string, xla::auto_reorder::InstrProfileInfo>*
          hlo_module_info);

  // LatencyEstimator interface
  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override;
  TimeCost NodeCost(const HloInstruction* instr) const override;
  int CyclesPerMicrosecond() const override {
    return latency_estimator_->CyclesPerMicrosecond();
  }

  static constexpr TimeCost kLowCost = 1.0;
  static constexpr TimeCost kLowLatency = 1.0;

 private:
  const xla::SchedulerConfig config_;
  std::unique_ptr<xla::LatencyEstimator> latency_estimator_;

  // communicate hash,include process_group info
  static std::string CommunicateInstOperandHash(
      const xla::HloInstruction& inst);
  // custom call hash,include custom call target
  static std::string CustomCallInstOperandHash(const xla::HloInstruction& inst);

  // default inst hash, using CommonHash
  static std::string DefaultInstOperandHash(const xla::HloInstruction& inst);

  // update common hash logic,such as operand size, operand type
  static void CommonHash(const xla::HloInstruction& inst,
                         llvm::MD5* md5_instance);
  static void CommonOperandsHash(const InstructionVector& operands,
                                 llvm::MD5* md5_instance);
  static std::string GetHash(llvm::MD5* md5_instance);
  // convert InstrProfileInfo to sql values; using sqlite3_bind_* so it's sql
  // safe
  static absl::Status BindInstInfoToSql(
      xla::auto_reorder::InstrProfileInfo info, sqlite3_stmt* stmt,
      size_t index);
  absl::StatusOr<double> QueryInstCostByCode(HloOpcode code,
                                             std::string hash) const;

 protected:
  bool is_memory_db_;
  sqlite3* client_;
  HloLatencyStats stats;
};
}  // namespace auto_reorder
}  // namespace xla

#endif  // XLA_AUTO_REORDER_OFFLINE_SQLITE_PGLE_H_
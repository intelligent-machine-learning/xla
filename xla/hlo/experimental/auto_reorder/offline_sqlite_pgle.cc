#include "xla/hlo/experimental/auto_reorder/offline_sqlite_pgle.h"

namespace xla {
namespace auto_reorder {
#define EXPECT_SQL_EQ(a, b)                             \
  {                                                     \
    auto ret = a;                                       \
    if (ret != b) {                                     \
      VLOG(2) << "expect " << b << " but error:" << ret \
              << sqlite3_errstr(ret);                   \
      return absl::InternalError("expect failed");      \
    }                                                   \
  }

OfflineSQLitePgle::OfflineSQLitePgle(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const std::string& db_path)
    : config_(config), latency_estimator_(std::move(latency_estimator)) {
  if (db_path == ":memory:") {
    is_memory_db_ = true;
  } else {
    is_memory_db_ = false;
  }
  auto status = OpenDB(db_path);
  if (!status.ok()) {
    throw std::runtime_error("OpenDB failed");
  }
}
OfflineSQLitePgle::~OfflineSQLitePgle() {
  if (client_ != nullptr) {
    sqlite3_close(client_);
  }
}

absl::Status OfflineSQLitePgle::ParseToInstProfileInfo(
    HloComputation* computation,
    absl::flat_hash_map<std::string, xla::auto_reorder::InstrProfileInfo>*
        hlo_module_info) {
  for (auto* instr : computation->instructions()) {
    // instr to json
    //{name:"name",opcode:"opcode",operand_count:1,operand_names:["a"],operand_types:["f32"],shape:"[1,2,3]",result_type:"f32",result_shape:"[1,2,3]",result_element_type:"f32",result_element_shape:"[1,2,3]",result_element_count:6}
    //  TODO: should we need shard info?
    //  TODO: custom call
    //  there are 3 category  instrs:
    //  1. custom call, include GEMM now; record its input shape/dtype
    //  2. communicate call, include async reducescatter ; record its input
    //  shape/dtype
    //  3. other,
    HloInstructionProto instr_origin_proto = instr->ToProto();
    auto_reorder::InstrProfileInfo instr_info;
    auto_reorder::Size ret_size;
    instr_info.set_name(instr_origin_proto.name());
    HloOpcode code = instr->opcode();
    instr_info.set_opname(HloOpcodeString(code).data());
    instr_info.set_opcode(static_cast<uint32_t>(code));

    // set operand count/type/size
    instr_info.set_operand_count(instr->operand_count());

    for (auto operand : instr->operands()) {
      Shape op_shape = operand->shape();
      // operand dtype

      instr_info.add_operand_types(PrimitiveType_Name(op_shape.element_type()));
      auto_reorder::Size* op_size = instr_info.add_operand_sizes();
      op_size->set_dim(op_shape.dimensions_size());
      for (size_t i = 0; i < op_shape.dimensions_size(); i++) {
        op_size->add_sizes(op_shape.dimensions(i));
      }
    }

    Shape shape = instr->shape();
    instr_info.mutable_result_size()->set_dim(shape.dimensions_size());
    for (size_t i = 0; i < shape.dimensions_size(); i++) {
      instr_info.mutable_result_size()->add_sizes(shape.dimensions(i));
    }
    // result_types
    instr_info.add_result_types(PrimitiveType_Name(shape.element_type()));
    instr_info.set_operand_hash(
        xla::auto_reorder::OfflineSQLitePgle::InstOperandHash(*instr));
    // TODO: set hw info; for now, we don't have hw info
    //  instr_info.set_hwinfo

    switch (code) {
      // TODO: fusion
      case HloOpcode::kFusion: {
        std::vector<absl::string_view> fusion_names;
        for (const HloComputation* computation : instr->called_computations()) {
          fusion_names.push_back(computation->name());
        }
        instr_info.set_custom_call_target(absl::StrJoin(fusion_names, ";"));
        break;
      }
      case HloOpcode::kCustomCall: {
        instr_info.set_custom_call_target(instr->custom_call_target());
        break;
      }
      case HloOpcode::kReduceScatter:
      case HloOpcode::kAllGather:
      case HloOpcode::kAllGatherStart:
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllReduceStart:
      case HloOpcode::kCollectivePermuteStart: {  // comm op need record
                                                  // process group
        // example :{{1,2,3,4}}, {{1,2},{3,4}}
        std::vector<xla::ReplicaGroup> replica_groups = instr->replica_groups();
        uint16_t group_id = 0;
        uint32_t replica_group_size = 1;
        for (auto replica_group : replica_groups) {
          xla::auto_reorder::ReplicaGroup* group =
              instr_info.add_process_groups();
          group->set_replica_group_id(group_id);
          group_id++;
          replica_group_size *= replica_group.replica_ids_size();
          for (auto replica : replica_group.replica_ids()) {
            group->add_replica_ids(replica);
          }
        }
        // for sql group by and index
        instr_info.set_replica_group_size(replica_group_size);

        // instr_info.set_process_group();
        break;
      }
      case HloOpcode::kAsyncStart: {
        // if(instr)
      }
      default:
        break;

    }  // end switch
    hlo_module_info->emplace(instr_origin_proto.name(), instr_info);
  }  // end for instrs
  return absl::OkStatus();
}
absl::Status OfflineSQLitePgle::SaveMemoryDBToFile(const std::string& db_path) {
  if (is_memory_db_) {
    sqlite3* file_client;
    int rc = sqlite3_open(db_path.c_str(), &file_client);
    if (rc) {
      VLOG(2) << "Can't open database: " << sqlite3_errmsg(file_client);
      return absl::InternalError("Can't open database");
    } else {
      VLOG(2) << "Opened database for save memory database successfully";
    }
    // ATTACH DATABASE ':memory:' AS memdb;
    sqlite3_backup* pBackup =
        sqlite3_backup_init(file_client, "main", client_, "main");
    if (pBackup) {
      sqlite3_backup_step(pBackup, -1);
      sqlite3_backup_finish(pBackup);
      VLOG(2) << "backup finish";
    }
    sqlite3_close(file_client);
    // sqlite3_close(client_);
    // client_=nullptr;
    return absl::OkStatus();
  }
}
std::string BatchInsertSQL(size_t insert_number) {
  std::vector<std::string> values;
  std::vector<std::string> params_sql_vector;
  for (size_t i = 0; i < kParamsCount; i++) {
    params_sql_vector.push_back("?");
  }
  std::string params_sql =
      absl::StrCat("(", absl::StrJoin(params_sql_vector, ","), ")");
  for (size_t i = 0; i < insert_number; i++) {
    // 12 params
    values.push_back(params_sql);
  }
  return absl::StrCat(kSQLInsert, absl::StrJoin(values, ","));
}
absl::Status OfflineSQLitePgle::BindInstInfoToSql(
    xla::auto_reorder::InstrProfileInfo info, sqlite3_stmt* stmt,
    size_t index) {
  // where there is string type, use single quote '
  // ('data3', 1, 2,'asd'),

  // some nested struct only show string
  std::string operand_sizes_str;
  std::vector<std::string> operand_sizes_strs;
  google::protobuf::util::JsonPrintOptions options;
  options.always_print_primitive_fields = true;
  google::protobuf::util::Status st;
  for (const auto& operand_size : info.operand_sizes()) {
    std::string operand_size_str;
    st = google::protobuf::util::MessageToJsonString(
        operand_size, &operand_size_str, options);
    if (!st.ok()) {
      VLOG(2) << "MessageToJsonString failed: " << st.message();
      return absl::InternalError("MessageToJsonString parse operand_size fail");
    }
    operand_sizes_strs.push_back(operand_size_str);
  }
  operand_sizes_str = absl::StrJoin(operand_sizes_strs, ",");
  operand_sizes_strs.clear();
  std::string process_groups_str;
  std::vector<std::string> process_groups_strs;
  for (const auto& process_group : info.process_groups()) {
    std::string process_group_str;
    st = google::protobuf::util::MessageToJsonString(
        process_group, &process_group_str, options);
    process_group.SerializeToString(&process_group_str);
    if (!st.ok()) {
      VLOG(2) << "MessageToJsonString failed: " << st.message();
      return absl::InternalError(
          "MessageToJsonString parse process_group fail");
    }
    process_groups_strs.push_back(process_group_str);
  }
  process_groups_str = absl::StrJoin(process_groups_strs, ",");
  // sqlite index start from 1
  size_t current_pos = index * kParamsCount + 1;
  process_groups_strs.clear();
  // 25   /* 2nd parameter to sqlite3_bind out of range */
  EXPECT_SQL_EQ(sqlite3_bind_text(stmt, current_pos + 0, info.name().c_str(),
                                  -1, SQLITE_TRANSIENT),
                SQLITE_OK);
  EXPECT_SQL_EQ(sqlite3_bind_int(stmt, current_pos + 1, info.operand_count()),
                SQLITE_OK);
  EXPECT_SQL_EQ(sqlite3_bind_int(stmt, current_pos + 2, info.opcode()),
                SQLITE_OK);
  EXPECT_SQL_EQ(sqlite3_bind_int(stmt, current_pos + 3, info.version()),
                SQLITE_OK);
  EXPECT_SQL_EQ(
      sqlite3_bind_text(stmt, current_pos + 4,
                        absl::StrJoin(info.operand_types(), ",").c_str(), -1,
                        SQLITE_TRANSIENT),
      SQLITE_OK);

  EXPECT_SQL_EQ(
      sqlite3_bind_text(stmt, current_pos + 5,
                        absl::StrJoin(info.result_types(), ",").c_str(), -1,
                        SQLITE_TRANSIENT),
      SQLITE_OK);
  EXPECT_SQL_EQ(
      sqlite3_bind_text(stmt, current_pos + 6, info.operand_hash().c_str(), -1,
                        SQLITE_TRANSIENT),
      SQLITE_OK);
  EXPECT_SQL_EQ(
      sqlite3_bind_text(stmt, current_pos + 7, operand_sizes_str.c_str(), -1,
                        SQLITE_TRANSIENT),
      SQLITE_OK);
  EXPECT_SQL_EQ(
      sqlite3_bind_text(stmt, current_pos + 8, process_groups_str.c_str(), -1,
                        SQLITE_TRANSIENT),
      SQLITE_OK);
  EXPECT_SQL_EQ(sqlite3_bind_text(stmt, current_pos + 9,
                                  info.custom_call_target().c_str(), -1,
                                  SQLITE_TRANSIENT),
                SQLITE_OK);

  EXPECT_SQL_EQ(sqlite3_bind_double(stmt, current_pos + 10, info.cost()),
                SQLITE_OK);
  EXPECT_SQL_EQ(sqlite3_bind_text(stmt, current_pos + 11, info.hwinfo().c_str(),
                                  -1, SQLITE_TRANSIENT),
                SQLITE_OK);
  return absl::OkStatus();
}
absl::Status OfflineSQLitePgle::BatchInsertInstrProfileInfo(
    std::vector<xla::auto_reorder::InstrProfileInfo>& infos) {
  std::string params_sql = BatchInsertSQL(infos.size());
  sqlite3_stmt* stmt;
  int code = sqlite3_prepare_v2(client_, params_sql.c_str(), -1, &stmt, 0);
  if (code != SQLITE_OK) {
    return absl::InternalError(
        absl::StrCat("Can't prepare statement:", sqlite3_errmsg(client_)));
  }
  VLOG(5) << "finish prepare stmt, it have parameters: "
          << sqlite3_bind_parameter_count(stmt);
  size_t index = 0;
  for (auto info : infos) {
    auto st = BindInstInfoToSql(info, stmt, index);
    TF_RET_CHECK(st.ok()) << "BindInstInfoToSql " << index << " failed";
    index++;
  }

  EXPECT_SQL_EQ(sqlite3_step(stmt), SQLITE_DONE);
  VLOG(5) << "insert sql:" << params_sql << "sucess";
  EXPECT_SQL_EQ(sqlite3_reset(stmt), SQLITE_OK);
  return absl::OkStatus();
}
absl::Status OfflineSQLitePgle::CreateDB() {
  int code =
      sqlite3_exec(client_, kSQLCreate.c_str(), nullptr, nullptr, nullptr);
  if (code != SQLITE_OK) {
    VLOG(2) << "Can't create table: " << sqlite3_errmsg(client_) << std::endl;
    return absl::InternalError("Can't create table");
  }
  for (auto sql : kSQLCreateIndexes) {
    code = sqlite3_exec(client_, sql.c_str(), nullptr, nullptr, nullptr);
    if (code != SQLITE_OK) {
      VLOG(2) << "Can't create index: " << sqlite3_errmsg(client_) << std::endl;
      return absl::InternalError("Can't create index");
    }
  }
  return absl::OkStatus();
}
absl::Status OfflineSQLitePgle::OpenDB(const std::string& db_path) {
  int rc = sqlite3_open(db_path.c_str(), &client_);
  if (rc) {
    VLOG(2) << "Can't open database: " << sqlite3_errmsg(client_) << std::endl;
    return absl::InternalError("Can't open database");
  } else {
    VLOG(3) << "Opened database successfully" << std::endl;
  }
  return absl::OkStatus();
}

std::string OfflineSQLitePgle::InstOperandHash(
    const xla::HloInstruction& inst) {
  HloOpcode code = inst.opcode();
  switch (code) {
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCollectivePermuteStart:
      return CommunicateInstOperandHash(inst);
    case HloOpcode::kCustomCall:
      return CustomCallInstOperandHash(inst);
    default:
      return DefaultInstOperandHash(inst);
  }
}
void OfflineSQLitePgle::CommonOperandsHash(const InstructionVector& operands,
                                           llvm::MD5* md5_instance) {
  for (auto operand : operands) {
    // BF16;1,2,24,.FP16;1,2,24,.
    Shape op_shape = operand->shape();
    // operand dtype BF16
    md5_instance->update(PrimitiveType_Name(op_shape.element_type()));
    md5_instance->update(";");
    md5_instance->update(std::to_string(op_shape.dimensions_size()));
    for (size_t i = 0; i < op_shape.dimensions_size(); i++) {
      // split by , so 12,4 is diff with 1,24
      md5_instance->update(std::to_string(op_shape.dimensions(i)));
      md5_instance->update(",");
    }
    md5_instance->update(".");
  }
}
void OfflineSQLitePgle::CommonHash(const xla::HloInstruction& inst,
                                   llvm::MD5* md5_instance) {
  CommonOperandsHash(inst.operands(), md5_instance);
}
std::string OfflineSQLitePgle::GetHash(llvm::MD5* md5_instance) {
  llvm::MD5::MD5Result result;
  md5_instance->final(result);
  // toHex(ArrayRef<uint8_t> Input, bool LowerCase = false)-> std::string

  return llvm::toHex(result, true);
}
void UpdateCommonicationHash(std::vector<xla::ReplicaGroup> replica_groups,
                             llvm::MD5* md5_instance) {
  uint16_t group_id = 0;
  uint32_t replica_group_size = 1;
  for (auto replica_group : replica_groups) {
    group_id++;
    md5_instance->update(";");
    md5_instance->update(std::to_string(group_id));

    replica_group_size *= replica_group.replica_ids_size();
    for (auto replica : replica_group.replica_ids()) {
      md5_instance->update(std::to_string(replica));
      md5_instance->update(",");
    }
    md5_instance->update(".");
  }
}
std::string OfflineSQLitePgle::CommunicateInstOperandHash(
    const xla::HloInstruction& inst) {
  std::unique_ptr<llvm::MD5> md5_instance(new llvm::MD5());
  CommonHash(inst, md5_instance.get());
  std::vector<xla::ReplicaGroup> replica_groups = inst.replica_groups();
  UpdateCommonicationHash(replica_groups, md5_instance.get());
  return GetHash(md5_instance.get());
}
/*static*/

std::string OfflineSQLitePgle::CustomCallInstOperandHash(
    const xla::HloInstruction& inst) {
  std::unique_ptr<llvm::MD5> md5_instance(new llvm::MD5());
  CommonHash(inst, md5_instance.get());
  md5_instance->update(inst.custom_call_target());
  return GetHash(md5_instance.get());
}
/*static*/
std::string OfflineSQLitePgle::DefaultInstOperandHash(
    const xla::HloInstruction& inst) {
  std::unique_ptr<llvm::MD5> md5_instance(new llvm::MD5());
  CommonHash(inst, md5_instance.get());
  return GetHash(md5_instance.get());
}
absl::StatusOr<double> OfflineSQLitePgle::QueryInstCostByCode(
    HloOpcode code, std::string hash) const {
  std::string groupby_func;
  switch (code) {
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCollectivePermuteStart:
      groupby_func = "min";
    default:
      groupby_func = "avg";
  }
  std::string sql = absl::StrCat(
      "select ", groupby_func, "(cost) from inst_profiler where opcode=", code,
      " and operand_hash='", hash, "';");
  sqlite3_stmt* stmt;
  int sqlcode = sqlite3_prepare_v2(client_, sql.c_str(), -1, &stmt, 0);
  if (sqlcode != SQLITE_OK) {
    return absl::InternalError(
        absl::StrCat("Can't prepare statement:", sqlite3_errmsg(client_)));
  }
  sqlcode = sqlite3_step(stmt);
  if (sqlcode != SQLITE_ROW) {
    VLOG(2) << "Can't query table: " << sqlite3_errmsg(client_) << std::endl;
    return absl::NotFoundError("Can't find inst in table");
  }
  double cost = sqlite3_column_double(stmt, 0);
  return cost;
}
absl::StatusOr<double> OfflineSQLitePgle::QueryInstCost(
    const xla::HloInstruction& inst) const {
  /*
  1. HloGraphNode GetInstr() return const xla::HloInstruction
  2. NodeCost(const xla::HloInstruction* inst) input is const
  xla::HloInstruction*
  */
  std::string hash = OfflineSQLitePgle::InstOperandHash(inst);
  HloOpcode code = inst.opcode();
  return QueryInstCostByCode(code, hash);
}
// reference: xla/service/profile_guided_latency_estimator.cc
// ProfileGuidedLatencyEstimator load from database
LatencyEstimator::TimeCost OfflineSQLitePgle::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  /*
     load from sqlite. when from&target is async start/done, there is latency
     between them
  */
  static constexpr HloGraphNode::TimeCost kLowLatency = 1.0;
  const HloOpcode from_op = from.GetInstr().opcode();
  if (!config_.schedule_send_recvs &&
      (from_op == HloOpcode::kSend || from_op == HloOpcode::kRecv)) {
    return kLowLatency;
  }
  absl::StatusOr<double> cost_or_status = QueryInstCost(from.GetInstr());
  // can't found inst and it's async wrapper inst, query inner inst
  if (!cost_or_status.ok() &&
      (from.GetInstr().opcode() == HloOpcode::kAsyncStart ||
       from.GetInstr().opcode() == HloOpcode::kAsyncDone)) {
    const HloInstruction* wrapped_inst =
        from.GetInstr().async_wrapped_instruction();
    VLOG(10) << "PGLE lookup async wrapped instruction: "
             << wrapped_inst->name() << " in " << from.GetInstr().name();
    cost_or_status = QueryInstCost(*wrapped_inst);
    if (!cost_or_status.ok()) {
      return latency_estimator_->GetLatencyBetween(from, target);
    }
    return cost_or_status.value() * CyclesPerMicrosecond();
  }

  if (!cost_or_status.ok()) {
    return latency_estimator_->GetLatencyBetween(from, target);
  }

  absl::StatusOr<double> cost_or_status2 = QueryInstCost(target.GetInstr());
  if (!cost_or_status2.ok() &&
      (target.GetInstr().opcode() == HloOpcode::kAsyncStart ||
       target.GetInstr().opcode() == HloOpcode::kAsyncDone)) {
    const HloInstruction* wrapped_inst =
        target.GetInstr().async_wrapped_instruction();
    cost_or_status2 = QueryInstCost(*wrapped_inst);
  }
  if (cost_or_status2.ok()) {
    VLOG(10) << "PGLE found latency between " << from.GetInstr().name()
             << " and " << target.GetInstr().name() << " in latency info";
    return cost_or_status2.value() * CyclesPerMicrosecond();
  }

  // For async-start/done instructions, if there is no entry in latencies, fall
  // back to using instruction cost as the latency.
  if (cost_or_status.ok() && IsAsyncPair(from, target)) {
    VLOG(10) << "PGLE found latency for async op " << from.GetInstr().name()
             << " and (assumed)" << target.GetInstr().name()
             << " in instruction costs";
    return cost_or_status.value() * CyclesPerMicrosecond();
  }

  return latency_estimator_->GetLatencyBetween(from, target);
}
LatencyEstimator::TimeCost OfflineSQLitePgle::NodeCost(
    const HloInstruction* instr) const {
  if (hlo_query::IsAsyncCollectiveStartOp(instr, /*include_send_recv=*/true) ||
      hlo_query::IsAsyncCollectiveDoneOp(instr, /*include_send_recv=*/true)) {
    static constexpr TimeCost kLowCost = 1.0;
    return kLowCost;
  }
  const absl::StatusOr<double> cost_or_status = QueryInstCost(*instr);
  if (cost_or_status.ok()) {
    return cost_or_status.value();
  }
  return latency_estimator_->NodeCost(instr);
}
}  // namespace auto_reorder
}  // namespace xla
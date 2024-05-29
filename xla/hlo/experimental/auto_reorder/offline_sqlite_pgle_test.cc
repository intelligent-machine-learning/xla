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
#include <stdlib.h>
#include "xla/hlo/experimental/auto_reorder/offline_sqlite_pgle.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/hlo/experimental/auto_reorder/convert_xplane.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include <gtest/gtest.h>
#include "xla/hlo/experimental/auto_reorder/auto_reorder.h"

namespace xla {
class OfflineSqlitePGLETestcase : public HloTestBase {
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
  StatusOr<HloComputation*> MakeTestComputation(HloModule* module) {
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
    auto dot0 = builder.AddInstruction(
        HloInstruction::CreateDot(shape, p0, p1, dot_dnums, precision_config));

    HloInstruction* ar_start0 =
        builder.AddInstruction(HloInstruction::CreateAllReduceStart(
            shape, {p0}, add_reducer,
            /*replica_groups=*/CreateReplicaGroups({{0, 1}}),
            /*constrain_layout=*/false, /*channel_id=*/1,
            /*use_global_device_ids=*/true));
    HloInstruction* ar_done0 =
        builder.AddInstruction(HloInstruction::CreateUnary(
            shape, HloOpcode::kAllReduceDone, ar_start0));
    std::vector<HloInstruction*> compute_vec = {ar_done0, dot0};
    auto ret = builder.AddInstruction(HloInstruction::CreateTuple(compute_vec));
    std::unique_ptr<xla::HloComputation> computation = builder.Build();
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
    auto post_insts_order =
        module->entry_computation()->MakeInstructionPostOrder();
    module->schedule().set_sequence(module->entry_computation(),
                                    post_insts_order);
    return entry_computation;
  }

 protected:
  void SetUp() override { setenv("XLA_AUTOREORDER_TIMEOUT", "60", 1); }
  void TearDown() override { unsetenv("XLA_AUTOREORDER_TIMEOUT"); }
  std::vector<ReplicaGroup> CreateReplicaGroups(
      absl::Span<const std::vector<int64_t>> groups) {
    std::vector<xla::ReplicaGroup> replica_groups(groups.size());
    for (int64_t i = 0; i < groups.size(); ++i) {
      *replica_groups[i].mutable_replica_ids() = {groups[i].begin(),
                                                  groups[i].end()};
    }
    return replica_groups;
  }
};
TEST_F(OfflineSqlitePGLETestcase, ConvertPDO) {
  GTEST_SKIP() << "convert Database here; dev will disable";
  // get filepath from env
  const char* env = std::getenv("XLA_AUTOREORDER_XPLANE_DIR");
  if (env == nullptr) {
    GTEST_SKIP() << "have no set XLA_AUTOREORDER_XPLANE_DIR env skip";
  }
  auto status =
      ConvertXplaneToFile(env, "/root/tb/llama_xla_trace_2n16g/llama_fdo.db");
  std::cout << status.message() << std::endl;
  EXPECT_TRUE(status.ok());
}
TEST_F(OfflineSqlitePGLETestcase, SQLitePGLEWithAutoReorderTest) {
  auto gpu_latency_estimator = std::make_unique<GpuLatencyEstimator>();
  int64_t memory_limit = 80 * 1000 * 1000;
  SchedulerConfig sched_config = GetSchedulerConfig(memory_limit);

  std::cout << "start test sqlite pgle" << std::endl;
  std::unique_ptr<HloModule> hlo_module = CreateNewUnverifiedModule(TestName());
  auto st = MakeTestComputation(hlo_module.get());
  std::cout << "start create module finish" << std::endl;

  HloComputation* entry_computation = st.value();
  absl::flat_hash_map<std::string, xla::auto_reorder::InstrProfileInfo>
      hlo_module_info;
  std::unique_ptr<xla::auto_reorder::OfflineSQLitePgle> dbbase_pgle =
      std::make_unique<xla::auto_reorder::OfflineSQLitePgle>(
          sched_config, std::move(gpu_latency_estimator), ":memory:");
  auto status = dbbase_pgle->CreateDB();
  EXPECT_TRUE(status.ok());

  std::cout << "get instructions success" << std::endl;

  // from computation,save it to hlo_module_info
  status = xla::auto_reorder::OfflineSQLitePgle::ParseToInstProfileInfo(
      entry_computation, &hlo_module_info);
  EXPECT_TRUE(status.ok()) << "parse to inst profile info failed"
                           << status.message();

  std::vector<xla::auto_reorder::InstrProfileInfo> waiting_insert_profile;
  for (auto& [name, info] : hlo_module_info) {
    info.set_cost(10.0);
    waiting_insert_profile.push_back(info);
  }
  status = dbbase_pgle->BatchInsertInstrProfileInfo(waiting_insert_profile);
  EXPECT_TRUE(status.ok());

  // query
  for (const xla::HloInstruction* inst : entry_computation->instructions()) {
    absl::StatusOr<double> cost_or_status = dbbase_pgle->QueryInstCost(inst);
    EXPECT_TRUE(cost_or_status.ok());
    EXPECT_DOUBLE_EQ(cost_or_status.value(), 10.0);
  }
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
  // use pass
  auto async_tracker = std::make_unique<AsyncTracker>(sched_config);
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_bytes, async_tracker.get(), dbbase_pgle.get(), sched_config);

  HloPassPipeline pipeline("latency-hiding-scheduler");

  auto test_pass =
      AutoReorderPass(std::move(dbbase_pgle), std::move(async_tracker),
                      std::move(scheduler_core), shape_size_bytes);
  StatusOr<bool> success_or_status = test_pass.Run(hlo_module.get());
  EXPECT_TRUE(success_or_status.ok());
  EXPECT_TRUE(success_or_status.value());
}
TEST_F(OfflineSqlitePGLETestcase, SqliteTest) {
  sqlite3* db;
  char* zErrMsg = 0;
  int rc;
  const char* sql;
  sqlite3_stmt* stmt;

  /* 打开数据库 */
  rc = sqlite3_open(":memory:", &db);
  EXPECT_TRUE(rc == SQLITE_OK) << "无法打开数据库:" << sqlite3_errmsg(db);

  std::string createTablesql =
      "CREATE TABLE COMPANY("
      "ID INT PRIMARY KEY     NOT NULL,"
      "NAME           TEXT    NOT NULL);";
  /* 创建表 */
  rc = sqlite3_exec(db, createTablesql.c_str(), 0, 0, &zErrMsg);
  EXPECT_TRUE(rc == SQLITE_OK) << "Table created successfully";

  /* 创建SQL语句 */
  sql = "INSERT INTO COMPANY (ID,NAME) VALUES (?,?),(?,?)";

  /* 准备语句 */
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  EXPECT_TRUE(rc == SQLITE_OK) << "Prepared statement created successfully";

  /* 插入多条数据 */
  for (int i = 0; i < 2; i++) {
    /* 绑定参数 */
    rc = sqlite3_bind_int(stmt, i * 2 + 1, i);
    EXPECT_TRUE(rc == SQLITE_OK) << "无法绑定参数:" << sqlite3_errmsg(db);

    char name[50];
    sprintf(name, "Company'asd' %d", i);
    rc = sqlite3_bind_text(stmt, i * 2 + 2, name, -1, SQLITE_TRANSIENT);
    EXPECT_TRUE(rc == SQLITE_OK);
  }
  /* 执行语句 */
  rc = sqlite3_step(stmt);
  EXPECT_TRUE(rc == SQLITE_DONE);
  /* 重置语句 */
  rc = sqlite3_reset(stmt);
  EXPECT_TRUE(rc == SQLITE_OK);
  // sqlite3_finalize(stmt);
  /*select count*/
  sql = "SELECT COUNT(*) FROM COMPANY";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  EXPECT_TRUE(rc == SQLITE_OK);
  rc = sqlite3_step(stmt);
  EXPECT_TRUE(rc == SQLITE_ROW);
  int count = sqlite3_column_int(stmt, 0);
  EXPECT_TRUE(count == 2);

  /* 销毁准备语句 */
  sqlite3_finalize(stmt);

  /* 关闭数据库 */
  sqlite3_close(db);
}
}  // namespace xla

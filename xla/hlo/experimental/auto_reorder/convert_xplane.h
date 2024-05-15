// decouple xla/python deps, xla/python need
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"

// #include "xla/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla.pb.h"

#include "tsl/platform/env.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/convert/xla_op_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/utils/file_system_utils.h"
#include "tsl/profiler/utils/tf_xplane_visitor.h"
#include "tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {

constexpr char kXPlanePb[] = "xplane.pb";
constexpr char kCostNameSep[] = "::";

using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;
using tsl::profiler::CreateTfXPlaneVisitor;
using tsl::profiler::FindPlanesWithPrefix;
using tsl::profiler::FindPlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::HostEventType;
using tsl::profiler::IsInternalEvent;
using tsl::profiler::ProfilerJoinPath;
using tsl::profiler::StatType;
using tsl::profiler::XEventMetadataVisitor;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XLineVisitor;
using tsl::profiler::XPlaneVisitor;
using tsl::profiler::XStatVisitor;

// Latency info for a single HLO instruction.
struct HloLatencyInfo {
  std::vector<double> durations;
};

Status ConvertXplaneToProfiledInstructionsProto(
    std::vector<tensorflow::profiler::XSpace> xspaces,
    tensorflow::profiler::ProfiledInstructionsProto*
        profiled_instructions_proto);
Status ConvertXplaneUnderLogdirToProfiledInstructionsProto(
    const std::string& logdir, tensorflow::profiler::ProfiledInstructionsProto*
                                   profiled_instructions_proto);

Status ConvertXplaneToFile(const std::string& xplane_dir,
                           const std::string& output_filename);


} //namespace xla
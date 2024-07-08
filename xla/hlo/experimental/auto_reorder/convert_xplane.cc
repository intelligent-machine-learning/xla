#include "xla/hlo/experimental/auto_reorder/convert_xplane.h"

namespace xla {

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

// maintain info for next PGLE
/*
steps

1. parse module, maintain map of {instr_name: instr_info} func:
GetHloInstrProfileInfo
2. parse xplane, maintain map of {instr_name: instr_latency}, update instr_info
3. write to sqlite file, as origin DB
4. use origin DB,group instr_type/shape to generate summary DB
5. use summary DB for next PGLE
*/

void GetXPlaneLatencyInfo(
    const XPlaneVisitor& xplane,
    absl::flat_hash_map<std::string, HloLatencyInfo>* hlo_latency_info) {
  // Iterate events.
  xplane.ForEachLine([hlo_latency_info](const XLineVisitor& xline) {
    if (xline.DisplayName() == tsl::profiler::kXlaAsyncOpLineName) {
      return;
    }
    VLOG(5) << "Processing line: " << xline.DisplayName();
    xline.ForEachEvent([hlo_latency_info](const XEventVisitor& xevent) {
      int64_t event_type =
          xevent.Type().value_or(HostEventType::kUnknownHostEventType);
      if (IsInternalEvent(event_type)) return;
      std::optional<std::string> hlo_name = std::nullopt;

      auto for_each_stat = [&](const XStatVisitor& stat) {
        if (stat.ValueCase() == tsl::profiler::XStat::VALUE_NOT_SET) return;
        // Store latency information for HLOs.
        if (stat.Name() == GetStatTypeStr(StatType::kHloOp)) {
          hlo_name = stat.ToString();
        }
      };
      xevent.Metadata().ForEachStat(for_each_stat);
      xevent.ForEachStat(for_each_stat);
      double latency = static_cast<double>(xevent.DurationNs()) / 1e3;
      if (!hlo_name.has_value()) {
        // why some hlo have no name?
        return;
      }
      VLOG(5) << "hlo_name: " << hlo_name.value_or("N/A")
              << "latency:" << latency;

      std::string key = hlo_name.value();
      (*hlo_latency_info)[key].durations.emplace_back(latency);
    });
  });
}

std::unique_ptr<xla::HloModule> CreateModuleFromProto(
    const xla::HloModuleProto& proto) {
  auto config = xla::HloModule::CreateModuleConfigFromProto(proto, {});
  if (config.ok()) {
    auto module = xla::HloModule::CreateFromProto(proto, config.value());
    if (module.ok()) {
      return std::move(*module);
    }
  }
  return nullptr;
}

Status GetHloInstrProfileInfo(
    const xla::HloModuleProto& hlo_module_proto,
    absl::flat_hash_map<std::string, xla::auto_reorder::InstrProfileInfo>*
        hlo_module_info) {
  std::unique_ptr<xla::HloModule> hlo_module =
      CreateModuleFromProto(hlo_module_proto);
  if (hlo_module == nullptr) {
    return absl::InternalError("Failed to create HloModule from proto");
  }
  VLOG(4) << "success get hlo module from proto";
  for (HloComputation* computation :
       hlo_module->MakeNonfusionComputations({})) {
    auto status = xla::auto_reorder::OfflineSQLitePgle::ParseToInstProfileInfo(
        computation, hlo_module_info);
    if (!status.ok()) {
      return status;
    }
  }  // end for computations
  return absl::OkStatus();
}

void GetXPlaneHloModuleProfileInfo(
    const XPlaneVisitor& xplane,
    absl::flat_hash_map<std::string, xla::auto_reorder::InstrProfileInfo>*
        hlo_module_info) {
  // Iterate events.
  xplane.ForEachEventMetadata([&](const XEventMetadataVisitor& event_metadata) {
    event_metadata.ForEachStat([&](const XStatVisitor& stat) {
      xla::HloProto hlo_proto;
      if (tsl::ParseProtoUnlimited(&hlo_proto, stat.BytesValue().data(),
                                   stat.BytesValue().size())) {
        const xla::HloModuleProto& hlo_module_proto = hlo_proto.hlo_module();

        Status st = GetHloInstrProfileInfo(hlo_module_proto, hlo_module_info);
        if (!st.ok()) {
          VLOG(4) << "Failed to get HloInstrProfileInfo from HloModuleProto";
        }
      }
    });
  });
}

Status ConvertXplaneToOfflineSQLitePgle(
    std::vector<tensorflow::profiler::XSpace> xspaces,
    xla::auto_reorder::OfflineSQLitePgle* dbbase_pgle) {
  // name to HloLatencyInfo
  absl::flat_hash_map<std::string, HloLatencyInfo> hlo_latency_info;
  // name to HloInstructionProto
  absl::flat_hash_map<std::string, xla::auto_reorder::InstrProfileInfo>
      hlo_instr_profile_info;
  google::protobuf::util::JsonPrintOptions options;
  options.always_print_primitive_fields = true;
  google::protobuf::util::Status st;
  // st = google::protobuf::util::MessageToJsonString(profile_proto,
  // &json_string, options); if(!st.ok()) {
  //   return absl::InternalError("Failed to convert ProfiledInstructionsProto
  //   to json");
  // }
  // Iterate through each host.
  for (const XSpace& xspace : xspaces) {
    const XPlane* metadata_plane =
        FindPlaneWithName(xspace, tsl::profiler::kMetadataPlaneName);
    if (metadata_plane != nullptr) {
      XPlaneVisitor xplane = CreateTfXPlaneVisitor(metadata_plane);
      GetXPlaneHloModuleProfileInfo(xplane, &hlo_instr_profile_info);
    }
    std::vector<const XPlane*> device_planes =
        FindPlanesWithPrefix(xspace, tsl::profiler::kGpuPlanePrefix);
    // We don't expect GPU and TPU planes and custom devices to be present in
    // the same XSpace.
    if (device_planes.empty()) {
      VLOG(4) << "No GPU plane found, try to find TPU plane.";
      device_planes =
          FindPlanesWithPrefix(xspace, tsl::profiler::kTpuPlanePrefix);
    }
    if (device_planes.empty()) {
      VLOG(4) << "No TPU plane found, try to find custom device plane.";
      device_planes =
          FindPlanesWithPrefix(xspace, tsl::profiler::kCustomPlanePrefix);
    }
    // Go over each device plane.
    for (const XPlane* device_plane : device_planes) {
      XPlaneVisitor xplane = CreateTfXPlaneVisitor(device_plane);
      GetXPlaneLatencyInfo(xplane, &hlo_latency_info);
    }
  }
  if (hlo_instr_profile_info.empty()) {
    VLOG(4) << "No HLO instruction info found in xplane protobuf.";
    return absl::InternalError("No HLO latency info found in xplane");
  }
  if (hlo_latency_info.empty()) {
    VLOG(4) << "No HLO latency info found in xplane.";
    return absl::InternalError("No HLO latency info found in xplane");
  }
  xla::auto_reorder::HloLatencyStats stats;
  std::vector<xla::auto_reorder::InstrProfileInfo> waiting_insert_profile;
  // Get the mean duration for each hlo and store into the proto.
  for (const auto& iter : hlo_latency_info) {
    // auto* cost = profiled_instructions_proto->add_costs();
    auto profile_it = hlo_instr_profile_info.find(iter.first);
    if (profile_it == hlo_instr_profile_info.end()) {
      VLOG(4) << "No instr info found for instr: " << iter.first;
      stats.misses++;
      continue;
    } else {
      stats.hits++;
    }

    auto_reorder::InstrProfileInfo cost = profile_it->second;
    for (auto duration : iter.second.durations) {
      // here need copy
      auto_reorder::InstrProfileInfo copyed_cost;
      copyed_cost.CopyFrom(cost);
      copyed_cost.set_cost(duration);
      waiting_insert_profile.push_back(copyed_cost);
      if (waiting_insert_profile.size() >= kBatchInsertSize) {
        auto status =
            dbbase_pgle->BatchInsertInstrProfileInfo(waiting_insert_profile);
        if (!status.ok()) {
          return status;
        }
        waiting_insert_profile.clear();
      }
    }
  }
  if (waiting_insert_profile.size() > 0) {
    auto status =
        dbbase_pgle->BatchInsertInstrProfileInfo(waiting_insert_profile);
    if (!status.ok()) {
      return status;
    }
    waiting_insert_profile.clear();
  }
  VLOG(4) << "Lookup inst profiler, Hits: " << stats.hits
          << " Misses: " << stats.misses;
  return OkStatus();
}
Status ConvertXplaneUnderLogdirToOfflineSQLitePgle(
    const std::string& logdir,
    xla::auto_reorder::OfflineSQLitePgle* database_pgle) {
  // Find the xplane files for each host under logdir.
  std::vector<std::string> children_path;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetChildren(logdir, &children_path));
  if (children_path.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find file under: ", logdir));
  }
  std::vector<tensorflow::profiler::XSpace> xspaces;
  for (const std::string& child_path : children_path) {
    if (absl::StrContains(child_path, kXPlanePb)) {
      std::string xspace_path = ProfilerJoinPath(logdir, child_path);
      tensorflow::profiler::XSpace xspace;
      TF_RETURN_IF_ERROR(
          ReadBinaryProto(tsl::Env::Default(), xspace_path, &xspace));
      xspaces.emplace_back(xspace);
    }
  }
  if (xspaces.size() == 0) {
    return absl::NotFoundError(
        absl::StrCat("Could not find xplane file under: ", logdir));
  }
  VLOG(3) << "Have load " << xspaces.size() << " xspaces";
  return ConvertXplaneToOfflineSQLitePgle(xspaces, database_pgle);
}

Status ConvertXplaneToFile(const std::string& xplane_dir,
                           const std::string& output_filename) {
  // tensorflow::profiler::ProfiledInstructionsProto profile_proto;
  std::vector<std::string> jsonline_vector;
  auto gpu_latency_estimator = std::make_unique<GpuLatencyEstimator>();
  int64_t memory_limit = 80 * 1000 * 1000;
  SchedulerConfig config = GetSchedulerConfig(memory_limit);

  std::unique_ptr<xla::auto_reorder::OfflineSQLitePgle> dbbase_pgle =
      std::make_unique<xla::auto_reorder::OfflineSQLitePgle>(
          config, std::move(gpu_latency_estimator), ":memory:");
  auto status = dbbase_pgle->CreateDB();
  if (!status.ok()) {
    return status;
  }
  status = ConvertXplaneUnderLogdirToOfflineSQLitePgle(xplane_dir,
                                                       dbbase_pgle.get());
  if (!status.ok()) {
    return status;
  }
  // open file,write jsonline
  status = dbbase_pgle->SaveMemoryDBToFile(output_filename);
  if (!status.ok()) {
    return status;
  }

  return OkStatus();
}

}  // namespace xla
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
void GetXPlaneLatencyInfo(
    const XPlaneVisitor& xplane,
    const absl::flat_hash_map<std::string, std::string>& hlo_module_info,
    absl::flat_hash_map<std::string, HloLatencyInfo>* hlo_latency_info) {
  // Iterate events.
  xplane.ForEachLine([hlo_latency_info,
                      hlo_module_info](const XLineVisitor& xline) {
    if (xline.DisplayName() == tsl::profiler::kXlaAsyncOpLineName) {
      return;
    }
    xline.ForEachEvent([hlo_latency_info,
                        hlo_module_info](const XEventVisitor& xevent) {
      int64_t event_type =
          xevent.Type().value_or(HostEventType::kUnknownHostEventType);
      if (IsInternalEvent(event_type)) return;
      std::optional<std::string> hlo_name = std::nullopt;
      std::optional<std::string> hlo_module_name = std::nullopt;
      std::optional<std::string> fingerprint = std::nullopt;
      std::optional<int64_t> program_id = std::nullopt;

      auto for_each_stat = [&](const XStatVisitor& stat) {
        if (stat.ValueCase() == tsl::profiler::XStat::VALUE_NOT_SET) return;
        // Store latency information for HLOs.
        if (stat.Name() == GetStatTypeStr(StatType::kHloOp)) {
          hlo_name = stat.ToString();
        }
        if (stat.Name() == GetStatTypeStr(StatType::kProgramId)) {
          program_id = stat.IntValue();
        }
        if (stat.Name() == GetStatTypeStr(StatType::kHloModule)) {
          hlo_module_name = stat.ToString();
        }
      };
      xevent.Metadata().ForEachStat(for_each_stat);
      xevent.ForEachStat(for_each_stat);
      if (!hlo_name.has_value() || !hlo_module_name.has_value()) {
        return;
      }

      if (hlo_module_name.has_value()) {
        std::string fingerprint_key = hlo_module_name.value();
        if (program_id.has_value()) {
          fingerprint_key = tsl::profiler::HloModuleNameWithProgramId(
              hlo_module_name.value(), program_id.value());
        }
        if (hlo_module_info.contains(fingerprint_key)) {
          fingerprint = hlo_module_info.at(fingerprint_key);
        }
      }
      double latency = static_cast<double>(xevent.DurationNs()) / 1e3;
      std::string key = hlo_name.value();
      if (fingerprint.has_value()) {
        key = absl::StrCat(fingerprint.value(), kCostNameSep, hlo_name.value());
      }
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

std::optional<std::string> GetHloModuleFingerprint(
    const xla::HloModuleProto& hlo_module_proto) {
  std::unique_ptr<xla::HloModule> hlo_module =
      CreateModuleFromProto(hlo_module_proto);
  if (hlo_module == nullptr) {
    return std::nullopt;
  }
  const auto& map = hlo_module->entry_computation()
                        ->root_instruction()
                        ->frontend_attributes()
                        .map();
  auto it = map.find("fingerprint_before_lhs");
  if (it != map.end()) {
    return it->second;
  }
  return std::nullopt;
}

void GetXPlaneHloModuleInfo(
    const XPlaneVisitor& xplane,
    absl::flat_hash_map<std::string, std::string>* hlo_module_info) {
  // Iterate events.
  xplane.ForEachEventMetadata([&](const XEventMetadataVisitor& event_metadata) {
    event_metadata.ForEachStat([&](const XStatVisitor& stat) {
      xla::HloProto hlo_proto;
      if (tsl::ParseProtoUnlimited(&hlo_proto, stat.BytesValue().data(),
                                   stat.BytesValue().size())) {
        const xla::HloModuleProto& hlo_module_proto = hlo_proto.hlo_module();

        std::optional<std::string> fingerprint =
            GetHloModuleFingerprint(hlo_module_proto);
        if (fingerprint.has_value()) {
          std::string key_with_id = tsl::profiler::HloModuleNameWithProgramId(
              hlo_module_proto.name(), hlo_module_proto.id());
          (*hlo_module_info)[key_with_id] = fingerprint.value();
        }
      }
    });
  });
}

Status ConvertXplaneToProfiledInstructionsProto(
    std::vector<tensorflow::profiler::XSpace> xspaces,
    tensorflow::profiler::ProfiledInstructionsProto*
        profiled_instructions_proto) {
  absl::flat_hash_map<std::string, HloLatencyInfo> hlo_latency_info;
  absl::flat_hash_map<std::string, std::string> hlo_module_info;
  // Iterate through each host.
  for (const XSpace& xspace : xspaces) {
    const XPlane* metadata_plane =
        FindPlaneWithName(xspace, tsl::profiler::kMetadataPlaneName);
    if (metadata_plane != nullptr) {
      XPlaneVisitor xplane = CreateTfXPlaneVisitor(metadata_plane);
      GetXPlaneHloModuleInfo(xplane, &hlo_module_info);
    }
    std::vector<const XPlane*> device_planes =
        FindPlanesWithPrefix(xspace, tsl::profiler::kGpuPlanePrefix);
    // We don't expect GPU and TPU planes and custom devices to be present in
    // the same XSpace.
    if (device_planes.empty()) {
      device_planes =
          FindPlanesWithPrefix(xspace, tsl::profiler::kTpuPlanePrefix);
    }
    if (device_planes.empty()) {
      device_planes =
          FindPlanesWithPrefix(xspace, tsl::profiler::kCustomPlanePrefix);
    }
    // Go over each device plane.
    for (const XPlane* device_plane : device_planes) {
      XPlaneVisitor xplane = CreateTfXPlaneVisitor(device_plane);
      GetXPlaneLatencyInfo(xplane, hlo_module_info, &hlo_latency_info);
    }
  }

  // Get the mean duration for each hlo and store into the proto.
  for (const auto& iter : hlo_latency_info) {
    auto* cost = profiled_instructions_proto->add_costs();
    std::vector<double> durations = iter.second.durations;
    double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
    cost->set_cost_us(sum / durations.size());
    cost->set_name(iter.first);
  }

  return OkStatus();
}
Status ConvertXplaneUnderLogdirToProfiledInstructionsProto(
    const std::string& logdir, tensorflow::profiler::ProfiledInstructionsProto*
                                   profiled_instructions_proto) {
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
  return ConvertXplaneToProfiledInstructionsProto(xspaces,
                                                  profiled_instructions_proto);
}

Status ConvertXplaneToFile(const std::string& xplane_dir,
                           const std::string& output_filename) {
  tensorflow::profiler::ProfiledInstructionsProto profile_proto;
  auto status = ConvertXplaneUnderLogdirToProfiledInstructionsProto(
      xplane_dir, &profile_proto);
  if (!status.ok()) {
    return status;
  }
  std::string profile_proto_str = profile_proto.SerializeAsString();
  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
      tsl::Env::Default(), output_filename, profile_proto_str));
  return OkStatus();
}

}  // namespace xla
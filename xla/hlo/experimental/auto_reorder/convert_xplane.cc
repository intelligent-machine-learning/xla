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
  VLOG(5) << "success get hlo module from proto";
  for (HloComputation* computation :
       hlo_module->MakeNonfusionComputations({})) {
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

      instr_info.set_opcode(static_cast<uint32_t>(code));

      // set operand count/type/size
      instr_info.set_operand_count(instr->operand_count());
      for (auto operand : instr->operands()) {
        Shape op_shape = operand->shape();
        // operand dtype

        instr_info.add_operand_types(
            PrimitiveType_Name(op_shape.element_type()));
        auto_reorder::Size* op_size = instr_info.add_operand_sizes();
        op_size->set_rank(op_shape.dimensions_size());
        for (size_t i = 0; i < op_shape.dimensions_size(); i++) {
          op_size->add_sizes(op_shape.dimensions(i));
        }
      }

      Shape shape = instr->shape();
      instr_info.mutable_result_size()->set_rank(shape.dimensions_size());
      for (size_t i = 0; i < shape.dimensions_size(); i++) {
        /* code */
        instr_info.mutable_result_size()->add_sizes(shape.dimensions(i));
      }
      // custom call
      switch (code) {
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
          std::vector<ReplicaGroup> replica_groups = instr->replica_groups();
          uint16_t group_id = 0;
          for (auto replica_group : replica_groups) {
            xla::auto_reorder::ReplicaGroup* group =
                instr_info.add_process_groups();
            group->set_replica_group_id(group_id);
            group_id++;
            for (auto replica : replica_group.replica_ids()) {
              group->add_replica_ids(replica);
            }
          }

          // instr_info.set_process_group();
          break;
        }
        case HloOpcode::kAsyncStart: {
          // get async inner instr
        }
        default:
          break;

      }  // end switch
      hlo_module_info->emplace(instr_origin_proto.name(), instr_info);
    }  // end for instrs
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
          VLOG(5) << "Failed to get HloInstrProfileInfo from HloModuleProto";
        }
      }
    });
  });
}

Status ConvertXplaneToProfiledJSONLine(
    std::vector<tensorflow::profiler::XSpace> xspaces,
    std::vector<std::string>* jsonline_vector) {
  // name to HloLatencyInfo
  absl::flat_hash_map<std::string, HloLatencyInfo> hlo_latency_info;
  // name to HloInstructionProto
  absl::flat_hash_map<std::string, xla::auto_reorder::InstrProfileInfo>
      hlo_instr_profile_info;
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
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
      VLOG(5) << "No GPU plane found, try to find TPU plane.";
      device_planes =
          FindPlanesWithPrefix(xspace, tsl::profiler::kTpuPlanePrefix);
    }
    if (device_planes.empty()) {
      VLOG(5) << "No TPU plane found, try to find custom device plane.";
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
    VLOG(5) << "No HLO instruction info found in xplane protobuf.";
    return absl::InternalError("No HLO latency info found in xplane");
  }
  if (hlo_latency_info.empty()) {
    VLOG(5) << "No HLO latency info found in xplane.";
    return absl::InternalError("No HLO latency info found in xplane");
  }
  HloLatencyStats stats;

  // Get the mean duration for each hlo and store into the proto.
  for (const auto& iter : hlo_latency_info) {
    // auto* cost = profiled_instructions_proto->add_costs();
    auto profile_it = hlo_instr_profile_info.find(iter.first);
    if (profile_it == hlo_instr_profile_info.end()) {
      VLOG(5) << "No instr info found for instr: " << iter.first;
      stats.misses++;
      continue;
    } else {
      stats.hits++;
    }

    auto_reorder::InstrProfileInfo cost = profile_it->second;
    for (auto duration : iter.second.durations) {
      // cost->add_durations(d);
      cost.set_cost(duration);
      std::string json_string;
      auto st = google::protobuf::util::MessageToJsonString(cost, &json_string,
                                                            options);
      if (!st.ok()) {
        return absl::InternalError(
            "Failed to convert ProfiledInstructionsProto to json");
      }
      jsonline_vector->push_back(json_string);
    }
  }
  VLOG(5) << "Lookup inst profiler, Hits: " << stats.hits
          << " Misses: " << stats.misses;
  return OkStatus();
}
Status ConvertXplaneUnderLogdirToProfiledInstructionsProto(
    const std::string& logdir, std::vector<std::string>* jsonline_vector) {
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
  return ConvertXplaneToProfiledJSONLine(xspaces, jsonline_vector);
}

Status ConvertXplaneToFile(const std::string& xplane_dir,
                           const std::string& output_filename) {
  tensorflow::profiler::ProfiledInstructionsProto profile_proto;
  std::vector<std::string> jsonline_vector;
  auto status = ConvertXplaneUnderLogdirToProfiledInstructionsProto(
      xplane_dir, &jsonline_vector);
  if (!status.ok()) {
    return status;
  }
  // open file,write jsonline
  std::ofstream fout = std::ofstream(output_filename);
  if (!fout.is_open()) {
    return absl::InternalError("Failed to open file for writing");
  }
  for (const std::string& jsonline : jsonline_vector) {
    fout << jsonline << std::endl;
  }
  return OkStatus();
}

}  // namespace xla
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
#include "xla/tests/hlo_test_base.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include <gtest/gtest.h>

namespace xla {

class FusionRultTestcase : public HloTestBase {
    public:
    StatusOr<std::unique_ptr<HloModule>> ParseHloText(
      absl::string_view hlo_string) {
    TF_ASSIGN_OR_RETURN(
        auto hlo_module,
        ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest()));
    return StatusOr<std::unique_ptr<HloModule>>(std::move(hlo_module));
    }
    absl::string_view origin_string = R"(
        HloModule module
        ENTRY fused_convert {
        param_0.30218 = f32[4096]{0} parameter(0)
        broadcast.8326.10 = f32[10,512,4096]{2,1,0} broadcast(param_0.30218), dimensions={2}
        param_1.32688 = f32[10,512,4096]{2,1,0} parameter(1)
        p2 = f32[10,512]{1,0} parameter(2)
        broadcast.8327.12 = f32[10,512,4096]{2,1,0} broadcast(p2), dimensions={0,1}
        multiply.1 = f32[10,512,4096]{2,1,0} multiply(param_1.32688, broadcast.8327.12)
        multiply.2 = f32[10,512,4096]{2,1,0} multiply(broadcast.8326.10, multiply.1)
        ROOT convert.6823.3 = bf16[10,512,4096]{2,1,0} convert(multiply.2)
        })";
    absl::string_view optimize_string = R"(
        HloModule module
        ENTRY fused_convert {
        p0 = f32[4096]{0} parameter(0)
        convert.p0 = bf16[4096]{0} convert(p0)
        broadcast.p0 = bf16[10,512,4096]{2,1,0} broadcast(convert.p0), dimensions={2}

        p1 = f32[10,512,4096]{2,1,0} parameter(1)
        convert.p1 = bf16[10,512,4096] convert(p1)

        p2 = f32[10,512]{1,0} parameter(2)
        convert.p2 = bf16[10,512]{1,0} convert(p2)
        broadcast.p2 = bf16[10,512,4096]{2,1,0} broadcast(convert.p2), dimensions={0,1}

        multiply.1 = bf16[10,512,4096]{2,1,0} multiply(convert.p1, broadcast.p2)
        ROOT multiply.2 = bf16[10,512,4096]{2,1,0} multiply(broadcast.p0, multiply.1)
        })";
};//FusionRultTestcase
TEST_F(FusionRultTestcase, TestFusion){
    //measure: how many memory access

    TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
    TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(origin_string, *client));
//   TF_ASSERT_OK_AND_ASSIGN(auto origin_module, ParseHloText(origin_string));
//   TF_ASSERT_OK_AND_ASSIGN(auto optimized_module, ParseHloText(optimize_string));

}//TestFusion

}//xla
#include "xla/hlo/experimental/auto_reorder/convert_xplane.h"
int main(int argc, char* argv[]) {
  std::string xplane_dir;
  std::string output_filename;
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <xplane_dir> <output_filename>\n";
    return 1;
  }
  xplane_dir = argv[1];
  output_filename = argv[2];
  auto status = xla::ConvertXplaneToFile(xplane_dir, output_filename);
  if (!status.ok()) {
    std::cerr << "Error: " << status.message() << "\n";
    return 1;
  }
  return 0;
}
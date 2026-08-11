#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/types.h"

namespace {

std::vector<char> read_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open input: " + path);
  const auto size = input.tellg();
  if (size < 0) throw std::runtime_error("cannot determine input size: " + path);
  std::vector<char> bytes(static_cast<size_t>(size));
  input.seekg(0);
  if (!input.read(bytes.data(), size)) throw std::runtime_error("cannot read input: " + path);
  return bytes;
}

void write_file(const std::string &path, const void *data, size_t size) {
  std::ofstream output(path, std::ios::binary);
  if (!output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size))) {
    throw std::runtime_error("cannot write output: " + path);
  }
}

std::string json_string(const std::string &value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character)
                 << std::dec;
        } else {
          output << character;
        }
    }
  }
  output << '"';
  return output.str();
}

std::string data_type(mindspore::DataType type) {
  switch (type) {
    case mindspore::DataType::kNumberTypeBool: return "bool";
    case mindspore::DataType::kNumberTypeInt8: return "int8";
    case mindspore::DataType::kNumberTypeInt16: return "int16";
    case mindspore::DataType::kNumberTypeInt32: return "int32";
    case mindspore::DataType::kNumberTypeInt64: return "int64";
    case mindspore::DataType::kNumberTypeUInt8: return "uint8";
    case mindspore::DataType::kNumberTypeUInt16: return "uint16";
    case mindspore::DataType::kNumberTypeUInt32: return "uint32";
    case mindspore::DataType::kNumberTypeUInt64: return "uint64";
    case mindspore::DataType::kNumberTypeFloat16: return "float16";
    case mindspore::DataType::kNumberTypeFloat32: return "float32";
    case mindspore::DataType::kNumberTypeFloat64: return "float64";
    default: return "unknown";
  }
}

std::string shape_json(const std::vector<int64_t> &shape) {
  std::ostringstream output;
  output << '[';
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) output << ',';
    output << shape[index];
  }
  output << ']';
  return output.str();
}

std::string quantization_json(const std::vector<mindspore::QuantParam> &parameters) {
  std::ostringstream output;
  output << '[';
  for (size_t index = 0; index < parameters.size(); ++index) {
    if (index != 0) output << ',';
    const auto &parameter = parameters[index];
    output << "{\"bitNum\":" << parameter.bit_num << ",\"scale\":" << std::setprecision(17)
           << parameter.scale << ",\"zeroPoint\":" << parameter.zero_point << '}';
  }
  output << ']';
  return output.str();
}

std::string tensor_json(const mindspore::MSTensor &tensor, size_t index) {
  std::ostringstream output;
  output << "{\"bytes\":" << tensor.DataSize() << ",\"dtype\":" << json_string(data_type(tensor.DataType()))
         << ",\"index\":" << index << ",\"name\":" << json_string(tensor.Name())
         << ",\"quantization\":" << quantization_json(tensor.QuantParams())
         << ",\"shape\":" << shape_json(tensor.Shape()) << '}';
  return output.str();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: mindspore-host-runner MODEL.ms INPUT.f32le OUTPUT.f32le\n";
    return 64;
  }
  try {
    auto context = std::make_shared<mindspore::Context>();
    context->SetThreadNum(1);
    context->MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());
    mindspore::Model model;
    const auto build = model.Build(argv[1], mindspore::ModelType::kMindIR, context);
    if (!build.IsOk()) {
      std::cerr << "Model::Build failed: " << build.ToString() << '\n';
      return 2;
    }

    auto inputs = model.GetInputs();
    if (inputs.size() != 1) throw std::runtime_error("expected exactly one model input");
    const auto input_bytes = read_file(argv[2]);
    if (inputs[0].DataType() != mindspore::DataType::kNumberTypeFloat32) {
      throw std::runtime_error("model input is not float32");
    }
    if (inputs[0].DataSize() != input_bytes.size()) throw std::runtime_error("input byte size mismatch");
    std::memcpy(inputs[0].MutableData(), input_bytes.data(), input_bytes.size());

    std::vector<mindspore::MSTensor> outputs;
    const auto predict = model.Predict(inputs, &outputs);
    if (!predict.IsOk()) {
      std::cerr << "Model::Predict failed: " << predict.ToString() << '\n';
      return 3;
    }
    if (outputs.size() != 1) throw std::runtime_error("expected exactly one model output");
    const auto output_data = outputs[0].Data();
    if (!output_data) throw std::runtime_error("model output data is null");
    write_file(argv[3], output_data.get(), outputs[0].DataSize());

    std::cout << "{\"build\":\"success\",\"inputs\":[" << tensor_json(inputs[0], 0)
              << "],\"outputs\":[" << tensor_json(outputs[0], 0) << "],\"predict\":\"success\"}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

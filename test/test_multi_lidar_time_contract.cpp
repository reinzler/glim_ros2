#include <glim_ros/multi_lidar_cloud_merger.hpp>

#include <glim/util/config.hpp>
#include <glim/util/ros_cloud_converter.hpp>
#include <glim/util/time_keeper.hpp>

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace {

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

enum class TimeEncoding {
  None,
  UInt32RelNs,
  Float64AbsNs,
  Float64AbsSec,
  Float64NaN,
};

void check(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_near(double actual, double expected, double eps, const std::string& message) {
  if (std::abs(actual - expected) > eps) {
    throw std::runtime_error(
      message + " actual=" + std::to_string(actual) +
      " expected=" + std::to_string(expected));
  }
}

builtin_interfaces::msg::Time stamp_from_sec(const double stamp) {
  builtin_interfaces::msg::Time out;
  double sec_floor = std::floor(stamp);
  long long nsec = std::llround((stamp - sec_floor) * 1e9);
  if (nsec >= 1000000000LL) {
    nsec -= 1000000000LL;
    sec_floor += 1.0;
  }
  out.sec = static_cast<std::int32_t>(sec_floor);
  out.nanosec = static_cast<std::uint32_t>(nsec);
  return out;
}

double stamp_to_sec(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

PointField make_field(
    const std::string& name,
    const std::uint32_t offset,
    const std::uint8_t datatype) {
  PointField field;
  field.name = name;
  field.offset = offset;
  field.datatype = datatype;
  field.count = 1;
  return field;
}

template <typename T>
void write_value(PointCloud2& msg, const std::size_t point_index, const std::uint32_t offset, const T& value) {
  const std::size_t byte_index = static_cast<std::size_t>(msg.point_step) * point_index + offset;
  std::memcpy(msg.data.data() + byte_index, &value, sizeof(T));
}

template <typename T>
T read_value(const PointCloud2& msg, const std::size_t point_index, const std::uint32_t offset) {
  T value{};
  const std::size_t byte_index = static_cast<std::size_t>(msg.point_step) * point_index + offset;
  std::memcpy(&value, msg.data.data() + byte_index, sizeof(T));
  return value;
}

PointCloud2::ConstSharedPtr make_cloud(
    const std::string& frame_id,
    const double header_stamp,
    const std::string& time_field_name,
    const TimeEncoding encoding,
    const std::vector<double>& relative_times_sec) {
  auto msg = std::make_shared<PointCloud2>();
  msg->header.frame_id = frame_id;
  msg->header.stamp = stamp_from_sec(header_stamp);
  msg->height = 1;
  msg->width = static_cast<std::uint32_t>(relative_times_sec.size());
  msg->is_bigendian = false;
  msg->is_dense = true;

  std::uint32_t offset = 0;
  msg->fields.push_back(make_field("x", offset, PointField::FLOAT32));
  offset += sizeof(float);
  msg->fields.push_back(make_field("y", offset, PointField::FLOAT32));
  offset += sizeof(float);
  msg->fields.push_back(make_field("z", offset, PointField::FLOAT32));
  offset += sizeof(float);
  msg->fields.push_back(make_field("intensity", offset, PointField::FLOAT32));
  offset += sizeof(float);

  const std::uint32_t time_offset = offset;
  std::uint8_t time_datatype = 0;
  if (encoding == TimeEncoding::UInt32RelNs) {
    time_datatype = PointField::UINT32;
    msg->fields.push_back(make_field(time_field_name, offset, time_datatype));
    offset += sizeof(std::uint32_t);
  } else if (
      encoding == TimeEncoding::Float64AbsNs ||
      encoding == TimeEncoding::Float64AbsSec ||
      encoding == TimeEncoding::Float64NaN) {
    time_datatype = PointField::FLOAT64;
    msg->fields.push_back(make_field(time_field_name, offset, time_datatype));
    offset += sizeof(double);
  }

  const std::uint32_t line_offset = offset;
  msg->fields.push_back(make_field("line", offset, PointField::UINT16));
  offset += sizeof(std::uint16_t);

  msg->point_step = offset;
  msg->row_step = msg->point_step * msg->width;
  msg->data.resize(msg->row_step);

  for (std::size_t i = 0; i < relative_times_sec.size(); i++) {
    const float x = static_cast<float>(i);
    const float y = static_cast<float>(i + 1);
    const float z = static_cast<float>(i + 2);
    const float intensity = static_cast<float>(10 + i);
    const std::uint16_t line = static_cast<std::uint16_t>(i);

    write_value(*msg, i, 0, x);
    write_value(*msg, i, 4, y);
    write_value(*msg, i, 8, z);
    write_value(*msg, i, 12, intensity);

    if (encoding == TimeEncoding::UInt32RelNs) {
      const auto t_nsec = static_cast<std::uint32_t>(
        std::llround(relative_times_sec[i] * 1e9));
      write_value(*msg, i, time_offset, t_nsec);
    } else if (encoding == TimeEncoding::Float64AbsNs) {
      const double timestamp_nsec = (header_stamp + relative_times_sec[i]) * 1e9;
      write_value(*msg, i, time_offset, timestamp_nsec);
    } else if (encoding == TimeEncoding::Float64AbsSec) {
      const double timestamp_sec = header_stamp + relative_times_sec[i];
      write_value(*msg, i, time_offset, timestamp_sec);
    } else if (encoding == TimeEncoding::Float64NaN) {
      const double timestamp_sec =
        i == 0 ? std::numeric_limits<double>::quiet_NaN() : relative_times_sec[i];
      write_value(*msg, i, time_offset, timestamp_sec);
    }

    (void)time_datatype;
    write_value(*msg, i, line_offset, line);
  }

  return msg;
}

PointCloud2::ConstSharedPtr make_conflicting_time_cloud(const double header_stamp) {
  const std::vector<double> relative_times_sec = {0.0, 0.1};
  auto msg = std::make_shared<PointCloud2>();
  msg->header.frame_id = "right_lidar";
  msg->header.stamp = stamp_from_sec(header_stamp);
  msg->height = 1;
  msg->width = static_cast<std::uint32_t>(relative_times_sec.size());
  msg->is_bigendian = false;
  msg->is_dense = true;

  msg->fields.push_back(make_field("x", 0, PointField::FLOAT32));
  msg->fields.push_back(make_field("y", 4, PointField::FLOAT32));
  msg->fields.push_back(make_field("z", 8, PointField::FLOAT32));
  msg->fields.push_back(make_field("intensity", 12, PointField::FLOAT32));
  msg->fields.push_back(make_field("timestamp", 16, PointField::FLOAT64));
  msg->fields.push_back(make_field("t", 24, PointField::UINT32));
  msg->fields.push_back(make_field("line", 28, PointField::UINT16));
  msg->point_step = 30;
  msg->row_step = msg->point_step * msg->width;
  msg->data.resize(msg->row_step);

  for (std::size_t i = 0; i < relative_times_sec.size(); i++) {
    const float x = static_cast<float>(i);
    const float y = static_cast<float>(i + 1);
    const float z = static_cast<float>(i + 2);
    const float intensity = static_cast<float>(20 + i);
    const double misleading_timestamp = 1759332564.0 + relative_times_sec[i];
    const auto t_nsec = static_cast<std::uint32_t>(
      std::llround(relative_times_sec[i] * 1e9));
    const std::uint16_t line = static_cast<std::uint16_t>(i);

    write_value(*msg, i, 0, x);
    write_value(*msg, i, 4, y);
    write_value(*msg, i, 8, z);
    write_value(*msg, i, 12, intensity);
    write_value(*msg, i, 16, misleading_timestamp);
    write_value(*msg, i, 24, t_nsec);
    write_value(*msg, i, 28, line);
  }

  return msg;
}

glim_ros::MultiLidarCloudMerger make_merger(
    const std::vector<std::string>& topics,
    const bool allow_incomplete = false) {
  glim_ros::MultiLidarMergerConfig config;
  config.enabled = true;
  config.target_frame = "right_lidar";
  config.lidar_topics = topics;
  config.lidar_serials.resize(topics.size());
  config.lidar_ids.resize(topics.size());
  for (std::size_t i = 0; i < topics.size(); i++) {
    config.lidar_serials[i] = "lidar_" + std::to_string(i);
    config.lidar_ids[i] = static_cast<int>(i);
  }
  config.sync_tolerance_sec = 0.02;
  config.allow_incomplete_lidar_group = allow_incomplete;
  config.wait_for_imu = false;
  config.max_cloud_buffer_size = 10;
  return glim_ros::MultiLidarCloudMerger(config);
}

const PointField* find_field(const PointCloud2& msg, const std::string& name) {
  for (const auto& field : msg.fields) {
    if (field.name == name) {
      return &field;
    }
  }
  return nullptr;
}

void check_merged_schema(const PointCloud2& msg) {
  check(msg.fields.size() == 7, "merged field count");
  check(msg.point_step == 23, "merged point_step must be 23");
  check(msg.row_step == msg.point_step * msg.width, "merged row_step");

  const auto* x = find_field(msg, "x");
  const auto* y = find_field(msg, "y");
  const auto* z = find_field(msg, "z");
  const auto* intensity = find_field(msg, "intensity");
  const auto* t = find_field(msg, "t");
  const auto* line = find_field(msg, "line");
  const auto* scanner_id = find_field(msg, "scanner_id");

  check(x && x->offset == 0 && x->datatype == PointField::FLOAT32, "merged x field");
  check(y && y->offset == 4 && y->datatype == PointField::FLOAT32, "merged y field");
  check(z && z->offset == 8 && z->datatype == PointField::FLOAT32, "merged z field");
  check(intensity && intensity->offset == 12 && intensity->datatype == PointField::FLOAT32, "merged intensity field");
  check(t && t->offset == 16 && t->datatype == PointField::UINT32, "merged t field");
  check(line && line->offset == 20 && line->datatype == PointField::UINT16, "merged line field");
  check(scanner_id && scanner_id->offset == 22 && scanner_id->datatype == PointField::UINT8, "merged scanner_id field");
}

std::uint32_t max_merged_t_nsec(const PointCloud2& msg) {
  const auto* t = find_field(msg, "t");
  check(t, "merged t field missing");
  std::uint32_t max_t = 0;
  for (std::size_t i = 0; i < msg.width * msg.height; i++) {
    max_t = std::max(max_t, read_value<std::uint32_t>(msg, i, t->offset));
  }
  return max_t;
}

void check_converter_and_timekeeper(
    const PointCloud2& merged,
    const double expected_stamp,
    const double expected_max_time,
    const std::string& case_name) {
  auto raw = glim::extract_raw_points(merged, "intensity", "line");
  check(raw != nullptr, case_name + ": extract_raw_points failed");
  check_near(raw->stamp, expected_stamp, 2e-6, case_name + ": raw stamp");

  auto before_minmax = std::minmax_element(raw->times.begin(), raw->times.end());
  std::cout
    << std::fixed << std::setprecision(9)
    << "[time_contract] " << case_name
    << " before_timekeeper raw_points_stamp=" << raw->stamp
    << " times_min=" << *before_minmax.first
    << " times_max=" << *before_minmax.second
    << " scan_end=" << raw->stamp + *before_minmax.second
    << std::endl;

  glim::TimeKeeper time_keeper;
  check(time_keeper.process(raw), case_name + ": TimeKeeper rejected points");

  const auto minmax = std::minmax_element(raw->times.begin(), raw->times.end());
  std::cout
    << std::fixed << std::setprecision(9)
    << "[time_contract] " << case_name
    << " after_timekeeper raw_points_stamp=" << raw->stamp
    << " times_min=" << *minmax.first
    << " times_max=" << *minmax.second
    << " scan_end=" << raw->stamp + *minmax.second
    << std::endl;

  check_near(*minmax.first, 0.0, 2e-6, case_name + ": min relative time");
  check_near(*minmax.second, expected_max_time, 5e-6, case_name + ": max relative time");
  check_near(raw->stamp + *minmax.second, expected_stamp + expected_max_time, 6e-6, case_name + ": scan_end");
  check(std::abs((raw->stamp + *minmax.second) - (2.0 * expected_stamp + expected_max_time)) > 10.0,
        case_name + ": scan_end contains doubled epoch");
  check(raw->attrs.timestamp && raw->attrs.timestamp->size() == raw->times.size(),
        case_name + ": attrs.timestamp not synchronized");
  check_near(raw->attrs.timestamp->back(), raw->times.back(), 1e-12, case_name + ": attrs.timestamp units");
}

PointCloud2::ConstSharedPtr merge_one(
    glim_ros::MultiLidarCloudMerger& merger,
    const std::string& topic,
    const PointCloud2::ConstSharedPtr& cloud) {
  const auto out = merger.add_cloud(topic, cloud);
  check(out.size() == 1, "one-topic merger should publish immediately");
  check(out.front() != nullptr, "merged cloud is null");
  return out.front();
}

void setup_timekeeper_config() {
  const std::string dir = "/tmp/glim_multilidar_time_contract";
  mkdir(dir.c_str(), 0755);

  std::ofstream ofs(dir + "/config.json");
  ofs <<
    "{\n"
    "  \"global\": {\n"
    "    \"config_sensors\": \"config.json\"\n"
    "  },\n"
    "  \"sensors\": {\n"
    "    \"autoconf_perpoint_times\": false,\n"
    "    \"autoconf_prefer_frame_time\": false,\n"
    "    \"perpoint_relative_time\": true,\n"
    "    \"perpoint_time_scale\": 1.0\n"
    "  }\n"
    "}\n";
  ofs.close();

  glim::GlobalConfig::instance(dir, true);
}

void test_two_lidar_uint32_relative() {
  auto merger = make_merger({"/lidar_a", "/lidar_b"});
  auto a = make_cloud("right_lidar", 1000.000, "t", TimeEncoding::UInt32RelNs, {0.0, 0.100});
  auto b = make_cloud("left_lidar", 1000.005, "t", TimeEncoding::UInt32RelNs, {0.0, 0.100});

  check(merger.add_cloud("/lidar_a", a).empty(), "first lidar should wait for sync partner");
  const auto out = merger.add_cloud("/lidar_b", b);
  check(out.size() == 1, "two-lidar merger should publish one cloud");

  const auto& merged = *out.front();
  check_merged_schema(merged);
  check_near(stamp_to_sec(merged.header.stamp), 1000.0, 2e-6, "two-lidar merged header");
  check(max_merged_t_nsec(merged) >= 104999000 && max_merged_t_nsec(merged) <= 105001000,
        "two-lidar t max should be about 105 ms");
  check_converter_and_timekeeper(merged, 1000.0, 0.105, "two_lidar_uint32_relative");
}

void test_float64_absolute_ns() {
  constexpr double kEpoch = 1759332564.0;
  auto merger = make_merger({"/lidar"});
  auto cloud = make_cloud("right_lidar", kEpoch, "timestamp", TimeEncoding::Float64AbsNs, {0.0, 0.050, 0.100});
  const auto merged = merge_one(merger, "/lidar", cloud);
  check_merged_schema(*merged);
  check_near(stamp_to_sec(merged->header.stamp), kEpoch, 2e-6, "abs-ns merged header");
  check_converter_and_timekeeper(*merged, kEpoch, 0.100, "float64_absolute_ns");
}

void test_float64_absolute_sec() {
  constexpr double kEpoch = 1759332564.0;
  auto merger = make_merger({"/lidar"});
  auto cloud = make_cloud("right_lidar", kEpoch, "timestamp", TimeEncoding::Float64AbsSec, {0.0, 0.020, 0.100});
  const auto merged = merge_one(merger, "/lidar", cloud);
  check_merged_schema(*merged);
  check_near(stamp_to_sec(merged->header.stamp), kEpoch, 2e-6, "abs-sec merged header");
  check_converter_and_timekeeper(*merged, kEpoch, 0.100, "float64_absolute_sec");
}

void test_raw_single_float64_absolute_ns() {
  constexpr double kEpoch = 1759332564.0;
  auto cloud = make_cloud("right_lidar", kEpoch, "timestamp", TimeEncoding::Float64AbsNs, {0.0, 0.050, 0.100});
  check_converter_and_timekeeper(*cloud, kEpoch, 0.100, "raw_single_float64_absolute_ns");
}

void test_raw_single_float64_absolute_sec() {
  constexpr double kEpoch = 1759332564.0;
  auto cloud = make_cloud("right_lidar", kEpoch, "timestamp", TimeEncoding::Float64AbsSec, {0.0, 0.050, 0.100});
  check_converter_and_timekeeper(*cloud, kEpoch, 0.100, "raw_single_float64_absolute_sec");
}

void test_no_time_field() {
  auto merger = make_merger({"/lidar"});
  auto cloud = make_cloud("right_lidar", 1000.0, "", TimeEncoding::None, {0.0, 0.0, 0.0});
  const auto merged = merge_one(merger, "/lidar", cloud);
  check_merged_schema(*merged);
  check(max_merged_t_nsec(*merged) == 0, "no-time cloud should produce zero relative t");
  check_converter_and_timekeeper(*merged, 1000.0, 0.0, "no_time_field");
}

void test_timestamp_priority_prefers_t() {
  auto merger = make_merger({"/lidar"});
  auto cloud = make_conflicting_time_cloud(1000.0);
  const auto merged = merge_one(merger, "/lidar", cloud);
  check_merged_schema(*merged);
  check_near(stamp_to_sec(merged->header.stamp), 1000.0, 2e-6, "t priority merged header");
  check_converter_and_timekeeper(*merged, 1000.0, 0.100, "timestamp_priority_prefers_t");
}

void test_monotonic_output_correction() {
  auto merger = make_merger({"/lidar"});
  auto first = make_cloud("right_lidar", 1000.0, "t", TimeEncoding::UInt32RelNs, {0.0, 0.100});
  auto second = make_cloud("right_lidar", 1000.0, "t", TimeEncoding::UInt32RelNs, {0.0, 0.100});

  const auto merged_first = merge_one(merger, "/lidar", first);
  const auto merged_second = merge_one(merger, "/lidar", second);

  check(stamp_to_sec(merged_second->header.stamp) > stamp_to_sec(merged_first->header.stamp),
        "monotonic correction should advance repeated header");
  check_merged_schema(*merged_second);
  check(max_merged_t_nsec(*merged_second) <= 100000000, "monotonic correction should not create large t");
}

void test_malformed_nan_time() {
  auto merger = make_merger({"/lidar"});
  auto cloud = make_cloud("right_lidar", 1000.0, "timestamp", TimeEncoding::Float64NaN, {0.0, 0.100});
  const auto merged = merge_one(merger, "/lidar", cloud);
  check_merged_schema(*merged);
  check_converter_and_timekeeper(*merged, 1000.0, 0.100, "malformed_nan_time");
}

}  // namespace

int main() {
  setup_timekeeper_config();

  test_two_lidar_uint32_relative();
  test_float64_absolute_ns();
  test_float64_absolute_sec();
  test_raw_single_float64_absolute_ns();
  test_raw_single_float64_absolute_sec();
  test_no_time_field();
  test_timestamp_priority_prefers_t();
  test_monotonic_output_correction();
  test_malformed_nan_time();

  return 0;
}

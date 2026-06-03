#include <glim_ros/multi_lidar_cloud_merger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace glim_ros {
namespace {

const sensor_msgs::msg::PointField* find_field(
    const sensor_msgs::msg::PointCloud2& msg,
    const std::string& name) {
  for (const auto& field : msg.fields) {
    if (field.name == name) {
      return &field;
    }
  }
  return nullptr;
}

const sensor_msgs::msg::PointField* find_timestamp_field(
    const sensor_msgs::msg::PointCloud2& msg) {
  if (const auto* f = find_field(msg, "timestamp")) return f;
  if (const auto* f = find_field(msg, "time")) return f;
  if (const auto* f = find_field(msg, "time_stamp")) return f;
  if (const auto* f = find_field(msg, "t")) return f;
  return nullptr;
}

const sensor_msgs::msg::PointField* find_line_field(
    const sensor_msgs::msg::PointCloud2& msg) {
  if (const auto* f = find_field(msg, "line")) return f;
  if (const auto* f = find_field(msg, "ring")) return f;
  return nullptr;
}

template <typename T>
bool read_raw(
    const sensor_msgs::msg::PointCloud2& msg,
    const std::size_t byte_index,
    T& value) {
  if (byte_index + sizeof(T) > msg.data.size()) {
    return false;
  }

  std::memcpy(&value, msg.data.data() + byte_index, sizeof(T));
  return true;
}

bool read_field_as_double(
    const sensor_msgs::msg::PointCloud2& msg,
    const sensor_msgs::msg::PointField& field,
    const std::size_t point_base,
    double& value) {
  const std::size_t byte_index = point_base + field.offset;

  switch (field.datatype) {
    case sensor_msgs::msg::PointField::INT8: {
      std::int8_t v = 0;
      if (!read_raw(msg, byte_index, v)) return false;
      value = static_cast<double>(v);
      return true;
    }
    case sensor_msgs::msg::PointField::UINT8: {
      std::uint8_t v = 0;
      if (!read_raw(msg, byte_index, v)) return false;
      value = static_cast<double>(v);
      return true;
    }
    case sensor_msgs::msg::PointField::INT16: {
      std::int16_t v = 0;
      if (!read_raw(msg, byte_index, v)) return false;
      value = static_cast<double>(v);
      return true;
    }
    case sensor_msgs::msg::PointField::UINT16: {
      std::uint16_t v = 0;
      if (!read_raw(msg, byte_index, v)) return false;
      value = static_cast<double>(v);
      return true;
    }
    case sensor_msgs::msg::PointField::INT32: {
      std::int32_t v = 0;
      if (!read_raw(msg, byte_index, v)) return false;
      value = static_cast<double>(v);
      return true;
    }
    case sensor_msgs::msg::PointField::UINT32: {
      std::uint32_t v = 0;
      if (!read_raw(msg, byte_index, v)) return false;
      value = static_cast<double>(v);
      return true;
    }
    case sensor_msgs::msg::PointField::FLOAT32: {
      float v = 0.0f;
      if (!read_raw(msg, byte_index, v)) return false;
      value = static_cast<double>(v);
      return true;
    }
    case sensor_msgs::msg::PointField::FLOAT64: {
      double v = 0.0;
      if (!read_raw(msg, byte_index, v)) return false;
      value = v;
      return true;
    }
    default:
      return false;
  }
}

bool is_integer_type(const std::uint8_t datatype) {
  return datatype == sensor_msgs::msg::PointField::INT8 ||
         datatype == sensor_msgs::msg::PointField::UINT8 ||
         datatype == sensor_msgs::msg::PointField::INT16 ||
         datatype == sensor_msgs::msg::PointField::UINT16 ||
         datatype == sensor_msgs::msg::PointField::INT32 ||
         datatype == sensor_msgs::msg::PointField::UINT32;
}

double decode_point_time(
    const sensor_msgs::msg::PointField* field,
    const double raw_time,
    const double header_time) {
  if (!field || !std::isfinite(raw_time)) {
    return header_time;
  }

  // Livox PointCloud2 commonly has "t" as UINT32 nanoseconds from scan start.
  if (field->name == "t" && is_integer_type(field->datatype)) {
    return header_time + raw_time * 1e-9;
  }

  if (is_integer_type(field->datatype)) {
    return header_time + raw_time * 1e-9;
  }

  // absolute epoch nanoseconds
  if (raw_time > 1e15) {
    return raw_time * 1e-9;
  }

  // absolute UNIX seconds
  if (raw_time > 1e9) {
    return raw_time;
  }

  // relative seconds
  return header_time + raw_time;
}

Eigen::Isometry3d matrix_from_json_calibration(const nlohmann::json& entry) {
  if (entry.contains("identity")) {
    const auto& v = entry.at("identity");
    if ((v.is_boolean() && v.get<bool>()) ||
        (v.is_string() && v.get<std::string>() == "true")) {
      return Eigen::Isometry3d::Identity();
    }
  }

  if (!entry.contains("data")) {
    throw std::runtime_error("calibration entry has no data[] and is not identity");
  }

  const auto data = entry.at("data").get<std::vector<double>>();
  if (data.size() != 16) {
    throw std::runtime_error("calibration data[] must contain exactly 16 values");
  }

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  const std::string order = entry.value("order", "ROW");

  if (order == "COLUMN") {
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        T(r, c) = data[c * 4 + r];
      }
    }
  } else {
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        T(r, c) = data[r * 4 + c];
      }
    }
  }

  Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
  iso.matrix() = T;

  if (entry.value("inverted", false)) {
    iso = iso.inverse();
  }

  return iso;
}

}  // namespace

double MultiLidarCloudMerger::stamp_to_sec(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

MultiLidarCloudMerger::MultiLidarCloudMerger(const MultiLidarMergerConfig& config)
: config_(config) {
  if (config_.lidar_ids.size() < config_.lidar_topics.size()) {
    config_.lidar_ids.resize(config_.lidar_topics.size());
    for (std::size_t i = 0; i < config_.lidar_ids.size(); i++) {
      config_.lidar_ids[i] = static_cast<int>(i);
    }
  }

  if (config_.lidar_serials.size() < config_.lidar_topics.size()) {
    config_.lidar_serials.resize(config_.lidar_topics.size());
    for (std::size_t i = 0; i < config_.lidar_serials.size(); i++) {
      config_.lidar_serials[i] = "lidar_" + std::to_string(i);
    }
  }

  buffers_.resize(config_.lidar_topics.size());
  T_target_lidar_.assign(config_.lidar_topics.size(), Eigen::Isometry3d::Identity());

  for (std::size_t i = 0; i < config_.lidar_topics.size(); i++) {
    topic_to_index_[config_.lidar_topics[i]] = i;
  }

  if (!enabled()) {
    return;
  }

  load_calibration();

  spdlog::info(
    "[multi_lidar] enabled target_frame={} lidars={} sync_tol={} allow_incomplete={}",
    config_.target_frame,
    config_.lidar_topics.size(),
    config_.sync_tolerance_sec,
    config_.allow_incomplete_lidar_group);

  for (std::size_t i = 0; i < config_.lidar_topics.size(); i++) {
    spdlog::info(
      "[multi_lidar] lidar{} topic={} serial={} id={}",
      i,
      config_.lidar_topics[i],
      config_.lidar_serials[i],
      config_.lidar_ids[i]);
  }
}

void MultiLidarCloudMerger::load_calibration() {
  if (config_.calibration_file.empty()) {
    spdlog::warn("[multi_lidar] calibration_file is empty; all T_target_lidar are Identity");
    return;
  }

  std::ifstream f(config_.calibration_file);
  if (!f.is_open()) {
    spdlog::warn("[multi_lidar] cannot open calibration_file={}; use Identity", config_.calibration_file);
    return;
  }

  nlohmann::json j;
  f >> j;

  if (!j.contains("calibration")) {
    spdlog::warn("[multi_lidar] calibration_file has no top-level calibration object; use Identity");
    return;
  }

  for (std::size_t i = 0; i < config_.lidar_topics.size(); i++) {
    const auto& serial = config_.lidar_serials[i];

    if (!j.at("calibration").contains(serial)) {
      spdlog::warn("[multi_lidar] calibration missing serial={}; use Identity", serial);
      continue;
    }

    T_target_lidar_[i] = matrix_from_json_calibration(j.at("calibration").at(serial));
  }
}

bool MultiLidarCloudMerger::is_lidar_topic(const std::string& topic) const {
  return topic_to_index_.find(topic) != topic_to_index_.end();
}

std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr>
MultiLidarCloudMerger::add_cloud(
    const std::string& topic,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
  std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr> out;

  const auto found = topic_to_index_.find(topic);
  if (found == topic_to_index_.end()) {
    return out;
  }

  const std::size_t idx = found->second;

  CloudItem item;
  item.topic = topic;
  item.lidar_index = idx;
  item.stamp = stamp_to_sec(msg->header.stamp);
  item.msg = msg;

  buffers_[idx].push_back(item);

  while (static_cast<int>(buffers_[idx].size()) > config_.max_cloud_buffer_size) {
    buffers_[idx].pop_front();
  }

  while (true) {
    auto merged = try_merge_once();
    if (!merged) {
      break;
    }

    out.push_back(*merged);
  }

  return out;
}

std::optional<sensor_msgs::msg::PointCloud2::ConstSharedPtr>
MultiLidarCloudMerger::try_merge_once() {
  if (buffers_.empty()) {
    return std::nullopt;
  }

  std::size_t anchor_idx = 0;
  double anchor_time = std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < buffers_.size(); i++) {
    if (buffers_[i].empty()) {
      if (!config_.allow_incomplete_lidar_group) {
        return std::nullopt;
      }
      continue;
    }

    if (buffers_[i].front().stamp < anchor_time) {
      anchor_time = buffers_[i].front().stamp;
      anchor_idx = i;
    }
  }

  if (!std::isfinite(anchor_time)) {
    return std::nullopt;
  }

  std::vector<int> selected(buffers_.size(), -1);

  for (std::size_t i = 0; i < buffers_.size(); i++) {
    if (buffers_[i].empty()) {
      continue;
    }

    int best = -1;
    double best_dt = std::numeric_limits<double>::max();

    for (std::size_t j = 0; j < buffers_[i].size(); j++) {
      const double dt = std::abs(buffers_[i][j].stamp - anchor_time);
      if (dt < best_dt) {
        best_dt = dt;
        best = static_cast<int>(j);
      }
    }

    if (best >= 0 && best_dt <= config_.sync_tolerance_sec) {
      selected[i] = best;
      continue;
    }

    // If another lidar is already newer than this anchor by more than tolerance,
    // this anchor can no longer be matched.
    if (i != anchor_idx && !buffers_[i].empty() &&
        buffers_[i].front().stamp > anchor_time + config_.sync_tolerance_sec) {
      spdlog::warn(
        "[multi_lidar] drop unsynchronized anchor topic={} stamp={}",
        buffers_[anchor_idx].front().topic,
        buffers_[anchor_idx].front().stamp);
      buffers_[anchor_idx].pop_front();
      return std::nullopt;
    }

    if (!config_.allow_incomplete_lidar_group) {
      return std::nullopt;
    }
  }

  if (selected[anchor_idx] < 0) {
    selected[anchor_idx] = 0;
  }

  std::vector<CloudItem> group;
  group.reserve(buffers_.size());

  for (std::size_t i = 0; i < buffers_.size(); i++) {
    if (selected[i] < 0) {
      continue;
    }

    group.push_back(buffers_[i][selected[i]]);

    for (int k = 0; k <= selected[i]; k++) {
      buffers_[i].pop_front();
    }
  }

  if (group.empty()) {
    return std::nullopt;
  }

  std::vector<MergedPoint> points;
  double group_min_time = std::numeric_limits<double>::max();

  for (const auto& item : group) {
    convert_cloud(item, points, group_min_time);
  }

  if (points.empty()) {
    return std::nullopt;
  }

  std::stable_sort(
    points.begin(),
    points.end(),
    [](const MergedPoint& a, const MergedPoint& b) {
      return a.abs_time < b.abs_time;
    });

  return build_msg(points, group_min_time);
}

bool MultiLidarCloudMerger::convert_cloud(
    const CloudItem& item,
    std::vector<MergedPoint>& points,
    double& group_min_time) const {
  const auto& msg = *item.msg;

  const auto* field_x = find_field(msg, "x");
  const auto* field_y = find_field(msg, "y");
  const auto* field_z = find_field(msg, "z");

  if (!field_x || !field_y || !field_z) {
    spdlog::warn("[multi_lidar] missing XYZ fields in topic={}", item.topic);
    return false;
  }

  const auto* field_i = find_field(msg, "intensity");
  const auto* field_t = find_timestamp_field(msg);
  const auto* field_line = find_line_field(msg);

  const int num_points = static_cast<int>(msg.width * msg.height);
  const double header_time = stamp_to_sec(msg.header.stamp);
  const auto& T = T_target_lidar_[item.lidar_index];

  points.reserve(points.size() + num_points);

  for (int i = 0; i < num_points; i++) {
    const std::size_t base = static_cast<std::size_t>(msg.point_step) * static_cast<std::size_t>(i);

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    if (!read_field_as_double(msg, *field_x, base, x) ||
        !read_field_as_double(msg, *field_y, base, y) ||
        !read_field_as_double(msg, *field_z, base, z)) {
      continue;
    }

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }

    double intensity = 0.0;
    if (field_i) {
      read_field_as_double(msg, *field_i, base, intensity);
    }

    double raw_time = 0.0;
    double abs_time = header_time;
    if (field_t && read_field_as_double(msg, *field_t, base, raw_time)) {
      abs_time = decode_point_time(field_t, raw_time, header_time);
    }

    double line_value = 0.0;
    std::uint16_t line = 0;
    if (field_line && read_field_as_double(msg, *field_line, base, line_value)) {
      line = static_cast<std::uint16_t>(
        std::max(0.0, std::min(65535.0, line_value)));
    }

    const Eigen::Vector4d p_lidar(x, y, z, 1.0);
    const Eigen::Vector4d p_target = T * p_lidar;

    MergedPoint p;
    p.x = static_cast<float>(p_target.x());
    p.y = static_cast<float>(p_target.y());
    p.z = static_cast<float>(p_target.z());
    p.intensity = static_cast<float>(intensity);
    p.abs_time = abs_time;
    p.line = line;
    p.scanner_id = static_cast<std::uint8_t>(
      std::max(0, std::min(255, config_.lidar_ids[item.lidar_index])));

    group_min_time = std::min(group_min_time, abs_time);
    points.push_back(p);
  }

  return true;
}

sensor_msgs::msg::PointCloud2::ConstSharedPtr
MultiLidarCloudMerger::build_msg(
    std::vector<MergedPoint>& points,
    const double group_start_time) const {
  auto out = std::make_shared<sensor_msgs::msg::PointCloud2>();

  out->header.frame_id = config_.target_frame;
  out->header.stamp.sec = static_cast<int32_t>(std::floor(group_start_time));
  out->header.stamp.nanosec =
    static_cast<std::uint32_t>((group_start_time - std::floor(group_start_time)) * 1e9);

  out->height = 1;
  out->width = static_cast<std::uint32_t>(points.size());
  out->is_bigendian = false;
  out->is_dense = true;

  out->fields.resize(7);

  out->fields[0].name = "x";
  out->fields[0].offset = 0;
  out->fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out->fields[0].count = 1;

  out->fields[1].name = "y";
  out->fields[1].offset = 4;
  out->fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out->fields[1].count = 1;

  out->fields[2].name = "z";
  out->fields[2].offset = 8;
  out->fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out->fields[2].count = 1;

  out->fields[3].name = "intensity";
  out->fields[3].offset = 12;
  out->fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
  out->fields[3].count = 1;

  out->fields[4].name = "t";
  out->fields[4].offset = 16;
  out->fields[4].datatype = sensor_msgs::msg::PointField::UINT32;
  out->fields[4].count = 1;

  out->fields[5].name = "line";
  out->fields[5].offset = 20;
  out->fields[5].datatype = sensor_msgs::msg::PointField::UINT16;
  out->fields[5].count = 1;

  out->fields[6].name = "scanner_id";
  out->fields[6].offset = 22;
  out->fields[6].datatype = sensor_msgs::msg::PointField::UINT8;
  out->fields[6].count = 1;

  out->point_step = 23;
  out->row_step = out->point_step * out->width;
  out->data.resize(static_cast<std::size_t>(out->row_step));

  for (std::size_t i = 0; i < points.size(); i++) {
    const auto& p = points[i];
    const std::size_t base = static_cast<std::size_t>(out->point_step) * i;

    std::memcpy(out->data.data() + base + 0, &p.x, sizeof(float));
    std::memcpy(out->data.data() + base + 4, &p.y, sizeof(float));
    std::memcpy(out->data.data() + base + 8, &p.z, sizeof(float));
    std::memcpy(out->data.data() + base + 12, &p.intensity, sizeof(float));

    double dt_sec = p.abs_time - group_start_time;
    if (!std::isfinite(dt_sec) || dt_sec < 0.0) {
      dt_sec = 0.0;
    }

    const double max_u32_nsec =
      static_cast<double>(std::numeric_limits<std::uint32_t>::max());
    const auto t_nsec = static_cast<std::uint32_t>(
      std::max(0.0, std::min(max_u32_nsec, dt_sec * 1e9)));

    std::memcpy(out->data.data() + base + 16, &t_nsec, sizeof(std::uint32_t));
    std::memcpy(out->data.data() + base + 20, &p.line, sizeof(std::uint16_t));
    std::memcpy(out->data.data() + base + 22, &p.scanner_id, sizeof(std::uint8_t));
  }

  return out;
}

}  // namespace glim_ros

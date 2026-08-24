#include <glim_ros/multi_lidar_cloud_merger.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

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
  // Prefer Livox/merged relative nanoseconds over legacy absolute timestamp aliases.
  if (const auto* f = find_field(msg, "t")) return f;
  if (const auto* f = find_field(msg, "time")) return f;
  if (const auto* f = find_field(msg, "time_stamp")) return f;
  if (const auto* f = find_field(msg, "timestamp")) return f;
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

const char* datatype_name(const std::uint8_t datatype) {
  switch (datatype) {
    case sensor_msgs::msg::PointField::INT8:
      return "INT8";
    case sensor_msgs::msg::PointField::UINT8:
      return "UINT8";
    case sensor_msgs::msg::PointField::INT16:
      return "INT16";
    case sensor_msgs::msg::PointField::UINT16:
      return "UINT16";
    case sensor_msgs::msg::PointField::INT32:
      return "INT32";
    case sensor_msgs::msg::PointField::UINT32:
      return "UINT32";
    case sensor_msgs::msg::PointField::FLOAT32:
      return "FLOAT32";
    case sensor_msgs::msg::PointField::FLOAT64:
      return "FLOAT64";
    default:
      return "UNKNOWN";
  }
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
    "[multi_lidar] enabled target_frame={} lidars={} sync_tol={} allow_incomplete={} wait_for_imu={}",
    config_.target_frame,
    config_.lidar_topics.size(),
    config_.sync_tolerance_sec,
    config_.allow_incomplete_lidar_group,
    config_.wait_for_imu);

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

    const auto& T = T_target_lidar_[i].matrix();
    spdlog::info(
      "[multi_lidar] calibration serial={} T_target_lidar=rowmajor=[{:.6f} {:.6f} {:.6f} {:.6f}; {:.6f} {:.6f} {:.6f} {:.6f}; {:.6f} {:.6f} {:.6f} {:.6f}; {:.6f} {:.6f} {:.6f} {:.6f}]",
      serial,
      T(0,0), T(0,1), T(0,2), T(0,3),
      T(1,0), T(1,1), T(1,2), T(1,3),
      T(2,0), T(2,1), T(2,2), T(2,3),
      T(3,0), T(3,1), T(3,2), T(3,3));
  }
}

bool MultiLidarCloudMerger::is_lidar_topic(const std::string& topic) const {
  return topic_to_index_.find(topic) != topic_to_index_.end();
}

bool MultiLidarCloudMerger::accept_stamp(
    const double stamp,
    const char* source,
    const std::string& topic) const {
  if (!std::isfinite(stamp)) {
    static int nonfinite_warn_count = 0;
    if (nonfinite_warn_count < 20) {
      nonfinite_warn_count++;
      spdlog::warn(
        "[stamp_filter] drop multi_lidar {}{} non-finite stamp={}",
        source,
        topic.empty() ? "" : (" topic=" + topic),
        stamp);
    }
    return false;
  }

  if (stamp < config_.stamp_filter_min_sec || stamp > config_.stamp_filter_max_sec) {
    static int range_warn_count = 0;
    if (range_warn_count < 50) {
      range_warn_count++;
      spdlog::warn(
        "[stamp_filter] drop multi_lidar {}{} stamp={:.9f} allowed=[{:.9f},{:.9f}]",
        source,
        topic.empty() ? "" : (" topic=" + topic),
        stamp,
        config_.stamp_filter_min_sec,
        config_.stamp_filter_max_sec);
    }
    return false;
  }

  return true;
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

  if (!accept_stamp(item.stamp, "cloud", topic)) {
    return out;
  }

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

void MultiLidarCloudMerger::notify_imu(const double stamp_sec) {
  if (!enabled() || !std::isfinite(stamp_sec)) {
    return;
  }

  if (!accept_stamp(stamp_sec, "imu")) {
    return;
  }

  last_imu_stamp_ = std::max(last_imu_stamp_, stamp_sec);
}

std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr>
MultiLidarCloudMerger::flush_ready_merges() {
  std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr> out;

  while (true) {
    auto merged = try_merge_once();
    if (!merged) {
      break;
    }

    out.push_back(*merged);
  }

  return out;
}

bool MultiLidarCloudMerger::cloud_time_range(
    const CloudItem& item,
    double& scan_beg_time,
    double& scan_end_time) const {
  const auto& msg = *item.msg;

  const auto* field_x = find_field(msg, "x");
  const auto* field_y = find_field(msg, "y");
  const auto* field_z = find_field(msg, "z");
  if (!field_x || !field_y || !field_z) {
    return false;
  }

  const auto* field_t = find_timestamp_field(msg);
  const double header_time = stamp_to_sec(msg.header.stamp);
  scan_beg_time = header_time;
  scan_end_time = header_time;

  double raw_time_min = std::numeric_limits<double>::infinity();
  double raw_time_max = -std::numeric_limits<double>::infinity();
  double decoded_time_min = std::numeric_limits<double>::infinity();
  double decoded_time_max = -std::numeric_limits<double>::infinity();
  int valid_points = 0;
  int timed_points = 0;

  const int num_points = static_cast<int>(msg.width * msg.height);
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

    valid_points++;

    double abs_time = header_time;
    if (field_t) {
      double raw_time = 0.0;
      if (read_field_as_double(msg, *field_t, base, raw_time)) {
        if (std::isfinite(raw_time)) {
          raw_time_min = std::min(raw_time_min, raw_time);
          raw_time_max = std::max(raw_time_max, raw_time);
        }
        abs_time = decode_point_time(field_t, raw_time, header_time);
        if (std::isfinite(abs_time)) {
          timed_points++;
          decoded_time_min = std::min(decoded_time_min, abs_time);
          decoded_time_max = std::max(decoded_time_max, abs_time);
        }
      }
    }

    scan_beg_time = std::min(scan_beg_time, abs_time);
    scan_end_time = std::max(scan_end_time, abs_time);
  }

  {
    static std::unordered_set<std::string> logged_schemas;
    const std::string time_name = field_t ? field_t->name : "<none>";
    const std::string time_type = field_t ? datatype_name(field_t->datatype) : "<none>";
    const std::string key =
      item.topic + "|" + msg.header.frame_id + "|" + std::to_string(msg.point_step) +
      "|" + time_name + "|" + time_type;

    if (logged_schemas.insert(key).second) {
      spdlog::info(
        "[multi_lidar] input schema topic={} frame={} header={:.9f} point_step={} width={} time_field={} time_type={} raw_time=[{:.9f},{:.9f}] decoded_abs=[{:.9f},{:.9f}] scan_duration={:.9f} valid_points={} timed_points={}",
        item.topic,
        msg.header.frame_id,
        header_time,
        msg.point_step,
        num_points,
        time_name,
        time_type,
        std::isfinite(raw_time_min) ? raw_time_min : 0.0,
        std::isfinite(raw_time_max) ? raw_time_max : 0.0,
        std::isfinite(decoded_time_min) ? decoded_time_min : header_time,
        std::isfinite(decoded_time_max) ? decoded_time_max : header_time,
        scan_end_time - scan_beg_time,
        valid_points,
        timed_points);
    }

    if (field_t && std::isfinite(decoded_time_min) &&
        (std::abs(decoded_time_min - header_time) > 5.0 ||
         std::abs(decoded_time_max - header_time) > 5.0)) {
      static int sanity_warn_count = 0;
      if (sanity_warn_count < 20) {
        sanity_warn_count++;
        spdlog::warn(
          "[multi_lidar] suspicious point time topic={} field={} type={} header={:.9f} decoded_abs=[{:.9f},{:.9f}] delta=[{:.9f},{:.9f}]",
          item.topic,
          time_name,
          time_type,
          header_time,
          decoded_time_min,
          decoded_time_max,
          decoded_time_min - header_time,
          decoded_time_max - header_time);
      }
    }
  }

  return true;
}

bool MultiLidarCloudMerger::select_sync_group(
    std::vector<int>& selected,
    double& anchor_time) const {
  if (buffers_.empty()) {
    return false;
  }

  selected.assign(buffers_.size(), -1);

  std::size_t anchor_idx = 0;
  anchor_time = std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < buffers_.size(); i++) {
    if (buffers_[i].empty()) {
      if (!config_.allow_incomplete_lidar_group) {
        return false;
      }
      continue;
    }

    if (buffers_[i].front().stamp < anchor_time) {
      anchor_time = buffers_[i].front().stamp;
      anchor_idx = i;
    }
  }

  if (!std::isfinite(anchor_time)) {
    return false;
  }

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

    if (i != anchor_idx && !buffers_[i].empty() &&
        buffers_[i].front().stamp > anchor_time + config_.sync_tolerance_sec) {
      return false;
    }

    if (!config_.allow_incomplete_lidar_group) {
      return false;
    }
  }

  if (selected[anchor_idx] < 0) {
    selected[anchor_idx] = 0;
  }

  return true;
}

std::optional<sensor_msgs::msg::PointCloud2::ConstSharedPtr>
MultiLidarCloudMerger::try_merge_once() {
  double anchor_time = 0.0;
  std::vector<int> selected;
  if (!select_sync_group(selected, anchor_time)) {
    // Drop stale anchor when the partner lidar has moved too far ahead.
    std::size_t anchor_idx = 0;
    double earliest = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < buffers_.size(); i++) {
      if (!buffers_[i].empty() && buffers_[i].front().stamp < earliest) {
        earliest = buffers_[i].front().stamp;
        anchor_idx = i;
      }
    }

    for (std::size_t i = 0; i < buffers_.size(); i++) {
      if (i == anchor_idx || buffers_[i].empty() || buffers_[anchor_idx].empty()) {
        continue;
      }

      if (buffers_[i].front().stamp > buffers_[anchor_idx].front().stamp + config_.sync_tolerance_sec) {
        spdlog::warn(
          "[multi_lidar] drop unsynchronized anchor topic={} stamp={}",
          buffers_[anchor_idx].front().topic,
          buffers_[anchor_idx].front().stamp);
        buffers_[anchor_idx].pop_front();
      }
    }

    return std::nullopt;
  }

  std::vector<CloudItem> group;
  group.reserve(buffers_.size());

  for (std::size_t i = 0; i < buffers_.size(); i++) {
    if (selected[i] < 0) {
      continue;
    }

    group.push_back(buffers_[i][selected[i]]);
  }

  if (group.empty()) {
    return std::nullopt;
  }

  if (config_.wait_for_imu) {
    double required_imu_time = 0.0;
    double group_scan_beg = std::numeric_limits<double>::infinity();
    double group_scan_end = -std::numeric_limits<double>::infinity();
    std::string source_ranges;
    for (const auto& item : group) {
      double scan_beg = item.stamp;
      double scan_end = item.stamp;
      if (!cloud_time_range(item, scan_beg, scan_end)) {
        scan_end = item.stamp;
      }
      required_imu_time = std::max(required_imu_time, scan_end);
      group_scan_beg = std::min(group_scan_beg, scan_beg);
      group_scan_end = std::max(group_scan_end, scan_end);
      source_ranges += item.topic + ":[" + std::to_string(scan_beg) + "," +
                       std::to_string(scan_end) + "] ";
    }

    if (last_imu_stamp_ < required_imu_time) {
      static auto last_wait_log_time = std::chrono::steady_clock::time_point{};
      static int wait_log_count = 0;
      const auto now = std::chrono::steady_clock::now();
      if (wait_log_count < 5 || now - last_wait_log_time > std::chrono::seconds(1)) {
        wait_log_count++;
        last_wait_log_time = now;

        std::string buffer_sizes;
        for (std::size_t i = 0; i < buffers_.size(); i++) {
          buffer_sizes += config_.lidar_topics[i] + ":" + std::to_string(buffers_[i].size()) + " ";
        }

        spdlog::warn(
          "[multi_lidar] waiting for IMU anchor={:.9f} source_scan=[{:.9f},{:.9f}] required_imu_time={:.9f} last_imu_stamp={:.9f} delta={:.9f} buffers={} sources={}",
          anchor_time,
          std::isfinite(group_scan_beg) ? group_scan_beg : anchor_time,
          std::isfinite(group_scan_end) ? group_scan_end : anchor_time,
          required_imu_time,
          last_imu_stamp_,
          required_imu_time - last_imu_stamp_,
          buffer_sizes,
          source_ranges);
      }
      return std::nullopt;
    }
  }

  for (std::size_t i = 0; i < buffers_.size(); i++) {
    if (selected[i] < 0) {
      continue;
    }

    for (int k = 0; k <= selected[i]; k++) {
      buffers_[i].pop_front();
    }
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

  // Use the earliest point time as frame stamp so deskew/IMU integration align
  // with per-point offsets (FAST-LIO uses min lidar_beg_time across lidars).
  double publish_time = group_min_time;
  if (!std::isfinite(publish_time)) {
    publish_time = anchor_time;
  }

  constexpr double kMonotonicEps = 1e-6;
  if (last_published_stamp_ >= 0.0 && publish_time <= last_published_stamp_) {
    publish_time = last_published_stamp_ + kMonotonicEps;
  }
  last_published_stamp_ = publish_time;

  return build_msg(points, publish_time);
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
    const double header_stamp) const {
  auto out = std::make_shared<sensor_msgs::msg::PointCloud2>();

  out->header.frame_id = config_.target_frame;
  double stamp_sec_floor = std::floor(header_stamp);
  long long stamp_nsec = std::llround((header_stamp - stamp_sec_floor) * 1e9);
  if (stamp_nsec >= 1000000000LL) {
    stamp_nsec -= 1000000000LL;
    stamp_sec_floor += 1.0;
  } else if (stamp_nsec < 0) {
    stamp_nsec += 1000000000LL;
    stamp_sec_floor -= 1.0;
  }
  out->header.stamp.sec = static_cast<int32_t>(stamp_sec_floor);
  out->header.stamp.nanosec = static_cast<std::uint32_t>(stamp_nsec);

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

  // Keep each point record aligned to four bytes. The final byte is padding.
  out->point_step = 24;
  out->row_step = out->point_step * out->width;
  out->data.resize(static_cast<std::size_t>(out->row_step));

  std::uint32_t t_min_nsec = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t t_max_nsec = 0;
  std::array<std::size_t, 256> scanner_counts{};

  for (std::size_t i = 0; i < points.size(); i++) {
    const auto& p = points[i];
    const std::size_t base = static_cast<std::size_t>(out->point_step) * i;

    std::memcpy(out->data.data() + base + 0, &p.x, sizeof(float));
    std::memcpy(out->data.data() + base + 4, &p.y, sizeof(float));
    std::memcpy(out->data.data() + base + 8, &p.z, sizeof(float));
    std::memcpy(out->data.data() + base + 12, &p.intensity, sizeof(float));

    double dt_sec = p.abs_time - header_stamp;
    if (!std::isfinite(dt_sec)) {
      static int nonfinite_warn_count = 0;
      if (nonfinite_warn_count < 10) {
        nonfinite_warn_count++;
        spdlog::warn(
          "[multi_lidar] non-finite point time in merged output abs_time={} header={}",
          p.abs_time,
          header_stamp);
      }
      dt_sec = 0.0;
    }

    constexpr double kSmallNegativeTimeEps = 1e-5;
    if (dt_sec < 0.0 && dt_sec > -kSmallNegativeTimeEps) {
      dt_sec = 0.0;
    }

    if (dt_sec < 0.0) {
      static int negative_warn_count = 0;
      if (negative_warn_count < 20) {
        negative_warn_count++;
        spdlog::warn(
          "[multi_lidar] negative point time in merged output dt={:.9f} abs_time={:.9f} header={:.9f}; clamp to zero",
          dt_sec,
          p.abs_time,
          header_stamp);
      }
      dt_sec = 0.0;
    }

    const long long t_nsec_ll = std::clamp(
      std::llround(dt_sec * 1e9),
      0LL,
      static_cast<long long>(std::numeric_limits<std::uint32_t>::max()));
    const auto t_nsec = static_cast<std::uint32_t>(t_nsec_ll);

    t_min_nsec = std::min(t_min_nsec, t_nsec);
    t_max_nsec = std::max(t_max_nsec, t_nsec);
    scanner_counts[p.scanner_id]++;

    std::memcpy(out->data.data() + base + 16, &t_nsec, sizeof(std::uint32_t));
    std::memcpy(out->data.data() + base + 20, &p.line, sizeof(std::uint16_t));
    std::memcpy(out->data.data() + base + 22, &p.scanner_id, sizeof(std::uint8_t));
  }

  {
    static int output_log_count = 0;
    if (output_log_count < 20) {
      output_log_count++;

      std::string scanner_count_text;
      for (std::size_t i = 0; i < scanner_counts.size(); i++) {
        if (scanner_counts[i] == 0) {
          continue;
        }
        scanner_count_text += std::to_string(i) + ":" + std::to_string(scanner_counts[i]) + " ";
      }

      spdlog::info(
        "[multi_lidar] merged output header={:.9f} points={} t_nsec=[{},{}] duration={:.9f} point_step={} scanner_counts={}",
        header_stamp,
        points.size(),
        points.empty() ? 0 : t_min_nsec,
        t_max_nsec,
        static_cast<double>(t_max_nsec) * 1e-9,
        out->point_step,
        scanner_count_text);
    }
  }

  {
    static bool debug_written = false;

    if (!debug_written) {
      debug_written = true;

      const std::string debug_path = "/tmp/glim_multilidar_merged_debug.ply";
      std::ofstream ofs(debug_path);

      if (ofs) {
        ofs << "ply\n";
        ofs << "format ascii 1.0\n";
        ofs << "element vertex " << points.size() << "\n";
        ofs << "property float x\n";
        ofs << "property float y\n";
        ofs << "property float z\n";
        ofs << "property uchar red\n";
        ofs << "property uchar green\n";
        ofs << "property uchar blue\n";
        ofs << "end_header\n";

        for (const auto& p : points) {
          int r = 255;
          int g = 255;
          int b = 255;

          if (p.scanner_id == 0) {
            r = 0;
            g = 255;
            b = 0;
          } else if (p.scanner_id == 1) {
            r = 255;
            g = 0;
            b = 0;
          }

          ofs << p.x << " "
              << p.y << " "
              << p.z << " "
              << r << " "
              << g << " "
              << b << "\n";
        }

        spdlog::warn(
          "[multi_lidar] wrote debug merged PLY: {} points={} front_green/back_red",
          debug_path,
          points.size());
      } else {
        spdlog::warn("[multi_lidar] failed to write debug merged PLY");
      }
    }
  }

  return out;
}

}  // namespace glim_ros

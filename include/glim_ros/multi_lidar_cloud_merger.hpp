#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace glim_ros {

struct MultiLidarMergerConfig {
  bool enabled = false;

  std::string target_frame = "lidar";

  std::vector<std::string> lidar_topics;
  std::vector<std::string> lidar_serials;
  std::vector<int> lidar_ids;

  std::string calibration_file;

  double sync_tolerance_sec = 0.03;
  bool allow_incomplete_lidar_group = false;
  int max_cloud_buffer_size = 50;
};

class MultiLidarCloudMerger {
public:
  explicit MultiLidarCloudMerger(const MultiLidarMergerConfig& config);

  bool enabled() const { return config_.enabled && !config_.lidar_topics.empty(); }

  const std::vector<std::string>& topics() const {
    return config_.lidar_topics;
  }

  bool is_lidar_topic(const std::string& topic) const;

  std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr> add_cloud(
    const std::string& topic,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);

private:
  struct CloudItem {
    std::string topic;
    std::size_t lidar_index = 0;
    double stamp = 0.0;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
  };

  struct MergedPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float intensity = 0.0f;
    double abs_time = 0.0;
    std::uint16_t line = 0;
    std::uint8_t scanner_id = 0;
  };

  void load_calibration();

  std::optional<sensor_msgs::msg::PointCloud2::ConstSharedPtr> try_merge_once();

  bool convert_cloud(
    const CloudItem& item,
    std::vector<MergedPoint>& points,
    double& group_min_time) const;

  sensor_msgs::msg::PointCloud2::ConstSharedPtr build_msg(
    std::vector<MergedPoint>& points,
    double group_start_time) const;

  static double stamp_to_sec(const builtin_interfaces::msg::Time& stamp);

private:
  MultiLidarMergerConfig config_;

  std::unordered_map<std::string, std::size_t> topic_to_index_;
  std::vector<Eigen::Isometry3d> T_target_lidar_;
  std::vector<std::deque<CloudItem>> buffers_;
};

}  // namespace glim_ros

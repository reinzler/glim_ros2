#pragma once

#include <any>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <glim_ros/multi_lidar_cloud_merger.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#ifdef BUILD_WITH_CV_BRIDGE
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#endif

namespace glim {
class TimeKeeper;
class CloudPreprocessor;
class AsyncOdometryEstimation;
class AsyncSubMapping;
class AsyncGlobalMapping;

class ExtensionModule;
class GenericTopicSubscription;

class GlimROS : public rclcpp::Node {
public:
  GlimROS(const rclcpp::NodeOptions& options);
  ~GlimROS();

  bool needs_wait();
  void timer_callback();

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
#ifdef BUILD_WITH_CV_BRIDGE
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void camera_image_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg,
    int camera_id,
    const std::string& camera_name,
    const std::string& default_frame_id);

#endif
  size_t points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void multi_lidar_points_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg,
    const std::string& topic);


  void wait(bool auto_quit = false);
  void save(const std::string& path);

  const std::vector<std::shared_ptr<GenericTopicSubscription>>& extension_subscriptions();

  size_t odometry_workload();
  size_t local_mapping_workload();
  size_t global_mapping_workload();
  size_t total_submaps() const { return total_submaps_; }

  /** Request force-close of the open local submap (workload_guard antidote). */
  void request_local_mapping_force_close();

private:
  bool accept_stamp(double stamp, const char* source) const;

private:
  std::unique_ptr<glim::TimeKeeper> time_keeper;
  std::unique_ptr<glim::CloudPreprocessor> preprocessor;

  std::shared_ptr<glim::AsyncOdometryEstimation> odometry_estimation;
  std::unique_ptr<glim::AsyncSubMapping> sub_mapping;
  std::unique_ptr<glim::AsyncGlobalMapping> global_mapping;
  std::atomic<size_t> total_submaps_{0};

  bool keep_raw_points;
  double imu_time_offset;
  double points_time_offset;
  double acc_scale;
  double stamp_filter_min_sec;
  double stamp_filter_max_sec;
  bool dump_on_unload;

  std::string intensity_field, ring_field;

  // Extension modulles
  std::vector<std::shared_ptr<ExtensionModule>> extension_modules;
  std::vector<std::shared_ptr<GenericTopicSubscription>> extension_subs;

  // ROS-related
  rclcpp::TimerBase::SharedPtr timer;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> points_subs;
  std::unique_ptr<glim_ros::MultiLidarCloudMerger> multi_lidar_merger;

#ifdef BUILD_WITH_CV_BRIDGE
  image_transport::Subscriber image_sub;
  std::vector<image_transport::Subscriber> camera_image_subs;

#endif
};

}  // namespace glim

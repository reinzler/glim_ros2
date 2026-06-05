#include <glim_ros/glim_ros.hpp>

#define GLIM_ROS2

#include <deque>
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <boost/format.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>

#include <glim/util/debug.hpp>
#include <glim/util/config.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/time_keeper.hpp>
#include <glim/util/ros_cloud_converter.hpp>
#include <glim/util/extension_module.hpp>
#include <glim/util/extension_module_ros2.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/odometry/async_odometry_estimation.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/mapping/async_sub_mapping.hpp>
#include <glim/mapping/async_global_mapping.hpp>
#include <glim_ros/ros_compatibility.hpp>
#include <glim_ros/ros_qos.hpp>
#include <glim_ros/multi_lidar_cloud_merger.hpp>
#include <chrono>
#include <future>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace glim {

GlimROS::GlimROS(const rclcpp::NodeOptions& options) : Node("glim_ros", options) {
  // Setup logger
  auto logger = spdlog::stdout_color_mt("glim");
  logger->sinks().push_back(get_ringbuffer_sink());
  spdlog::set_default_logger(logger);

  bool debug = false;
  this->declare_parameter<bool>("debug", false);
  this->get_parameter<bool>("debug", debug);

  if (debug) {
    spdlog::info("enable debug printing");
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("/tmp/glim_log.log", true);
    logger->sinks().push_back(file_sink);
    logger->set_level(spdlog::level::trace);

    print_system_info(logger);
  }

  dump_on_unload = false;
  this->declare_parameter<bool>("dump_on_unload", false);
  this->get_parameter<bool>("dump_on_unload", dump_on_unload);

  if (dump_on_unload) {
    spdlog::info("dump_on_unload={}", dump_on_unload);
  }

  std::string config_path;
  this->declare_parameter<std::string>("config_path", "config");
  this->get_parameter<std::string>("config_path", config_path);

  if (config_path[0] != '/') {
    // config_path is relative to the glim directory
    config_path = ament_index_cpp::get_package_share_directory("glim") + "/" + config_path;
  }

  logger->info("config_path: {}", config_path);
  glim::GlobalConfig::instance(config_path);
  glim::Config config_ros(glim::GlobalConfig::get_config_path("config_ros"));

  keep_raw_points = config_ros.param<bool>("glim_ros", "keep_raw_points", false);
  imu_time_offset = config_ros.param<double>("glim_ros", "imu_time_offset", 0.0);
  points_time_offset = config_ros.param<double>("glim_ros", "points_time_offset", 0.0);
  acc_scale = config_ros.param<double>("glim_ros", "acc_scale", 0.0);

  glim::Config config_sensors(glim::GlobalConfig::get_config_path("config_sensors"));
  intensity_field = config_sensors.param<std::string>("sensors", "intensity_field", "intensity");
  ring_field = config_sensors.param<std::string>("sensors", "ring_field", "");

  // Setup GPU-based linearization
#ifdef BUILD_GTSAM_POINTS_GPU
  gtsam_points::LinearizationHook::register_hook([]() { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif

  // Preprocessing
  time_keeper.reset(new glim::TimeKeeper);
  preprocessor.reset(new glim::CloudPreprocessor);

  // Odometry estimation
  glim::Config config_odometry(glim::GlobalConfig::get_config_path("config_odometry"));
  const std::string odometry_estimation_so_name = config_odometry.param<std::string>("odometry_estimation", "so_name", "libodometry_estimation_cpu.so");
  spdlog::info("load {}", odometry_estimation_so_name);

  std::shared_ptr<glim::OdometryEstimationBase> odom = OdometryEstimationBase::load_module(odometry_estimation_so_name);
  if (!odom) {
    spdlog::critical("failed to load odometry estimation module");
    abort();
  }
  odometry_estimation.reset(new glim::AsyncOdometryEstimation(odom, odom->requires_imu()));

  // Sub mapping
  if (config_ros.param<bool>("glim_ros", "enable_local_mapping", true)) {
    const std::string sub_mapping_so_name =
      glim::Config(glim::GlobalConfig::get_config_path("config_sub_mapping")).param<std::string>("sub_mapping", "so_name", "libsub_mapping.so");
    if (!sub_mapping_so_name.empty()) {
      spdlog::info("load {}", sub_mapping_so_name);
      auto sub = SubMappingBase::load_module(sub_mapping_so_name);
      if (sub) {
        sub_mapping.reset(new AsyncSubMapping(sub));
      }
    }
  }

  // Global mapping
  if (config_ros.param<bool>("glim_ros", "enable_global_mapping", true)) {
    const std::string global_mapping_so_name =
      glim::Config(glim::GlobalConfig::get_config_path("config_global_mapping")).param<std::string>("global_mapping", "so_name", "libglobal_mapping.so");
    if (!global_mapping_so_name.empty()) {
      spdlog::info("load {}", global_mapping_so_name);
      auto global = GlobalMappingBase::load_module(global_mapping_so_name);
      if (global) {
        global_mapping.reset(new AsyncGlobalMapping(global));
      }
    }
  }

  // Extention modules
  const auto extensions = config_ros.param<std::vector<std::string>>("glim_ros", "extension_modules");
  if (extensions && !extensions->empty()) {
    for (const auto& extension : *extensions) {
      if (extension.find("viewer") == std::string::npos && extension.find("monitor") == std::string::npos) {
        spdlog::warn("Extension modules are enabled!!");
        spdlog::warn("You must carefully check and follow the licenses of ext modules");

        try {
          const std::string config_ext_path = ament_index_cpp::get_package_share_directory("glim_ext") + "/config";
          spdlog::info("config_ext_path: {}", config_ext_path);
          glim::GlobalConfig::instance()->override_param<std::string>("global", "config_ext", config_ext_path);
        } catch (ament_index_cpp::PackageNotFoundError& e) {
          spdlog::warn("glim_ext package path was not found!!");
        }

        break;
      }
    }

    for (const auto& extension : *extensions) {
      spdlog::info("load {}", extension);
      auto ext_module = ExtensionModule::load_module(extension);
      if (ext_module == nullptr) {
        spdlog::error("failed to load {}", extension);
        continue;
      } else {
        extension_modules.push_back(ext_module);

        auto ext_module_ros = std::dynamic_pointer_cast<ExtensionModuleROS2>(ext_module);
        if (ext_module_ros) {
          const auto subs = ext_module_ros->create_subscriptions(*this);
          extension_subs.insert(extension_subs.end(), subs.begin(), subs.end());
        }
      }
    }
  }

  // ROS-related
  using std::placeholders::_1;
  const std::string imu_topic = config_ros.param<std::string>("glim_ros", "imu_topic", "");
  const std::string points_topic = config_ros.param<std::string>("glim_ros", "points_topic", "");

  // Subscribers
  rclcpp::SensorDataQoS default_imu_qos;
  default_imu_qos.get_rmw_qos_profile().depth = 1000;

  auto qos = get_qos_settings(config_ros, "glim_ros", "imu_qos", default_imu_qos);
  imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic,
    qos,
    std::bind(&GlimROS::imu_callback, this, _1));

  qos = get_qos_settings(config_ros, "glim_ros", "points_qos");

  glim_ros::MultiLidarMergerConfig multi_lidar_config;
  multi_lidar_config.enabled = config_ros.param<bool>("multi_lidar", "enabled", false);
  multi_lidar_config.target_frame = config_ros.param<std::string>("multi_lidar", "target_frame", "lidar");
  multi_lidar_config.lidar_topics = config_ros.param<std::vector<std::string>>("multi_lidar", "lidar_topics", {});
  multi_lidar_config.lidar_serials = config_ros.param<std::vector<std::string>>("multi_lidar", "lidar_serials", {});
  multi_lidar_config.lidar_ids = config_ros.param<std::vector<int>>("multi_lidar", "lidar_ids", {});
  multi_lidar_config.calibration_file = config_ros.param<std::string>("multi_lidar", "calibration_file", "");
  multi_lidar_config.sync_tolerance_sec = config_ros.param<double>("multi_lidar", "sync_tolerance_sec", 0.03);
  multi_lidar_config.allow_incomplete_lidar_group =
    config_ros.param<bool>("multi_lidar", "allow_incomplete_lidar_group", false);
  multi_lidar_config.max_cloud_buffer_size =
    config_ros.param<int>("multi_lidar", "max_cloud_buffer_size", 50);

  multi_lidar_merger.reset(new glim_ros::MultiLidarCloudMerger(multi_lidar_config));

  if (multi_lidar_merger && multi_lidar_merger->enabled()) {
    for (const auto& lidar_topic : multi_lidar_merger->topics()) {
      spdlog::info("[multi_lidar] subscribe to {}", lidar_topic);

      points_subs.emplace_back(this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic,
        qos,
        [this, lidar_topic](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          this->multi_lidar_points_callback(msg, lidar_topic);
        }));
    }
  } else {
    points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic,
      qos,
      std::bind(&GlimROS::points_callback, this, _1));
  }

#ifdef BUILD_WITH_CV_BRIDGE
  qos = get_qos_settings(config_ros, "glim_ros", "image_qos");

  const auto image_topics = config_ros.param<std::vector<std::string>>(
    "glim_ros",
    "image_topics",
    std::vector<std::string>());

  const auto image_names = config_ros.param<std::vector<std::string>>(
    "glim_ros",
    "image_names",
    std::vector<std::string>());

  const auto image_frames = config_ros.param<std::vector<std::string>>(
    "glim_ros",
    "image_frames",
    std::vector<std::string>());

  if (!image_topics.empty()) {
    for (size_t i = 0; i < image_topics.size(); i++) {
      if (image_topics[i].empty()) {
        continue;
      }

      const int camera_id = static_cast<int>(i);
      const std::string camera_name =
        i < image_names.size() && !image_names[i].empty()
          ? image_names[i]
          : ("camera" + std::to_string(i));

      const std::string camera_frame =
        i < image_frames.size()
          ? image_frames[i]
          : "";

      spdlog::info(
        "image_topic[{}]: {} name={} frame={}",
        camera_id,
        image_topics[i],
        camera_name,
        camera_frame);

      camera_image_subs.emplace_back(image_transport::create_subscription(
        this,
        image_topics[i],
        [this, camera_id, camera_name, camera_frame](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
          this->camera_image_callback(msg, camera_id, camera_name, camera_frame);
        },
        "raw",
        qos.get_rmw_qos_profile()));
    }
  } else {
    const std::string image_topic = config_ros.param<std::string>("glim_ros", "image_topic", "");

    if (!image_topic.empty()) {
      spdlog::info("image_topic: {}", image_topic);

      image_sub = image_transport::create_subscription(
        this,
        image_topic,
        std::bind(&GlimROS::image_callback, this, _1),
        "raw",
        qos.get_rmw_qos_profile());
    }
  }
#endif

  for (const auto& sub : this->extension_subscriptions()) {
    spdlog::debug("subscribe to {}", sub->topic);
    sub->create_subscriber(*this);
  }

  // Start timer
  timer = this->create_wall_timer(std::chrono::milliseconds(1), [this]() { timer_callback(); });

  spdlog::debug("initialized");
}

GlimROS::~GlimROS() {
  spdlog::debug("quit");
  extension_modules.clear();

  if (dump_on_unload) {
    std::string dump_path = "/tmp/dump";
    wait(true);
    save(dump_path);
  }
}

const std::vector<std::shared_ptr<GenericTopicSubscription>>& GlimROS::extension_subscriptions() {
  return extension_subs;
}

void GlimROS::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  spdlog::trace("IMU: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (!GlobalConfig::instance()->has_param("meta", "imu_frame_id")) {
    spdlog::debug("auto-detecting IMU frame ID: {}", msg->header.frame_id);
    GlobalConfig::instance()->override_param<std::string>("meta", "imu_frame_id", msg->header.frame_id);
  }

  if (std::abs(acc_scale) < 1e-6) {
    const double norm = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z).norm();
    if (norm > 7.0 && norm < 12.0) {
      acc_scale = 1.0;
      spdlog::debug("assuming [m/s^2] for acceleration unit (acc_scale={}, norm={})", acc_scale, norm);
    } else if (norm > 0.8 && norm < 1.2) {
      acc_scale = 9.80665;
      spdlog::debug("assuming [g] for acceleration unit (acc_scale={}, norm={})", acc_scale, norm);
    } else {
      acc_scale = 1.0;
      spdlog::warn("unexpected acceleration norm {}. assuming [m/s^2] for acceleration unit (acc_scale={})", norm, acc_scale);
    }
  }

  const double imu_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9 + imu_time_offset;
  const Eigen::Vector3d linear_acc = acc_scale * Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  const Eigen::Vector3d angular_vel(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  if (!time_keeper->validate_imu_stamp(imu_stamp)) {
    spdlog::warn("skip an invalid IMU data (stamp={})", imu_stamp);
    return;
  }

  odometry_estimation->insert_imu(imu_stamp, linear_acc, angular_vel);
  if (sub_mapping) {
    sub_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
  }
  if (global_mapping) {
    global_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
  }
}

#ifdef BUILD_WITH_CV_BRIDGE
void GlimROS::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  camera_image_callback(msg, 0, "camera0", "");
}

void GlimROS::camera_image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr msg,
  int camera_id,
  const std::string& camera_name,
  const std::string& default_frame_id) {
  if (!msg) {
    return;
  }

  spdlog::trace(
    "image[{}:{}]: {}.{}",
    camera_id,
    camera_name,
    msg->header.stamp.sec,
    msg->header.stamp.nanosec);

  const std::string frame_id = !default_frame_id.empty() ? default_frame_id : msg->header.frame_id;

  if (camera_id == 0 && !GlobalConfig::instance()->has_param("meta", "image_frame")) {
    spdlog::debug("auto-detecting primary image frame ID: {}", frame_id);
    GlobalConfig::instance()->override_param<std::string>("meta", "image_frame", frame_id);
  }

  auto cv_image = cv_bridge::toCvCopy(msg, "bgr8");
  const double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;

  auto image_ptr = std::make_shared<cv::Mat>(cv_image->image.clone());

  auto image_frame = std::make_shared<glim::CameraImageFrame>(
    stamp,
    camera_id,
    camera_name,
    frame_id,
    image_ptr);

  glim::OdometryEstimationCallbacks::on_insert_image_frame(image_frame);

  // Backward compatibility:
  // only primary camera enters the old single-camera GLIM image pipeline.
  if (camera_id == 0) {
    odometry_estimation->insert_image(stamp, cv_image->image);

    if (sub_mapping) {
      sub_mapping->insert_image(stamp, cv_image->image);
    }

    if (global_mapping) {
      global_mapping->insert_image(stamp, cv_image->image);
    }
  }
}

#endif

void GlimROS::multi_lidar_points_callback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg,
  const std::string& topic) {
  if (!msg || !multi_lidar_merger) {
    return;
  }

  const auto merged_clouds = multi_lidar_merger->add_cloud(topic, msg);
  for (const auto& merged_cloud : merged_clouds) {
    if (merged_cloud) {
      points_callback(merged_cloud);
    }
  }
}

size_t GlimROS::points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
  spdlog::trace("points: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (!GlobalConfig::instance()->has_param("meta", "lidar_frame_id")) {
    spdlog::debug("auto-detecting LiDAR frame ID: {}", msg->header.frame_id);
    GlobalConfig::instance()->override_param<std::string>("meta", "lidar_frame_id", msg->header.frame_id);
  }

  auto raw_points = glim::extract_raw_points(*msg, intensity_field, ring_field);
  if (raw_points == nullptr) {
    spdlog::warn("failed to extract points from message");
    return 0;
  }

  if (raw_points->size() == 0) {
    spdlog::warn(
      "[scan_guard] raw point cloud is empty, skip LiDAR update stamp={}",
      raw_points->stamp);
    return 0;
  }

  raw_points->stamp += points_time_offset;
  if (!time_keeper->process(raw_points)) {
    spdlog::warn("skip an invalid point cloud (stamp={})", raw_points->stamp);
    return 0;
  }
  auto preprocessed = preprocessor->preprocess(raw_points);
  if (!preprocessed) {
    spdlog::warn(
      "[scan_guard] preprocessed frame is nullptr, skip LiDAR update stamp={} raw_points={}",
      raw_points ? raw_points->stamp : 0.0,
      raw_points ? raw_points->size() : 0);
    return 0;
  }

  if (keep_raw_points) {
    // note: Raw points are used only in extension modules for visualization purposes.
    //       If you need to reduce the memory footprint, you can safely comment out the following line.
    preprocessed->raw_points = raw_points;
  }

  odometry_estimation->insert_frame(preprocessed);

  const size_t workload = odometry_estimation->workload();
  spdlog::debug("workload={}", workload);

  return workload;
}


size_t GlimROS::odometry_workload() {
  return odometry_estimation ? odometry_estimation->workload() : 0;
}

size_t GlimROS::local_mapping_workload() {
  return sub_mapping ? sub_mapping->workload() : 0;
}

size_t GlimROS::global_mapping_workload() {
  return global_mapping ? global_mapping->workload() : 0;
}

bool GlimROS::needs_wait() {
  for (const auto& ext_module : extension_modules) {
    if (ext_module->needs_wait()) {
      return true;
    }
  }

  return false;
}

void GlimROS::timer_callback() {
  for (const auto& ext_module : extension_modules) {
    if (!ext_module->ok()) {
      rclcpp::shutdown();
    }
  }

  std::vector<glim::EstimationFrame::ConstPtr> estimation_frames;
  std::vector<glim::EstimationFrame::ConstPtr> marginalized_frames;
  odometry_estimation->get_results(estimation_frames, marginalized_frames);

  if (sub_mapping) {
    for (const auto& frame : marginalized_frames) {
      sub_mapping->insert_frame(frame);
    }

    auto submaps = sub_mapping->get_results();
    if (global_mapping) {
      for (const auto& submap : submaps) {
        global_mapping->insert_submap(submap);
      }
    }
  }
}

void GlimROS::wait(bool auto_quit) {
  auto memory_usage_ratio = []() -> double {
    std::ifstream ifs("/proc/meminfo");
    if (!ifs) {
      return -1.0;
    }

    long long mem_total_kb = -1;
    long long mem_available_kb = -1;

    std::string key;
    long long value = 0;
    std::string unit;

    while (ifs >> key >> value >> unit) {
      if (key == "MemTotal:") {
        mem_total_kb = value;
      } else if (key == "MemAvailable:") {
        mem_available_kb = value;
      }

      if (mem_total_kb > 0 && mem_available_kb >= 0) {
        break;
      }
    }

    if (mem_total_kb <= 0 || mem_available_kb < 0) {
      return -1.0;
    }

    return static_cast<double>(mem_total_kb - mem_available_kb) /
           static_cast<double>(mem_total_kb);
  };

  auto make_bar = [](double ratio, int width = 28) -> std::string {
    ratio = std::max(0.0, std::min(1.0, ratio));
    const int filled = static_cast<int>(std::round(ratio * width));

    std::string bar;
    bar.reserve(width + 2);
    bar.push_back('[');
    for (int i = 0; i < width; i++) {
      bar.push_back(i < filled ? '#' : '.');
    }
    bar.push_back(']');
    return bar;
  };

  auto wait_with_progress = [&](
    const std::string& label,
    const std::string& unit_name,
    const std::function<size_t()>& workload_fn,
    const std::function<void()>& join_fn) {
    const auto begin = std::chrono::steady_clock::now();

    size_t initial_pending = 0;
    try {
      initial_pending = workload_fn();
    } catch (...) {
      initial_pending = 0;
    }

    spdlog::info(
      "\033[1;35m[drain]\033[0m {} started: initial_pending={} unit={}",
      label,
      initial_pending,
      unit_name);

    auto future = std::async(std::launch::async, [&]() {
      join_fn();
    });

    size_t last_pending = initial_pending;
    int stable_pending_sec = 0;

    while (rclcpp::ok()) {
      const auto status = future.wait_for(std::chrono::milliseconds(500));
      const auto now = std::chrono::steady_clock::now();
      const auto elapsed_sec =
        std::chrono::duration_cast<std::chrono::seconds>(now - begin).count();

      size_t pending = 0;
      try {
        pending = workload_fn();
      } catch (...) {
        pending = 0;
      }

      if (pending == last_pending) {
        stable_pending_sec++;
      } else {
        stable_pending_sec = 0;
      }
      last_pending = pending;

      const size_t done_est =
        initial_pending > pending ? initial_pending - pending : 0;

      const double queue_ratio =
        initial_pending > 0
          ? static_cast<double>(done_est) / static_cast<double>(initial_pending)
          : (pending == 0 ? 1.0 : 0.0);

      const std::string phase =
        pending > 0 ? "processing_queue" : "finalizing_join";

      const double mem = memory_usage_ratio();

      std::ostringstream oss;
      oss
        << "\r\033[1;36m[drain_progress]\033[0m "
        << label << " "
        << make_bar(queue_ratio)
        << " queue=" << done_est << "/" << initial_pending
        << " pending=" << pending
        << " unit=" << unit_name
        << " phase=" << phase
        << " stable=" << stable_pending_sec / 2 << "s"
        << " elapsed=" << elapsed_sec << "s";

      if (mem >= 0.0) {
        oss << " mem=" << std::fixed << std::setprecision(1) << (mem * 100.0) << "%";
      }

      oss << "      ";

      std::cerr << oss.str() << std::flush;

      if (status == std::future_status::ready) {
        future.get();
        std::cerr << std::endl;

        spdlog::info(
          "\033[1;32m[drain_done]\033[0m {} done: initial_pending={} final_pending={} elapsed={}s",
          label,
          initial_pending,
          pending,
          elapsed_sec);
        return;
      }
    }

    future.wait();
    future.get();
    std::cerr << std::endl;
  };

  wait_with_progress(
    "odometry_estimation",
    "lidar_frames",
    [this]() -> size_t {
      return odometry_estimation ? odometry_estimation->workload() : 0;
    },
    [this]() {
      odometry_estimation->join();
    });

  if (sub_mapping) {
    std::vector<glim::EstimationFrame::ConstPtr> estimation_results;
    std::vector<glim::EstimationFrame::ConstPtr> marginalized_frames;
    odometry_estimation->get_results(estimation_results, marginalized_frames);
    for (const auto& marginalized_frame : marginalized_frames) {
      sub_mapping->insert_frame(marginalized_frame);
    }

    wait_with_progress(
      "local_mapping",
      "marginalized_frames",
      [this]() -> size_t {
        return sub_mapping ? sub_mapping->workload() : 0;
      },
      [this]() {
        sub_mapping->join();
      });

    const auto submaps = sub_mapping->get_results();
    if (global_mapping) {
      for (const auto& submap : submaps) {
        global_mapping->insert_submap(submap);
      }

      wait_with_progress(
        "global_mapping",
        "submaps",
        [this]() -> size_t {
          return global_mapping ? global_mapping->workload() : 0;
        },
        [this]() {
          global_mapping->join();
        });
    }
  }

  if (!auto_quit) {
    spdlog::info("\033[1;35m[drain]\033[0m waiting for extension modules");
    bool terminate = false;
    while (!terminate && rclcpp::ok()) {
      for (const auto& ext_module : extension_modules) {
        terminate |= (!ext_module->ok());
      }
    }
    spdlog::info("\033[1;32m[drain_done]\033[0m extension modules finished");
  }
}

void GlimROS::save(const std::string& path) {
  if (global_mapping) global_mapping->save(path);
  for (auto& module : extension_modules) {
    module->at_exit(path);
  }
}

}  // namespace glim

RCLCPP_COMPONENTS_REGISTER_NODE(glim::GlimROS);
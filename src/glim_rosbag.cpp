#include <glob.h>
#include <termios.h>
#include <unistd.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <future>
#include <numeric>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <boost/format.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_compression/sequential_compression_reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/metadata_io.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <glim/util/config.hpp>
#include <glim/util/extension_module_ros2.hpp>
#include <glim_ros/glim_ros.hpp>
#include <glim_ros/ros_compatibility.hpp>
#include <glim_ros/bag_progress.hpp>
#include <glim_ros/multi_lidar_cloud_merger.hpp>
#include <fstream>
#include <algorithm>

class SpeedCounter {
public:
  SpeedCounter() : last_sim_time(0.0), last_real_time(std::chrono::high_resolution_clock::now()) {}

  void update(const double& stamp) {
    const auto now = std::chrono::high_resolution_clock::now();
    if (now - last_real_time < std::chrono::seconds(5)) {
      return;
    }

    if (last_sim_time > 0.0) {
      const auto real = now - last_real_time;
      const auto sim = stamp - last_sim_time;
      const double playback_speed = sim / (std::chrono::duration_cast<std::chrono::nanoseconds>(real).count() / 1e9);
      spdlog::info("playback speed: {:.3f}x", playback_speed);
    }

    last_sim_time = stamp;
    last_real_time = now;
  }

private:
  double last_sim_time;
  std::chrono::high_resolution_clock::time_point last_real_time;
};

class KeyboardHandler {
public:
  KeyboardHandler() : paused_(false), active_(false) {
    if (isatty(STDIN_FILENO)) {
      tcgetattr(STDIN_FILENO, &original_termios_);
      struct termios raw = original_termios_;
      raw.c_lflag &= ~(ICANON | ECHO);
      raw.c_cc[VMIN] = 0;
      raw.c_cc[VTIME] = 0;
      tcsetattr(STDIN_FILENO, TCSANOW, &raw);
      active_ = true;
    }
  }

  ~KeyboardHandler() {
    if (active_) {
      tcsetattr(STDIN_FILENO, TCSANOW, &original_termios_);
    }
  }

  void update() {
    if (!active_) return;
    char c;
    while (read(STDIN_FILENO, &c, 1) > 0) {
      if (c == ' ') {
        paused_ = !paused_;
        if (paused_) {
          spdlog::info("playback paused (press space to resume)");
        } else {
          spdlog::info("playback resumed");
        }
      }
    }
  }

  bool is_paused() const { return paused_; }

private:
  bool paused_;
  bool active_;
  struct termios original_termios_;
};


namespace {

std::unordered_map<std::string, std::size_t> topic_message_counts_from_metadata(
    const rosbag2_storage::BagMetadata& metadata,
    const std::vector<std::string>& selected_topics) {
  std::unordered_map<std::string, std::size_t> out;

  for (const auto& topic : selected_topics) {
    out[topic] = 0;
  }

  for (const auto& topic_info : metadata.topics_with_message_count) {
    const auto& name = topic_info.topic_metadata.name;

    if (std::find(selected_topics.begin(), selected_topics.end(), name) == selected_topics.end()) {
      continue;
    }

    out[name] += static_cast<std::size_t>(topic_info.message_count);
  }

  return out;
}

double duration_sec_from_metadata(const rosbag2_storage::BagMetadata& metadata) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(metadata.duration).count();
}

}  // namespace



namespace {

struct MemoryPauseConfig {
  bool enabled = true;
  double high_ratio = 0.91;
  double resume_ratio = 0.86;
  int check_interval_ms = 500;
  double log_interval_sec = 5.0;
};

bool read_meminfo_kb(long long& mem_total_kb, long long& mem_available_kb) {
  std::ifstream ifs("/proc/meminfo");
  if (!ifs) {
    return false;
  }

  mem_total_kb = -1;
  mem_available_kb = -1;

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
      return true;
    }
  }

  return mem_total_kb > 0 && mem_available_kb >= 0;
}

double system_memory_usage_ratio() {
  long long total_kb = 0;
  long long available_kb = 0;

  if (!read_meminfo_kb(total_kb, available_kb) || total_kb <= 0) {
    return 0.0;
  }

  const double used = static_cast<double>(total_kb - available_kb);
  return std::clamp(used / static_cast<double>(total_kb), 0.0, 1.0);
}

void wait_for_memory_if_needed(
  const MemoryPauseConfig& cfg,
  const std::shared_ptr<glim::GlimROS>& glim) {
  if (!cfg.enabled) {
    return;
  }

  double usage = system_memory_usage_ratio();
  if (usage < cfg.high_ratio) {
    return;
  }

  spdlog::warn(
    "[mem_guard] pausing rosbag reading: RAM usage {:.2f}% >= {:.2f}%",
    usage * 100.0,
    cfg.high_ratio * 100.0);

  auto last_log = std::chrono::steady_clock::now();

  while (rclcpp::ok()) {
    rclcpp::spin_some(glim);

    // Let GLIM process already queued work while we stop feeding new bag messages.
    glim->timer_callback();

    usage = system_memory_usage_ratio();
    if (usage <= cfg.resume_ratio) {
      spdlog::warn(
        "[mem_guard] resuming rosbag reading: RAM usage {:.2f}% <= {:.2f}%",
        usage * 100.0,
        cfg.resume_ratio * 100.0);
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(now - last_log).count();

    if (elapsed >= cfg.log_interval_sec) {
      spdlog::warn(
        "[mem_guard] still paused: RAM usage {:.2f}% > resume {:.2f}%",
        usage * 100.0,
        cfg.resume_ratio * 100.0);
      last_log = now;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.check_interval_ms));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: glim_rosbag input_rosbag_path" << std::endl;
    return 0;
  }

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto glim = std::make_shared<glim::GlimROS>(options);

  // List topics
  glim::Config config_ros(glim::GlobalConfig::get_config_path("config_ros"));

  const std::string imu_topic = config_ros.param<std::string>("glim_ros", "imu_topic", "/imu");
  const std::string points_topic = config_ros.param<std::string>("glim_ros", "points_topic", "/points");
  const std::string image_topic = config_ros.param<std::string>("glim_ros", "image_topic", "/image");

  glim_ros::MultiLidarMergerConfig multi_lidar_config;
  multi_lidar_config.enabled = config_ros.param<bool>("multi_lidar", "enabled", false);
  multi_lidar_config.target_frame = config_ros.param<std::string>("multi_lidar", "target_frame", "lidar");
  multi_lidar_config.lidar_topics = config_ros.param<std::vector<std::string>>("multi_lidar", "lidar_topics", {});
  multi_lidar_config.lidar_serials = config_ros.param<std::vector<std::string>>("multi_lidar", "lidar_serials", {});
  multi_lidar_config.lidar_ids = config_ros.param<std::vector<int>>("multi_lidar", "lidar_ids", {});
  multi_lidar_config.calibration_file = config_ros.param<std::string>("multi_lidar", "calibration_file", "");
  multi_lidar_config.sync_tolerance_sec = config_ros.param<double>("multi_lidar", "sync_tolerance_sec", 0.03);
  multi_lidar_config.allow_incomplete_lidar_group = config_ros.param<bool>("multi_lidar", "allow_incomplete_lidar_group", false);
  multi_lidar_config.max_cloud_buffer_size = config_ros.param<int>("multi_lidar", "max_cloud_buffer_size", 50);

  glim_ros::MultiLidarCloudMerger multi_lidar(multi_lidar_config);

  std::vector<std::string> topics = {imu_topic, image_topic};

  if (multi_lidar.enabled()) {
    topics.insert(topics.end(), multi_lidar.topics().begin(), multi_lidar.topics().end());
  } else {
    topics.push_back(points_topic);
  }

  rosbag2_storage::StorageFilter filter;
  spdlog::info("topics:");
  for (const auto& topic : topics) {
    spdlog::info("- {}", topic);
    filter.topics.push_back(topic);
  }

  //
  std::unordered_map<std::string, std::vector<glim::GenericTopicSubscription::Ptr>> subscription_map;
  for (const auto& sub : glim->extension_subscriptions()) {
    spdlog::info("- {} (ext)", sub->topic);
    filter.topics.push_back(sub->topic);
    subscription_map[sub->topic].push_back(sub);
  }

  // List input rosbag filenames
  std::vector<std::string> bag_filenames;

  for (int i = 1; i < argc; i++) {
    std::vector<std::string> filenames;
    glob_t globbuf;
    int ret = glob(argv[i], 0, nullptr, &globbuf);
    for (int i = 0; i < globbuf.gl_pathc; i++) {
      filenames.push_back(globbuf.gl_pathv[i]);
    }
    globfree(&globbuf);

    bag_filenames.insert(bag_filenames.end(), filenames.begin(), filenames.end());
  }
  std::sort(bag_filenames.begin(), bag_filenames.end());

  spdlog::info("bag_filenames:");
  for (const auto& bag_filename : bag_filenames) {
    spdlog::info("- {}", bag_filename);
  }

  std::unordered_map<std::string, std::size_t> progress_total_counts;
  for (const auto& topic : filter.topics) {
    progress_total_counts[topic] = 0;
  }

  double progress_total_duration_sec = 0.0;

  for (const auto& bag_filename : bag_filenames) {
    try {
      rosbag2_storage::MetadataIo metadata_io;
      const auto metadata = metadata_io.read_metadata(bag_filename);

      const auto counts = topic_message_counts_from_metadata(metadata, filter.topics);
      for (const auto& [topic, count] : counts) {
        progress_total_counts[topic] += count;
      }

      progress_total_duration_sec += duration_sec_from_metadata(metadata);
    } catch (const std::exception& e) {
      spdlog::warn("[bag_progress] failed to read metadata for {}: {}", bag_filename, e.what());
    }
  }

  glim_ros::BagProgress bag_progress(2.0);
  bag_progress.set_totals(progress_total_counts, progress_total_duration_sec);
  bag_progress.force_log();

  // Playback range settings
  double delay = 0.0;
  glim->declare_parameter<double>("delay", delay);
  glim->get_parameter<double>("delay", delay);

  double start_offset = 0.0;
  glim->declare_parameter<double>("start_offset", start_offset);
  glim->get_parameter<double>("start_offset", start_offset);

  double playback_duration = 0.0;
  glim->declare_parameter<double>("playback_duration", playback_duration);
  glim->get_parameter<double>("playback_duration", playback_duration);

  double playback_until = 0.0;
  glim->declare_parameter<double>("playback_until", playback_until);
  glim->get_parameter<double>("playback_until", playback_until);

  // Playback speed settings
  const double playback_speed = config_ros.param<double>("glim_ros", "playback_speed", 100.0);
  std::chrono::system_clock::time_point real_t0;
  rcutils_time_point_value_t bag_t0 = 0;
  SpeedCounter speed_counter;

  double end_time = std::numeric_limits<double>::max();
  glim->declare_parameter<double>("end_time", end_time);
  glim->get_parameter<double>("end_time", end_time);

  if (delay > 0.0) {
    spdlog::info("delaying {} sec", delay);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(delay * 1000)));
  }

  // Keyboard handler for pause/resume
  KeyboardHandler keyboard;
  // Memory guard for UAV/offline processing.
  MemoryPauseConfig memory_pause_cfg;
  glim->declare_parameter("glim_rosbag/memory_pause_enabled", memory_pause_cfg.enabled);
  glim->declare_parameter("glim_rosbag/memory_pause_high_ratio", memory_pause_cfg.high_ratio);
  glim->declare_parameter("glim_rosbag/memory_pause_resume_ratio", memory_pause_cfg.resume_ratio);
  glim->declare_parameter("glim_rosbag/memory_pause_check_interval_ms", memory_pause_cfg.check_interval_ms);
  glim->declare_parameter("glim_rosbag/memory_pause_log_interval_sec", memory_pause_cfg.log_interval_sec);

  glim->get_parameter("glim_rosbag/memory_pause_enabled", memory_pause_cfg.enabled);
  glim->get_parameter("glim_rosbag/memory_pause_high_ratio", memory_pause_cfg.high_ratio);
  glim->get_parameter("glim_rosbag/memory_pause_resume_ratio", memory_pause_cfg.resume_ratio);
  glim->get_parameter("glim_rosbag/memory_pause_check_interval_ms", memory_pause_cfg.check_interval_ms);
  glim->get_parameter("glim_rosbag/memory_pause_log_interval_sec", memory_pause_cfg.log_interval_sec);

  spdlog::info(
    "[mem_guard] enabled={} high={:.2f}% resume={:.2f}% check={}ms",
    memory_pause_cfg.enabled,
    memory_pause_cfg.high_ratio * 100.0,
    memory_pause_cfg.resume_ratio * 100.0,
    memory_pause_cfg.check_interval_ms);


  // Bag read function
  const auto read_bag = [&](const std::string& bag_filename) {
    spdlog::info("opening {}", bag_filename);
    rosbag2_storage::StorageOptions options;
    options.uri = bag_filename;

    bool is_mcap = bag_filename.size() > 5 && bag_filename.rfind(".mcap") == (bag_filename.size() - 5);
    if (is_mcap) {
      options.storage_id = "mcap";
    } else if (std::filesystem::is_directory(bag_filename)) {
      try {
        rosbag2_storage::MetadataIo metadata_io;
        const auto metadata = metadata_io.read_metadata(bag_filename);
        options.storage_id = metadata.storage_identifier;

        if (options.storage_id.empty()) {
          spdlog::warn("storage_identifier not found in metadata.yaml (uri={}), fallback to sqlite3", bag_filename);
          options.storage_id = "sqlite3";
        } else {
          spdlog::info("detected storage_id={} from metadata.yaml", options.storage_id);
        }
      } catch (const std::exception& e) {
        spdlog::warn("failed to read metadata.yaml (uri={}): {} (fallback to sqlite3)", bag_filename, e.what());
        options.storage_id = "sqlite3";
      }
    } else {
      options.storage_id = "sqlite3";
    }

    rosbag2_cpp::ConverterOptions converter_options;

    // rosbag2_cpp::Reader reader;
    std::unique_ptr<rosbag2_cpp::reader_interfaces::BaseReaderInterface> reader_;
    reader_ = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
    reader_->open(options, converter_options);

    if (reader_->get_metadata().compression_format != "") {
      spdlog::info("compression detected (format={})", reader_->get_metadata().compression_format);
      spdlog::info("opening bag with SequentialCompressionReader");
      reader_ = std::make_unique<rosbag2_compression::SequentialCompressionReader>();
      reader_->open(options, converter_options);
    }

    auto& reader = *reader_;
    reader.set_filter(filter);

    const auto topics_and_types = reader.get_all_topics_and_types();
    std::unordered_map<std::string, std::string> topic_type_map;
    for (const auto& topic : topics_and_types) {
      topic_type_map[topic.name] = topic.type;
    }

    rclcpp::Serialization<sensor_msgs::msg::Imu> imu_serialization;
    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> points_serialization;
#ifdef BUILD_WITH_CV_BRIDGE
    rclcpp::Serialization<sensor_msgs::msg::Image> image_serialization;
    rclcpp::Serialization<sensor_msgs::msg::CompressedImage> compressed_image_serialization;
#endif

    while (reader.has_next()) {
      if (!rclcpp::ok()) {
        return false;
      }
      rclcpp::spin_some(glim);

      wait_for_memory_if_needed(memory_pause_cfg, glim);
      const auto msg = reader.read_next();
      const std::string topic_type = topic_type_map[msg->topic_name];
      const rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);

      if (real_t0.time_since_epoch().count() == 0) {
        real_t0 = std::chrono::system_clock::now();
      }

      const auto msg_time = get_msg_recv_timestamp(*msg);
      if (bag_t0 == 0) {
        bag_t0 = msg_time;
      }
      bag_progress.on_message(msg->topic_name, msg_time / 1e9);
      spdlog::debug("msg_time: {} ({} sec)", msg_time / 1e9, (msg_time - bag_t0) / 1e9);

      if (start_offset > 0.0) {
        spdlog::info("skipping msg for start_offset {}", start_offset);
        reader.seek(bag_t0 + start_offset * 1e9);

        start_offset = 0.0;
        bag_t0 = 0;
        real_t0 = std::chrono::system_clock::from_time_t(0);
        continue;
      }

      if (playback_until > 0.0 && msg_time / 1e9 > playback_until) {
        spdlog::info("reached playback_until ({} < {})", msg_time / 1e9, playback_until);
        return false;
      }

      if (playback_duration > 0.0 && (msg_time - bag_t0) / 1e9 > playback_duration) {
        spdlog::info("reached playback_duration ({} > {})", (msg_time - bag_t0) / 1e9, playback_duration);
        return false;
      }

      // Pause/resume handling
      keyboard.update();
      if (keyboard.is_paused()) {
        auto pause_start = std::chrono::system_clock::now();
        while (keyboard.is_paused() && rclcpp::ok()) {
          rclcpp::spin_some(glim);
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          keyboard.update();
        }
        if (!rclcpp::ok()) {
          return false;
        }
        // Adjust real_t0 to account for pause duration to avoid fast-forward
        real_t0 += std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::system_clock::now() - pause_start);
      }

      const auto bag_elapsed = std::chrono::nanoseconds(msg_time - bag_t0);
      while (playback_speed > 0.0 && (std::chrono::system_clock::now() - real_t0) * playback_speed < bag_elapsed) {
        const double real_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - real_t0).count() / 1e9;
        spdlog::debug("throttling (real_elapsed={} bag_elapsed={} playback_speed={})", real_elapsed, bag_elapsed.count() / 1e9, playback_speed);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      if (msg->topic_name == imu_topic) {
        if (topic_type != "sensor_msgs/msg/Imu") {
          spdlog::error("topic_type mismatch: {} != sensor_msgs/msg/Imu (topic={})", topic_type, msg->topic_name);
          return false;
        }
        auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
        imu_serialization.deserialize_message(&serialized_msg, imu_msg.get());
        glim->imu_callback(imu_msg);
      } else if (((!multi_lidar.enabled() && msg->topic_name == points_topic) || (multi_lidar.enabled() && multi_lidar.is_lidar_topic(msg->topic_name)))) {
        if (topic_type != "sensor_msgs/msg/PointCloud2") {
          spdlog::error("topic_type mismatch: {} != sensor_msgs/msg/PointCloud2 (topic={})", topic_type, msg->topic_name);
          return false;
        }
        auto points_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        points_serialization.deserialize_message(&serialized_msg, points_msg.get());
        const size_t workload = glim->points_callback(points_msg);

        if (points_msg->header.stamp.sec + points_msg->header.stamp.nanosec * 1e-9 > end_time) {
          spdlog::info("end_time reached");
          return false;
        }

        if (workload > 5) {
          // Odometry estimation is behind
          const size_t sleep_msec = (workload - 4) * 5;
          spdlog::debug("throttling: {} msec (workload={})", sleep_msec, workload);
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
        }
      }
#ifdef BUILD_WITH_CV_BRIDGE
      else if (msg->topic_name == image_topic) {
        if (topic_type == "sensor_msgs/msg/Image") {
          auto image_msg = std::make_shared<sensor_msgs::msg::Image>();
          image_serialization.deserialize_message(&serialized_msg, image_msg.get());
          glim->image_callback(image_msg);
        } else if (topic_type == "sensor_msgs/msg/CompressedImage") {
          auto compressed_image_msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
          compressed_image_serialization.deserialize_message(&serialized_msg, compressed_image_msg.get());

          auto image_msg = std::make_shared<sensor_msgs::msg::Image>();
          cv_bridge::toCvCopy(*compressed_image_msg, "bgr8")->toImageMsg(*image_msg);
          glim->image_callback(image_msg);
        } else {
          spdlog::error("topic_type mismatch: {} != sensor_msgs/msg/(Image|CompressedImage) (topic={})", topic_type, msg->topic_name);
          return false;
        }
      }
#endif

      auto found = subscription_map.find(msg->topic_name);
      if (found != subscription_map.end()) {
        for (const auto& sub : found->second) {
          sub->insert_message_instance(serialized_msg, topic_type);
        }
      }

      glim->timer_callback();
      speed_counter.update(msg_time / 1e9);

      const auto t0 = std::chrono::high_resolution_clock::now();
      while (glim->needs_wait()) {
        rclcpp::spin_some(glim);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        spdlog::debug("throttling (waiting for odometry estimation)");
        if (std::chrono::high_resolution_clock::now() - t0 > std::chrono::seconds(1)) {
          spdlog::warn("throttling timeout (an extension module may be hanged)");
          break;
        }
      }
    }

    bag_progress.finish_log();
    return true;
  };

  // Read all rosbags
  bool auto_quit = false;
  glim->declare_parameter<bool>("auto_quit", auto_quit);
  glim->get_parameter<bool>("auto_quit", auto_quit);

  std::string dump_path = "/tmp/dump";
  glim->declare_parameter<std::string>("dump_path", dump_path);
  glim->get_parameter<std::string>("dump_path", dump_path);

  for (const auto& bag_filename : bag_filenames) {
    if (!read_bag(bag_filename)) {
      auto_quit = true;
      break;
    }
  }

  if (!auto_quit) {
    rclcpp::spin(glim);
  }

  {
    const double drain_start = glim_ros::BagProgress::now_sec();
    spdlog::info("[bag_progress] drain begin auto_quit={}", auto_quit);

    auto wait_future = std::async(std::launch::async, [&] {
      glim->wait(auto_quit);
    });

    while (wait_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      glim_ros::BagProgress::drain_log("glim_wait", drain_start);
    }

    wait_future.get();
    glim_ros::BagProgress::drain_log("glim_wait_done", drain_start);
  }

  {
    const double save_start = glim_ros::BagProgress::now_sec();
    spdlog::info("[bag_progress] save begin path={}", dump_path);

    auto save_future = std::async(std::launch::async, [&] {
      glim->save(dump_path);
    });

    while (save_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      glim_ros::BagProgress::drain_log("save_dump", save_start);
    }

    save_future.get();
    glim_ros::BagProgress::drain_log("save_dump_done", save_start);
  }

  return 0;
}

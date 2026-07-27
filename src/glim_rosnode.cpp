#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <spdlog/spdlog.h>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <glim_ros/glim_ros.hpp>
#include <glim/util/config.hpp>
#include <glim/util/extension_module_ros2.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  auto glim = std::make_shared<glim::GlimROS>(options);

  std::string dump_path = "/tmp/dump";

  if (!glim->has_parameter("dump_path")) {
    glim->declare_parameter<std::string>("dump_path", dump_path);
  }

  glim->get_parameter("dump_path", dump_path);

  std::atomic_bool finish_requested{false};

  auto save_and_finish_service =
    glim->create_service<std_srvs::srv::Trigger>(
      "~/save_and_finish",
      [&finish_requested](
        const std::shared_ptr<
          std_srvs::srv::Trigger::Request> /* request */,
        std::shared_ptr<
          std_srvs::srv::Trigger::Response> response) {
        const bool already_requested =
          finish_requested.exchange(
            true,
            std::memory_order_acq_rel);

        if (already_requested) {
          response->success = false;
          response->message =
            "Save and finish has already been requested";
          return;
        }

        response->success = true;
        response->message =
          "Save accepted; GLIM is draining queues and saving the dump";
      });

  spdlog::info("[direct_node] started");
  spdlog::info("[direct_node] dump_path={}", dump_path);
  spdlog::info(
    "[direct_node] save service={}/save_and_finish",
    glim->get_fully_qualified_name());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(glim);

  // Обычный direct/live режим. Узел получает IMU/LiDAR через ROS topics.
  while (
    rclcpp::ok() &&
    !finish_requested.load(std::memory_order_acquire)) {
    executor.spin_some();
  }

  if (!finish_requested.load(std::memory_order_acquire)) {
    spdlog::error(
      "[direct_node] ROS shutdown happened before save service call; "
      "dump was not saved");

    executor.remove_node(glim);

    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }

    return 2;
  }

  // Callback уже вернул ответ клиенту. После этого новые сообщения
  // больше не должны приниматься.
  executor.remove_node(glim);
  save_and_finish_service.reset();

  try {
    spdlog::info(
      "[direct_node] finish requested; workloads before drain: "
      "odom={} local={} global={}",
      glim->odometry_workload(),
      glim->local_mapping_workload(),
      glim->global_mapping_workload());

    // Перед wait переносим готовые результаты между стадиями pipeline.
    glim->timer_callback();

    spdlog::info(
      "[direct_node] workloads after final timer callback: "
      "odom={} local={} global={}",
      glim->odometry_workload(),
      glim->local_mapping_workload(),
      glim->global_mapping_workload());

    spdlog::info("[direct_node] drain begin");

    // true соответствует offline/auto-quit завершению:
    // дождаться рабочих очередей и остановить async modules.
    glim->wait(true);

    spdlog::info("[direct_node] drain done");
    spdlog::info(
      "[direct_node] save begin path={}",
      dump_path);

    glim->save(dump_path);

    spdlog::info(
      "[direct_node] save done path={}",
      dump_path);
  } catch (const std::exception& e) {
    spdlog::critical(
      "[direct_node] drain/save exception: {}",
      e.what());

    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }

    return 3;
  } catch (...) {
    spdlog::critical(
      "[direct_node] unknown drain/save exception");

    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }

    return 4;
  }

  glim.reset();

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  return 0;
}

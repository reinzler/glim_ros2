#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>

namespace glim_ros {

struct BagProgressTopicInfo {
  std::string topic;
  std::size_t total_messages = 0;
  std::size_t read_messages = 0;
};

class BagProgress {
public:
  explicit BagProgress(double log_period_sec = 2.0)
  : log_period_sec_(log_period_sec),
    wall_start_(Clock::now()),
    last_log_(wall_start_) {}

  void set_totals(
      const std::unordered_map<std::string, std::size_t>& totals,
      double bag_duration_sec) {
    topics_.clear();
    total_messages_ = 0;

    for (const auto& [topic, count] : totals) {
      BagProgressTopicInfo info;
      info.topic = topic;
      info.total_messages = count;
      topics_[topic] = info;
      total_messages_ += count;
    }

    bag_duration_sec_ = std::max(0.0, bag_duration_sec);
  }

  void on_message(const std::string& topic, double msg_stamp_sec) {
    read_messages_++;

    auto found = topics_.find(topic);
    if (found != topics_.end()) {
      found->second.read_messages++;
    }

    if (first_stamp_sec_ < 0.0) {
      first_stamp_sec_ = msg_stamp_sec;
    }
    last_stamp_sec_ = msg_stamp_sec;

    maybe_log(false);
  }

  void force_log() {
    maybe_log(true);
  }

  void finish_log() const {
    const double elapsed = wall_elapsed_sec();
    const double msg_rate = elapsed > 1e-9 ? static_cast<double>(read_messages_) / elapsed : 0.0;

    spdlog::info(
      "[bag_progress] playback finished read={}/{} wall={} msg/s={}",
      read_messages_,
      total_messages_,
      format_duration(elapsed),
      format_double(msg_rate, 1));
  }

  static double now_sec() {
    const auto now = Clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now).count();
  }

  static void drain_log(const std::string& phase, double phase_start_sec) {
    const double elapsed = now_sec() - phase_start_sec;
    spdlog::info(
      "[bag_progress] drain phase={} elapsed={} eta=n/a",
      phase,
      format_duration(elapsed));
  }

private:
  using Clock = std::chrono::steady_clock;

  double wall_elapsed_sec() const {
    const auto now = Clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now - wall_start_).count();
  }

  static std::string format_double(double v, int precision) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << v;
    return ss.str();
  }

  static std::string format_duration(double sec) {
    if (!std::isfinite(sec) || sec < 0.0) {
      return "n/a";
    }

    const int total = static_cast<int>(sec + 0.5);
    const int h = total / 3600;
    const int m = (total % 3600) / 60;
    const int s = total % 60;

    std::ostringstream ss;
    if (h > 0) {
      ss << h << "h";
    }
    if (h > 0 || m > 0) {
      ss << m << "m";
    }
    ss << s << "s";
    return ss.str();
  }

  void maybe_log(bool force) {
    const auto now = Clock::now();
    const double since_last =
      std::chrono::duration_cast<std::chrono::duration<double>>(now - last_log_).count();

    if (!force && since_last < log_period_sec_) {
      return;
    }

    last_log_ = now;

    const double wall_elapsed = wall_elapsed_sec();
    const double msg_rate =
      wall_elapsed > 1e-9 ? static_cast<double>(read_messages_) / wall_elapsed : 0.0;

    double pct = 0.0;
    if (total_messages_ > 0) {
      pct = 100.0 * static_cast<double>(read_messages_) / static_cast<double>(total_messages_);
    }

    double eta_sec = -1.0;
    if (msg_rate > 1e-9 && total_messages_ > read_messages_) {
      eta_sec = static_cast<double>(total_messages_ - read_messages_) / msg_rate;
    }

    double bag_elapsed = 0.0;
    if (first_stamp_sec_ >= 0.0 && last_stamp_sec_ >= first_stamp_sec_) {
      bag_elapsed = last_stamp_sec_ - first_stamp_sec_;
    }

    double replay_speed = 0.0;
    if (wall_elapsed > 1e-9 && bag_elapsed > 0.0) {
      replay_speed = bag_elapsed / wall_elapsed;
    }

    spdlog::info(
      "[bag_progress] read={}/{} {}% bag={}/{} wall={} speed={}x msg/s={} eta={}",
      read_messages_,
      total_messages_,
      format_double(pct, 1),
      format_duration(bag_elapsed),
      bag_duration_sec_ > 0.0 ? format_duration(bag_duration_sec_) : "n/a",
      format_duration(wall_elapsed),
      format_double(replay_speed, 2),
      format_double(msg_rate, 1),
      eta_sec >= 0.0 ? format_duration(eta_sec) : "n/a");

    if (!topics_.empty()) {
      std::vector<std::string> keys;
      keys.reserve(topics_.size());
      for (const auto& [topic, _] : topics_) {
        keys.push_back(topic);
      }
      std::sort(keys.begin(), keys.end());

      std::ostringstream ss;
      bool first = true;
      for (const auto& topic : keys) {
        const auto& item = topics_.at(topic);
        if (!first) {
          ss << " ";
        }
        first = false;
        ss << topic << "=" << item.read_messages << "/" << item.total_messages;
      }

      spdlog::info("[bag_progress] topics {}", ss.str());
    }
  }

private:
  double log_period_sec_;
  Clock::time_point wall_start_;
  Clock::time_point last_log_;

  std::size_t total_messages_ = 0;
  std::size_t read_messages_ = 0;

  double first_stamp_sec_ = -1.0;
  double last_stamp_sec_ = -1.0;
  double bag_duration_sec_ = 0.0;

  std::unordered_map<std::string, BagProgressTopicInfo> topics_;
};

}  // namespace glim_ros

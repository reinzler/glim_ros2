#pragma once

// DrainMonitor — многостадийный монитор пропускной способности для glim_ros.
//
// Отвечает на вопрос "успевает ли одометрия за сенсором", а не только
// "сколько осталось ждать".
//
// Ключевая метрика — realtime factor (RTF):
//   RTF = фактическая частота обработки / номинальная частота сенсора
//   RTF >= 1.0  -> стадия успевает, очередь не растёт
//   RTF <  1.0  -> отставание; в онлайне очередь растёт до OOM
//
// Стадии опрашиваются (poll), а не дёргаются из горячего пути:
// нужны только две функции на стадию — глубина очереди и монотонный
// счётчик обработанного. Ничего в самих стадиях править не нужно.
//
// Использование:
//   glim::DrainMonitor monitor;
//   monitor.add_stage({"odom", [&]{ return async_odom->input_queue_size(); },
//                              [&]{ return async_odom->processed_count(); }, 10.0});
//   monitor.add_stage({"sub",  [&]{ return async_sub->input_queue_size();  },
//                              [&]{ return async_sub->processed_count();  }, 0.0});
//   monitor.add_stage({"glob", [&]{ return async_glob->input_queue_size(); },
//                              [&]{ return async_glob->processed_count(); }, 0.0});
//   monitor.start();
//   ...
//   monitor.set_input_closed();   // бага кончилась / сенсор отключён -> включается ETA
//   monitor.stop();
//
// Если стадия не отдаёт processed_count(), передай nullptr — счётчик будет
// восстановлен как (всего подано - глубина очереди), см. add_stage().

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace glim {

class DrainMonitor {
public:
  struct StageSpec {
    std::string name;
    std::function<size_t()> queue_size;
    std::function<uint64_t()> processed;
    // Номинальная частота входа, Гц. 0 = неизвестна (RTF не считается).
    // Для одометрии с MID-360 это 10.0 — именно она и есть бюджет.
    double nominal_hz = 0.0;
  };

  struct Options {
    int poll_interval_ms = 500;
    int print_interval_ms = 1000;
    // Окно усреднения скорости, с. Слишком короткое -> ETA скачет.
    double rate_window_sec = 30.0;
    // Порог занятости RAM, до которого экстраполируется тренд, %.
    double mem_limit_percent = 90.0;
    int bar_width = 24;
    bool use_ansi = true;
  };

  DrainMonitor() = default;

  explicit DrainMonitor(const Options& opts) : opts_(opts) {}

  ~DrainMonitor() { stop(); }

  void add_stage(const StageSpec& spec) {
    std::lock_guard<std::mutex> lock(mutex_);
    Stage s;
    s.spec = spec;
    stages_.push_back(std::move(s));
  }

  // Общее число элементов на входе (если известно заранее, напр. кадров в баге).
  // В онлайне не вызывается -> прогресс-бар и ETA не показываются.
  void set_total(const std::string& stage_name, uint64_t total) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& s : stages_) {
      if (s.spec.name == stage_name) {
        s.total = total;
      }
    }
  }

  // Вход закрылся (бага доиграла). До этого момента ETA не имеет смысла:
  // total ещё растёт, и процент дёргался бы назад.
  void set_input_closed() { input_closed_.store(true); }

  void start() {
    if (running_.exchange(true)) {
      return;
    }
    start_time_ = clock::now();
    thread_ = std::thread([this] { run(); });
  }

  void stop() {
    if (!running_.exchange(false)) {
      return;
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    print_final();
  }

private:
  using clock = std::chrono::steady_clock;
  using time_point = clock::time_point;

  struct Sample {
    time_point t;
    uint64_t processed;
  };

  struct Stage {
    StageSpec spec;
    std::deque<Sample> samples;
    uint64_t total = 0;
    uint64_t last_processed = 0;
    size_t last_queue = 0;
    double rate = 0.0;      // элементов/с, сглаженная
    double rtf = 0.0;       // realtime factor
    double queue_growth = 0.0;  // элементов/с, + = растёт
  };

  struct MemSample {
    time_point t;
    double used_percent;
  };

  // ---------- сбор данных ----------

  static double read_mem_used_percent() {
    std::ifstream ifs("/proc/meminfo");
    if (!ifs) {
      return -1.0;
    }
    double total_kb = 0.0;
    double avail_kb = 0.0;
    std::string line;
    while (std::getline(ifs, line)) {
      std::istringstream iss(line);
      std::string key;
      double value = 0.0;
      iss >> key >> value;
      if (key == "MemTotal:") {
        total_kb = value;
      } else if (key == "MemAvailable:") {
        avail_kb = value;
      }
      if (total_kb > 0.0 && avail_kb > 0.0) {
        break;
      }
    }
    if (total_kb <= 0.0) {
      return -1.0;
    }
    return 100.0 * (total_kb - avail_kb) / total_kb;
  }

  // Линейная регрессия по окну: возвращает наклон (единиц в секунду).
  template <typename Container, typename GetT, typename GetY>
  static double slope(const Container& c, GetT get_t, GetY get_y) {
    const size_t n = c.size();
    if (n < 2) {
      return 0.0;
    }
    const auto t0 = get_t(c.front());
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (const auto& e : c) {
      const double x = std::chrono::duration<double>(get_t(e) - t0).count();
      const double y = get_y(e);
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
    }
    const double denom = n * sxx - sx * sx;
    if (std::fabs(denom) < 1e-9) {
      return 0.0;
    }
    return (n * sxy - sx * sy) / denom;
  }

  void poll() {
    const time_point now = clock::now();
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& s : stages_) {
      const size_t q = s.spec.queue_size ? s.spec.queue_size() : 0;
      const uint64_t p = s.spec.processed ? s.spec.processed() : 0;

      s.samples.push_back({now, p});
      while (s.samples.size() > 1 &&
             std::chrono::duration<double>(now - s.samples.front().t).count() > opts_.rate_window_sec) {
        s.samples.pop_front();
      }

      s.rate = slope(
        s.samples,
        [](const Sample& x) { return x.t; },
        [](const Sample& x) { return static_cast<double>(x.processed); });
      if (s.rate < 0.0) {
        s.rate = 0.0;
      }

      s.rtf = (s.spec.nominal_hz > 0.0) ? (s.rate / s.spec.nominal_hz) : 0.0;

      // Рост очереди: разница между притоком и обработкой.
      const double dt = std::chrono::duration<double>(now - last_poll_).count();
      if (dt > 1e-6 && polled_once_) {
        const double dq = static_cast<double>(q) - static_cast<double>(s.last_queue);
        s.queue_growth = 0.7 * s.queue_growth + 0.3 * (dq / dt);
      }

      s.last_queue = q;
      s.last_processed = p;
    }

    const double mem = read_mem_used_percent();
    if (mem >= 0.0) {
      mem_samples_.push_back({now, mem});
      while (mem_samples_.size() > 1 &&
             std::chrono::duration<double>(now - mem_samples_.front().t).count() > opts_.rate_window_sec) {
        mem_samples_.pop_front();
      }
      mem_percent_ = mem;
      mem_slope_ = slope(
        mem_samples_,
        [](const MemSample& x) { return x.t; },
        [](const MemSample& x) { return x.used_percent; });
    }

    last_poll_ = now;
    polled_once_ = true;
  }

  // ---------- вывод ----------

  static std::string fmt_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
      return "n/a";
    }
    if (seconds > 86400.0 * 7) {
      return ">7d";
    }
    const int total = static_cast<int>(seconds);
    const int h = total / 3600;
    const int m = (total % 3600) / 60;
    const int sec = total % 60;
    char buf[32];
    if (h > 0) {
      std::snprintf(buf, sizeof(buf), "%dh%02dm", h, m);
    } else if (m > 0) {
      std::snprintf(buf, sizeof(buf), "%dm%02ds", m, sec);
    } else {
      std::snprintf(buf, sizeof(buf), "%ds", sec);
    }
    return buf;
  }

  std::string make_bar(double frac) const {
    frac = std::max(0.0, std::min(1.0, frac));
    const int filled = static_cast<int>(frac * opts_.bar_width);
    std::string bar(1, '[');
    bar.append(filled, '#');
    bar.append(opts_.bar_width - filled, '.');
    bar.push_back(']');
    return bar;
  }

  void print() {
    std::lock_guard<std::mutex> lock(mutex_);

    const double elapsed = std::chrono::duration<double>(clock::now() - start_time_).count();
    const bool closed = input_closed_.load();

    std::ostringstream oss;

    if (opts_.use_ansi && lines_printed_ > 0) {
      oss << "\033[" << lines_printed_ << "A";  // курсор вверх
    }

    int lines = 0;
    const char* phase = closed ? "DRAIN" : "LIVE ";

    for (const auto& s : stages_) {
      oss << "\033[2K";  // очистить строку
      oss << "  " << phase << " " << pad(s.spec.name, 5) << " ";

      if (s.total > 0) {
        const double frac = static_cast<double>(s.last_processed) / static_cast<double>(s.total);
        oss << make_bar(frac) << " " << s.last_processed << "/" << s.total << " ";
      } else {
        oss << "done=" << s.last_processed << " ";
      }

      oss << "q=" << s.last_queue << " ";

      char buf[128];
      std::snprintf(buf, sizeof(buf), "%.2f/s ", s.rate);
      oss << buf;

      if (s.spec.nominal_hz > 0.0) {
        // Главная строка для реалтайма.
        const char* verdict = (s.rtf >= 1.0) ? "OK" : "LAG";
        std::snprintf(buf, sizeof(buf), "RTF=%.2f %s ", s.rtf, verdict);
        oss << buf;
      }

      if (s.queue_growth > 0.05) {
        std::snprintf(buf, sizeof(buf), "q+%.1f/s ", s.queue_growth);
        oss << buf;
      }

      // ETA только когда вход закрыт: иначе total ещё растёт и число врёт.
      if (closed && s.rate > 1e-6) {
        const double remaining = static_cast<double>(s.last_queue);
        oss << "eta=" << fmt_duration(remaining / s.rate) << " ";
      } else if (!closed) {
        oss << "eta=n/a ";
      }

      oss << "\n";
      ++lines;
    }

    // Память + экстраполяция до предела.
    {
      oss << "\033[2K";
      char buf[196];
      std::snprintf(buf, sizeof(buf), "  MEM   %.1f%%", mem_percent_);
      oss << buf;

      if (mem_slope_ > 1e-4) {
        const double to_limit = (opts_.mem_limit_percent - mem_percent_) / mem_slope_;
        std::snprintf(
          buf, sizeof(buf), "  (+%.2f%%/min -> %.0f%% через %s)",
          mem_slope_ * 60.0, opts_.mem_limit_percent, fmt_duration(to_limit).c_str());
        oss << buf;
      } else {
        oss << "  (стабильно)";
      }

      oss << "   elapsed=" << fmt_duration(elapsed) << "\n";
      ++lines;
    }

    lines_printed_ = lines;
    std::fputs(oss.str().c_str(), stderr);
    std::fflush(stderr);
  }

  void print_final() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fputs("\n", stderr);
    for (const auto& s : stages_) {
      char buf[256];
      if (s.spec.nominal_hz > 0.0) {
        std::snprintf(
          buf, sizeof(buf), "[drain] %s: done=%llu rate=%.2f/s RTF=%.2f (%s)\n",
          s.spec.name.c_str(), static_cast<unsigned long long>(s.last_processed), s.rate, s.rtf,
          (s.rtf >= 1.0 ? "успевает" : "ОТСТАЁТ"));
      } else {
        std::snprintf(
          buf, sizeof(buf), "[drain] %s: done=%llu rate=%.2f/s\n",
          s.spec.name.c_str(), static_cast<unsigned long long>(s.last_processed), s.rate);
      }
      std::fputs(buf, stderr);
    }
    std::fflush(stderr);
  }

  static std::string pad(const std::string& s, size_t n) {
    std::string r = s;
    if (r.size() < n) {
      r.append(n - r.size(), ' ');
    }
    return r;
  }

  void run() {
    auto next_print = clock::now();
    last_poll_ = clock::now();

    while (running_.load()) {
      poll();

      const auto now = clock::now();
      if (now >= next_print) {
        print();
        next_print = now + std::chrono::milliseconds(opts_.print_interval_ms);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(opts_.poll_interval_ms));
    }
  }

  Options opts_;
  std::vector<Stage> stages_;
  std::deque<MemSample> mem_samples_;

  double mem_percent_ = 0.0;
  double mem_slope_ = 0.0;

  time_point start_time_;
  time_point last_poll_;
  bool polled_once_ = false;

  std::atomic_bool running_{false};
  std::atomic_bool input_closed_{false};
  std::thread thread_;
  std::mutex mutex_;
  int lines_printed_ = 0;
};

}  // namespace glim

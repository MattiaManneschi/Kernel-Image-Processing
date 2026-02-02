#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cmath>
#include <fstream>





class Timer {
public:
    Timer() : running_(false), elapsed_(0.0) {}
    
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }
    
    void stop() {
        if (running_) {
            auto end_time = std::chrono::high_resolution_clock::now();
            elapsed_ += std::chrono::duration<double, std::milli>(end_time - start_time_).count();
            running_ = false;
        }
    }
    
    void reset() {
        elapsed_ = 0.0;
        running_ = false;
    }
    
    double elapsed_ms() const {
        if (running_) {
            auto now = std::chrono::high_resolution_clock::now();
            return elapsed_ + std::chrono::duration<double, std::milli>(now - start_time_).count();
        }
        return elapsed_;
    }
    
    double elapsed_seconds() const {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    bool running_;
    double elapsed_;
};





enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }
    
    void set_level(LogLevel level) { level_ = level; }
    void set_verbose(bool verbose) { verbose_ = verbose; }
    
    template<typename... Args>
    void debug(Args&&... args) {
        if (verbose_ && level_ <= LogLevel::DEBUG) {
            log("[DEBUG] ", std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    void info(Args&&... args) {
        if (level_ <= LogLevel::INFO) {
            log("[INFO] ", std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    void warning(Args&&... args) {
        if (level_ <= LogLevel::WARNING) {
            log("[WARNING] ", std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    void error(Args&&... args) {
        if (level_ <= LogLevel::ERROR) {
            log_err("[ERROR] ", std::forward<Args>(args)...);
        }
    }

private:
    Logger() : level_(LogLevel::INFO), verbose_(false) {}
    
    template<typename T, typename... Args>
    void log(T first, Args&&... args) {
        std::cout << first;
        if constexpr (sizeof...(args) > 0) {
            log(std::forward<Args>(args)...);
        } else {
            std::cout << std::endl;
        }
    }
    
    template<typename T, typename... Args>
    void log_err(T first, Args&&... args) {
        std::cerr << first;
        if constexpr (sizeof...(args) > 0) {
            log_err(std::forward<Args>(args)...);
        } else {
            std::cerr << std::endl;
        }
    }
    
    LogLevel level_;
    bool verbose_;
};


#define LOG_DEBUG(...) Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...) Logger::instance().info(__VA_ARGS__)
#define LOG_WARNING(...) Logger::instance().warning(__VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().error(__VA_ARGS__)





inline std::string format_time(double ms) {
    std::ostringstream oss;
    if (ms < 1.0) {
        oss << std::fixed << std::setprecision(3) << (ms * 1000.0) << " µs";
    } else if (ms < 1000.0) {
        oss << std::fixed << std::setprecision(3) << ms << " ms";
    } else {
        oss << std::fixed << std::setprecision(3) << (ms / 1000.0) << " s";
    }
    return oss.str();
}

inline std::string format_speedup(double speedup) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << speedup << "x";
    return oss.str();
}

inline std::string format_size(int width, int height) {
    std::ostringstream oss;
    oss << width << "x" << height;
    return oss.str();
}





template<typename T>
inline T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(value, max_val));
}

inline double compute_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / values.size();
}

inline double compute_stddev(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    double mean = compute_mean(values);
    double sum_sq = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / (values.size() - 1));
}

inline double compute_median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    }
    return values[n/2];
}





inline bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

inline std::string get_filename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

inline std::string get_extension(const std::string& path) {
    size_t pos = path.find_last_of('.');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return "";
}

inline std::string remove_extension(const std::string& path) {
    size_t pos = path.find_last_of('.');
    if (pos != std::string::npos) {
        return path.substr(0, pos);
    }
    return path;
}





class ProgressBar {
public:
    ProgressBar(int total, int width = 50) 
        : total_(total), current_(0), width_(width) {}
    
    void update(int current) {
        current_ = current;
        display();
    }
    
    void increment() {
        current_++;
        display();
    }
    
    void finish() {
        current_ = total_;
        display();
        std::cout << std::endl;
    }

private:
    void display() {
        float progress = static_cast<float>(current_) / total_;
        int filled = static_cast<int>(progress * width_);
        
        std::cout << "\r[";
        for (int i = 0; i < width_; i++) {
            if (i < filled) std::cout << "=";
            else if (i == filled) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "%";
        std::cout.flush();
    }
    
    int total_;
    int current_;
    int width_;
};

#endif 

#include "perf_cache.hpp"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

constexpr std::uint64_t kIntelLlcReferencesRaw = 0x4f2e;
constexpr std::uint64_t kIntelLlcMissesRaw = 0x412e;

std::atomic<bool> counters_enabled{true};
std::atomic<bool> warned_open_failure{false};

int perf_event_open(perf_event_attr *attr, pid_t pid, int cpu, int group_fd,
                    unsigned long flags) {
    return static_cast<int>(
        syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags));
}

void warn_open_failure(int error_number) {
    bool expected = false;
    if (!warned_open_failure.compare_exchange_strong(expected, true)) {
        return;
    }

    std::fprintf(stderr,
                 "[AMR-PERF] failed to open cache PMU counters (%s); "
                 "PMU cache columns will be -1\n",
                 std::strerror(error_number));
}

perf_event_attr make_event_attr(std::uint32_t type, std::uint64_t config) {
    perf_event_attr attr {};
    attr.type = type;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.inherit = 0;
    attr.read_format = PERF_FORMAT_GROUP |
                       PERF_FORMAT_TOTAL_TIME_ENABLED |
                       PERF_FORMAT_TOTAL_TIME_RUNNING;
    return attr;
}

std::uint64_t scaled_count(std::uint64_t value,
                           std::uint64_t time_enabled,
                           std::uint64_t time_running) {
    if (time_enabled == 0 || time_running == 0 || time_enabled == time_running) {
        return value;
    }

    const long double scaled =
        static_cast<long double>(value) *
        static_cast<long double>(time_enabled) /
        static_cast<long double>(time_running);
    const long double max_value =
        static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
    if (scaled >= max_value) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(std::llround(scaled));
}

struct GroupRead {
    std::uint64_t nr = 0;
    std::uint64_t time_enabled = 0;
    std::uint64_t time_running = 0;
    std::uint64_t values[2] = {0, 0};
};

class CounterGroup {
public:
    ~CounterGroup() {
        close_fds();
    }

    bool start() {
        if (!counters_enabled.load(std::memory_order_relaxed) || active) {
            return false;
        }
        if (!ensure_open()) {
            return false;
        }

        if (ioctl(reference_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
            return false;
        }
        if (ioctl(reference_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
            return false;
        }

        active = true;
        return true;
    }

    perf_cache::Delta stop() {
        perf_cache::Delta delta;
        if (!active) {
            return delta;
        }

        ioctl(reference_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        active = false;

        GroupRead data;
        const ssize_t bytes = read(reference_fd, &data, sizeof(data));
        if (bytes < static_cast<ssize_t>(sizeof(std::uint64_t) * 5) ||
            data.nr < 2) {
            return delta;
        }

        delta.valid = true;
        delta.references = scaled_count(data.values[0],
                                        data.time_enabled,
                                        data.time_running);
        delta.misses = scaled_count(data.values[1],
                                    data.time_enabled,
                                    data.time_running);
        delta.hits = delta.references >= delta.misses
                         ? delta.references - delta.misses
                         : 0;
        return delta;
    }

private:
    int reference_fd = -1;
    int miss_fd = -1;
    bool unavailable = false;
    bool active = false;

    bool ensure_open() {
        if (reference_fd >= 0 && miss_fd >= 0) {
            return true;
        }
        if (unavailable) {
            return false;
        }

        int error_number = 0;
        if (open_pair(PERF_TYPE_HARDWARE,
                      PERF_COUNT_HW_CACHE_REFERENCES,
                      PERF_COUNT_HW_CACHE_MISSES,
                      &error_number)) {
            return true;
        }

        if (open_pair(PERF_TYPE_RAW,
                      kIntelLlcReferencesRaw,
                      kIntelLlcMissesRaw,
                      &error_number)) {
            return true;
        }

        warn_open_failure(error_number == 0 ? EOPNOTSUPP : error_number);
        unavailable = true;
        return false;
    }

    bool open_pair(std::uint32_t type,
                   std::uint64_t reference_config,
                   std::uint64_t miss_config,
                   int *error_number) {
        close_fds();

        perf_event_attr reference_attr = make_event_attr(type, reference_config);
        reference_fd = perf_event_open(&reference_attr, 0, -1, -1, 0);
        if (reference_fd < 0) {
            if (error_number) {
                *error_number = errno;
            }
            return false;
        }

        perf_event_attr miss_attr = make_event_attr(type, miss_config);
        miss_fd = perf_event_open(&miss_attr, 0, -1, reference_fd, 0);
        if (miss_fd < 0) {
            if (error_number) {
                *error_number = errno;
            }
            close_fds();
            return false;
        }

        return true;
    }

    void close_fds() {
        if (miss_fd >= 0) {
            close(miss_fd);
            miss_fd = -1;
        }
        if (reference_fd >= 0) {
            close(reference_fd);
            reference_fd = -1;
        }
        active = false;
    }
};

thread_local CounterGroup thread_counters;

}  // namespace

namespace perf_cache {

void configure(bool enabled) {
    counters_enabled.store(enabled, std::memory_order_relaxed);
}

bool start() {
    return thread_counters.start();
}

Delta stop() {
    return thread_counters.stop();
}

}  // namespace perf_cache

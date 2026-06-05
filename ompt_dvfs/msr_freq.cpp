#include "msr_freq.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr int kMaxCpus = CPU_SETSIZE;

std::array<int, kMaxCpus> msr_fd_read;
std::array<int, kMaxCpus> msr_fd_write;
bool msr_initialized = false;

void init_msr_access() {
    if (msr_initialized) return;

    for (int cpu = 0; cpu < kMaxCpus; cpu++) {
        msr_fd_read[cpu] = -1;
        msr_fd_write[cpu] = -1;

        char path[64];
        std::snprintf(path, sizeof(path), "/dev/cpu/%d/msr", cpu);

        msr_fd_read[cpu] = open(path, O_RDONLY);
        msr_fd_write[cpu] = open(path, O_WRONLY);
    }

    msr_initialized = true;
}

int read_msr_on_cpu(int cpu, std::uint32_t msr, std::uint64_t* value) {
    if (!msr_initialized) init_msr_access();

    if (cpu < 0 || cpu >= kMaxCpus || msr_fd_read[cpu] < 0) return -1;

    ssize_t n = pread(msr_fd_read[cpu], value, sizeof(std::uint64_t), static_cast<off_t>(msr));
    if (n != static_cast<ssize_t>(sizeof(std::uint64_t))) return -2;
    return 0;
}

int write_msr_on_cpu(int cpu, std::uint32_t msr, std::uint64_t value) {
    if (!msr_initialized) init_msr_access();

    if (cpu < 0 || cpu >= kMaxCpus || msr_fd_write[cpu] < 0) return -1;

    ssize_t n = pwrite(msr_fd_write[cpu], &value, sizeof(std::uint64_t), static_cast<off_t>(msr));
    if (n != static_cast<ssize_t>(sizeof(std::uint64_t))) return -2;
    return 0;
}

bool hwp_available_on_cpu(int cpu) {
    std::uint64_t pm_enable = 0;
    if (read_msr_on_cpu(cpu, 0x770, &pm_enable) != 0) {
        return false;
    }
    return (pm_enable & 0x1ULL) != 0;
}

std::vector<int> get_allowed_cpus() {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        return {};
    }

    std::vector<int> target_cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) {
            target_cpus.push_back(cpu);
        }
    }

    return target_cpus;
}

}  // namespace

namespace msr {

void msr_init() {
    init_msr_access();
}

int current_cpu() {
    int cpu = sched_getcpu();
    if (cpu < 0) return -1;
    return cpu;
}

CounterSample sample() {
    int cpu = sched_getcpu();
    return sample_on_cpu(cpu);
}

CounterSample sample_on_cpu(int cpu) {
    CounterSample out{0ULL, 0ULL};
    if (cpu < 0) return out;

    std::uint64_t aperf = 0;
    std::uint64_t mperf = 0;
    if (read_msr_on_cpu(cpu, 0xE8, &aperf) != 0) aperf = 0;
    if (read_msr_on_cpu(cpu, 0xE7, &mperf) != 0) mperf = 0;

    out.aperf = aperf;
    out.mperf = mperf;
    return out;
}

double compute_freq_mhz(double base_mhz,
                        const CounterSample& first,
                        const CounterSample& second) {
    if (second.aperf <= first.aperf || second.mperf <= first.mperf) return -1.0;

    double ratio = static_cast<double>(second.aperf - first.aperf) /
                   static_cast<double>(second.mperf - first.mperf);
    return base_mhz * ratio;
}

double platform_base_mhz_on_cpu(int cpu, double bus_mhz) {
    std::uint64_t platform_info = 0;
    if (read_msr_on_cpu(cpu, 0xCE, &platform_info) != 0) return -1.0;

    const int max_non_turbo_ratio = static_cast<int>((platform_info >> 8) & 0xFFULL);
    if (max_non_turbo_ratio <= 0 || bus_mhz <= 0.0) return -1.0;

    return static_cast<double>(max_non_turbo_ratio) * bus_mhz;
}

bool set_freq_on_cpu(int cpu, double freq_mhz, double bus_mhz) {
    (void)bus_mhz;

    bool use_hwp = hwp_available_on_cpu(cpu);

    int ratio = 0xFF;
    if (freq_mhz <= 0.0) {
        std::uint64_t hwp_cap = 0;
        if (read_msr_on_cpu(cpu, 0x771, &hwp_cap) == 0) {
            ratio = static_cast<int>(hwp_cap & 0xFFULL);
        }
    } else {
        ratio = static_cast<int>(freq_mhz / 100.0);
    }

    if (use_hwp) {
        std::uint64_t hwp_value = 0;
        if (read_msr_on_cpu(cpu, 0x774, &hwp_value) == 0) {
            std::uint64_t new_hwp = (ratio & 0xFF) |
                                    ((static_cast<std::uint64_t>(ratio) & 0xFFULL) << 8) |
                                    ((static_cast<std::uint64_t>(ratio) & 0xFFULL) << 16) |
                                    (0x80ULL << 24);
            if (write_msr_on_cpu(cpu, 0x774, new_hwp) == 0) return true;
        }
    }

    std::uint64_t value = (static_cast<std::uint64_t>(ratio) << 8);
    if (write_msr_on_cpu(cpu, 0x199, value) != 0) {
        std::fprintf(stderr, "Failed to write MSR 0x199 on CPU %d\n", cpu);
        if (cpu >= 0 && cpu < kMaxCpus && msr_fd_write[cpu] < 0) {
            std::fprintf(stderr, "MSR write fd not open (root required)\n");
        }
        return false;
    }

    return true;
}


bool reset_freq_all() {
    bool result = true;
    auto target_cpus = get_allowed_cpus();
    for(int cpu: target_cpus) {
        //std::fprintf(stderr, "[MSR] Resetting frequency on CPU %d\n", cpu);
        if(!set_freq_on_cpu(cpu, 0, 100)) {
            std::fprintf(stderr, "[MSR] Failed to reset frequency on CPU %d\n", cpu);
            result = false;
        }
    }
    return result;
}

bool set_freq(double freq_mhz, double bus_mhz) {
    int cpu = current_cpu();
    if (cpu < 0) return false;
    return set_freq_on_cpu(cpu, freq_mhz, bus_mhz);
}

bool sysfs_set_freq_on_cpu(int cpu, int freq_khz) {
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",
                  cpu);

    FILE* fp = std::fopen(path, "w");
    if (!fp) {
        std::perror("Failed to open scaling_setspeed");
        return false;
    }

    if (std::fprintf(fp, "%d\n", freq_khz) < 0) {
        std::fclose(fp);
        return false;
    }

    std::fclose(fp);
    return true;
}

}  // namespace msr

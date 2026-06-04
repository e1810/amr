#ifndef MSR_FREQ_HPP
#define MSR_FREQ_HPP

#include <cstdint>
#include <vector>

namespace msr {

void msr_init();

struct CounterSample {
    std::uint64_t aperf;
    std::uint64_t mperf;
};

int current_cpu();
CounterSample sample_on_cpu(int cpu);
CounterSample sample();
double compute_freq_mhz(double base_mhz,
                        const CounterSample& first,
                        const CounterSample& second);

bool set_freq_on_cpu(int cpu, double freq_mhz, double bus_mhz = 100.0);
bool set_freq_multi(const std::vector<int>& cpu_list,
                    double freq_mhz,
                    double bus_mhz = 100.0);
bool set_freq(double freq_mhz, double bus_mhz = 100.0);
bool reset_freq_all();
bool sysfs_set_freq_on_cpu(int cpu, int freq_khz);

}  // namespace msr

#endif

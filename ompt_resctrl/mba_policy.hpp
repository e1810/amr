#ifndef AMR_OMPT_RESCTRL_MBA_POLICY_HPP
#define AMR_OMPT_RESCTRL_MBA_POLICY_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mba_policy {

// Metrics captured during one region execution and used to plan the next one.
struct History {
    std::vector<double> elapsed_ms;
    std::vector<double> target_mb_percent;
    std::vector<unsigned char> cache_counter_valid;
    std::vector<std::uint64_t> cache_references;
    std::vector<std::uint64_t> cache_misses;
};

struct Config {
    // Values above 1.0 make PMU pressure penalties more aggressive.
    double strictness = 1.15;
};

Config strict_config();

// The slowest measured thread remains unrestricted. Other threads receive a
// relaxed level that targets the slowest thread's elapsed time without pushing
// non-critical threads too aggressively toward the lowest MBA levels.
std::vector<int> plan_levels(const History &history,
                             std::size_t slots,
                             int level_count,
                             const Config &config);

}  // namespace mba_policy
#endif

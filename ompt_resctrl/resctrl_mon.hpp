#ifndef RESCTRL_MON_HPP
#define RESCTRL_MON_HPP

#include <cstdint>
#include <string>

namespace resctrl_mon {

struct Sample {
    bool valid = false;
    int domain_count = 0;
    std::int64_t llc_occupancy_bytes = -1;
    std::int64_t mbm_total_bytes = -1;
    std::int64_t mbm_local_bytes = -1;
    std::int64_t mbm_remote_bytes = -1;
};

void configure(const char *root_path, const char *group_prefix);
Sample sample_slot(int slot);

std::string root_path();
std::string group_prefix();

}  // namespace resctrl_mon

#endif

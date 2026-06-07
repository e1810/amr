#ifndef RESCTRL_CAT_HPP
#define RESCTRL_CAT_HPP

#include <cstdint>
#include <sys/types.h>

#include <string>
#include <vector>

namespace resctrl_cat {

struct L3Info {
    bool available = false;
    int min_cbm_bits = 1;
    int max_cbm_bits = 0;
    std::uint64_t cbm_mask = 0;
    std::vector<int> domains;
};

struct AssignmentResult {
    bool applied = false;
    int cpu = -1;
    pid_t tid = -1;
    int ways = 0;
    std::string mask;
};

void configure(const char *root_path, const char *group_prefix);
L3Info l3_info(int fallback_min_cbm_bits);
bool control_available();

AssignmentResult assign_current_thread_l3(int slot, int requested_ways);
AssignmentResult assign_current_thread_l3_mask(int slot, const std::string &requested_mask);
bool release_task(pid_t tid);
bool cleanup_created_groups();

std::string root_path();
std::string group_prefix();
std::string mask_for_ways_text(int requested_ways);
int effective_ways(int requested_ways);

}  // namespace resctrl_cat

#endif

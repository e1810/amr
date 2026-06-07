#ifndef RESCTRL_MBA_HPP
#define RESCTRL_MBA_HPP

#include <sys/types.h>

#include <string>
#include <vector>

namespace resctrl {

struct MbaInfo {
    bool available = false;
    int min_percent = 10;
    int granularity_percent = 10;
    int max_percent = 100;
    std::vector<int> domains;
};

struct AssignmentResult {
    bool applied = false;
    int cpu = -1;
    pid_t tid = -1;
};

void configure(const char *root_path, const char *group_prefix);
MbaInfo mba_info(int fallback_min_percent,
                 int fallback_granularity_percent,
                 int max_percent);
bool control_available();

int current_cpu();
pid_t current_tid();

AssignmentResult assign_current_thread_mba(int slot, double percent);
bool release_task(pid_t tid);
bool cleanup_created_groups();

std::string root_path();
std::string group_prefix();

}  // namespace resctrl

#endif
